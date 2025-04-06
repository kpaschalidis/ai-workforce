"""
Planning pattern implementation using LangGraph directly.

This module provides an implementation of a planning-based reasoning and
execution pattern for agents, where the agent creates a plan, executes each step,
and evaluates its progress.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END

from .config import PatternConfig
from src.core.infrastructure.events import AgentEventType
from .base import BaseAgentPattern


class PlanningPattern(BaseAgentPattern):
    """
    Implements a planning-based execution pattern for agents.

    This pattern follows a planning approach:
    1. Create a plan with multiple steps
    2. Execute each step in sequence (potentially using tools)
    3. Evaluate the execution and determine if replanning is needed
    4. Continue until plan is complete or replanning is needed
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Any] = None,
        config: Optional[Union[Dict[str, Any], PatternConfig]] = None,
        **kwargs,
    ):
        """
        Initialize the PlanningPattern.

        Args:
            llm: Language model to use
            tools: List of tools available to the pattern
            config: Configuration dictionary
            **kwargs: Additional arguments for BaseAgentPattern
        """
        super().__init__(llm=llm, tools=tools, config=config, **kwargs)

        # Get configuration values
        self.max_plan_steps = self.config.get("max_plan_steps", 5)
        self.max_iterations = self.config.get("max_iterations", 10)
        self.stop_on_error = self.config.get("stop_on_error", False)

    def build_graph(self) -> StateGraph:
        """
        Build and return the LangGraph state graph for Planning pattern.

        Returns:
            Configured StateGraph for Planning pattern
        """
        # Create a new state graph
        graph = StateGraph(self.state_type)

        # Add nodes for the planning cycle
        graph.add_node("plan", self.execute_planning_step)
        graph.add_node("execute", self.execute_plan_step)
        graph.add_node("evaluate", self.evaluate_plan_step)

        # Add conditional edge from planning
        # If planning creates a valid plan, go to execute, otherwise end
        graph.add_conditional_edges(
            "plan", self._has_plan, {True: "execute", False: END}
        )

        # Execute step leads to evaluate
        graph.add_edge("execute", "evaluate")

        # Add conditional edge from evaluate based on evaluation result
        graph.add_conditional_edges(
            "evaluate",
            self._evaluate_condition,
            {
                "replan": "plan",  # Need to create a new plan
                "continue": "execute",  # Continue with next step in current plan
                "complete": END,  # Plan is complete, end workflow
            },
        )

        # Set entry point
        graph.set_entry_point("plan")

        return graph

    def _has_plan(self, state: Any) -> bool:
        """
        Check if the state has a valid plan.

        Args:
            state: Current state

        Returns:
            True if state has a valid plan, False otherwise
        """
        return hasattr(state, "plan") and len(state.plan) > 0

    def _evaluate_condition(self, state: Any) -> str:
        """
        Determine next action based on evaluation results.

        Args:
            state: Current state

        Returns:
            "replan", "continue", or "complete"
        """
        # Check if replanning is needed
        if self.needs_replanning(state):
            return "replan"

        # Check if plan is complete
        if self.is_plan_complete(state):
            return "complete"

        # Otherwise continue with next step
        return "continue"

    def execute_planning_step(self, state: Any) -> Any:
        """
        Execute the planning step to create a plan.

        Args:
            state: Current state

        Returns:
            Updated state with plan
        """
        self._publish_event(AgentEventType.STEP_STARTED, {"step": "planning"})

        try:
            # Initialize or reset plan-related state
            if not hasattr(state, "plan"):
                state.plan = []

            if not hasattr(state, "current_step_index"):
                state.current_step_index = 0

            if not hasattr(state, "evaluations"):
                state.evaluations = []

            if not hasattr(state, "iteration_count"):
                state.iteration_count = 0

            # Increment iteration count
            state.iteration_count += 1

            # Check for max iterations
            if state.iteration_count > self.max_iterations:
                state.add_system_message(
                    f"Reached maximum iterations ({self.max_iterations}). Completing with best response so far."
                )
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "planning", "status": "max_iterations"},
                )
                return state

            # Add planning prompt to state
            planning_prompt = self._get_planning_prompt()
            state.add_system_message(planning_prompt)

            # Get messages from state
            messages = state.get_messages()

            # Call LLM to generate plan
            self._publish_event(
                AgentEventType.BEFORE_LLM_CALL, {"messages": str(messages)}
            )

            response = self.llm.invoke(messages)

            self._publish_event(
                AgentEventType.AFTER_LLM_CALL, {"response": response.content}
            )

            # Add assistant message to state
            state.add_assistant_message(response.content)

            # Extract plan from LLM response
            plan_steps = self._extract_plan_from_text(response.content)

            # Validate and limit plan steps
            if len(plan_steps) > self.max_plan_steps:
                self.logger.warning(
                    f"Plan has {len(plan_steps)} steps, limiting to {self.max_plan_steps}"
                )
                plan_steps = plan_steps[: self.max_plan_steps]
                state.add_system_message(
                    f"Plan too long, limiting to {self.max_plan_steps} steps."
                )

            # Reset current step index and plan
            state.plan = plan_steps
            state.current_step_index = 0

            # Log the plan
            plan_summary = "\n".join(
                [f"{i+1}. {step['description']}" for i, step in enumerate(plan_steps)]
            )
            self.logger.info(
                f"Generated plan with {len(plan_steps)} steps:\n{plan_summary}"
            )

            self._publish_event(
                AgentEventType.STEP_COMPLETED,
                {"step": "planning", "plan_steps": len(plan_steps)},
            )
            return state

        except Exception as e:
            self._handle_error(e, {"step": "planning"})

            # Add error message to state
            state.add_system_message(f"Error during planning: {str(e)}")

            # Create a simple plan even in case of error
            state.plan = [
                {
                    "id": 1,
                    "description": "Answer the user's question directly with available information",
                    "tool": None,
                    "completed": False,
                }
            ]
            state.current_step_index = 0

            self._publish_event(
                AgentEventType.STEP_COMPLETED, {"step": "planning", "status": "error"}
            )
            return state

    def execute_plan_step(self, state: Any) -> Any:
        """
        Execute a step in the plan.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        self._publish_event(AgentEventType.STEP_STARTED, {"step": "execute_plan"})

        try:
            # Get current step
            if not hasattr(state, "plan") or not state.plan:
                self.logger.warning("No plan found in state")
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "execute_plan", "status": "no_plan"},
                )
                return state

            current_step_index = getattr(state, "current_step_index", 0)
            if current_step_index >= len(state.plan):
                self.logger.warning("Current step index out of bounds")
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "execute_plan", "status": "invalid_step"},
                )
                return state

            current_step = state.plan[current_step_index]

            # Log the current step
            self.logger.info(
                f"Executing plan step {current_step_index + 1}: {current_step['description']}"
            )

            # Add message about executing step
            state.add_system_message(
                f"Executing step {current_step_index + 1}: {current_step['description']}"
            )

            # If step has a tool specified, try to execute it
            if current_step.get("tool"):
                # Try to execute specified tool
                result = self._execute_tool_for_plan(state, current_step)
                if result is not None:
                    # Tool executed successfully
                    state.current_step_evaluated = False
                    self._publish_event(
                        AgentEventType.STEP_COMPLETED,
                        {"step": "execute_plan", "status": "tool_executed"},
                    )
                    return state

            # If no tool or tool execution failed, ask LLM to execute step
            state.add_system_message(
                f"Please execute step {current_step_index + 1} of the plan: {current_step['description']}"
            )

            # Get messages from state
            messages = state.get_messages()

            # Call LLM
            self._publish_event(
                AgentEventType.BEFORE_LLM_CALL, {"messages": str(messages)}
            )
            response = self.llm.invoke(messages)
            self._publish_event(
                AgentEventType.AFTER_LLM_CALL, {"response": response.content}
            )

            # Add assistant message to state
            state.add_assistant_message(response.content)

            # Check if the response includes a tool call
            tool_call = self._extract_tool_call(response.content)
            if tool_call and self.tools:
                tool_name, tool_params = tool_call

                # Check if tool exists
                tool = self.get_tool_by_name(tool_name)
                if tool:
                    # Execute the tool
                    self._publish_event(
                        AgentEventType.BEFORE_TOOL_CALL,
                        {"tool": tool_name, "parameters": tool_params},
                    )

                    try:
                        # Run the tool
                        result = tool.run(**tool_params)

                        # Publish success event
                        self._publish_event(
                            AgentEventType.AFTER_TOOL_CALL,
                            {"tool": tool_name, "result": str(result)},
                        )

                        # Add tool message to state
                        state.add_tool_message(tool_name, str(result))

                    except Exception as tool_error:
                        # Tool execution error
                        error_msg = (
                            f"Error executing tool '{tool_name}': {str(tool_error)}"
                        )
                        self.logger.error(error_msg)
                        state.add_tool_message(tool_name, f"Error: {str(tool_error)}")

                        # Publish tool error event
                        self._publish_event(
                            AgentEventType.TOOL_ERROR,
                            {"tool": tool_name, "error": str(tool_error)},
                        )

            # Mark step as ready for evaluation
            state.current_step_evaluated = False

            self._publish_event(AgentEventType.STEP_COMPLETED, {"step": "execute_plan"})
            return state

        except Exception as e:
            self._handle_error(e, {"step": "execute_plan"})

            # Add error message to state
            state.add_system_message(f"Error executing plan step: {str(e)}")

            # Mark step as ready for evaluation
            state.current_step_evaluated = False

            self._publish_event(
                AgentEventType.STEP_COMPLETED,
                {"step": "execute_plan", "status": "error"},
            )
            return state

    def evaluate_plan_step(self, state: Any) -> Any:
        """
        Evaluate the execution of a plan step.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        self._publish_event(AgentEventType.STEP_STARTED, {"step": "evaluate_plan"})

        try:
            # Get current step
            if not hasattr(state, "plan") or not state.plan:
                self.logger.warning("No plan found in state")
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "evaluate_plan", "status": "no_plan"},
                )
                return state

            current_step_index = getattr(state, "current_step_index", 0)
            if current_step_index >= len(state.plan):
                self.logger.warning("Current step index out of bounds")
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "evaluate_plan", "status": "invalid_step"},
                )
                return state

            # Mark current step as completed
            state.plan[current_step_index]["completed"] = True

            # Add evaluation prompt
            evaluation_prompt = """
Evaluate the execution of the current plan step:
1. Did the step complete successfully?
2. Have we gained the information needed?
3. Should we proceed to the next step, or do we need to modify our plan?

Respond with one of the following:
- CONTINUE: If we should proceed to the next step
- REPLAN: If we need to create a new plan
- COMPLETE: If the plan is complete and we can provide a final answer
"""
            state.add_system_message(evaluation_prompt)

            # Get messages from state
            messages = state.get_messages()

            # Call LLM for evaluation
            self._publish_event(
                AgentEventType.BEFORE_LLM_CALL, {"messages": str(messages)}
            )
            response = self.llm.invoke(messages)
            self._publish_event(
                AgentEventType.AFTER_LLM_CALL, {"response": response.content}
            )

            # Add assistant message to state
            state.add_assistant_message(response.content)

            # Process evaluation result
            if "REPLAN" in response.content:
                # Need to replan
                self.logger.info("Evaluation result: REPLAN")
                state.need_replan = True
                state.add_system_message("Replanning needed based on evaluation.")

            elif "COMPLETE" in response.content:
                # Plan is complete
                self.logger.info("Evaluation result: COMPLETE")
                state.current_step_index = len(state.plan)  # Mark as complete
                state.add_system_message(
                    "Plan execution complete. Please provide a final comprehensive answer."
                )

                # Generate final response
                messages = state.get_messages()
                self._publish_event(
                    AgentEventType.BEFORE_LLM_CALL, {"messages": str(messages)}
                )
                final_response = self.llm.invoke(messages)
                self._publish_event(
                    AgentEventType.AFTER_LLM_CALL, {"response": final_response.content}
                )
                state.add_assistant_message(final_response.content)

            else:
                # Continue to next step
                self.logger.info("Evaluation result: CONTINUE")
                state.current_step_index += 1

                # Check if we've completed all steps
                if state.current_step_index >= len(state.plan):
                    self.logger.info("All plan steps completed")
                    state.add_system_message(
                        "All plan steps completed. Please provide a final comprehensive answer."
                    )

                    # Generate final response
                    messages = state.get_messages()
                    self._publish_event(
                        AgentEventType.BEFORE_LLM_CALL, {"messages": str(messages)}
                    )
                    final_response = self.llm.invoke(messages)
                    self._publish_event(
                        AgentEventType.AFTER_LLM_CALL,
                        {"response": final_response.content},
                    )
                    state.add_assistant_message(final_response.content)

            # Mark as evaluated
            state.current_step_evaluated = True

            # Store evaluation result
            if not hasattr(state, "evaluations"):
                state.evaluations = []

            state.evaluations.append(
                {
                    "step_index": current_step_index,
                    "step_description": state.plan[current_step_index]["description"],
                    "evaluation": response.content,
                }
            )

            self._publish_event(
                AgentEventType.STEP_COMPLETED, {"step": "evaluate_plan"}
            )
            return state

        except Exception as e:
            self._handle_error(e, {"step": "evaluate_plan"})

            # Add error message to state
            state.add_system_message(f"Error evaluating plan step: {str(e)}")

            # Move to next step even in case of error, to avoid getting stuck
            if hasattr(state, "current_step_index"):
                state.current_step_index += 1

            # Mark as evaluated
            state.current_step_evaluated = True

            self._publish_event(
                AgentEventType.STEP_COMPLETED,
                {"step": "evaluate_plan", "status": "error"},
            )
            return state

    def is_plan_complete(self, state: Any) -> bool:
        """
        Check if the plan is complete.

        Args:
            state: Current state

        Returns:
            True if the plan is complete, False otherwise
        """
        if not hasattr(state, "plan") or not state.plan:
            return True

        if not hasattr(state, "current_step_index"):
            return False

        return state.current_step_index >= len(state.plan)

    def needs_replanning(self, state: Any) -> bool:
        """
        Check if replanning is needed.

        Args:
            state: Current state

        Returns:
            True if replanning is needed, False otherwise
        """
        return hasattr(state, "need_replan") and state.need_replan

    def _extract_plan_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract a plan from text.

        Args:
            text: Text to extract plan from

        Returns:
            List of plan steps
        """
        # Try to extract numbered steps
        steps = []

        # Look for numbered steps like "1. Step description" or "Step 1: Description"
        step_patterns = [
            r"\b(\d+)[\.\)] (.+?)(?=\n\d+[\.\)]|\Z)",  # "1. Step" or "1) Step"
            r"Step (\d+)[\:\-] (.+?)(?=\nStep \d+[\:\-]|\Z)",  # "Step 1: Step" or "Step 1 - Step"
            r"\n\s*(\d+)[\.\)] (.+?)(?=\n\s*\d+[\.\)]|\Z)",  # Indented "1. Step"
        ]

        all_matches = []
        for pattern in step_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            all_matches.extend([(int(m.group(1)), m.group(2).strip()) for m in matches])

        # If no matches, try to split by double newlines
        if not all_matches:
            paragraphs = text.split("\n\n")
            all_matches = [
                (i + 1, p.strip()) for i, p in enumerate(paragraphs) if p.strip()
            ]

        # Sort by step number
        all_matches.sort()

        # Convert to plan steps
        for step_num, description in all_matches:
            # Try to extract tool name if mentioned
            tool_name = None
            tool_match = re.search(r"use tool[:\s]+(\w+)", description, re.IGNORECASE)
            if tool_match:
                tool_name = tool_match.group(1)

            steps.append(
                {
                    "id": step_num,
                    "description": description,
                    "tool": tool_name,
                    "completed": False,
                }
            )

        return steps

    def _extract_tool_call(self, message: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Extract a tool call from a message.

        Args:
            message: Message to extract tool call from

        Returns:
            Tuple of (tool_name, parameters) if a tool call is found, None otherwise
        """
        if not message:
            return None

        # Try to extract tool name
        tool_match = re.search(r"Tool:\s*(\w+)", message, re.IGNORECASE)
        if not tool_match:
            return None

        tool_name = tool_match.group(1)

        # Try to extract parameters
        params_match = re.search(
            r"Parameters:\s*(\{.*?\})", message, re.DOTALL | re.IGNORECASE
        )
        parameters = {}

        if params_match:
            try:
                params_text = params_match.group(1)
                parameters = json.loads(params_text)
            except json.JSONDecodeError:
                # Try to extract parameters with regex as fallback
                param_matches = re.finditer(r'"(\w+)":\s*"([^"]*)"', params_text)
                for match in param_matches:
                    key, value = match.groups()
                    parameters[key] = value

        return tool_name, parameters

    def _get_planning_prompt(self) -> str:
        """
        Get the planning prompt for the LLM.

        Returns:
            Planning prompt
        """
        return f"""
Please create a step-by-step plan to answer the user's request.

For each step, include:
1. A clear description of what should be done
2. If a specific tool should be used, mention it explicitly

Use at most {self.max_plan_steps} steps, and focus on breaking down the task in a logical sequence.
Format each step as "Step 1: Description", "Step 2: Description", etc.
"""

    def _execute_tool_for_plan(self, state: Any, step: Dict[str, Any]) -> Optional[str]:
        """
        Execute a tool for a plan step.

        Args:
            state: Current state
            step: Plan step with tool specification

        Returns:
            Tool result or None if tool execution failed
        """
        tool_name = step.get("tool")
        if not tool_name or not self.tools:
            return None

        # Check if tool exists
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            state.add_system_message(f"Tool '{tool_name}' specified in plan not found.")
            return None

        # Need to get parameters for the tool
        tool_prompt = f"""
To execute the current plan step: "{step['description']}"
I need to use the tool: {tool_name}

Please provide the parameters for the tool in JSON format.
Respond with ONLY the JSON object, nothing else.
"""
        state.add_system_message(tool_prompt)

        # Get messages
        messages = state.get_messages()

        # Call LLM to get parameters
        self._publish_event(AgentEventType.BEFORE_LLM_CALL, {"messages": str(messages)})
        response = self.llm.invoke(messages)
        self._publish_event(
            AgentEventType.AFTER_LLM_CALL, {"response": response.content}
        )

        # Add assistant message to state
        state.add_assistant_message(response.content)

        # Extract JSON parameters
        try:
            # Find JSON object in response
            json_match = re.search(r"({.*})", response.content, re.DOTALL)
            if not json_match:
                return None

            params_json = json_match.group(1)
            tool_params = json.loads(params_json)

            # Execute the tool
            self._publish_event(
                AgentEventType.BEFORE_TOOL_CALL,
                {"tool": tool_name, "parameters": tool_params},
            )

            # Run the tool
            result = tool.run(**tool_params)

            # Publish success event
            self._publish_event(
                AgentEventType.AFTER_TOOL_CALL,
                {"tool": tool_name, "result": str(result)},
            )

            # Add tool message to state
            state.add_tool_message(tool_name, str(result))

            return result

        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            self.logger.error(error_msg)
            state.add_tool_message(tool_name, f"Error: {str(e)}")

            # Publish tool error event
            self._publish_event(
                AgentEventType.TOOL_ERROR, {"tool": tool_name, "error": str(e)}
            )

            return None
