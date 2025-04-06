"""
ReAct pattern implementation using LangGraph directly.

This module provides an implementation of the ReAct pattern for agent reasoning
and execution, which alternates between reasoning steps and action steps.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END

from .config import PatternConfig
from src.core.infrastructure.events import AgentEventType
from .base import BaseAgentPattern


class ReactPattern(BaseAgentPattern):
    """
    Implements the ReAct (Reasoning + Acting) pattern for agent execution.

    The ReAct pattern alternates between:
    1. Reasoning: Using the LLM to analyze the situation and determine what to do
    2. Acting: Executing tools based on the reasoning

    This creates a cycle of reasoning and acting until a resolution is reached.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Any] = None,
        config: Optional[Union[Dict[str, Any], PatternConfig]] = None,
        **kwargs,
    ):
        """
        Initialize the ReactPattern.

        Args:
            llm: Language model to use
            tools: List of tools available to the pattern
            config: Configuration dictionary
            **kwargs: Additional arguments for BaseAgentPattern
        """
        super().__init__(llm=llm, tools=tools, config=config, **kwargs)

        # Get configuration values
        self.max_iterations = self.config.get("max_iterations", 10)
        self.stop_on_error = self.config.get("stop_on_error", False)

    def build_graph(self) -> StateGraph:
        """
        Build and return the LangGraph state graph for ReAct pattern.

        Returns:
            Configured StateGraph for ReAct pattern
        """
        # Create a new state graph
        graph = StateGraph(self.state_type)

        # Add nodes for reasoning and action
        graph.add_node("reasoning", self.execute_reasoning_step)
        graph.add_node("action", self.execute_action_step)

        # Add conditional edge from reasoning based on tool detection
        graph.add_conditional_edges(
            "reasoning", self.should_execute_tool, {True: "action", False: END}
        )

        # Add edge from action back to reasoning
        graph.add_edge("action", "reasoning")

        # Add iteration check if max_iterations is set
        if self.max_iterations > 0:
            # Add node for checking iteration count
            graph.add_node("check_iterations", self._check_iterations)

            # Modify the action node to go to check_iterations instead of directly to reasoning
            # Remove the direct edge from action to reasoning
            # (We need to recreate the graph since we cannot modify edges)
            graph = StateGraph(self.state_type)
            graph.add_node("reasoning", self.execute_reasoning_step)
            graph.add_node("action", self.execute_action_step)
            graph.add_node("check_iterations", self._check_iterations)

            graph.add_conditional_edges(
                "reasoning", self.should_execute_tool, {True: "action", False: END}
            )

            # Action goes to check_iterations
            graph.add_edge("action", "check_iterations")

            # Add conditional edge from check_iterations
            graph.add_conditional_edges(
                "check_iterations",
                self._max_iterations_reached,
                {True: END, False: "reasoning"},
            )

        # Set entry point
        graph.set_entry_point("reasoning")

        return graph

    def _check_iterations(self, state: Any) -> Any:
        """
        Check and update the iteration count.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        # Initialize counter if it doesn't exist
        if not hasattr(state, "iteration_count"):
            state.iteration_count = 0

        # Increment counter
        state.iteration_count += 1

        return state

    def _max_iterations_reached(self, state: Any) -> bool:
        """
        Check if maximum iterations have been reached.

        Args:
            state: Current state

        Returns:
            True if max iterations reached, False otherwise
        """
        iteration_count = getattr(state, "iteration_count", 0)
        return iteration_count >= self.max_iterations

    def execute_reasoning_step(self, state: Any) -> Any:
        """
        Execute the reasoning step of the ReAct pattern.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        self._publish_event(AgentEventType.STEP_STARTED, {"step": "reasoning"})

        try:
            # Get messages from state
            messages = state.get_messages()

            # Publish event before LLM call
            self._publish_event(
                AgentEventType.BEFORE_LLM_CALL, {"messages": str(messages)}
            )

            # Call LLM for reasoning
            response = self.llm.invoke(messages)

            # Publish event after LLM call
            self._publish_event(
                AgentEventType.AFTER_LLM_CALL, {"response": response.content}
            )

            # Add assistant message to state
            state.add_assistant_message(response.content)

            # Check if we've reached the max iterations
            if hasattr(state, "iteration_count"):
                self.logger.debug(
                    f"Iteration {state.iteration_count} of {self.max_iterations}"
                )

                if state.iteration_count >= self.max_iterations:
                    self.logger.warning(
                        f"Reached maximum iterations ({self.max_iterations})"
                    )
                    state.add_system_message(
                        f"Reached maximum iterations ({self.max_iterations}). Completing with the best response so far."
                    )

            self._publish_event(AgentEventType.STEP_COMPLETED, {"step": "reasoning"})
            return state

        except Exception as e:
            self._handle_error(e, {"step": "reasoning"})

            # Add error message to state
            state.add_system_message(f"Error during reasoning: {str(e)}")

            # Stop if configured to do so
            if self.stop_on_error:
                state.add_system_message("Stopping due to error.")
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "reasoning", "status": "error"},
                )
                return state

            # Otherwise try to continue
            self._publish_event(
                AgentEventType.STEP_COMPLETED,
                {"step": "reasoning", "status": "error_continue"},
            )
            return state

    def execute_action_step(self, state: Any) -> Any:
        """
        Execute the action step of the ReAct pattern.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        self._publish_event(AgentEventType.STEP_STARTED, {"step": "action"})

        try:
            # Extract tool call from the last assistant message
            tool_call = self._extract_tool_call(state.get_last_assistant_message())

            if not tool_call:
                self.logger.warning("No valid tool call found in assistant message")
                state.add_system_message(
                    "I couldn't identify a valid tool call. Please specify a tool name and parameters clearly."
                )
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "action", "status": "no_tool"},
                )
                return state

            tool_name, tool_params = tool_call

            # Check if tool exists
            tool = self.get_tool_by_name(tool_name)
            if not tool:
                error_msg = f"Tool '{tool_name}' not found. Available tools: {', '.join(t.name for t in self.tools)}"
                self.logger.warning(error_msg)
                state.add_tool_message("error", error_msg)
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "action", "status": "invalid_tool"},
                )
                return state

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
                error_msg = f"Error executing tool '{tool_name}': {str(tool_error)}"
                self.logger.error(error_msg)
                state.add_tool_message(tool_name, f"Error: {str(tool_error)}")

                # Publish tool error event
                self._publish_event(
                    AgentEventType.TOOL_ERROR,
                    {"tool": tool_name, "error": str(tool_error)},
                )

                # Stop if configured to do so
                if self.stop_on_error:
                    state.add_system_message("Stopping due to tool execution error.")
                    self._publish_event(
                        AgentEventType.STEP_COMPLETED,
                        {"step": "action", "status": "error_stop"},
                    )
                    return state

            self._publish_event(AgentEventType.STEP_COMPLETED, {"step": "action"})
            return state

        except Exception as e:
            self._handle_error(e, {"step": "action"})

            # Add error message to state
            state.add_system_message(f"Error during action execution: {str(e)}")

            # Stop if configured to do so
            if self.stop_on_error:
                state.add_system_message("Stopping due to error.")
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "action", "status": "error_stop"},
                )
                return state

            # Otherwise try to continue
            self._publish_event(
                AgentEventType.STEP_COMPLETED,
                {"step": "action", "status": "error_continue"},
            )
            return state

    def should_execute_tool(self, state: Any) -> bool:
        """
        Determine if a tool should be executed.

        Args:
            state: Current state

        Returns:
            True if a tool should be executed, False otherwise
        """
        # If there are no tools, never execute
        if not self.tools:
            return False

        # Get the last assistant message
        assistant_message = state.get_last_assistant_message()
        if not assistant_message:
            return False

        # Check if a valid tool call is present
        return self._extract_tool_call(assistant_message) is not None

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

    def _get_execution_instructions(self) -> str:
        """
        Get ReAct-specific instructions for the system prompt.

        Returns:
            ReAct instructions
        """
        return """
When you need to use a tool, respond in the following format:

Tool: <tool_name>
Parameters: {
    "param1": "value1",
    "param2": "value2"
}

The output of the tool will be provided to you for further processing.

When you don't need to use a tool and want to respond directly to the user,
simply provide your response without the Tool/Parameters format.
"""
