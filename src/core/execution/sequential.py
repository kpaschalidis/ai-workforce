"""
Sequential pattern implementation using LangGraph directly.

This module provides an implementation of a sequential execution pattern for agents,
where steps are performed in a predetermined, linear order.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END

from .config import PatternConfig
from src.core.infrastructure.events import AgentEventType
from .base import BaseAgentPattern


class SequentialPattern(BaseAgentPattern):
    """
    Implements a sequential execution pattern for agents.

    This pattern follows a linear, predetermined series of steps:
    1. Parse input
    2. Generate initial reasoning with LLM
    3. Execute tools (if needed)
    4. Generate final response with LLM
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Any] = None,
        config: Optional[Union[Dict[str, Any], PatternConfig]] = None,
        **kwargs,
    ):
        """
        Initialize the SequentialPattern.

        Args:
            llm: Language model to use
            tools: List of tools available to the pattern
            config: Configuration dictionary
            **kwargs: Additional arguments for BaseAgentPattern
        """
        super().__init__(llm=llm, tools=tools, config=config, **kwargs)

        # Get configuration values
        self.stop_on_error = self.config.get("stop_on_error", False)
        self.always_use_tools = self.config.get("always_use_tools", False)

    def build_graph(self) -> StateGraph:
        """
        Build and return the LangGraph state graph for Sequential pattern.

        Returns:
            Configured StateGraph for Sequential pattern
        """
        # Create a new state graph
        graph = StateGraph(self.state_type)

        # Add nodes for each step in the sequence
        graph.add_node("parse_input", self.execute_parse_step)
        graph.add_node("initial_reasoning", self.execute_reasoning_step)

        # Only add tool execution if tools are available
        if self.tools:
            graph.add_node("tool_execution", self.execute_tool_step)

        graph.add_node("final_response", self.execute_response_step)

        # Create edges for the sequence
        graph.add_edge("parse_input", "initial_reasoning")

        if self.tools:
            # Add conditional edge for tool execution based on whether tools are needed
            graph.add_conditional_edges(
                "initial_reasoning",
                self._should_use_tools,
                {True: "tool_execution", False: "final_response"},
            )
            graph.add_edge("tool_execution", "final_response")
        else:
            # Direct edge if no tools
            graph.add_edge("initial_reasoning", "final_response")

        # Final response leads to end
        graph.add_edge("final_response", END)

        # Set entry point
        graph.set_entry_point("parse_input")

        return graph

    def _should_use_tools(self, state: Any) -> bool:
        """
        Determine if tools should be used.

        Args:
            state: Current state

        Returns:
            True if tools should be used, False otherwise
        """
        # If always_use_tools is enabled, always return True
        if self.always_use_tools:
            return True

        # Otherwise, check the context or detect from the last assistant message
        use_tools = state.get_from_context("use_tools", False)

        if not use_tools:
            # Try to detect tool call in last assistant message
            use_tools = self.should_execute_tool(state)

        return use_tools

    def execute_parse_step(self, state: Any) -> Any:
        """
        Execute the input parsing step.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        self._publish_event(AgentEventType.STEP_STARTED, {"step": "parse_input"})

        try:
            # Get and process the user message
            user_message = state.get_last_user_message()
            self.logger.debug(f"Parsing input: {user_message}")

            # Add input analysis to state context if needed
            state.add_to_context(
                "input_length", len(user_message) if user_message else 0
            )

            # Check if we have a system message, add one if not
            if not state.has_system_message():
                system_prompt = self.get_system_prompt()
                state.add_system_message(system_prompt)

            self._publish_event(AgentEventType.STEP_COMPLETED, {"step": "parse_input"})
            return state

        except Exception as e:
            self._handle_error(e, {"step": "tool_execution"})

            # Add error message to state
            state.add_system_message(f"Error during tool execution: {str(e)}")

            # Stop if configured to do so
            if self.stop_on_error:
                state.add_system_message("Stopping due to error.")
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "tool_execution", "status": "error_stop"},
                )
                return state

            # Otherwise try to continue
            self._publish_event(
                AgentEventType.STEP_COMPLETED,
                {"step": "tool_execution", "status": "error_continue"},
            )
            return state
            self._handle_error(e, {"step": "parse_input"})

            # Add error message to state
            state.add_system_message(f"Error during input parsing: {str(e)}")

            # Stop if configured to do so
            if self.stop_on_error:
                state.add_system_message("Stopping due to error.")
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "parse_input", "status": "error_stop"},
                )
                return state

            # Otherwise try to continue
            self._publish_event(
                AgentEventType.STEP_COMPLETED,
                {"step": "parse_input", "status": "error_continue"},
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
        if not self.tools:
            return False

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

        tool_match = re.search(r"Tool:\s*(\w+)", message, re.IGNORECASE)
        if not tool_match:
            return None

        tool_name = tool_match.group(1)

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

    def execute_reasoning_step(self, state: Any) -> Any:
        """
        Execute the reasoning step.

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

            # Determine if tools are needed
            if self.tools:
                if self.always_use_tools:
                    state.add_to_context("use_tools", True)
                else:
                    # Check if the response indicates a tool should be used
                    state.add_to_context("use_tools", self.should_execute_tool(state))

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
                    {"step": "reasoning", "status": "error_stop"},
                )
                return state

            # Otherwise try to continue
            self._publish_event(
                AgentEventType.STEP_COMPLETED,
                {"step": "reasoning", "status": "error_continue"},
            )
            return state

    def execute_response_step(self, state: Any) -> Any:
        """
        Execute the final response step.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        self._publish_event(AgentEventType.STEP_STARTED, {"step": "response"})

        try:
            # Check if last message is already from the assistant
            last_role = state.get_last_message_role()

            if last_role == "assistant":
                # If last message is from assistant, we're already done
                self.logger.debug(
                    "Last message already from assistant, skipping final response"
                )
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "response", "status": "skipped"},
                )
                return state

            # Add system prompt to generate a final response
            response_prompt = """
Based on the information and tool results so far, please provide a final 
comprehensive response to the user's original question.
"""
            state.add_system_message(response_prompt)

            # Get messages from state
            messages = state.get_messages()

            # Call LLM for final response
            self._publish_event(
                AgentEventType.BEFORE_LLM_CALL, {"messages": str(messages)}
            )
            response = self.llm.invoke(messages)
            self._publish_event(
                AgentEventType.AFTER_LLM_CALL, {"response": response.content}
            )

            # Add assistant message to state
            state.add_assistant_message(response.content)

            self._publish_event(AgentEventType.STEP_COMPLETED, {"step": "response"})
            return state

        except Exception as e:
            self._handle_error(e, {"step": "response"})

            # Add error message to state
            state.add_system_message(f"Error generating final response: {str(e)}")

            # Generate a simple response even in case of error
            state.add_assistant_message(
                "I apologize, but I encountered an error while generating my response. Please try again or ask a different question."
            )

            self._publish_event(
                AgentEventType.STEP_COMPLETED, {"step": "response", "status": "error"}
            )
            return state

    def execute_tool_step(self, state: Any) -> Any:
        """
        Execute the tool step if tools are needed.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        self._publish_event(AgentEventType.STEP_STARTED, {"step": "tool_execution"})

        try:
            # Extract tool call from the last assistant message
            tool_call = self._extract_tool_call(state.get_last_assistant_message())

            if not tool_call:
                self.logger.warning("No valid tool call found in assistant message")

                # For sequential pattern, we'll prompt the LLM to explicitly use a tool
                tool_prompt = "Based on the user's request, I need you to select and use one of the available tools."
                tool_prompt += " Respond in the following format:\n\n"
                tool_prompt += 'Tool: <tool_name>\nParameters: {\n    "param1": "value1",\n    "param2": "value2"\n}'

                state.add_system_message(tool_prompt)

                # Get messages from state
                messages = state.get_messages()

                # Call LLM again to get explicit tool call
                self._publish_event(
                    AgentEventType.BEFORE_LLM_CALL, {"messages": str(messages)}
                )
                response = self.llm.invoke(messages)
                self._publish_event(
                    AgentEventType.AFTER_LLM_CALL, {"response": response.content}
                )

                # Add assistant message to state
                state.add_assistant_message(response.content)

                # Try again to extract tool call
                tool_call = self._extract_tool_call(response.content)

                if not tool_call:
                    state.add_system_message(
                        "I still couldn't identify a valid tool call. Proceeding without tool execution."
                    )
                    self._publish_event(
                        AgentEventType.STEP_COMPLETED,
                        {"step": "tool_execution", "status": "no_tool"},
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
                    {"step": "tool_execution", "status": "invalid_tool"},
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
                        {"step": "tool_execution", "status": "error_stop"},
                    )
                    return state

            self._publish_event(
                AgentEventType.STEP_COMPLETED, {"step": "tool_execution"}
            )
            return state
        except Exception as e:
            self._handle_error(e, {"step": "tool_execution"})

            # Add error message to state
            state.add_system_message(f"Error during tool execution: {str(e)}")

            # Stop if configured to do so
            if self.stop_on_error:
                state.add_system_message("Stopping due to error.")
                self._publish_event(
                    AgentEventType.STEP_COMPLETED,
                    {"step": "tool_execution", "status": "error_stop"},
                )
                return state

            # Otherwise try to continue
            self._publish_event(
                AgentEventType.STEP_COMPLETED,
                {"step": "tool_execution", "status": "error_continue"},
            )
            return state
