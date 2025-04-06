"""
Base reasoning pattern for the AI agent framework.

This module defines the base class for agent reasoning patterns,
providing a standardized way to implement different patterns
such as react, planning etc. directly with LangGraph.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Union

from langchain_core.language_models import BaseChatModel

from src.core.infrastructure.events import AgentEvent, AgentEventType, EventBus

# Import LangGraph conditionally to handle cases where it's not installed
try:
    from langgraph.graph import StateGraph, END

    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Create placeholders for type checking to work
    class StateGraph:
        def __init__(self, state_type):
            pass

    END = object()
    LANGGRAPH_AVAILABLE = False

from .config import PatternConfig
from src.core.infrastructure.logging import (
    AgentLogger,
    get_logger,
)


class BaseAgentPattern(ABC):
    """
    Abstract base class for different reasoning pattern implementations.

    This class provides a foundation for implementing various reasoning patterns
    like ReAct, Planning, and Sequential, directly using LangGraph.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Any] = None,
        config: Optional[Union[Dict[str, Any], PatternConfig]] = None,
        logger: Optional[AgentLogger] = None,
        event_bus: Optional[EventBus] = None,
        name: str = None,
    ):
        """
        Initialize the agent pattern.

        Args:
            llm: Language model to use
            tools: List of tools available to the pattern
            config: Configuration dictionary
            logger: Optional logger instance
            event_bus: Optional event bus for publishing events
            name: Optional name for the pattern
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is not installed. Install with 'pip install langgraph'"
            )

        self.llm = llm
        self.tools = tools or []
        self.config = config or {}
        self.logger = logger or get_logger(self.__class__.__name__)
        self.event_bus = event_bus
        self.name = name or self.__class__.__name__

        self.state_type = self.config.get("state_type", "AgentState")

        self.graph = None

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Build and return the LangGraph state graph for this pattern.

        This method should create and configure a StateGraph that implements
        the specific reasoning pattern.

        Returns:
            Configured StateGraph
        """
        pass

    def create_workflow(self) -> Callable:
        """
        Create and return an executable workflow.

        Returns:
            Executable workflow function
        """
        if self.graph is None:
            self.graph = self.build_graph()

        # Compile the graph
        return self.graph.compile()

    def is_valid_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is valid and available.

        Args:
            tool_name: Name of the tool

        Returns:
            True if the tool is valid, False otherwise
        """
        return any(tool.name == tool_name for tool in self.tools)

    def get_tool_by_name(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool or None if not found
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def get_system_prompt(self, additional_instructions: str = None) -> str:
        """
        Get the system prompt for the executor.

        This method generates a system prompt based on the available tools
        and execution strategy.

        Args:
            additional_instructions: Optional additional instructions to include

        Returns:
            System prompt
        """
        # Start with base prompt
        base_prompt = self.config.get(
            "system_prompt", "You are a helpful AI assistant."
        )

        # Add tool descriptions if available
        if self.tools:
            tool_descriptions = "\n".join(
                [f"- {tool.name}: {tool.description}" for tool in self.tools]
            )

            base_prompt += (
                f"\n\nYou have access to the following tools:\n{tool_descriptions}\n"
            )

            # Add execution-specific instructions
            base_prompt += self._get_execution_instructions()

        # Add additional instructions if provided
        if additional_instructions:
            base_prompt += f"\n\n{additional_instructions}"

        return base_prompt

    def _get_execution_instructions(self) -> str:
        """
        Get execution-specific instructions for the system prompt.

        Returns:
            Execution instructions
        """
        # Base implementation - subclasses should override
        return """
When using tools, respond in the following format:

Tool: <tool_name>
Parameters: {
    "param1": "value1",
    "param2": "value2"
}

The output of the tool will be provided to you for further processing.
"""

    def _publish_event(
        self,
        event_type: AgentEventType,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Publish an event to the event bus.

        Args:
            event_type: Type of event
            data: Optional event data
            metadata: Optional event metadata
        """
        if self.event_bus:
            event = AgentEvent(
                event_type=event_type,
                source=f"pattern.{self.name}",
                data=data or {},
                metadata=metadata or {},
            )
            self.event_bus.publish(event)

    def _handle_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Handle an error during execution.

        Args:
            error: Exception that occurred
            context: Error context
        """
        error_msg = f"Error in pattern '{self.name}': {str(error)}"
        self.logger.error(error_msg, data=context)

        if self.event_bus:
            self._publish_event(
                AgentEventType.AGENT_ERROR,
                data={
                    "error": str(error),
                    "error_type": error.__class__.__name__,
                    "context": context or {},
                },
            )
