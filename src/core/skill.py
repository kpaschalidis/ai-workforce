from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool

from src.core.state import AgentState
from .logger import get_logger


class BaseSkill(ABC):
    """
    Abstract base class for creating skills in the AI agent framework.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the skill with optional name.

        Args:
            name: Optional name for the skill
        """
        self.name = name or self.__class__.__name__
        self.logger = get_logger(self.name)
        self.tools: List[BaseTool] = []

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """
        Generate and return tools for this skill.

        Returns:
            List of tools associated with the skill
        """
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """
        Initialize the skill with specific configuration.

        Args:
            config: Configuration dictionary for the skill
        """
        pass

    def recommend_tool(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """
        Optionally suggest an action based on the agent state.

        Returns:
            Dictionary with keys: type, description (tool name), and optional details (tool inputs).
        """
        return None

    def validate_input(self, tool_name: str, inputs: Dict[str, Any]) -> bool:
        """
        Validate inputs for a specific tool.

        Args:
            tool_name: Name of the tool
            inputs: Input parameters to validate

        Returns:
            Boolean indicating if inputs are valid
        """
        # Default implementation, can be overridden
        return True

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Handle and log errors for the skill.

        Args:
            error: Exception that occurred
            context: Optional context for the error
        """
        self.logger.error(f"Error in skill {self.name}: {str(error)}")
        if context:
            self.logger.error(f"Error context: {context}")

    def log_execution(self, tool_name: str, inputs: Dict[str, Any], outputs: Any):
        """
        Log tool execution details.

        Args:
            tool_name: Name of the tool executed
            inputs: Input parameters
            outputs: Tool execution results
        """
        self.logger.info(f"Executed tool: {tool_name}")
        self.logger.debug(f"Inputs: {inputs}")
        self.logger.debug(f"Outputs: {outputs}")
