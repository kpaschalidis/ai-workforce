"""
Base agent abstractions for the AI agent framework.

This module defines the abstract base classes for agents in the framework,
providing common functionality and interfaces.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.core.agent.prompt import PromptManager
from src.core.infrastructure.logging import AgentLogger, get_logger

from .config import AgentConfig
from src.core.intelligence.skill import BaseSkill

from src.core.infrastructure.events import AgentEventType, EventBus, create_event


class BaseAgent(ABC):
    """
    Abstract base agent with core functionality.

    This class defines the foundation for all agents in the framework, providing
    common functionality like skill management, configuration, and logging.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        skills: Optional[List["BaseSkill"]] = None,
        llm: Optional[BaseChatModel] = None,
        logger: Optional[AgentLogger] = None,
        config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
        event_bus: Optional[EventBus] = None,
        agent_id: Optional[str] = None,
    ):
        """
        Initialize the base agent.

        Args:
            name: Optional name for the agent
            description: Optional description of the agent's capabilities
            skills: List of skills for the agent
            llm: Language model (defaults to OpenAI)
            logger: Custom logger (if None, creates one based on name)
            config: Agent configuration
            event_bus: Event bus for publishing events
            agent_id: Unique agent ID (if None, generates one)
        """
        self.name = name or self.__class__.__name__
        self.description = description or f"{self.name} Agent"
        self.agent_id = agent_id or str(uuid.uuid4())
        self.config = config or {}

        self.skills = skills or []
        self.tools = self._collect_tools()
        self.llm = llm or self._create_default_llm()
        self.logger = logger or get_logger(self.name)
        self.event_bus = event_bus or EventBus()

        # Track initialization
        self.initialized = False

        for skill in self.skills:
            skill.initialize(self.config)

        self._publish_event(AgentEventType.AGENT_INITIALIZED)
        self.logger.info(
            f"Agent '{self.name}' initialized",
            data={
                "agent_id": self.agent_id,
                "skills": [skill.name for skill in self.skills],
            },
        )

    def _create_default_llm(self) -> BaseChatModel:
        """
        Create a default language model based on configuration.

        Returns:
            Default language model
        """
        model = self.config.get("llm", {}).get("model", "gpt-4-turbo")
        temperature = self.config.get("llm", {}).get("temperature", 0.7)

        return ChatOpenAI(model=model, temperature=temperature)

    def _collect_tools(self) -> List[Any]:
        """
        Collect tools from all skills.

        Returns:
            List of tools from all skills
        """
        tools = []
        for skill in self.skills:
            if hasattr(skill, "get_tools"):
                tools.extend(skill.get_tools())
        return tools

    def add_skill(self, skill: "BaseSkill") -> None:
        """
        Add a skill to the agent.

        Args:
            skill: Skill to add
        """
        if not skill.initialized:
            skill.initialize(self.config)

        self.skills.append(skill)
        self.tools = self._collect_tools()

        self.logger.info(f"Added skill '{skill.name}' to agent '{self.name}'")

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent.

        Returns:
            System prompt
        """
        return PromptManager.create_system_message(self.config, self.tools, self.skills)

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """
        Run the agent with the given input.

        Args:
            input_text: Input text
            **kwargs: Additional arguments

        Returns:
            Agent response
        """
        pass

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
            event = create_event(
                event_type=event_type,
                source=f"agent.{self.name}",
                data=data or {},
                metadata=metadata or {"agent_id": self.agent_id},
            )
            self.event_bus.publish(event)

    def _handle_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Handle an error during agent execution.

        Args:
            error: Exception that occurred
            context: Error context
        """
        error_msg = f"Error in agent '{self.name}': {str(error)}"
        self.logger.error(error_msg, data=context)

        self._publish_event(
            AgentEventType.AGENT_ERROR,
            data={
                "error": str(error),
                "error_type": error.__class__.__name__,
                "context": context or {},
            },
        )
