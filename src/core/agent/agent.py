"""
GenericAgent implementation for the AI framework.

This module provides a streamlined AI agent implementation that uses the pattern-based
approach for reasoning and action, without task execution functionality.
"""

from typing import List, Dict, Any, Optional, Type, Union

from langchain_core.language_models import BaseChatModel

from src.core.agent.config import AgentConfig
from src.core.execution.base import BaseAgentPattern
from src.core.infrastructure.events import AgentEventType, EventBus
from src.core.infrastructure.logging import AgentLogger
from src.core.intelligence.skill import BaseSkill

from .base import BaseAgent
from ..execution.helpers import create_pattern
from .state import AgentState


class GenericAgent(BaseAgent):
    """
    Streamlined agent implementation with configurable behavior using patterns.

    This class provides a clean implementation of an AI agent that can be
    configured with different reasoning patterns, skills, and tools.
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
        pattern: Optional[BaseAgentPattern] = None,
        state_type: Optional[Type] = None,
    ):
        """
        Initialize the agent.

        Args:
            name: Optional name for the agent
            description: Optional description of the agent's capabilities
            skills: List of skills for the agent
            llm: Language model
            logger: Custom logger (if None, creates one based on name)
            config: Agent configuration
            event_bus: Optional event bus for publishing events
            agent_id: Unique agent ID (if None, generates one)
            pattern: Optional custom pattern for the agent
        """
        super().__init__(
            name=name,
            description=description,
            skills=skills,
            llm=llm,
            logger=logger,
            config=config or {},
            event_bus=event_bus,
            agent_id=agent_id,
        )
        self.state_type = state_type or AgentState

        if pattern is not None:
            self.pattern = pattern
        else:
            pattern_type = self.config.get("pattern_type", "react")
            self.pattern = create_pattern(
                pattern_type=pattern_type,
                llm=self.llm,
                tools=self.tools,
                config=self.config.get("pattern_config", {}),
            )

            self.logger.info(
                f"Agent '{self.name}' initialized with pattern '{pattern_type}'",
            )

    def run(self, input_text: str, **kwargs) -> str:
        """
        Run the agent with the given input.

        Args:
            input_text: Input text
            **kwargs: Additional arguments for context

        Returns:
            Agent response
        """
        try:
            self._publish_event(AgentEventType.AGENT_STARTED, {"input": input_text})

            state = self.state_type()

            if not state.has_system_message():
                system_prompt = self.get_system_prompt()
                state.add_system_message(system_prompt)

            state.add_user_message(input_text)

            for key, value in kwargs.items():
                state.add_to_context(key, value)

            workflow = self.pattern.create_workflow()

            self.logger.info(
                f"Running agent '{self.name}' with pattern '{self.pattern.__class__.__name__}'"
            )

            final_state = workflow.invoke(state)

            response = final_state.get_last_assistant_message()

            self._publish_event(AgentEventType.AGENT_STOPPED, {"output": response})

            return response

        except Exception as e:
            self._handle_error(e, {"input": input_text})
            return f"Error: {str(e)}"
