"""
Agent factory for the AI framework.

This module provides factory methods for creating different types of agents
with appropriate configurations.
"""

from typing import List, Optional

from langchain_core.language_models import BaseChatModel

from src.core.infrastructure.events import EventBus
from src.core.intelligence.skill import BaseSkill

from .agent import GenericAgent
from .config import AgentConfig, AgentConfigFactory


class AgentFactory:
    """
    Factory for creating agent instances.

    This class provides static methods for creating different types of agents
    with appropriate configurations and patterns.
    """

    @staticmethod
    def create_agent(
        config: AgentConfig,
        skills: Optional[List[BaseSkill]] = None,
        llm: Optional[BaseChatModel] = None,
        event_bus: Optional[EventBus] = None,
    ) -> GenericAgent:
        """
        Create an agent with the given configuration.

        Args:
            config: Agent configuration
            skills: Optional list of skills
            llm: Optional language model (overrides config)
            event_bus: Optional event bus

        Returns:
            Configured GenericAgent instance
        """
        return GenericAgent(
            config=config,
            skills=skills,
            llm=llm,
            event_bus=event_bus,
        )

    @staticmethod
    def create_conversational_agent(
        name: str = "Conversational Assistant",
        description: str = "a helpful AI assistant focused on natural conversations",
        skills: Optional[List[BaseSkill]] = None,
        llm: Optional[BaseChatModel] = None,
        event_bus: Optional[EventBus] = None,
        **kwargs
    ) -> GenericAgent:
        """
        Create a conversational agent using the ReAct pattern.

        Args:
            name: Agent name
            description: Agent description
            skills: Optional list of skills
            llm: Optional language model (overrides config)
            event_bus: Optional event bus
            **kwargs: Additional configuration parameters

        Returns:
            Configured conversational agent
        """
        config = AgentConfigFactory.create_conversational_config(
            name=name, description=description, **kwargs
        )

        return AgentFactory.create_agent(
            config=config,
            skills=skills,
            llm=llm,
            event_bus=event_bus,
        )

    @staticmethod
    def create_planning_agent(
        name: str = "Planning Assistant",
        description: str = "an AI assistant that creates and executes detailed plans",
        skills: Optional[List[BaseSkill]] = None,
        llm: Optional[BaseChatModel] = None,
        event_bus: Optional[EventBus] = None,
        max_plan_steps: int = 5,
        **kwargs
    ) -> GenericAgent:
        """
        Create a planning agent using the Planning pattern.

        Args:
            name: Agent name
            description: Agent description
            skills: Optional list of skills
            llm: Optional language model (overrides config)
            event_bus: Optional event bus
            max_plan_steps: Maximum steps in the plan
            **kwargs: Additional configuration parameters

        Returns:
            Configured planning agent
        """
        config = AgentConfigFactory.create_planning_config(
            name=name, description=description, max_plan_steps=max_plan_steps, **kwargs
        )

        return AgentFactory.create_agent(
            config=config,
            skills=skills,
            llm=llm,
            event_bus=event_bus,
        )

    @staticmethod
    def create_sequential_agent(
        name: str = "Sequential Assistant",
        description: str = "an AI assistant that processes tasks in a sequential manner",
        skills: Optional[List[BaseSkill]] = None,
        llm: Optional[BaseChatModel] = None,
        event_bus: Optional[EventBus] = None,
        always_use_tools: bool = False,
        **kwargs
    ) -> GenericAgent:
        """
        Create a sequential agent using the Sequential pattern.

        Args:
            name: Agent name
            description: Agent description
            skills: Optional list of skills
            llm: Optional language model (overrides config)
            event_bus: Optional event bus
            always_use_tools: Whether to always try to use tools
            **kwargs: Additional configuration parameters

        Returns:
            Configured sequential agent
        """
        config = AgentConfigFactory.create_sequential_config(
            name=name,
            description=description,
            always_use_tools=always_use_tools,
            **kwargs
        )

        return AgentFactory.create_agent(
            config=config,
            skills=skills,
            llm=llm,
            event_bus=event_bus,
        )

    @staticmethod
    def create_research_agent(
        name: str = "Research Assistant",
        description: str = "an AI assistant specialized in research and information gathering",
        skills: Optional[List[BaseSkill]] = None,
        llm: Optional[BaseChatModel] = None,
        event_bus: Optional[EventBus] = None,
        **kwargs
    ) -> GenericAgent:
        """
        Create a research-focused agent.

        Args:
            name: Agent name
            description: Agent description
            skills: Optional list of skills
            llm: Optional language model (overrides config)
            event_bus: Optional event bus
            **kwargs: Additional configuration parameters

        Returns:
            Configured research agent
        """
        config = AgentConfigFactory.create_research_config(
            name=name, description=description, **kwargs
        )

        return AgentFactory.create_agent(
            config=config,
            skills=skills,
            llm=llm,
            event_bus=event_bus,
        )

    @staticmethod
    def create_coding_agent(
        name: str = "Coding Assistant",
        description: str = "an AI assistant specialized in software development",
        skills: Optional[List[BaseSkill]] = None,
        llm: Optional[BaseChatModel] = None,
        event_bus: Optional[EventBus] = None,
        **kwargs
    ) -> GenericAgent:
        """
        Create a coding-focused agent.

        Args:
            name: Agent name
            description: Agent description
            skills: Optional list of skills
            llm: Optional language model (overrides config)
            event_bus: Optional event bus
            **kwargs: Additional configuration parameters

        Returns:
            Configured coding agent
        """
        config = AgentConfigFactory.create_coding_config(
            name=name, description=description, **kwargs
        )

        return AgentFactory.create_agent(
            config=config,
            skills=skills,
            llm=llm,
            event_bus=event_bus,
        )
