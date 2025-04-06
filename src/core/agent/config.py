"""
Agent configuration factory for the AI framework.

This module provides utilities for creating and managing agent configurations,
making it easy to create agents with specific behaviors.
"""

from typing import Dict, Any, Optional, Type, Union

from ..execution.config import PatternType


class AgentConfig:
    """
    Configuration class for agent settings.

    This class encapsulates all configuration parameters for agents and provides
    methods for setting and retrieving configuration values.
    """

    def __init__(
        self,
        name: str = "AI Assistant",
        description: str = "a helpful AI assistant",
        agent_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        pattern_type: Optional[Union[str, PatternType]] = PatternType.REACT,
        pattern_config: Optional[Dict[str, Any]] = None,
        state_type: Optional[Type] = "AgentState",
        **kwargs,
    ):
        """
        Initialize agent configuration.

        Args:
            name: Agent name
            description: Agent description
            agent_id: Optional agent ID
            system_prompt: Custom system prompt
            pattern_type: Type of reasoning pattern to use
            llm_config: Language model configuration
            pattern_type: Type of reasoning pattern to use (if pattern not provided)
            pattern_config: Pattern-specific configuration
            state_type: Type of state to use in the workflow
            **kwargs: Additional configuration parameters
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.system_prompt = system_prompt or f"You are {name}, {description}."
        self.state_type = state_type

        if pattern_type is not None:
            if isinstance(pattern_type, PatternType):
                self.pattern_type = pattern_type.value
            else:
                self.pattern_type = pattern_type
        else:
            self.pattern_type = PatternType.REACT.value

        self.pattern_config = pattern_config or {}

        self.llm_config = {"model": "gpt-4-turbo", "temperature": 0.7}
        if llm_config:
            self.llm_config.update(llm_config)

        self.extra_config = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        config = {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "llm": self.llm_config,
            "pattern_type": self.pattern_type,
            "pattern_config": self.pattern_config,
        }

        if self.agent_id:
            config["agent_id"] = self.agent_id

        # Add any extra configuration parameters
        config.update(self.extra_config)

        return config

    def set(self, key: str, value: Any) -> "AgentConfig":
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value

        Returns:
            Self for method chaining
        """
        if key in {"name", "description", "system_prompt", "pattern_type", "agent_id"}:
            setattr(self, key, value)
        elif key == "llm_config":
            self.llm_config.update(value if isinstance(value, dict) else {})
        elif key == "pattern_config":
            self.pattern_config.update(value if isinstance(value, dict) else {})
        else:
            self.extra_config[key] = value

        return self

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if key in {"name", "description", "system_prompt", "pattern_type", "agent_id"}:
            return getattr(self, key)
        elif key == "llm_config":
            return self.llm_config
        elif key == "pattern_config":
            return self.pattern_config
        else:
            return self.extra_config.get(key, default)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            AgentConfig instance
        """
        # Extract known parameters
        name = config_dict.get("name", "AI Assistant")
        description = config_dict.get("description", "a helpful AI assistant")
        system_prompt = config_dict.get("system_prompt")
        pattern_type = config_dict.get("pattern_type", PatternType.REACT)
        llm_config = config_dict.get("llm")
        pattern_config = config_dict.get("pattern_config")
        agent_id = config_dict.get("agent_id")

        # Create config, filtering out known keys
        known_keys = {
            "name",
            "description",
            "system_prompt",
            "pattern_type",
            "llm",
            "pattern_config",
            "agent_id",
        }
        extra_config = {k: v for k, v in config_dict.items() if k not in known_keys}

        return cls(
            name=name,
            description=description,
            system_prompt=system_prompt,
            llm_config=llm_config,
            pattern_type=pattern_type,
            pattern_config=pattern_config,
            agent_id=agent_id,
            **extra_config,
        )


class AgentConfigFactory:
    """
    Factory for creating agent configurations.

    This class provides methods for creating predefined configurations for different types of
    agents with specific pattern types and skills.
    """

    @staticmethod
    def create_conversational_config(
        name: str = "Conversational Assistant",
        description: str = "a helpful AI assistant focused on providing clear, concise, and accurate information",
        temperature: float = 0.7,
        max_iterations: int = 5,
        **kwargs,
    ) -> AgentConfig:
        """
        Create configuration for a conversational agent using ReAct pattern.

        Args:
            name: Agent name
            description: Agent description
            temperature: LLM temperature (creativity vs determinism)
            max_iterations: Maximum ReAct iterations
            **kwargs: Additional configuration parameters

        Returns:
            Conversational agent configuration
        """
        return AgentConfig(
            name=name,
            description=description,
            pattern_type=PatternType.REACT,
            llm_config={"temperature": temperature},
            pattern_config={
                "max_iterations": max_iterations,
                "stop_on_error": False,
            },
            **kwargs,
        )

    @staticmethod
    def create_planning_config(
        name: str = "Planning Assistant",
        description: str = "an AI assistant that creates and executes detailed plans to solve complex tasks",
        temperature: float = 0.5,
        max_iterations: int = 15,
        max_plan_steps: int = 5,
        **kwargs,
    ) -> AgentConfig:
        """
        Create configuration for a planning agent.

        Args:
            name: Agent name
            description: Agent description
            temperature: LLM temperature (lower for more consistent plans)
            max_iterations: Maximum planning iterations
            max_plan_steps: Maximum number of steps in the plan
            **kwargs: Additional configuration parameters

        Returns:
            Planning agent configuration
        """
        return AgentConfig(
            name=name,
            description=description,
            pattern_type=PatternType.PLANNING,
            llm_config={"temperature": temperature},
            pattern_config={
                "max_iterations": max_iterations,
                "max_plan_steps": max_plan_steps,
                "stop_on_error": False,
            },
            **kwargs,
        )

    @staticmethod
    def create_sequential_config(
        name: str = "Sequential Assistant",
        description: str = "an AI assistant that processes tasks in a linear, predictable sequence",
        temperature: float = 0.6,
        always_use_tools: bool = False,
        **kwargs,
    ) -> AgentConfig:
        """
        Create configuration for a sequential agent.

        Args:
            name: Agent name
            description: Agent description
            temperature: LLM temperature
            always_use_tools: Whether to always try to use tools
            **kwargs: Additional configuration parameters

        Returns:
            Sequential agent configuration
        """
        return AgentConfig(
            name=name,
            description=description,
            pattern_type=PatternType.SEQUENTIAL,
            llm_config={"temperature": temperature},
            pattern_config={
                "always_use_tools": always_use_tools,
                "stop_on_error": False,
            },
            **kwargs,
        )

    @staticmethod
    def create_research_config(
        name: str = "Research Assistant",
        description: str = "an AI assistant specialized in thorough research and information gathering",
        temperature: float = 0.3,
        max_iterations: int = 10,
        max_plan_steps: int = 7,
        **kwargs,
    ) -> AgentConfig:
        """
        Create configuration for a research-oriented agent.

        Args:
            name: Agent name
            description: Agent description
            temperature: LLM temperature (low for factual focus)
            max_iterations: Maximum iterations
            max_plan_steps: Maximum plan steps for research
            **kwargs: Additional configuration parameters

        Returns:
            Research agent configuration
        """
        return AgentConfig(
            name=name,
            description=description,
            pattern_type=PatternType.PLANNING,  # Research benefits from planning
            llm_config={"temperature": temperature},
            pattern_config={
                "max_iterations": max_iterations,
                "max_plan_steps": max_plan_steps,
                "stop_on_error": False,
            },
            **kwargs,
        )

    @staticmethod
    def create_coding_config(
        name: str = "Coding Assistant",
        description: str = "an AI assistant specialized in software development and code generation",
        temperature: float = 0.2,
        **kwargs,
    ) -> AgentConfig:
        """
        Create configuration for a coding-focused agent.

        Args:
            name: Agent name
            description: Agent description
            temperature: LLM temperature (low for precise code)
            **kwargs: Additional configuration parameters

        Returns:
            Coding agent configuration
        """
        return AgentConfig(
            name=name,
            description=description,
            pattern_type=PatternType.REACT,  # ReAct works well for coding
            llm_config={
                "temperature": temperature,
                "model": "gpt-4-turbo",  # Ensure latest model for coding
            },
            pattern_config={
                "max_iterations": 7,
                "stop_on_error": True,  # Stop on errors for coding tasks
            },
            **kwargs,
        )
