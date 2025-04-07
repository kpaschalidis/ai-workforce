from typing import Dict, Any, Optional, Type, Union, List


from ..execution.config import PatternType


class PersonaConfig:
    """
    Configuration for an agent's persona characteristics.

    This class encapsulates personality traits, communication style, and expertise
    levels that define how an agent presents itself during interactions.
    """

    def __init__(
        self,
        traits: Optional[Dict[str, float]] = None,
        communication_style: Optional[str] = None,
        expertise_areas: Optional[Dict[str, float]] = None,
        voice: Optional[str] = None,
        tone: Optional[str] = None,
        quirks: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize persona configuration.

        Args:
            traits: Dictionary of personality traits with values (0.0-1.0)
            communication_style: Communication style descriptor
            expertise_areas: Dictionary of knowledge areas with expertise levels (0.0-1.0)
            voice: Voice characteristic descriptor
            tone: Overall tone descriptor
            quirks: List of personality quirks or mannerisms
            **kwargs: Additional persona parameters
        """
        self.traits = traits or {}
        self.communication_style = communication_style or "neutral"
        self.expertise_areas = expertise_areas or {}
        self.voice = voice or "neutral"
        self.tone = tone or "professional"
        self.quirks = quirks or []
        self.extra_config = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona configuration to dictionary."""
        config = {
            "traits": self.traits,
            "communication_style": self.communication_style,
            "expertise_areas": self.expertise_areas,
            "voice": self.voice,
            "tone": self.tone,
            "quirks": self.quirks,
        }

        # Add any extra configuration parameters
        config.update(self.extra_config)

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PersonaConfig":
        """
        Create persona configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            PersonaConfig instance
        """
        # Extract known parameters
        traits = config_dict.get("traits", {})
        communication_style = config_dict.get("communication_style")
        expertise_areas = config_dict.get("expertise_areas", {})
        voice = config_dict.get("voice")
        tone = config_dict.get("tone")
        quirks = config_dict.get("quirks", [])

        # Create config, filtering out known keys
        known_keys = {
            "traits",
            "communication_style",
            "expertise_areas",
            "voice",
            "tone",
            "quirks",
        }
        extra_config = {k: v for k, v in config_dict.items() if k not in known_keys}

        return cls(
            traits=traits,
            communication_style=communication_style,
            expertise_areas=expertise_areas,
            voice=voice,
            tone=tone,
            quirks=quirks,
            **extra_config,
        )


class RoleConfig:
    """
    Configuration for an agent's role characteristics.

    This class encapsulates role-specific information such as goals, constraints,
    responsibilities, and background information.
    """

    def __init__(
        self,
        role: str = "Assistant",
        goals: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        responsibilities: Optional[List[str]] = None,
        backstory: Optional[str] = None,
        tools_required: Optional[List[str]] = None,
        skills_required: Optional[List[str]] = None,
        performance_evaluators: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize role configuration.

        Args:
            role: Role title or name
            goals: List of goals or objectives for this role
            constraints: List of constraints or limitations
            responsibilities: List of key responsibilities
            backstory: Background narrative for the role
            tools_required: List of tools this role requires
            skills_required: List of skills this role requires
            performance_evaluators: List of criteria to evaluate role performance
            **kwargs: Additional role parameters
        """
        self.role = role
        self.goals = goals or []
        self.constraints = constraints or []
        self.responsibilities = responsibilities or []
        self.backstory = backstory or ""
        self.tools_required = tools_required or []
        self.skills_required = skills_required or []
        self.performance_evaluators = performance_evaluators or []
        self.extra_config = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert role configuration to dictionary."""
        config = {
            "role": self.role,
            "goals": self.goals,
            "constraints": self.constraints,
            "responsibilities": self.responsibilities,
            "backstory": self.backstory,
            "tools_required": self.tools_required,
            "skills_required": self.skills_required,
            "performance_evaluators": self.performance_evaluators,
        }

        # Add any extra configuration parameters
        config.update(self.extra_config)

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RoleConfig":
        """
        Create role configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            RoleConfig instance
        """
        # Extract known parameters
        role = config_dict.get("role", "Assistant")
        goals = config_dict.get("goals", [])
        constraints = config_dict.get("constraints", [])
        responsibilities = config_dict.get("responsibilities", [])
        backstory = config_dict.get("backstory", "")
        tools_required = config_dict.get("tools_required", [])
        skills_required = config_dict.get("skills_required", [])
        performance_evaluators = config_dict.get("performance_evaluators", [])

        # Create config, filtering out known keys
        known_keys = {
            "role",
            "goals",
            "constraints",
            "responsibilities",
            "backstory",
            "tools_required",
            "skills_required",
            "performance_evaluators",
        }
        extra_config = {k: v for k, v in config_dict.items() if k not in known_keys}

        return cls(
            role=role,
            goals=goals,
            constraints=constraints,
            responsibilities=responsibilities,
            backstory=backstory,
            tools_required=tools_required,
            skills_required=skills_required,
            performance_evaluators=performance_evaluators,
            **extra_config,
        )


class AgentConfig:
    """
    Configuration class for agent settings with enhanced role support.

    This class encapsulates all configuration parameters for agents and provides
    methods for setting and retrieving configuration values, with added support
    for roles, personas, and enhanced prompt management.
    """

    def __init__(
        self,
        name: str = "AI Assistant",
        description: str = "a helpful AI assistant",
        agent_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        role_config: Optional[Union[Dict[str, Any], RoleConfig]] = None,
        persona_config: Optional[Union[Dict[str, Any], PersonaConfig]] = None,
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
            role_config: Role configuration
            persona_config: Persona configuration
            pattern_type: Type of reasoning pattern to use
            pattern_config: Pattern-specific configuration
            state_type: Type of state to use in the workflow
            **kwargs: Additional configuration parameters
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description

        # Initialize role config
        if isinstance(role_config, dict):
            self.role_config = RoleConfig.from_dict(role_config)
        elif isinstance(role_config, RoleConfig):
            self.role_config = role_config
        else:
            self.role_config = RoleConfig(role=name)

        # Initialize persona config
        if isinstance(persona_config, dict):
            self.persona_config = PersonaConfig.from_dict(persona_config)
        elif isinstance(persona_config, PersonaConfig):
            self.persona_config = persona_config
        else:
            self.persona_config = PersonaConfig()

        # Set system prompt or leave it for generation at runtime
        self.system_prompt = system_prompt

        self.state_type = state_type

        if pattern_type is not None:
            if isinstance(pattern_type, PatternType):
                self.pattern_type = pattern_type.value
            else:
                self.pattern_type = pattern_type
        else:
            self.pattern_type = PatternType.REACT.value

        self.pattern_config = pattern_config or {}

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
            "role_config": self.role_config.to_dict(),
            "persona_config": self.persona_config.to_dict(),
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
        elif key == "role_config":
            if isinstance(value, dict):
                self.role_config = RoleConfig.from_dict(value)
            elif isinstance(value, RoleConfig):
                self.role_config = value
        elif key == "persona_config":
            if isinstance(value, dict):
                self.persona_config = PersonaConfig.from_dict(value)
            elif isinstance(value, PersonaConfig):
                self.persona_config = value
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
        elif key == "role_config":
            return self.role_config
        elif key == "persona_config":
            return self.persona_config
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
        pattern_config = config_dict.get("pattern_config")
        agent_id = config_dict.get("agent_id")

        # Extract role and persona configs
        role_config = config_dict.get("role_config")
        persona_config = config_dict.get("persona_config")

        # Create config, filtering out known keys
        known_keys = {
            "name",
            "description",
            "system_prompt",
            "pattern_type",
            "llm",
            "pattern_config",
            "agent_id",
            "role_config",
            "persona_config",
        }
        extra_config = {k: v for k, v in config_dict.items() if k not in known_keys}

        return cls(
            name=name,
            description=description,
            system_prompt=system_prompt,
            role_config=role_config,
            persona_config=persona_config,
            pattern_type=pattern_type,
            pattern_config=pattern_config,
            agent_id=agent_id,
            **extra_config,
        )
