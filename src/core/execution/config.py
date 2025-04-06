"""
Pattern configuration for the AI agent framework.

This module provides configuration utilities and default settings for the
different reasoning patterns available in the framework.
"""

from typing import Dict, Any, Optional, Type
from enum import Enum


class PatternType(Enum):
    """Types of reasoning patterns available in the framework."""

    DEFAULT = "default"  # Default (usually ReAct)
    REACT = "react"  # Reasoning + Acting pattern
    PLANNING = "planning"  # Create plan, execute steps, evaluate
    SEQUENTIAL = "sequential"  # Fixed sequence of steps


class PatternConfig:
    """
    Configuration for reasoning patterns.

    This class provides a structured way to define and manage pattern configurations
    with defaults and validation.
    """

    def __init__(
        self,
        pattern_type: PatternType = PatternType.REACT,
        max_iterations: int = 10,
        stop_on_error: bool = False,
        state_type: Optional[Type] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize a pattern configuration.

        Args:
            pattern_type: Type of reasoning pattern to use
            max_iterations: Maximum number of iterations for the pattern
            stop_on_error: Whether to stop execution on errors
            state_type: Type of state to use in the pattern
            system_prompt: Custom system prompt for the pattern
            **kwargs: Additional pattern-specific configuration options
        """
        self.pattern_type = pattern_type
        self.config = {
            "max_iterations": max_iterations,
            "stop_on_error": stop_on_error,
            "system_prompt": system_prompt,
            **kwargs,
        }

        if state_type:
            self.config["state_type"] = state_type

    def update(self, **kwargs) -> "PatternConfig":
        """
        Update the configuration with new values.

        Args:
            **kwargs: Configuration values to update

        Returns:
            Self for method chaining
        """
        self.config.update(kwargs)
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
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> "PatternConfig":
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Value to set

        Returns:
            Self for method chaining
        """
        self.config[key] = value
        return self

    def as_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "pattern_type": self.pattern_type.value,
            **self.config,
        }


# Default configurations for each pattern type
DEFAULT_REACT_CONFIG = {
    "max_iterations": 10,
    "stop_on_error": False,
}

DEFAULT_PLANNING_CONFIG = {
    "max_iterations": 10,
    "max_plan_steps": 5,
    "stop_on_error": False,
}

DEFAULT_SEQUENTIAL_CONFIG = {
    "always_use_tools": False,
    "stop_on_error": False,
}


def create_react_config(
    max_iterations: int = 10,
    stop_on_error: bool = False,
    state_type: Optional[Type] = None,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> PatternConfig:
    """
    Create a configuration for the ReAct pattern.

    Args:
        max_iterations: Maximum number of iterations
        stop_on_error: Whether to stop on errors
        state_type: Type of state to use
        system_prompt: Custom system prompt
        **kwargs: Additional configuration options

    Returns:
        PatternConfig instance
    """
    return PatternConfig(
        pattern_type=PatternType.REACT,
        max_iterations=max_iterations,
        stop_on_error=stop_on_error,
        state_type=state_type,
        system_prompt=system_prompt,
        **kwargs,
    )


def create_planning_config(
    max_iterations: int = 10,
    max_plan_steps: int = 5,
    stop_on_error: bool = False,
    state_type: Optional[Type] = None,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> PatternConfig:
    """
    Create a configuration for the Planning pattern.

    Args:
        max_iterations: Maximum number of iterations
        max_plan_steps: Maximum number of steps in the plan
        stop_on_error: Whether to stop on errors
        state_type: Type of state to use
        system_prompt: Custom system prompt
        **kwargs: Additional configuration options

    Returns:
        PatternConfig instance
    """
    return PatternConfig(
        pattern_type=PatternType.PLANNING,
        max_iterations=max_iterations,
        max_plan_steps=max_plan_steps,
        stop_on_error=stop_on_error,
        state_type=state_type,
        system_prompt=system_prompt,
        **kwargs,
    )


def create_sequential_config(
    always_use_tools: bool = False,
    stop_on_error: bool = False,
    state_type: Optional[Type] = None,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> PatternConfig:
    """
    Create a configuration for the Sequential pattern.

    Args:
        always_use_tools: Whether to always try to use tools
        stop_on_error: Whether to stop on errors
        state_type: Type of state to use
        system_prompt: Custom system prompt
        **kwargs: Additional configuration options

    Returns:
        PatternConfig instance
    """
    return PatternConfig(
        pattern_type=PatternType.SEQUENTIAL,
        always_use_tools=always_use_tools,
        stop_on_error=stop_on_error,
        state_type=state_type,
        system_prompt=system_prompt,
        **kwargs,
    )


def validate_config(
    config: Dict[str, Any], pattern_type: PatternType
) -> Dict[str, Any]:
    """
    Validate and complete a configuration dictionary for a specific pattern type.

    Args:
        config: Configuration dictionary to validate
        pattern_type: Type of pattern to validate for

    Returns:
        Validated and completed configuration dictionary
    """
    # Create a copy of the input config
    validated_config = config.copy()

    # Add default values based on pattern type
    if pattern_type == PatternType.REACT:
        for key, value in DEFAULT_REACT_CONFIG.items():
            if key not in validated_config:
                validated_config[key] = value
    elif pattern_type == PatternType.PLANNING:
        for key, value in DEFAULT_PLANNING_CONFIG.items():
            if key not in validated_config:
                validated_config[key] = value
    elif pattern_type == PatternType.SEQUENTIAL:
        for key, value in DEFAULT_SEQUENTIAL_CONFIG.items():
            if key not in validated_config:
                validated_config[key] = value

    return validated_config
