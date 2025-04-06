"""
Pattern helper methods for the AI agent framework.

This module provides helper methods for creating and configuring reasoning patterns
that can be passed as optional parameters to agents. The helper functions create
pattern instances directly, without using builder or registry patterns.
"""

from typing import Dict, List, Any, Optional, Type, Union

from langchain_core.language_models import BaseChatModel

from .base import BaseAgentPattern
from .react import ReactPattern
from .planning import PlanningPattern
from .sequential import SequentialPattern
from .config import PatternType, PatternConfig, validate_config
from src.core.infrastructure.logging import get_logger

logger = get_logger("patterns")


def create_pattern(
    pattern_type: Union[PatternType, str],
    llm: BaseChatModel,
    tools: List[Any] = None,
    config: Optional[Union[Dict[str, Any], PatternConfig]] = None,
    state_type: Optional[Type] = None,
    **kwargs,
) -> BaseAgentPattern:
    """
    Create a reasoning pattern of the specified type.

    Args:
        pattern_type: Type of pattern to create (react, planning, sequential)
        llm: Language model to use
        tools: List of tools available to the pattern
        config: Configuration dictionary or PatternConfig instance
        state_type: Type of state to use (if not specified in config)
        **kwargs: Additional arguments specific to the pattern

    Returns:
        An instance of the requested pattern

    Raises:
        ValueError: If pattern_type is not recognized
    """
    # Convert dictionary config to PatternConfig if needed
    if isinstance(config, dict):
        # Extract and use state_type from kwargs or passed parameter
        if state_type and "state_type" not in config:
            config["state_type"] = state_type
        pattern_config = config
    elif isinstance(config, PatternConfig):
        pattern_config = config.as_dict()
    else:
        # Create a minimal config with state_type if provided
        pattern_config = {}
        if state_type:
            pattern_config["state_type"] = state_type

    # Add kwargs to config
    pattern_config.update(kwargs)

    # Normalize pattern type
    if isinstance(pattern_type, PatternType):
        pattern_type_enum = pattern_type
        pattern_type_str = pattern_type.value
    else:
        pattern_type_str = pattern_type.lower()
        try:
            pattern_type_enum = PatternType(pattern_type_str)
        except ValueError:
            if pattern_type_str == "react":
                pattern_type_enum = PatternType.REACT
            elif pattern_type_str == "planning":
                pattern_type_enum = PatternType.PLANNING
            elif pattern_type_str == "sequential":
                pattern_type_enum = PatternType.SEQUENTIAL
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type_str}")

    # Validate and complete the config for the specific pattern type
    validated_config = validate_config(pattern_config, pattern_type_enum)

    # Create the appropriate pattern
    if pattern_type_enum == PatternType.REACT:
        return create_react_pattern(llm, tools, validated_config)
    elif pattern_type_enum == PatternType.PLANNING:
        return create_planning_pattern(llm, tools, validated_config)
    elif pattern_type_enum == PatternType.SEQUENTIAL:
        return create_sequential_pattern(llm, tools, validated_config)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type_str}")


def create_react_pattern(
    llm: BaseChatModel,
    tools: List[Any] = None,
    config: Union[Dict[str, Any], PatternConfig] = None,
    **kwargs,
) -> ReactPattern:
    """
    Create a ReAct (Reasoning + Acting) pattern.

    Args:
        llm: Language model to use
        tools: List of tools available to the pattern
        config: Configuration dictionary or PatternConfig instance
        **kwargs: Additional arguments for ReactPattern

    Returns:
        Configured ReactPattern instance
    """
    # Convert PatternConfig to dict if needed
    if isinstance(config, PatternConfig):
        config_dict = config.as_dict()
    else:
        config_dict = config or {}

    # Add any additional kwargs to config
    if kwargs:
        config_dict.update(kwargs)

    # Create and return the pattern
    return ReactPattern(
        llm=llm,
        tools=tools,
        config=config_dict,
    )


def create_planning_pattern(
    llm: BaseChatModel,
    tools: List[Any] = None,
    config: Union[Dict[str, Any], PatternConfig] = None,
    **kwargs,
) -> PlanningPattern:
    """
    Create a Planning pattern.

    Args:
        llm: Language model to use
        tools: List of tools available to the pattern
        config: Configuration dictionary or PatternConfig instance
        **kwargs: Additional arguments for PlanningPattern

    Returns:
        Configured PlanningPattern instance
    """
    # Convert PatternConfig to dict if needed
    if isinstance(config, PatternConfig):
        config_dict = config.as_dict()
    else:
        config_dict = config or {}

    # Add any additional kwargs to config
    if kwargs:
        config_dict.update(kwargs)

    # Create and return the pattern
    return PlanningPattern(
        llm=llm,
        tools=tools,
        config=config_dict,
    )


def create_sequential_pattern(
    llm: BaseChatModel,
    tools: List[Any] = None,
    config: Union[Dict[str, Any], PatternConfig] = None,
    **kwargs,
) -> SequentialPattern:
    """
    Create a Sequential pattern.

    Args:
        llm: Language model to use
        tools: List of tools available to the pattern
        config: Configuration dictionary or PatternConfig instance
        **kwargs: Additional arguments for SequentialPattern

    Returns:
        Configured SequentialPattern instance
    """
    # Convert PatternConfig to dict if needed
    if isinstance(config, PatternConfig):
        config_dict = config.as_dict()
    else:
        config_dict = config or {}

    # Add any additional kwargs to config
    if kwargs:
        config_dict.update(kwargs)

    # Create and return the pattern
    return SequentialPattern(
        llm=llm,
        tools=tools,
        config=config_dict,
    )


def get_default_pattern(
    llm: BaseChatModel,
    tools: List[Any] = None,
    config: Optional[Union[Dict[str, Any], PatternConfig]] = None,
    state_type: Optional[Type] = None,
) -> BaseAgentPattern:
    """
    Get the default reasoning pattern (ReAct).

    Args:
        llm: Language model to use
        tools: List of tools available to the pattern
        config: Configuration dictionary or PatternConfig instance
        state_type: Type of state to use (if not specified in config)

    Returns:
        Default pattern instance (ReactPattern)
    """
    from .config import create_react_config

    if config is None and state_type is not None:
        # Create a config with the state type
        config = create_react_config(state_type=state_type)

    return create_react_pattern(llm, tools, config)


def configure_agent_with_pattern(
    agent_config: Dict[str, Any],
    pattern_type: Union[str, PatternType] = PatternType.DEFAULT,
    pattern_config: Optional[Union[Dict[str, Any], PatternConfig]] = None,
) -> Dict[str, Any]:
    """
    Configure an agent with a specific reasoning pattern.

    This helper method adds pattern configuration to an agent config.

    Args:
        agent_config: Original agent configuration
        pattern_type: Type of pattern to use
        pattern_config: Pattern-specific configuration

    Returns:
        Updated agent configuration
    """
    # Clone the config to avoid modifying the original
    config = agent_config.copy()

    # Normalize pattern type
    if isinstance(pattern_type, PatternType):
        pattern_type_str = pattern_type.value
    else:
        pattern_type_str = pattern_type

    # Only set pattern type if it's not default
    if pattern_type_str and pattern_type_str != "default":
        config["pattern_type"] = pattern_type_str

    # Process pattern config
    if pattern_config:
        if isinstance(pattern_config, PatternConfig):
            pattern_config_dict = pattern_config.as_dict()
            # Don't copy the pattern_type to avoid confusion
            if "pattern_type" in pattern_config_dict:
                del pattern_config_dict["pattern_type"]
        else:
            pattern_config_dict = pattern_config

        # Add pattern-specific configuration
        config.setdefault("pattern_config", {})
        config["pattern_config"].update(pattern_config_dict)

    return config
