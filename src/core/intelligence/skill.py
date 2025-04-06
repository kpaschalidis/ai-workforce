from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set

from langchain_core.tools import BaseTool

from src.core.infrastructure.logging import AgentLogger, get_logger


class BaseSkill(ABC):
    """
    Abstract base class for creating skills in the AI agent framework.

    Skills are modular capabilities that provide tools and domain knowledge
    to agents. They encapsulate related functionality and can be composed
    to create agents with specific capabilities.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        logger: Optional[AgentLogger] = None,
    ):
        """
        Initialize the skill with optional name and description.

        Args:
            name: Optional name for the skill
            description: Optional description of the skill's capabilities
            logger: Optional logger instance
        """
        self.name = name or self.__class__.__name__
        self.description = description or f"{self.name} Skill"
        self.logger = logger or get_logger(self.name)
        self.tools: List[BaseTool] = []
        self.initialized = False
        self.metadata: Dict[str, Any] = {}
        self.tag_set: Set[str] = set()

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """
        Generate and return tools for this skill.

        This method should create and return a list of tools associated with
        the skill. Tools are the primary way skills expose functionality to agents.

        Returns:
            List of tools associated with the skill
        """
        pass

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the skill with specific configuration.

        This method is called when the skill is added to an agent,
        allowing it to configure itself based on the agent's configuration.

        Args:
            config: Configuration dictionary for the skill
        """
        # Default implementation that can be overridden
        self.initialized = True

    def validate_input(self, tool_name: str, inputs: Dict[str, Any]) -> bool:
        """
        Validate inputs for a specific tool.

        This method validates that the inputs for a specific tool are correct
        before executing the tool. This helps prevent errors during execution.

        Args:
            tool_name: Name of the tool
            inputs: Input parameters to validate

        Returns:
            Boolean indicating if inputs are valid
        """
        # Default implementation, can be overridden
        return True

    def handle_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Handle and log errors for the skill.

        This method provides a standard way to handle errors that occur
        during tool execution or other skill operations.

        Args:
            error: Exception that occurred
            context: Optional context for the error
        """
        self.logger.error(f"Error in skill {self.name}: {str(error)}")
        if context:
            self.logger.error(f"Error context: {context}")

    def log_execution(
        self, tool_name: str, inputs: Dict[str, Any], outputs: Any
    ) -> None:
        """
        Log tool execution details.

        This method logs information about tool executions, which can be useful
        for debugging and monitoring.

        Args:
            tool_name: Name of the tool executed
            inputs: Input parameters
            outputs: Tool execution results
        """
        self.logger.info(f"Executed tool: {tool_name}")
        self.logger.debug(f"Inputs: {inputs}")
        self.logger.debug(f"Outputs: {outputs}")

    def get_system_prompt(self) -> str:
        """
        Get skill-specific system prompt for LLM.

        This method returns a prompt that informs the LLM about the skill's
        capabilities and how to use its tools. This is typically added to
        the agent's system prompt.

        Returns:
            Skill-specific system prompt text
        """
        # Get tool descriptions for the prompt
        tool_descriptions = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in self.get_tools()]
        )

        # Create the skill-specific prompt
        prompt = f"""This skill provides {self.description}.

Available tools from {self.name}:

{tool_descriptions}

When using this skill, focus on its specific domain and use the appropriate tools.
"""
        return prompt

    def get_relevant_context(self, query: str) -> Optional[str]:
        """
        Get skill-specific context relevant to the query.

        This method allows skills to provide domain-specific knowledge or context
        that might be helpful for responding to a particular query.

        Args:
            query: User query or input

        Returns:
            Relevant context information or None
        """
        # Default implementation returns None
        # Specialized skills can override to provide domain knowledge
        return None

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the skill.

        Metadata can be used to store additional information about the skill
        that doesn't fit into the standard attributes.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata from the skill.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)

    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the skill.

        Tags can be used for categorization and discovery of skills.

        Args:
            tag: Tag to add
        """
        self.tag_set.add(tag.lower())

    def add_tags(self, tags: List[str]) -> None:
        """
        Add multiple tags to the skill.

        Args:
            tags: List of tags to add
        """
        for tag in tags:
            self.add_tag(tag)

    def has_tag(self, tag: str) -> bool:
        """
        Check if the skill has a specific tag.

        Args:
            tag: Tag to check

        Returns:
            True if the skill has the tag, False otherwise
        """
        return tag.lower() in self.tag_set

    def get_tags(self) -> List[str]:
        """
        Get all tags for the skill.

        Returns:
            List of tags
        """
        return list(self.tag_set)


class CompoundSkill(BaseSkill):
    """
    A skill that combines multiple sub-skills.

    This class allows creating composite skills by combining
    multiple existing skills into a single skill.
    """

    def __init__(
        self,
        name: str,
        description: str,
        skills: List[BaseSkill],
        logger: Optional[AgentLogger] = None,
    ):
        """
        Initialize the compound skill.

        Args:
            name: Name for the compound skill
            description: Description of the skill's capabilities
            skills: List of sub-skills to include
            logger: Optional logger instance
        """
        super().__init__(name=name, description=description, logger=logger)
        self.sub_skills = skills

        # Combine tags from sub-skills
        for skill in skills:
            for tag in skill.get_tags():
                self.add_tag(tag)

    def get_tools(self) -> List[BaseTool]:
        """
        Get tools from all sub-skills.

        Returns:
            Combined list of tools from all sub-skills
        """
        all_tools = []
        for skill in self.sub_skills:
            all_tools.extend(skill.get_tools())
        return all_tools

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize all sub-skills.

        Args:
            config: Configuration dictionary
        """
        for skill in self.sub_skills:
            skill.initialize(config)

        self.initialized = True

    def get_system_prompt(self) -> str:
        """
        Get combined system prompt from all sub-skills.

        Returns:
            Combined system prompt
        """
        prompts = [f"Compound Skill: {self.description}"]
        for skill in self.sub_skills:
            prompts.append(skill.get_system_prompt())

        return "\n\n".join(prompts)

    def get_relevant_context(self, query: str) -> Optional[str]:
        """
        Get relevant context from all sub-skills.

        Args:
            query: User query

        Returns:
            Combined relevant context or None
        """
        contexts = []
        for skill in self.sub_skills:
            context = skill.get_relevant_context(query)
            if context:
                contexts.append(context)

        if contexts:
            return "\n\n".join(contexts)

        return None
