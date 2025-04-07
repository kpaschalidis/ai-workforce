from typing import Any, List, Optional
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from src.core.agent.config import AgentConfig, PersonaConfig, RoleConfig


class PromptManager:
    """
    Manager for creating and rendering prompts from agent configurations.

    This class handles the transformation of configuration data into formatted
    prompt text using LangChain's prompt templates.
    """

    @staticmethod
    def render_persona(persona_config: PersonaConfig) -> str:
        """
        Transform a PersonaConfig into prompt text.

        Args:
            persona_config: Persona configuration

        Returns:
            Formatted prompt text
        """
        prompt_parts = []

        if persona_config.communication_style:
            prompt_parts.append(
                f"Communication style: {persona_config.communication_style}"
            )

        if persona_config.voice:
            prompt_parts.append(f"Voice: {persona_config.voice}")

        if persona_config.tone:
            prompt_parts.append(f"Tone: {persona_config.tone}")

        # Add traits if present
        if persona_config.traits:
            traits_str = ", ".join(
                [
                    f"{trait} ({level:.1f})"
                    for trait, level in persona_config.traits.items()
                ]
            )
            prompt_parts.append(f"Personality traits: {traits_str}")

        # Add expertise areas if present
        if persona_config.expertise_areas:
            expertise_str = ", ".join(
                [
                    f"{area} ({level:.1f})"
                    for area, level in persona_config.expertise_areas.items()
                ]
            )
            prompt_parts.append(f"Areas of expertise: {expertise_str}")

        # Add quirks if present
        if persona_config.quirks:
            quirks_str = ", ".join(persona_config.quirks)
            prompt_parts.append(f"Unique characteristics: {quirks_str}")

        return "\n".join(prompt_parts)

    @staticmethod
    def render_role(role_config: RoleConfig) -> str:
        """
        Transform a RoleConfig into prompt text.

        Args:
            role_config: Role configuration

        Returns:
            Formatted prompt text
        """
        prompt_parts = [f"Role: {role_config.role}"]

        # Add backstory if present
        if role_config.backstory:
            prompt_parts.append(f"Background: {role_config.backstory}")

        # Add goals if present
        if role_config.goals:
            goals_str = "\n".join([f"- {goal}" for goal in role_config.goals])
            prompt_parts.append(f"Goals:\n{goals_str}")

        # Add responsibilities if present
        if role_config.responsibilities:
            resp_str = "\n".join([f"- {resp}" for resp in role_config.responsibilities])
            prompt_parts.append(f"Responsibilities:\n{resp_str}")

        # Add constraints if present
        if role_config.constraints:
            const_str = "\n".join([f"- {const}" for const in role_config.constraints])
            prompt_parts.append(f"Constraints:\n{const_str}")

        return "\n\n".join(prompt_parts)

    @staticmethod
    def render_tools(tools: List[Any]) -> str:
        """
        Format tools for inclusion in a prompt.

        Args:
            tools: List of tools

        Returns:
            Formatted tools text
        """
        if not tools:
            return ""

        tools_list = [f"- {tool.name}: {tool.description}" for tool in tools]
        return "You have access to the following tools:\n" + "\n".join(tools_list)

    @staticmethod
    def render_skills(skills: List[Any]) -> str:
        """
        Format skills for inclusion in a prompt.

        Args:
            skills: List of skills

        Returns:
            Formatted skills text
        """
        if not skills:
            return ""

        skill_parts = ["You are equipped with the following skills:"]

        for skill in skills:
            skill_prompt = (
                skill.get_system_prompt()
                if hasattr(skill, "get_system_prompt")
                else f"{skill.name} Skill"
            )
            skill_parts.append(f"{skill.name.upper()} SKILL:\n{skill_prompt}")

        return "\n\n".join(skill_parts)

    @staticmethod
    def create_system_message(
        agent_config: "AgentConfig",
        tools: Optional[List[Any]] = None,
        skills: Optional[List[Any]] = None,
    ) -> str:
        """
        Create a system message from agent configuration.

        Args:
            agent_config: Agent configuration
            tools: Optional list of tools
            skills: Optional list of skills

        Returns:
            Formatted system message
        """
        role_text = PromptManager.render_role(agent_config.role_config)
        persona_text = PromptManager.render_persona(agent_config.persona_config)
        tools_text = PromptManager.render_tools(tools or [])
        skills_text = PromptManager.render_skills(skills or [])

        sections = [
            f"You are {agent_config.name}, {agent_config.description}.",
            role_text,
        ]

        if skills_text:
            sections.append(skills_text)

        sections.append(persona_text)

        if tools_text:
            sections.append(tools_text)

        return "\n\n".join(sections)

    @staticmethod
    def create_chat_template(
        agent_config: "AgentConfig",
        tools: Optional[List[Any]] = None,
        skills: Optional[List[Any]] = None,
    ) -> ChatPromptTemplate:
        """
        Create a chat prompt template from agent configuration.

        Args:
            agent_config: Agent configuration
            tools: Optional list of tools
            skills: Optional list of skills

        Returns:
            Chat prompt template
        """
        system_message = SystemMessagePromptTemplate.from_template(
            """You are {name}, {description}.

{role_text}

{skills_text}

{persona_text}

{tools_text}"""
        )

        human_message = HumanMessagePromptTemplate.from_template("{input}")

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        # Prepare template variables
        role_text = PromptManager.render_role(agent_config.role_config)
        persona_text = PromptManager.render_persona(agent_config.persona_config)
        tools_text = PromptManager.render_tools(tools or [])
        skills_text = PromptManager.render_skills(skills or [])

        # Partial format with agent config
        return chat_prompt.partial(
            name=agent_config.name,
            description=agent_config.description,
            role_text=role_text,
            persona_text=persona_text,
            tools_text=tools_text,
            skills_text=skills_text,
        )
