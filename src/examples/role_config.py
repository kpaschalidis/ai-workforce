"""
Example usage of the enhanced agent configuration system.

This module demonstrates how to create and use the enhanced agent configurations
with the existing agent framework.
"""

from src.core.agent.agent import GenericAgent
from src.core.agent.config import (
    AgentConfig,
    RoleConfig,
    PersonaConfig,
    AgentConfigFactory,
)
from src.core.agent.prompt import PromptManager
from src.core.execution.config import PatternType
from src.core.intelligence.skill import BaseSkill
from langchain_openai import ChatOpenAI


# Example 1: Creating a research agent with custom role and persona
def create_research_agent():
    """Create a research agent with custom configuration."""
    # Create role and persona configurations
    role_config = RoleConfig(
        role="Research Specialist",
        goals=[
            "Find accurate information from reliable sources",
            "Synthesize information into comprehensive summaries",
            "Identify gaps in existing knowledge",
        ],
        responsibilities=[
            "Search for relevant information",
            "Evaluate source credibility",
            "Organize findings in a structured format",
        ],
    )

    persona_config = PersonaConfig(
        traits={"thorough": 0.9, "analytical": 0.9, "objective": 0.8},
        communication_style="clear and academic",
        expertise_areas={"research methodology": 0.9, "critical analysis": 0.8},
    )

    # Create agent configuration
    agent_config = AgentConfig(
        name="ResearchGPT",
        description="an advanced research assistant specialized in comprehensive information gathering",
        role_config=role_config,
        persona_config=persona_config,
        pattern_type=PatternType.PLANNING,
    )

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)

    # Create agent
    agent = GenericAgent(config=agent_config, llm=llm)

    return agent


# Example 2: Using the factory to create a teaching assistant
def create_teaching_assistant():
    """Create a teaching assistant using the factory."""
    # Use the factory to create a teacher configuration
    config = AgentConfigFactory.create_role_config(
        role_type="teacher",
        name="EduBot",
        description="an educational assistant specialized in explaining complex concepts simply",
        custom_config={
            "persona_config": {
                "traits": {"patient": 1.0, "encouraging": 0.9},
                "communication_style": "simple and engaging",
            }
        },
    )

    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0.7,
    )

    # Create agent
    agent = GenericAgent(config=config, llm=llm)

    return agent


# Example 3: Creating a team of complementary agents
def create_project_team():
    """Create a team of agents with complementary roles."""
    # Define team roles
    team_roles = ["project_manager", "writer", "analyst"]

    # Create custom configurations for each role
    custom_configs = {
        "project_manager": {
            "name": "ProjectLead",
            "description": "the project coordination lead",
            "persona_config": {"traits": {"decisive": 0.9, "organized": 0.9}},
        },
        "writer": {
            "name": "ContentCreator",
            "description": "the content creation specialist",
        },
        "analyst": {"name": "DataInsight", "description": "the data analysis expert"},
    }

    # Create the team
    team_configs = AgentConfigFactory.create_agent_team(
        team_name="ProjectSuccess", roles=team_roles, custom_configs=custom_configs
    )

    # Create agents from configs
    agents = []
    for config in team_configs:
        llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.7,
        )
        agent = GenericAgent(config=config, llm=llm)
        agents.append(agent)

    return agents


# Example 4: Adding skills to an agent with enhanced config
class ResearchSkill(BaseSkill):
    """Example research skill for agents."""

    def __init__(self):
        super().__init__(
            name="Research", description="Advanced information gathering capabilities"
        )

    def get_tools(self):
        """Return tools for this skill."""
        # In a real implementation, this would return actual tool instances
        return []

    def get_system_prompt(self):
        """Return skill-specific system prompt."""
        return """
This skill provides advanced research capabilities.

When conducting research, follow these principles:
- Start with a clear research question
- Use multiple sources for verification
- Consider source credibility and bias
- Synthesize information into coherent findings
- Acknowledge limitations and gaps in available information
"""


def create_skilled_agent():
    """Create an agent with enhanced skills."""
    # Create base agent configuration
    config = AgentConfigFactory.create_research_config(
        name="SkillfulResearcher",
        description="a research assistant with enhanced capabilities",
    )

    # Create skills
    research_skill = ResearchSkill()

    # Create LLM
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)

    # Create agent with skills
    agent = GenericAgent(config=config, llm=llm, skills=[research_skill])

    return agent


# Example 5: Demonstrating how system prompts are generated
def demonstrate_system_prompts():
    """Show generated system prompts for different agent types."""
    research_config = AgentConfigFactory.create_research_config(
        name="ResearchGPT", description="a specialized research assistant"
    )

    teacher_config = AgentConfigFactory.create_role_config(
        role_type="teacher", name="EduBot"
    )

    # Create a research skill
    research_skill = ResearchSkill()

    # Create example tools
    tools = [
        type(
            "Tool",
            (),
            {"name": "search", "description": "Search for information online"},
        ),
        type("Tool", (), {"name": "analyze", "description": "Analyze text or data"}),
    ]
    research_skill.tools = tools
    skills = [research_skill]

    # Generate system prompts
    research_prompt = PromptManager.create_system_message(
        agent_config=research_config, tools=tools, skills=skills
    )
    teacher_prompt = PromptManager.create_system_message(teacher_config)
    skilled_prompt = research_skill.get_system_prompt()

    return {
        "research_prompt": research_prompt,
        "teacher_prompt": teacher_prompt,
        "skilled_prompt": skilled_prompt,
    }


# Example usage as a module
if __name__ == "__main__":
    # Create different agent types
    research_agent = create_research_agent()
    # teaching_assistant = create_teaching_assistant()
    # team = create_project_team()
    # skilled_agent = create_skilled_agent()

    # Display system prompts
    # prompts = demonstrate_system_prompts()

    # Example interaction with an agent
    # response = teaching_assistant.run(
    #     "Can you explain how photosynthesis works to a 10-year-old?"
    # )

    response = research_agent.run(
        "Can you research about the best stocs or etfs to invest in?"
    )
    print(f"Agent response: {response}")
