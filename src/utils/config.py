from typing import Dict, Any, Optional, Union, List

from src.core.agent.config import AgentConfig, PersonaConfig, RoleConfig
from src.core.execution.config import PatternType


class AgentConfigFactory:
    """
    Factory for creating agent configurations with role support.

    This class provides methods for creating predefined configurations for different types of
    agents with specific roles, personas, pattern types, and skills.
    """

    @staticmethod
    def create_conversational_config(
        name: str = "Conversational Assistant",
        description: str = "a helpful AI assistant focused on providing clear, concise, and accurate information",
        role_config: Optional[Union[Dict[str, Any], RoleConfig]] = None,
        persona_config: Optional[Union[Dict[str, Any], PersonaConfig]] = None,
        temperature: float = 0.7,
        max_iterations: int = 5,
        **kwargs,
    ) -> AgentConfig:
        """
        Create configuration for a conversational agent using ReAct pattern.

        Args:
            name: Agent name
            description: Agent description
            role_config: Role configuration
            persona_config: Persona configuration
            temperature: LLM temperature (creativity vs determinism)
            max_iterations: Maximum ReAct iterations
            **kwargs: Additional configuration parameters

        Returns:
            Conversational agent configuration
        """
        # Create default role config if not provided
        if not role_config:
            role_config = RoleConfig(
                role="Conversational Assistant",
                goals=[
                    "Provide clear, accurate, and helpful information",
                    "Maintain a natural, engaging conversation flow",
                    "Adapt responses to the user's needs and context",
                ],
                responsibilities=[
                    "Answer questions accurately and completely",
                    "Ask clarifying questions when needed",
                    "Remember context from earlier in the conversation",
                ],
            )

        # Create default persona config if not provided
        if not persona_config:
            persona_config = PersonaConfig(
                traits={"helpful": 0.9, "friendly": 0.8, "precise": 0.9},
                communication_style="clear and conversational",
                tone="warm and professional",
            )

        return AgentConfig(
            name=name,
            description=description,
            role_config=role_config,
            persona_config=persona_config,
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
        role_config: Optional[Union[Dict[str, Any], RoleConfig]] = None,
        persona_config: Optional[Union[Dict[str, Any], PersonaConfig]] = None,
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
            role_config: Role configuration
            persona_config: Persona configuration
            temperature: LLM temperature (lower for more consistent plans)
            max_iterations: Maximum planning iterations
            max_plan_steps: Maximum number of steps in the plan
            **kwargs: Additional configuration parameters

        Returns:
            Planning agent configuration
        """
        # Create default role config if not provided
        if not role_config:
            role_config = RoleConfig(
                role="Planning Assistant",
                goals=[
                    "Create detailed, actionable plans for complex tasks",
                    "Break down problems into manageable steps",
                    "Adapt plans based on feedback and constraints",
                ],
                responsibilities=[
                    "Analyze problems thoroughly before planning",
                    "Create plans with appropriate level of detail",
                    "Consider potential obstacles and contingencies",
                    "Evaluate plan effectiveness",
                ],
            )

        # Create default persona config if not provided
        if not persona_config:
            persona_config = PersonaConfig(
                traits={"analytical": 0.9, "thorough": 0.9, "organized": 0.9},
                communication_style="structured and methodical",
                tone="professional and confident",
            )

        return AgentConfig(
            name=name,
            description=description,
            role_config=role_config,
            persona_config=persona_config,
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
    def create_research_config(
        name: str = "Research Assistant",
        description: str = "an AI assistant specialized in thorough research and information gathering",
        role_config: Optional[Union[Dict[str, Any], RoleConfig]] = None,
        persona_config: Optional[Union[Dict[str, Any], PersonaConfig]] = None,
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
            role_config: Role configuration
            persona_config: Persona configuration
            temperature: LLM temperature (low for factual focus)
            max_iterations: Maximum iterations
            max_plan_steps: Maximum plan steps for research
            **kwargs: Additional configuration parameters

        Returns:
            Research agent configuration
        """
        # Create default role config if not provided
        if not role_config:
            role_config = RoleConfig(
                role="Research Assistant",
                goals=[
                    "Gather comprehensive information on topics",
                    "Evaluate information quality and relevance",
                    "Synthesize findings into coherent summaries",
                ],
                responsibilities=[
                    "Find and evaluate relevant information",
                    "Cross-reference sources for accuracy",
                    "Present information in a structured, digestible format",
                    "Acknowledge limitations and gaps in knowledge",
                ],
            )

        # Create default persona config if not provided
        if not persona_config:
            persona_config = PersonaConfig(
                traits={"curious": 0.9, "methodical": 0.9, "objective": 0.9},
                communication_style="informative and clear",
                tone="scholarly yet accessible",
                expertise_areas={"research methods": 0.9, "critical analysis": 0.8},
            )

        return AgentConfig(
            name=name,
            description=description,
            role_config=role_config,
            persona_config=persona_config,
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
        role_config: Optional[Union[Dict[str, Any], RoleConfig]] = None,
        persona_config: Optional[Union[Dict[str, Any], PersonaConfig]] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> AgentConfig:
        """
        Create configuration for a coding-focused agent.

        Args:
            name: Agent name
            description: Agent description
            role_config: Role configuration
            persona_config: Persona configuration
            temperature: LLM temperature (low for precise code)
            **kwargs: Additional configuration parameters

        Returns:
            Coding agent configuration
        """
        # Create default role config if not provided
        if not role_config:
            role_config = RoleConfig(
                role="Coding Assistant",
                goals=[
                    "Generate high-quality, functional code",
                    "Solve programming problems efficiently",
                    "Explain code and programming concepts clearly",
                ],
                responsibilities=[
                    "Write well-structured, documented code",
                    "Debug and troubleshoot code issues",
                    "Provide explanations of code functionality",
                    "Follow best practices and coding standards",
                ],
            )

        # Create default persona config if not provided
        if not persona_config:
            persona_config = PersonaConfig(
                traits={"logical": 0.9, "precise": 0.9, "solution-oriented": 0.8},
                communication_style="clear and technical",
                tone="professional and educational",
                expertise_areas={"software development": 0.9, "algorithms": 0.8},
            )

        return AgentConfig(
            name=name,
            description=description,
            role_config=role_config,
            persona_config=persona_config,
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

    @staticmethod
    def create_agent_with_template(
        name: str,
        description: str,
        role_config: Optional[Union[Dict[str, Any], RoleConfig]] = None,
        persona_config: Optional[Union[Dict[str, Any], PersonaConfig]] = None,
        template_type: str = "default",
        pattern_type: Optional[Union[str, PatternType]] = PatternType.REACT,
        llm_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AgentConfig:
        """
        Create an agent configuration using a template type.

        Args:
            name: Agent name
            description: Agent description
            role_config: Role configuration
            persona_config: Persona configuration
            template_type: Type of agent (conversational, planning, research, coding)
            pattern_type: Type of reasoning pattern to use
            llm_config: Language model configuration
            **kwargs: Additional configuration parameters

        Returns:
            Configured AgentConfig
        """
        # Map template types to factory methods
        template_factory_map = {
            "conversational": AgentConfigFactory.create_conversational_config,
            "planning": AgentConfigFactory.create_planning_config,
            "research": AgentConfigFactory.create_research_config,
            "coding": AgentConfigFactory.create_coding_config,
        }

        # Use the appropriate factory method if available
        if template_type in template_factory_map:
            factory_method = template_factory_map[template_type]
            return factory_method(
                name=name,
                description=description,
                role_config=role_config,
                persona_config=persona_config,
                pattern_type=pattern_type,
                llm_config=llm_config,
                **kwargs,
            )

        # For unknown template types, create a basic config
        return AgentConfig(
            name=name,
            description=description,
            role_config=role_config,
            persona_config=persona_config,
            pattern_type=pattern_type,
            llm_config=llm_config or {"temperature": 0.7},
            **kwargs,
        )

    @staticmethod
    def create_role_config(
        role_type: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AgentConfig:
        """
        Create an agent configuration with a predefined role type.

        Args:
            role_type: Type of role (e.g., "teacher", "analyst", "writer")
            name: Optional name override
            description: Optional description override
            custom_config: Additional role-specific configuration
            **kwargs: Additional agent configuration parameters

        Returns:
            Agent configuration with the specified role
        """
        role_configs = {
            "teacher": {
                "name": "Educational Assistant",
                "description": "an AI assistant focused on teaching and explaining concepts clearly",
                "role_config": RoleConfig(
                    role="Teacher",
                    goals=[
                        "Explain concepts clearly and accurately",
                        "Adapt explanations to learner's level of understanding",
                        "Encourage critical thinking and curiosity",
                    ],
                    responsibilities=[
                        "Break down complex topics into manageable parts",
                        "Provide examples and analogies to illustrate concepts",
                        "Check for understanding and provide feedback",
                        "Answer questions with patience and clarity",
                    ],
                    backstory="Experienced educator with expertise in adapting content to different learning styles",
                ),
                "persona_config": PersonaConfig(
                    traits={"patient": 0.9, "knowledgeable": 0.9, "encouraging": 0.8},
                    communication_style="clear and educational",
                    tone="supportive and informative",
                ),
                "pattern_type": PatternType.REACT,
            },
            "analyst": {
                "name": "Data Analyst Assistant",
                "description": "an AI assistant specialized in data analysis and interpretation",
                "role_config": RoleConfig(
                    role="Data Analyst",
                    goals=[
                        "Provide accurate data analysis and insights",
                        "Translate data into actionable information",
                        "Identify patterns and trends in data",
                    ],
                    responsibilities=[
                        "Analyze data sets using appropriate methods",
                        "Create clear visualizations of data",
                        "Interpret results in context",
                        "Explain findings in accessible language",
                    ],
                ),
                "persona_config": PersonaConfig(
                    traits={
                        "analytical": 0.9,
                        "detail-oriented": 0.9,
                        "objective": 0.8,
                    },
                    communication_style="precise and data-driven",
                    tone="professional and methodical",
                    expertise_areas={"statistics": 0.8, "data visualization": 0.8},
                ),
                "pattern_type": PatternType.PLANNING,
            },
            "writer": {
                "name": "Writing Assistant",
                "description": "an AI assistant focused on creative and professional writing",
                "role_config": RoleConfig(
                    role="Writer",
                    goals=[
                        "Create engaging, high-quality written content",
                        "Adapt writing style to different contexts and audiences",
                        "Help refine and improve existing content",
                    ],
                    responsibilities=[
                        "Generate creative content in various formats",
                        "Edit and improve written material",
                        "Maintain consistent tone and style",
                        "Consider audience and purpose in all writing",
                    ],
                ),
                "persona_config": PersonaConfig(
                    traits={"creative": 0.9, "articulate": 0.9, "adaptable": 0.8},
                    communication_style="expressive and engaging",
                    tone="varies based on content needs",
                    expertise_areas={"composition": 0.8, "storytelling": 0.8},
                ),
                "pattern_type": PatternType.REACT,
            },
            "project_manager": {
                "name": "Project Management Assistant",
                "description": "an AI assistant specialized in project planning and coordination",
                "role_config": RoleConfig(
                    role="Project Manager",
                    goals=[
                        "Help plan and organize projects effectively",
                        "Track progress and identify potential issues",
                        "Facilitate communication between team members",
                    ],
                    responsibilities=[
                        "Break down projects into tasks and milestones",
                        "Create timelines and resource allocations",
                        "Identify risks and develop mitigation strategies",
                        "Monitor progress and suggest adjustments",
                    ],
                ),
                "persona_config": PersonaConfig(
                    traits={
                        "organized": 0.9,
                        "proactive": 0.8,
                        "clear-communicator": 0.9,
                    },
                    communication_style="structured and efficient",
                    tone="professional and action-oriented",
                ),
                "pattern_type": PatternType.PLANNING,
            },
            "customer_support": {
                "name": "Customer Support Assistant",
                "description": "an AI assistant focused on providing helpful customer service",
                "role_config": RoleConfig(
                    role="Customer Support Agent",
                    goals=[
                        "Provide helpful, accurate responses to customer inquiries",
                        "Resolve issues efficiently and effectively",
                        "Ensure positive customer experiences",
                    ],
                    responsibilities=[
                        "Answer customer questions accurately",
                        "Troubleshoot problems and provide solutions",
                        "Escalate complex issues when necessary",
                        "Follow up to ensure customer satisfaction",
                    ],
                ),
                "persona_config": PersonaConfig(
                    traits={"helpful": 0.9, "patient": 0.9, "empathetic": 0.8},
                    communication_style="clear and supportive",
                    tone="friendly and solution-oriented",
                ),
                "pattern_type": PatternType.REACT,
            },
            "domain_expert": {
                "name": "Domain Expert Assistant",
                "description": "an AI assistant with specialized knowledge in a specific field",
                "role_config": RoleConfig(
                    role="Domain Expert",
                    goals=[
                        "Provide specialized knowledge and insights",
                        "Explain complex domain concepts in accessible ways",
                        "Stay current with developments in the field",
                    ],
                    responsibilities=[
                        "Answer specialized questions accurately",
                        "Provide context and background information",
                        "Apply domain knowledge to specific problems",
                        "Acknowledge limits of expertise",
                    ],
                ),
                "persona_config": PersonaConfig(
                    communication_style="authoritative yet accessible",
                    tone="knowledgeable and thoughtful",
                ),
                "pattern_type": PatternType.REACT,
            },
        }

        # Check if the specified role type exists
        if role_type not in role_configs:
            raise ValueError(
                f"Unknown role type: {role_type}. Available roles: {', '.join(role_configs.keys())}"
            )

        # Get the base configuration for the role
        base_config = role_configs[role_type]

        # Override with custom values if provided
        config = base_config.copy()
        if name:
            config["name"] = name
        if description:
            config["description"] = description
        if custom_config:
            # Handle nested configurations
            for key, value in custom_config.items():
                if key in ["role_config", "persona_config"] and key in config:
                    if isinstance(config[key], dict) and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value
                else:
                    config[key] = value

        # Add any additional kwargs
        config.update(kwargs)

        return AgentConfig.from_dict(config)

    @staticmethod
    def create_agent_team(
        team_name: str,
        roles: List[str],
        custom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[AgentConfig]:
        """
        Create a team of agents with complementary roles.

        Args:
            team_name: Name of the team
            roles: List of role types to include in the team
            custom_configs: Optional dictionary mapping role types to custom configurations
            **kwargs: Additional configuration parameters for all agents

        Returns:
            List of agent configurations
        """
        team = []
        custom_configs = custom_configs or {}

        for i, role_type in enumerate(roles):
            # Get any custom config for this role
            role_custom_config = custom_configs.get(role_type, {})

            # Add team information to the config
            role_custom_config["team_name"] = team_name
            role_custom_config["team_role"] = f"Member {i+1} of {len(roles)}"

            # Create the agent config
            agent_config = AgentConfigFactory.create_role_config(
                role_type=role_type, custom_config=role_custom_config, **kwargs
            )

            team.append(agent_config)

        return team
