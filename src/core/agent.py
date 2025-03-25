from typing import List, Dict, Any, Literal, Optional, Callable
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .state import AgentState
from .skill import BaseSkill
from .logger import AgentLogger


class GenericAIAgent:
    """
    Comprehensive, configurable AI agent framework.
    """

    def __init__(
        self,
        skills: List[BaseSkill],
        llm: Optional[BaseChatModel] = None,
        logger: Optional[AgentLogger] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the generic AI agent.

        Args:
            skills: List of skills for the agent
            llm: Language model (defaults to OpenAI)
            logger: Custom logger
            config: Agent configuration
        """
        self.skills = skills
        self.tools = self._collect_tools()
        self.llm = llm or ChatOpenAI(model="gpt-4-turbo")
        self.logger = logger or AgentLogger()
        self.config = config or {}

        # Initialize skills
        for skill in self.skills:
            skill.initialize(self.config)

    def _collect_tools(self) -> List[Any]:
        """
        Collect tools from all skills.

        Returns:
            List of tools from all skills
        """
        return [tool for skill in self.skills for tool in skill.get_tools()]

    def create_workflow(self) -> Callable:
        """
        Create a generic agent workflow.

        Returns:
            Compiled workflow graph
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._agent_executor)
        workflow.add_node("tools", self._tool_executor)
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges(
            "agent",
            lambda state: self._route(state),
            {"tools": "tools", "__end__": END},
        )

        workflow.set_entry_point("agent")

        return workflow.compile()

    def _agent_executor(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute agent reasoning.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        try:
            # LLM reasoning logic
            response = self.llm.invoke(state.messages)

            # Update state
            state.add_message("assistant", response.content)

            for skill in self.skills:
                suggested_action = skill.suggest_action(state)
                if suggested_action:
                    state.add_action(**suggested_action)

            return state.model_dump()
        except Exception as e:
            self.logger.log_agent_event(
                "agent_execution_error", {"error": str(e)}, level="ERROR"
            )
            raise

    def _tool_executor(self, state: AgentState) -> Dict[str, Any]:
        """Default tool execution does nothing unless overridden."""
        self.logger.log_agent_event(
            "tool_executor_skipped", {"reason": "No tools defined"}
        )
        return state.model_dump()

    def _route(self, state: AgentState) -> Literal["agent", "tools", "__end__"]:
        """Default routing based on whether tools are defined."""
        if self.tools:
            return "tools"
        return "__end__"
