from typing import List, Dict, Any, Literal, Optional, Callable
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
import asyncio

from .state import AgentState
from .skill import BaseSkill
from .logger import AgentLogger
from .tool import EnhancedTool


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

        for skill in self.skills:
            skill.initialize(self.config)

    def _collect_tools(self) -> List[EnhancedTool]:
        """
        Collect tools from all skills.

        Returns:
            List of tools from all skills
        """
        tools = []
        for skill in self.skills:
            for tool in skill.get_tools():
                # Attach extra metadata
                tool.metadata = {
                    "skill": skill.name,
                    "input_schema": getattr(tool, "args_schema", None),
                    "examples": getattr(tool, "examples", []),
                    "category": getattr(tool, "category", "general"),
                }
                tools.append(tool)
        return tools

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

    async def _agent_executor(self, state: AgentState) -> Dict[str, Any]:
        """
        Async execution of agent reasoning.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        try:
            response = await self.llm.ainvoke(state.messages)
            state.add_message("assistant", response.content)
            return state.model_dump()
        except Exception as e:
            self.logger.log_agent_event(
                "agent_execution_error", {"error": str(e)}, level="ERROR"
            )
            raise

    async def _tool_executor(self, state: AgentState) -> Dict[str, Any]:
        """Async tool execution with basic tool selection logic."""
        try:
            user_input = state.get_last_user_message()
            if not user_input:
                return state.model_dump()

            for skill in self.skills:
                recommendation = skill.recommend_tool(state)
                if recommendation:
                    tool_name, tool_inputs = recommendation
                    matching_tool = next(
                        (t for t in self.tools if t.name == tool_name), None
                    )
                    if matching_tool:
                        result = await asyncio.to_thread(
                            matching_tool.run, **tool_inputs
                        )
                        state.add_tool_message(tool_name, str(result))
                        state.record_tool_execution(tool_name, tool_inputs, result)
                        return state.model_dump()

            self.logger.log_agent_event(
                "tool_not_found",
                {"reason": "No recommended tool matched"},
                level="WARNING",
            )
            return state.model_dump()

        except Exception as e:
            self.logger.log_agent_event(
                "tool_execution_error", {"error": str(e)}, level="ERROR"
            )
            raise

    def _route(self, state: AgentState) -> Literal["agent", "tools", "__end__"]:
        """Routing based on tool recommendation presence."""
        for skill in self.skills:
            if skill.recommend_tool(state):
                return "tools"
        return "__end__"
