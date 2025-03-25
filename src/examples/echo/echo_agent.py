from src.core.agent import GenericAIAgent
from src.core.state import AgentState
from typing import Dict, Any, Literal


class EchoAgent(GenericAIAgent):
    def _tool_executor(self, state: AgentState) -> Dict[str, Any]:
        tool = next((t for t in self.tools if t.name == "echo"), None)
        if not tool:
            return state.model_dump()

        last_input = state.get_last_user_message()
        result = tool.invoke({"input_text": last_input})

        state.record_tool_execution(tool.name, {"input_text": last_input}, result)
        state.add_tool_message(tool.name, result)

        return state.model_dump()

    def _route(self, state: AgentState) -> Literal["agent", "tools", "__end__"]:
        if not state.get_last_user_message():
            return "__end__"
        return "tools"
