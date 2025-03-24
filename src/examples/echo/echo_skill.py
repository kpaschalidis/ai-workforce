from src.core.skill import BaseSkill
from src.core.tool import EnhancedTool


class EchoSkill(BaseSkill):
    def get_tools(self):
        echo_tool = EnhancedTool(
            skill=self,
            name="echo",
            description="Echoes back the input text.",
            func=self.echo,
        )
        self.tools = [echo_tool]
        return self.tools

    def initialize(self, config):
        self.logger.info("EchoSkill initialized")

    def echo(self, input_text: str) -> str:
        return f"Echo: {input_text}"
