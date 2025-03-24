from typing import Callable, Dict, Any, Optional
from langchain_core.tools import BaseTool
from core.skill import BaseSkill


class EnhancedTool(BaseTool):
    """
    Enhanced tool with advanced capabilities and tight skill integration.
    """

    def __init__(
        self,
        skill: BaseSkill,
        name: str,
        description: str,
        func: Callable,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        input_validator: Optional[Callable] = None,
    ):
        """
        Initialize an enhanced tool.

        Args:
            skill: Parent skill
            name: Tool name
            description: Tool description
            func: Function to execute
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            input_validator: Optional custom input validator
        """
        super().__init__(name=name, description=description, func=func)
        self.skill = skill
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.input_validator = input_validator or skill.validate_input

    def _run(self, *args, **kwargs):
        """
        Enhanced run method with:
        - Input validation
        - Retry mechanism
        - Comprehensive error handling
        """
        import time

        # Validate inputs
        if not self.input_validator(self.name, kwargs):
            raise ValueError(f"Invalid input for tool {self.name}")

        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                result = super()._run(*args, **kwargs)

                # Log successful execution
                self.skill.log_execution(self.name, kwargs, result)

                return result

            except Exception as e:
                self.skill.handle_error(e, {"attempt": attempt + 1})

                if attempt == self.max_retries - 1:
                    raise

                time.sleep(self.retry_delay)
