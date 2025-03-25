from typing import Callable, Optional
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr
from .skill import BaseSkill


class EnhancedTool(BaseTool):
    """
    Enhanced tool with advanced capabilities and tight skill integration.
    """

    _skill: BaseSkill = PrivateAttr()
    _max_retries: int = PrivateAttr()
    _retry_delay: float = PrivateAttr()
    _input_validator: Callable = PrivateAttr()

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

        self._skill = skill
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._input_validator = input_validator or skill.validate_input

    def _run(self, *args, **kwargs):
        """
        Enhanced run method with:
        - Input validation
        - Retry mechanism
        - Comprehensive error handling
        """
        import time

        if not self._input_validator(self.name, kwargs):
            raise ValueError(f"Invalid input for tool {self.name}")

        # Retry mechanism
        for attempt in range(self._max_retries):
            try:
                result = super()._run(*args, **kwargs)
                self._skill.log_execution(self.name, kwargs, result)
                return result

            except Exception as e:
                self._skill.handle_error(e, {"attempt": attempt + 1})

                if attempt == self._max_retries - 1:
                    raise

                time.sleep(self._retry_delay)
