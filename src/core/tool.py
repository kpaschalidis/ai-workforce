from typing import Callable, Optional, Any
from langchain_core.tools import BaseTool
from pydantic import Field, field_validator, model_validator
from .skill import BaseSkill


class EnhancedTool(BaseTool):
    """
    Enhanced tool with advanced capabilities and tight skill integration.
    """

    # Public model fields
    skill: BaseSkill = Field(description="Parent skill for this tool")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )
    input_validator: Optional[Callable] = Field(
        default=None, description="Custom input validator function"
    )

    def __init__(
        self,
        skill: BaseSkill,
        name: str,
        description: str,
        func: Callable,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        input_validator: Optional[Callable] = None,
        **kwargs,
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
        # Initialize BaseTool first
        super().__init__(name=name, description=description, func=func, **kwargs)

        # Set our fields
        self.skill = skill
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.input_validator = input_validator or skill.validate_input

    # Field validators
    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_retries must be at least 1")
        return v

    @field_validator("retry_delay")
    @classmethod
    def validate_retry_delay(cls, v: float) -> float:
        if v < 0:
            raise ValueError("retry_delay cannot be negative")
        return v

    # Ensure input_validator is set properly
    @model_validator(mode="after")
    def set_default_validator(self) -> "EnhancedTool":
        if self.input_validator is None and hasattr(self, "skill"):
            self.input_validator = self.skill.validate_input
        return self

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Enhanced run method with:
        - Input validation
        - Retry mechanism
        - Comprehensive error handling
        """
        import time

        # Validate input
        if not self.input_validator(self.name, kwargs):
            raise ValueError(f"Invalid input for tool {self.name}")

        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                result = super()._run(*args, **kwargs)
                self.skill.log_execution(self.name, kwargs, result)
                return result

            except Exception as e:
                self.skill.handle_error(e, {"attempt": attempt + 1})

                # If this was the last attempt, re-raise the exception
                if attempt == self.max_retries - 1:
                    raise

                # Wait before retrying
                time.sleep(self.retry_delay)
