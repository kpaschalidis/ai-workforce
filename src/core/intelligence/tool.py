"""
Enhanced tool implementation for the AI agent framework.

This module provides enhanced tool capabilities beyond the basic LangChain tools,
including better error handling, retries, validation, and more.
"""

import asyncio
from typing import Callable, Optional, Dict, Any, Type
import time
import functools
import traceback
import inspect

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from ..infrastructure import get_logger

logger = get_logger("tools")


class EnhancedTool(BaseTool):
    """
    Enhanced tool with advanced capabilities and tight skill integration.

    This class extends LangChain's BaseTool with additional functionality
    for validation, retries, and error handling.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        skill: Optional[Any] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        input_validator: Optional[Callable] = None,
        output_validator: Optional[Callable] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
        verbose: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an enhanced tool.

        Args:
            name: Tool name
            description: Tool description
            func: Function to execute
            skill: Parent skill
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            input_validator: Optional custom input validator
            output_validator: Optional custom output validator
            args_schema: Optional schema for arguments
            return_direct: Whether to return the output directly to the user
            verbose: Whether to output verbose logging
            metadata: Additional metadata for the tool
        """
        # Store additional attributes
        self._skill = skill
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._input_validator = input_validator
        self._output_validator = output_validator
        self._verbose = verbose

        # Create args schema if not provided
        if args_schema is None:
            args_schema = self._create_args_schema_from_function(func)

        # Call parent constructor
        super().__init__(
            name=name,
            description=description,
            func=self._enhanced_func_wrapper(func),
            args_schema=args_schema,
            return_direct=return_direct,
        )

        # Add metadata
        self.metadata = metadata or {}

    def _create_args_schema_from_function(self, func: Callable) -> Type[BaseModel]:
        """
        Create a pydantic model schema from a function's signature.

        Args:
            func: Function to analyze

        Returns:
            Pydantic model for function arguments
        """
        # Get function signature
        signature = inspect.signature(func)

        # Create field definitions
        fields = {}

        for param_name, param in signature.parameters.items():
            # Skip self, *args, and **kwargs
            if param_name == "self" or param.kind in (
                param.VAR_POSITIONAL,
                param.VAR_KEYWORD,
            ):
                continue

            # Get type annotation or default to Any
            annotation = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )

            # Get default value or make field required
            default = param.default if param.default != inspect.Parameter.empty else ...

            # Create field with description from docstring if available
            fields[param_name] = (
                annotation,
                Field(default, description=f"{param_name} parameter"),
            )

        # Create the model dynamically
        model_name = f"{func.__name__.title()}Schema"
        return create_model(model_name, **fields)

    def _enhanced_func_wrapper(self, func: Callable) -> Callable:
        """
        Wrap the function with validation, retry, and error handling.

        Args:
            func: Original function

        Returns:
            Enhanced function
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get start time for performance tracking
            start_time = time.time()

            # Log invocation if verbose
            if self._verbose:
                logger.info(f"Invoking tool: {self.name}")
                logger.debug(f"Args: {args}, Kwargs: {kwargs}")

            # Validate input if validator exists
            if self._input_validator:
                try:
                    is_valid = self._input_validator(self.name, kwargs)
                    if not is_valid:
                        error_msg = f"Invalid input for tool {self.name}"
                        logger.warning(error_msg)
                        return error_msg
                except Exception as e:
                    error_msg = (
                        f"Error in input validation for tool {self.name}: {str(e)}"
                    )
                    logger.error(error_msg)
                    return error_msg

            # Retry mechanism
            for attempt in range(self._max_retries):
                try:
                    # Call the original function
                    result = func(*args, **kwargs)

                    # Validate output if validator exists
                    if self._output_validator:
                        try:
                            is_valid = self._output_validator(self.name, result)
                            if not is_valid:
                                error_msg = f"Invalid output from tool {self.name}"
                                logger.warning(error_msg)
                                return error_msg
                        except Exception as e:
                            error_msg = f"Error in output validation for tool {self.name}: {str(e)}"
                            logger.error(error_msg)
                            return error_msg

                    # Log execution in skill if available
                    if self._skill and hasattr(self._skill, "log_execution"):
                        self._skill.log_execution(self.name, kwargs, result)

                    # Log success if verbose
                    if self._verbose:
                        duration = time.time() - start_time
                        logger.info(
                            f"Tool {self.name} executed successfully in {duration:.2f}s"
                        )
                        logger.debug(f"Result: {result}")

                    return result

                except Exception as e:
                    # Handle error in skill if available
                    if self._skill and hasattr(self._skill, "handle_error"):
                        self._skill.handle_error(
                            e, {"attempt": attempt + 1, "args": kwargs}
                        )

                    # Log the error
                    if attempt == self._max_retries - 1:
                        # Last attempt, log as error
                        logger.error(
                            f"Tool {self.name} failed after {attempt + 1} attempts: {str(e)}"
                        )
                        logger.debug(f"Error details: {traceback.format_exc()}")
                        return f"Error executing tool {self.name}: {str(e)}"
                    else:
                        # More attempts left, log as warning
                        logger.warning(
                            f"Tool {self.name} failed (attempt {attempt + 1}): {str(e)}"
                        )
                        logger.debug(f"Retrying in {self._retry_delay}s...")

                        # Wait before retrying
                        time.sleep(self._retry_delay)

            # Should never reach here, but just in case
            return f"Error: Tool {self.name} failed after all retry attempts"

        return wrapper


class AsyncTool(EnhancedTool):
    """
    Tool that supports asynchronous execution.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        async_func: Callable,
        **kwargs,
    ):
        """
        Initialize an async tool.

        Args:
            name: Tool name
            description: Tool description
            func: Synchronous function to execute
            async_func: Asynchronous function to execute
            **kwargs: Additional arguments for EnhancedTool
        """
        super().__init__(name=name, description=description, func=func, **kwargs)
        self._async_func = self._enhanced_async_func_wrapper(async_func)

    async def _arun(self, *args, **kwargs):
        """
        Run the tool asynchronously.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool result
        """
        return await self._async_func(*args, **kwargs)

    def _enhanced_async_func_wrapper(self, async_func: Callable) -> Callable:
        """
        Wrap the async function with validation, retry, and error handling.

        Args:
            async_func: Original async function

        Returns:
            Enhanced async function
        """

        @functools.wraps(async_func)
        async def wrapper(*args, **kwargs):
            # Get start time for performance tracking
            start_time = time.time()

            # Log invocation if verbose
            if self._verbose:
                logger.info(f"Invoking async tool: {self.name}")
                logger.debug(f"Args: {args}, Kwargs: {kwargs}")

            # Validate input if validator exists
            if self._input_validator:
                try:
                    is_valid = self._input_validator(self.name, kwargs)
                    if not is_valid:
                        error_msg = f"Invalid input for async tool {self.name}"
                        logger.warning(error_msg)
                        return error_msg
                except Exception as e:
                    error_msg = f"Error in input validation for async tool {self.name}: {str(e)}"
                    logger.error(error_msg)
                    return error_msg

            # Retry mechanism
            for attempt in range(self._max_retries):
                try:
                    # Call the original async function
                    result = await async_func(*args, **kwargs)

                    # Validate output if validator exists
                    if self._output_validator:
                        try:
                            is_valid = self._output_validator(self.name, result)
                            if not is_valid:
                                error_msg = (
                                    f"Invalid output from async tool {self.name}"
                                )
                                logger.warning(error_msg)
                                return error_msg
                        except Exception as e:
                            error_msg = f"Error in output validation for async tool {self.name}: {str(e)}"
                            logger.error(error_msg)
                            return error_msg

                    # Log execution in skill if available
                    if self._skill and hasattr(self._skill, "log_execution"):
                        self._skill.log_execution(self.name, kwargs, result)

                    # Log success if verbose
                    if self._verbose:
                        duration = time.time() - start_time
                        logger.info(
                            f"Async tool {self.name} executed successfully in {duration:.2f}s"
                        )
                        logger.debug(f"Result: {result}")

                    return result

                except Exception as e:
                    # Handle error in skill if available
                    if self._skill and hasattr(self._skill, "handle_error"):
                        self._skill.handle_error(
                            e, {"attempt": attempt + 1, "args": kwargs}
                        )

                    # Log the error
                    if attempt == self._max_retries - 1:
                        # Last attempt, log as error
                        logger.error(
                            f"Async tool {self.name} failed after {attempt + 1} attempts: {str(e)}"
                        )
                        logger.debug(f"Error details: {traceback.format_exc()}")
                        return f"Error executing async tool {self.name}: {str(e)}"
                    else:
                        # More attempts left, log as warning
                        logger.warning(
                            f"Async tool {self.name} failed (attempt {attempt + 1}): {str(e)}"
                        )
                        logger.debug(f"Retrying in {self._retry_delay}s...")

                        # Wait before retrying
                        await asyncio.sleep(self._retry_delay)

            # Should never reach here, but just in case
            return f"Error: Async tool {self.name} failed after all retry attempts"

        return wrapper
