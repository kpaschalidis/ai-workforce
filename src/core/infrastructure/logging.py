"""
Enhanced logging system for the AI agent framework.

This module provides a comprehensive logging system with features like
structured logging, log rotation, performance tracking, and more.
"""

import logging
import os
import sys
import json
import time
import functools
from datetime import datetime
from typing import Dict, Any, Optional, Union
from logging.handlers import RotatingFileHandler


class LogFormatter(logging.Formatter):
    """Enhanced log formatter with support for structured data."""

    def __init__(self, fmt=None, datefmt=None, style="%", validate=True, colored=False):
        """
        Initialize the formatter.

        Args:
            fmt: Format string
            datefmt: Date format string
            style: Format style
            validate: Whether to validate the format
            colored: Whether to use colored output
        """
        super().__init__(fmt, datefmt, style, validate)
        self.colored = colored

        # ANSI color codes
        self.colors = {
            "DEBUG": "\033[94m",  # Blue
            "INFO": "\033[92m",  # Green
            "WARNING": "\033[93m",  # Yellow
            "ERROR": "\033[91m",  # Red
            "CRITICAL": "\033[95m",  # Magenta
            "RESET": "\033[0m",  # Reset
        }

    def format(self, record):
        """
        Format a log record.

        Args:
            record: Log record

        Returns:
            Formatted log message
        """
        # Process structured data if available
        if hasattr(record, "data") and record.data:
            # Convert data to a string representation
            if isinstance(record.data, dict):
                data_str = json.dumps(record.data)
            else:
                data_str = str(record.data)

            record.msg = f"{record.msg} | Data: {data_str}"

        # Apply colors if enabled
        if self.colored and record.levelname in self.colors:
            levelname = record.levelname
            record.levelname = (
                f"{self.colors[levelname]}{levelname}{self.colors['RESET']}"
            )

        return super().format(record)


class AgentLogger:
    """
    Advanced logging utility for AI agents.
    """

    def __init__(
        self,
        name: str = "agent",
        log_dir: str = "logs",
        log_level: str = "INFO",
        console_level: Optional[str] = None,
        file_level: Optional[str] = None,
        max_log_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        log_format: Optional[str] = None,
        date_format: str = "%Y-%m-%d %H:%M:%S",
        colored_console: bool = True,
        capture_exceptions: bool = True,
        log_to_file: bool = True,
    ):
        """
        Initialize advanced logger.

        Args:
            name: Logger name
            log_dir: Directory to store log files
            log_level: Master logging level
            console_level: Console log level (defaults to log_level)
            file_level: File log level (defaults to log_level)
            max_log_size: Maximum log file size
            backup_count: Number of backup log files
            log_format: Log format string
            date_format: Date format string
            colored_console: Whether to use colored console output
            capture_exceptions: Whether to automatically capture uncaught exceptions
            log_to_file: Whether to log to file
        """
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        self.logger.handlers = []

        # Default log format
        if log_format is None:
            log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

        # Console level defaults to main level if not specified
        if console_level is None:
            console_level = log_level

        # File level defaults to main level if not specified
        if file_level is None:
            file_level = log_level

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = LogFormatter(
            fmt=log_format, datefmt=date_format, colored=colored_console
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Create file handler if enabled
        if log_to_file:
            # Ensure log directory exists
            os.makedirs(log_dir, exist_ok=True)

            # Create file handler
            log_file = os.path.join(log_dir, f"{name}.log")
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_log_size, backupCount=backup_count
            )
            file_handler.setLevel(getattr(logging, file_level.upper()))
            file_formatter = LogFormatter(
                fmt=log_format, datefmt=date_format, colored=False
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Set up exception capturing
        if capture_exceptions:
            self._setup_exception_capture()

        # Track performance
        self.performance_data = {}

        # Store configuration
        self.config = {
            "name": name,
            "log_dir": log_dir,
            "log_level": log_level,
            "console_level": console_level,
            "file_level": file_level,
            "max_log_size": max_log_size,
            "backup_count": backup_count,
            "log_format": log_format,
            "date_format": date_format,
            "colored_console": colored_console,
            "capture_exceptions": capture_exceptions,
            "log_to_file": log_to_file,
        }

    def _setup_exception_capture(self):
        """Set up capturing of uncaught exceptions."""

        def exception_handler(exctype, value, tb):
            """Handle uncaught exception."""
            self.logger.critical("Uncaught exception", exc_info=(exctype, value, tb))
            # Call the default exception handler
            sys.__excepthook__(exctype, value, tb)

        # Set the exception hook
        sys.excepthook = exception_handler

    def debug(self, msg, *args, data=None, **kwargs):
        """Log a debug message."""
        if data:
            extra = kwargs.get("extra", {})
            extra["data"] = data
            kwargs["extra"] = extra
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, data=None, **kwargs):
        """Log an info message."""
        if data:
            extra = kwargs.get("extra", {})
            extra["data"] = data
            kwargs["extra"] = extra
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, data=None, **kwargs):
        """Log a warning message."""
        if data:
            extra = kwargs.get("extra", {})
            extra["data"] = data
            kwargs["extra"] = extra
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, data=None, **kwargs):
        """Log an error message."""
        if data:
            extra = kwargs.get("extra", {})
            extra["data"] = data
            kwargs["extra"] = extra
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, data=None, **kwargs):
        """Log a critical message."""
        if data:
            extra = kwargs.get("extra", {})
            extra["data"] = data
            kwargs["extra"] = extra
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, data=None, **kwargs):
        """Log an exception."""
        if data:
            extra = kwargs.get("extra", {})
            extra["data"] = data
            kwargs["extra"] = extra
        self.logger.exception(msg, *args, **kwargs)

    def log_structured(self, level: Union[int, str], msg: str, data: Dict[str, Any]):
        """
        Log a structured message with data.

        Args:
            level: Log level
            msg: Log message
            data: Structured data
        """
        # Convert string level to int if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        # Log with data
        self.logger.log(level, msg, extra={"data": data})

    def track_performance(
        self, metric_name: str, value: float, context: Optional[Dict[str, Any]] = None
    ):
        """
        Track performance metrics.

        Args:
            metric_name: Name of the metric
            value: Metric value
            context: Optional context details
        """
        timestamp = datetime.now().isoformat()

        if metric_name not in self.performance_data:
            self.performance_data[metric_name] = []

        self.performance_data[metric_name].append(
            {"value": value, "timestamp": timestamp, "context": context or {}}
        )

        # Log the metric
        self.info(
            f"Performance Metric: {metric_name} = {value}",
            data={"metric": metric_name, "value": value, "context": context},
        )

    def get_performance_data(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get tracked performance data.

        Args:
            metric_name: Optional specific metric name

        Returns:
            Performance data
        """
        if metric_name:
            return {metric_name: self.performance_data.get(metric_name, [])}
        return self.performance_data

    def log_execution_time(self, func=None, *, level="INFO", threshold=None):
        """
        Decorator to log the execution time of a function.

        Args:
            func: Function to decorate
            level: Log level
            threshold: Optional time threshold in seconds

        Returns:
            Decorated function
        """

        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                # Log the start of the function
                self.debug(f"Starting {f.__name__}")

                # Record the start time
                start_time = time.time()

                try:
                    # Execute the function
                    result = f(*args, **kwargs)

                    # Record the end time
                    end_time = time.time()

                    # Calculate the duration
                    duration = end_time - start_time

                    # Only log if above threshold (if specified)
                    if threshold is None or duration >= threshold:
                        # Log the completion
                        log_method = getattr(self, level.lower())
                        log_method(
                            f"Finished {f.__name__} in {duration:.4f}s",
                            data={"function": f.__name__, "duration": duration},
                        )

                    # Track performance
                    self.track_performance(
                        f"function_execution_time.{f.__name__}", duration
                    )

                    return result
                except Exception as e:
                    # Record the end time
                    end_time = time.time()

                    # Calculate the duration
                    duration = end_time - start_time

                    # Log the error
                    self.exception(
                        f"Error in {f.__name__} after {duration:.4f}s: {str(e)}",
                        data={
                            "function": f.__name__,
                            "duration": duration,
                            "error": str(e),
                        },
                    )

                    # Re-raise the exception
                    raise

            return wrapper

        # Handle both @log_execution_time and @log_execution_time()
        if func is None:
            return decorator
        return decorator(func)

    def async_log_execution_time(self, func=None, *, level="INFO", threshold=None):
        """
        Decorator to log the execution time of an async function.

        Args:
            func: Async function to decorate
            level: Log level
            threshold: Optional time threshold in seconds

        Returns:
            Decorated async function
        """

        def decorator(f):
            @functools.wraps(f)
            async def wrapper(*args, **kwargs):
                # Log the start of the function
                self.debug(f"Starting async {f.__name__}")

                # Record the start time
                start_time = time.time()

                try:
                    # Execute the function
                    result = await f(*args, **kwargs)

                    # Record the end time
                    end_time = time.time()

                    # Calculate the duration
                    duration = end_time - start_time

                    # Only log if above threshold (if specified)
                    if threshold is None or duration >= threshold:
                        # Log the completion
                        log_method = getattr(self, level.lower())
                        log_method(
                            f"Finished async {f.__name__} in {duration:.4f}s",
                            data={"function": f.__name__, "duration": duration},
                        )

                    # Track performance
                    self.track_performance(
                        f"async_function_execution_time.{f.__name__}", duration
                    )

                    return result
                except Exception as e:
                    # Record the end time
                    end_time = time.time()

                    # Calculate the duration
                    duration = end_time - start_time

                    # Log the error
                    self.exception(
                        f"Error in async {f.__name__} after {duration:.4f}s: {str(e)}",
                        data={
                            "function": f.__name__,
                            "duration": duration,
                            "error": str(e),
                        },
                    )

                    # Re-raise the exception
                    raise

            return wrapper

        # Handle both @async_log_execution_time and @async_log_execution_time()
        if func is None:
            return decorator
        return decorator(func)


# Helper functions
def get_logger(name: str, **kwargs) -> AgentLogger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name
        **kwargs: Additional configuration options

    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = AgentLogger(name=name, **kwargs)
    _loggers[name] = logger
    return logger


# Global logger registry
_loggers = {}


# Default logger
default_logger = get_logger("agent")
