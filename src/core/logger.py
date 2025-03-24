from datetime import datetime
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any


class AgentLogger:
    """
    Advanced logging utility for AI agents.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        max_log_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
    ):
        """
        Initialize advanced logger.

        Args:
            log_dir: Directory to store log files
            log_level: Logging level
            max_log_size: Maximum log file size
            backup_count: Number of backup log files
        """
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger("agent_logger")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Create formatters
        self.console_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "agent.log"),
            maxBytes=max_log_size,
            backupCount=backup_count,
        )
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)

    def log_agent_event(
        self, event_type: str, details: Dict[str, Any], level: str = "INFO"
    ):
        """
        Log a structured agent event.

        Args:
            event_type: Type of event
            details: Event details
            level: Logging level
        """
        log_method = getattr(self.logger, level.lower())
        log_method(f"Event: {event_type} - {details}")

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
        log_entry = f"Performance Metric: {metric_name} = {value}"
        if context:
            log_entry += f" | Context: {context}"

        self.logger.info(log_entry)

    def log_execution_time(func):
        """Decorator to log the execution time of a function."""

        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)

            # Log the start of the function
            logger.debug(f"Starting {func.__name__}")

            # Record the start time
            start_time = datetime.now()

            # Execute the function
            result = func(*args, **kwargs)

            # Record the end time
            end_time = datetime.now()

            # Calculate the duration
            duration = end_time - start_time

            # Log the end of the function
            logger.debug(f"Finished {func.__name__} in {duration.total_seconds():.2f}s")

            return result

        return wrapper

    def log_async_execution_time(func):
        """Decorator to log the execution time of an async function."""

        async def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)

            # Log the start of the function
            logger.debug(f"Starting {func.__name__}")

            # Record the start time
            start_time = datetime.now()

            # Execute the function
            result = await func(*args, **kwargs)

            # Record the end time
            end_time = datetime.now()

            # Calculate the duration
            duration = end_time - start_time

            # Log the end of the function
            logger.debug(f"Finished {func.__name__} in {duration.total_seconds():.2f}s")

            return result

        return wrapper


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module or component name

    Returns:
        Configured logger
    """
    return logging.getLogger(f"agent_logger.{name}")
