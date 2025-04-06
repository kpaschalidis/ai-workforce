"""
Event system for the AI agent framework.

This module provides an event-based communication system for the AI agent framework,
allowing components to communicate with each other in a loosely coupled way.
"""

from typing import Dict, List, Any, Callable, Optional, Set, Union
from enum import Enum, auto
import asyncio
import threading
import uuid
import time
from abc import ABC, abstractmethod


class AgentEventType(Enum):
    """Standard event types for the agent framework."""

    # Lifecycle events
    AGENT_INITIALIZED = auto()
    AGENT_STARTED = auto()
    AGENT_STOPPED = auto()
    AGENT_ERROR = auto()

    # Execution events
    BEFORE_LLM_CALL = auto()
    AFTER_LLM_CALL = auto()
    BEFORE_TOOL_CALL = auto()
    AFTER_TOOL_CALL = auto()
    TOOL_ERROR = auto()
    STEP_STARTED = auto()
    STEP_COMPLETED = auto()

    # Message events
    MESSAGE_RECEIVED = auto()
    MESSAGE_SENT = auto()
    SYSTEM_MESSAGE_ADDED = auto()

    # Skill events
    SKILL_ACTIVATED = auto()
    SKILL_COMPLETED = auto()

    # State events
    STATE_UPDATED = auto()
    CONTEXT_UPDATED = auto()
    MEMORY_STORED = auto()
    MEMORY_RETRIEVED = auto()

    # Task events
    TASK_CREATED = auto()
    TASK_ASSIGNED = auto()
    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()

    # Custom event for extensions
    CUSTOM = auto()


class AgentEvent:
    """Event object for agent framework events."""

    def __init__(
        self,
        event_type: AgentEventType,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        event_id: Optional[str] = None,
    ):
        """
        Initialize an agent event.

        Args:
            event_type: Type of event
            source: Source component of the event
            data: Optional event data
            metadata: Optional event metadata
            timestamp: Optional timestamp (defaults to current time)
            event_id: Optional event ID (defaults to UUID)
        """
        self.event_type = event_type
        self.source = source
        self.data = data or {}
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
        self.event_id = event_id or str(uuid.uuid4())

    def __str__(self) -> str:
        """String representation of the event."""
        return (
            f"AgentEvent(type={self.event_type.name}, "
            f"source={self.source}, id={self.event_id})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "source": self.source,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class EventListener(ABC):
    """Interface for event listeners."""

    @abstractmethod
    def handle_event(self, event: AgentEvent) -> None:
        """
        Handle an event.

        Args:
            event: Event to handle
        """
        pass


class AsyncEventListener(EventListener):
    """Interface for asynchronous event listeners."""

    @abstractmethod
    async def handle_event_async(self, event: AgentEvent) -> None:
        """
        Handle an event asynchronously.

        Args:
            event: Event to handle
        """
        pass

    def handle_event(self, event: AgentEvent) -> None:
        """
        Synchronous wrapper for async event handling.

        Args:
            event: Event to handle
        """
        # This is a fallback for sync contexts - prefer to use handle_event_async
        # Create a new event loop for this thread if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async handler
        loop.run_until_complete(self.handle_event_async(event))


class EventFilter:
    """Filter for events based on type, source, and other criteria."""

    def __init__(
        self,
        event_types: Optional[Union[AgentEventType, List[AgentEventType]]] = None,
        sources: Optional[Union[str, List[str]]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an event filter.

        Args:
            event_types: Event types to include
            sources: Event sources to include
            metadata_filter: Filter based on metadata
        """
        # Convert single values to lists
        self.event_types = self._to_set(event_types, AgentEventType)
        self.sources = self._to_set(sources, str)
        self.metadata_filter = metadata_filter or {}

    def matches(self, event: AgentEvent) -> bool:
        """
        Check if an event matches the filter.

        Args:
            event: Event to check

        Returns:
            True if the event matches, False otherwise
        """
        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check source
        if self.sources and event.source not in self.sources:
            return False

        # Check metadata
        for key, value in self.metadata_filter.items():
            if key not in event.metadata or event.metadata[key] != value:
                return False

        return True

    def _to_set(self, value, item_type):
        """Convert a value to a set of the specified type."""
        if value is None:
            return set()
        if isinstance(value, item_type):
            return {value}
        return set(value)


class EventBus:
    """
    Event bus for agent framework events with both synchronous and asynchronous support.
    """

    def __init__(self, async_mode: bool = False):
        """
        Initialize the event bus.

        Args:
            async_mode: Whether to use async mode
        """
        self.listeners: Dict[AgentEventType, List[EventListener]] = {}
        self.async_mode = async_mode
        self.async_loop = None
        self.async_thread = None

        # Initialize for all event types
        for event_type in AgentEventType:
            self.listeners[event_type] = []

    def subscribe(
        self,
        listener: EventListener,
        event_types: Optional[Union[AgentEventType, List[AgentEventType]]] = None,
        filter: Optional[EventFilter] = None,
    ) -> None:
        """
        Subscribe a listener to event types.

        Args:
            listener: Listener to subscribe
            event_types: Event types to subscribe to (None for all)
            filter: Optional filter for events
        """
        # Use all event types if none specified
        if event_types is None:
            subscribe_types = list(AgentEventType)
        elif isinstance(event_types, AgentEventType):
            subscribe_types = [event_types]
        else:
            subscribe_types = event_types

        # Wrap with filter if provided
        if filter:
            wrapped_listener = FilteredEventListener(listener, filter)
        else:
            wrapped_listener = listener

        # Subscribe to each event type
        for event_type in subscribe_types:
            if event_type not in self.listeners:
                self.listeners[event_type] = []

            self.listeners[event_type].append(wrapped_listener)

    def unsubscribe(
        self,
        listener: EventListener,
        event_types: Optional[Union[AgentEventType, List[AgentEventType]]] = None,
    ) -> None:
        """
        Unsubscribe a listener from event types.

        Args:
            listener: Listener to unsubscribe
            event_types: Event types to unsubscribe from (None for all)
        """
        # Use all event types if none specified
        if event_types is None:
            unsubscribe_types = list(AgentEventType)
        elif isinstance(event_types, AgentEventType):
            unsubscribe_types = [event_types]
        else:
            unsubscribe_types = event_types

        # Unsubscribe from each event type
        for event_type in unsubscribe_types:
            if event_type in self.listeners:
                # Check for wrapped listeners too
                self.listeners[event_type] = [
                    l
                    for l in self.listeners[event_type]
                    if (
                        l != listener
                        and not (
                            isinstance(l, FilteredEventListener)
                            and l.listener == listener
                        )
                    )
                ]

    def publish(self, event: AgentEvent) -> None:
        """
        Publish an event to subscribers.

        Args:
            event: Event to publish
        """
        if self.async_mode:
            self._ensure_async_loop()
            asyncio.run_coroutine_threadsafe(
                self._publish_async(event), self.async_loop
            )
        else:
            self._publish_sync(event)

    def _publish_sync(self, event: AgentEvent) -> None:
        """
        Synchronously publish an event.

        Args:
            event: Event to publish
        """
        for listener in self.listeners.get(event.event_type, []):
            try:
                listener.handle_event(event)
            except Exception as e:
                print(f"Error in event listener: {e}")

    async def _publish_async(self, event: AgentEvent) -> None:
        """
        Asynchronously publish an event.

        Args:
            event: Event to publish
        """
        for listener in self.listeners.get(event.event_type, []):
            try:
                # If listener has async support
                if isinstance(listener, AsyncEventListener):
                    await listener.handle_event_async(event)
                else:
                    # Run synchronous handler in executor
                    await asyncio.get_event_loop().run_in_executor(
                        None, listener.handle_event, event
                    )
            except Exception as e:
                print(f"Error in async event listener: {e}")

    def _ensure_async_loop(self) -> None:
        """Ensure async event loop is running."""
        if (
            self.async_loop is None
            or not self.async_thread
            or not self.async_thread.is_alive()
        ):
            self.async_loop = asyncio.new_event_loop()
            self.async_thread = threading.Thread(
                target=self._run_async_loop, daemon=True, name="EventBusAsyncThread"
            )
            self.async_thread.start()

    def _run_async_loop(self) -> None:
        """Run the async event loop."""
        asyncio.set_event_loop(self.async_loop)
        self.async_loop.run_forever()

    def shutdown(self) -> None:
        """Shutdown the event bus."""
        if self.async_loop and self.async_thread and self.async_thread.is_alive():
            self.async_loop.call_soon_threadsafe(self.async_loop.stop)
            self.async_thread.join(timeout=1.0)
            self.async_loop = None
            self.async_thread = None


class FilteredEventListener(EventListener):
    """Wrapper for event listeners that applies a filter."""

    def __init__(self, listener: EventListener, filter: EventFilter):
        """
        Initialize a filtered event listener.

        Args:
            listener: Listener to wrap
            filter: Filter to apply
        """
        self.listener = listener
        self.filter = filter

    def handle_event(self, event: AgentEvent) -> None:
        """
        Handle an event if it matches the filter.

        Args:
            event: Event to handle
        """
        if self.filter.matches(event):
            self.listener.handle_event(event)

    async def handle_event_async(self, event: AgentEvent) -> None:
        """
        Handle an event asynchronously if it matches the filter.

        Args:
            event: Event to handle
        """
        if self.filter.matches(event):
            if isinstance(self.listener, AsyncEventListener):
                await self.listener.handle_event_async(event)
            else:
                self.listener.handle_event(event)


# Example listeners
class LoggingEventListener(EventListener):
    """Event listener that logs events."""

    def __init__(self, logger):
        """
        Initialize a logging event listener.

        Args:
            logger: Logger to use
        """
        self.logger = logger

    def handle_event(self, event: AgentEvent) -> None:
        """
        Log the event.

        Args:
            event: Event to log
        """
        self.logger.info(
            f"Event: {event.event_type.name} | "
            f"Source: {event.source} | "
            f"ID: {event.event_id}"
        )
        if event.data:
            self.logger.debug(f"Event data: {event.data}")
        if event.metadata:
            self.logger.debug(f"Event metadata: {event.metadata}")


class MetricsEventListener(EventListener):
    """Event listener that collects metrics."""

    def __init__(self):
        """Initialize a metrics event listener."""
        self.counters = {}
        self.timers = {}
        self.start_times = {}

    def handle_event(self, event: AgentEvent) -> None:
        """
        Update metrics based on the event.

        Args:
            event: Event to process
        """
        event_name = event.event_type.name

        # Count event
        if event_name not in self.counters:
            self.counters[event_name] = 0
        self.counters[event_name] += 1

        # Track timing for paired events (BEFORE/AFTER)
        if event_name.startswith("BEFORE_"):
            # Store start time for this event pair
            base_name = event_name[7:]  # Remove "BEFORE_"
            event_pair_key = f"{base_name}_{event.source}_{event.data.get('id', '')}"
            self.start_times[event_pair_key] = event.timestamp

        elif event_name.startswith("AFTER_"):
            # Calculate duration if we have a matching start time
            base_name = event_name[6:]  # Remove "AFTER_"
            event_pair_key = f"{base_name}_{event.source}_{event.data.get('id', '')}"

            if event_pair_key in self.start_times:
                start_time = self.start_times.pop(event_pair_key)
                duration = event.timestamp - start_time

                timer_key = f"{base_name}_duration"
                if timer_key not in self.timers:
                    self.timers[timer_key] = []

                self.timers[timer_key].append(duration)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the collected metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = {"counters": self.counters.copy(), "timers": {}}

        # Calculate statistics for timers
        for timer_key, durations in self.timers.items():
            if durations:
                metrics["timers"][timer_key] = {
                    "count": len(durations),
                    "mean": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "total": sum(durations),
                }

        return metrics

    def reset(self) -> None:
        """Reset all metrics."""
        self.counters = {}
        self.timers = {}
        self.start_times = {}


# Helper function to create a simple event
def create_event(
    event_type: AgentEventType,
    source: str,
    data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentEvent:
    """
    Create an event with the given type and source.

    Args:
        event_type: Type of event
        source: Source of the event
        data: Optional event data
        metadata: Optional event metadata

    Returns:
        Created event
    """
    return AgentEvent(
        event_type=event_type, source=source, data=data, metadata=metadata
    )
