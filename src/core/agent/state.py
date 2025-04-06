"""
State management for the AI agent framework.

This module defines the AgentState class which is used to maintain the state of agent
executions, including conversation history, context, and execution tracking.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    State management for AI agents.

    This class maintains the execution state of an agent, including:
    - Conversation history (messages between user and agent)
    - Context information (key-value store for data used during execution)
    - Tool execution history
    - Scratchpad for temporary calculations
    - Performance and error tracking

    The state is the central object passed through workflow nodes during execution.
    """

    # Identifiers
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None

    # Conversation context
    messages: List[BaseMessage] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)

    # Execution tracking
    scratchpad: List[Dict[str, Any]] = Field(default_factory=list)
    tool_executions: List[Dict[str, Any]] = Field(default_factory=list)

    # Agent-specific management
    current_skill: Optional[str] = None
    available_skills: List[str] = Field(default_factory=list)
    actions: List[Dict[str, Any]] = Field(default_factory=list)

    # Performance and error tracking
    metrics: Dict[str, Any] = Field(default_factory=dict)
    execution_times: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    # Plan-related tracking (for planning workflows)
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    current_step_index: int = 0
    evaluations: List[Dict[str, Any]] = Field(default_factory=list)
    current_step_evaluated: bool = True
    need_replan: bool = False
    iteration_count: int = 0

    # Timestamp management
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Special Sets/Flags
    processed_message_ids: Set[str] = Field(default_factory=set)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for extensibility

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to the conversation with optional metadata.

        Args:
            role: Role of the message sender ('user', 'assistant', 'system', etc.)
            content: Message content
            metadata: Optional metadata for the message
        """
        if role == "user":
            self.add_user_message(content)
        elif role == "assistant":
            self.add_assistant_message(content)
        elif role == "system":
            self.add_system_message(content)
        else:
            # For other roles, create a dictionary message
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now(),
            }
            if metadata:
                message.update(metadata)
            self.messages.append(message)

        self.updated_at = datetime.now()

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.

        Args:
            content: Message content
        """
        self.messages.append(HumanMessage(content=content))
        self.updated_at = datetime.now()

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation.

        Args:
            content: Message content
        """
        self.messages.append(AIMessage(content=content))
        self.updated_at = datetime.now()

    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation.

        Args:
            content: Message content
        """
        self.messages.append(SystemMessage(content=content))
        self.updated_at = datetime.now()

    def add_tool_message(self, tool_name: str, content: str) -> None:
        """
        Add a tool message to the conversation.

        Args:
            tool_name: Name of the tool
            content: Tool output content
        """
        # Generate a unique ID for this tool call
        tool_call_id = str(uuid.uuid4())

        # Create and add the tool message
        self.messages.append(
            ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)
        )
        self.updated_at = datetime.now()

    def get_messages(self) -> List[BaseMessage]:
        """
        Get all messages in the conversation.

        Returns:
            List of messages
        """
        return self.messages

    def get_last_message(self) -> Optional[BaseMessage]:
        """
        Get the last message in the conversation.

        Returns:
            Last message or None if no messages
        """
        if self.messages:
            return self.messages[-1]
        return None

    def get_last_message_role(self) -> Optional[str]:
        """
        Get the role of the last message.

        Returns:
            Role of the last message or None if no messages
        """
        last_message = self.get_last_message()
        if last_message is None:
            return None

        if isinstance(last_message, HumanMessage):
            return "user"
        elif isinstance(last_message, AIMessage):
            return "assistant"
        elif isinstance(last_message, SystemMessage):
            return "system"
        elif isinstance(last_message, ToolMessage):
            return "tool"
        elif isinstance(last_message, dict) and "role" in last_message:
            return last_message["role"]

        return None

    def get_last_user_message(self) -> Optional[str]:
        """
        Get the last user message from the conversation.

        Returns:
            Content of the last user message or None if no user messages
        """
        for message in reversed(self.messages):
            if isinstance(message, HumanMessage):
                return message.content
            elif isinstance(message, dict) and message.get("role") == "user":
                return message.get("content")
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """
        Get the last assistant message from the conversation.

        Returns:
            Content of the last assistant message or None if no assistant messages
        """
        for message in reversed(self.messages):
            if isinstance(message, AIMessage):
                return message.content
            elif isinstance(message, dict) and message.get("role") == "assistant":
                return message.get("content")
        return None

    def has_system_message(self) -> bool:
        """
        Check if the conversation has a system message.

        Returns:
            True if a system message exists, False otherwise
        """
        for message in self.messages:
            if isinstance(message, SystemMessage):
                return True
            elif isinstance(message, dict) and message.get("role") == "system":
                return True
        return False

    def add_to_context(self, key: str, value: Any) -> None:
        """
        Add a value to the context.

        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
        self.updated_at = datetime.now()

    def get_from_context(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the context.

        Args:
            key: Context key
            default: Default value if key not found

        Returns:
            Context value or default
        """
        return self.context.get(key, default)

    def add_to_scratchpad(self, content: Dict[str, Any]) -> None:
        """
        Add content to the scratchpad.

        Args:
            content: Content to add
        """
        self.scratchpad.append(content)
        self.updated_at = datetime.now()

    def clear_scratchpad(self) -> None:
        """Clear the scratchpad."""
        self.scratchpad = []
        self.updated_at = datetime.now()

    def record_tool_execution(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Any,
        status: str = "success",
    ) -> None:
        """
        Record a tool execution with comprehensive details.

        Args:
            tool_name: Name of the tool
            inputs: Tool inputs
            outputs: Tool outputs
            status: Execution status
        """
        execution_record = {
            "tool_name": tool_name,
            "inputs": inputs,
            "outputs": outputs,
            "status": status,
            "timestamp": datetime.now(),
        }
        self.tool_executions.append(execution_record)
        self.updated_at = datetime.now()

    def add_action(
        self,
        action_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an action to the state.

        Args:
            action_type: Type of action
            description: Action description
            details: Optional action details
        """
        self.actions.append(
            {
                "type": action_type,
                "description": description,
                "details": details or {},
                "timestamp": datetime.now(),
            }
        )
        self.updated_at = datetime.now()

    def add_error(self, error_type: str, details: Dict[str, Any]) -> None:
        """
        Add an error to the state's error log.

        Args:
            error_type: Type of error
            details: Error details
        """
        error_record = {
            "type": error_type,
            "details": details,
            "timestamp": datetime.now(),
        }
        self.errors.append(error_record)
        self.updated_at = datetime.now()

    def update_metric(self, metric_name: str, value: Any) -> None:
        """
        Update or add a metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.metrics[metric_name] = value
        self.updated_at = datetime.now()

    def record_execution_time(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record execution time for performance tracking.

        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            metadata: Optional metadata
        """
        self.execution_times.append(
            {
                "operation": operation,
                "duration_ms": duration_ms,
                "metadata": metadata or {},
                "timestamp": datetime.now(),
            }
        )
        self.updated_at = datetime.now()

    def set_plan(self, plan: List[Dict[str, Any]]) -> None:
        """
        Set the execution plan.

        Args:
            plan: List of plan steps
        """
        self.plan = plan
        self.current_step_index = 0
        self.evaluations = []
        self.current_step_evaluated = True
        self.need_replan = False
        self.updated_at = datetime.now()

    def save_to_file(self, path: str) -> None:
        """
        Save the current state to a JSON file.

        Args:
            path: File path
        """
        # Convert to dictionary
        state_dict = self.model_dump(exclude={"processed_message_ids"})

        # Convert datetime objects to strings
        for key in ["created_at", "updated_at"]:
            if key in state_dict:
                state_dict[key] = state_dict[key].isoformat()

        # Convert messages to dictionaries
        messages = []
        for msg in self.messages:
            if isinstance(msg, BaseMessage):
                msg_dict = {"type": msg.__class__.__name__, "content": msg.content}
                if isinstance(msg, ToolMessage):
                    msg_dict["tool_name"] = msg.name
                    msg_dict["tool_call_id"] = msg.tool_call_id
                messages.append(msg_dict)
            else:
                messages.append(msg)

        state_dict["messages"] = messages

        # Write to file
        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> "AgentState":
        """
        Load agent state from a JSON file.

        Args:
            path: File path

        Returns:
            Loaded AgentState
        """
        # Read from file
        with open(path, "r") as f:
            state_dict = json.load(f)

        # Convert message dictionaries to proper message objects
        if "messages" in state_dict:
            messages = []
            for msg in state_dict["messages"]:
                if isinstance(msg, dict) and "type" in msg:
                    if msg["type"] == "HumanMessage":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["type"] == "AIMessage":
                        messages.append(AIMessage(content=msg["content"]))
                    elif msg["type"] == "SystemMessage":
                        messages.append(SystemMessage(content=msg["content"]))
                    elif msg["type"] == "ToolMessage":
                        messages.append(
                            ToolMessage(
                                content=msg["content"],
                                tool_call_id=msg.get("tool_call_id", str(uuid.uuid4())),
                                name=msg.get("tool_name", "unknown"),
                            )
                        )
                else:
                    messages.append(msg)

            state_dict["messages"] = messages

        # Convert datetime strings back to datetime objects
        for key in ["created_at", "updated_at"]:
            if key in state_dict and isinstance(state_dict[key], str):
                state_dict[key] = datetime.fromisoformat(state_dict[key])

        # Create state instance
        return cls(**state_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to a dictionary.

        Returns:
            State as a dictionary
        """
        # Convert to dictionary
        state_dict = self.model_dump(exclude={"processed_message_ids"})

        # Convert datetime objects to strings
        for key in ["created_at", "updated_at"]:
            if key in state_dict:
                state_dict[key] = state_dict[key].isoformat()

        # Convert messages to dictionaries
        messages = []
        for msg in self.messages:
            if isinstance(msg, BaseMessage):
                msg_dict = {"type": msg.__class__.__name__, "content": msg.content}
                if isinstance(msg, ToolMessage):
                    msg_dict["tool_name"] = msg.name
                    msg_dict["tool_call_id"] = msg.tool_call_id
                messages.append(msg_dict)
            else:
                messages.append(msg)

        state_dict["messages"] = messages

        return state_dict

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "AgentState":
        """
        Create state from a dictionary.

        Args:
            state_dict: State dictionary

        Returns:
            AgentState instance
        """
        # Make a copy of the dictionary to avoid modifying the original
        state_dict = state_dict.copy()

        # Convert message dictionaries to proper message objects
        if "messages" in state_dict:
            messages = []
            for msg in state_dict["messages"]:
                if isinstance(msg, dict) and "type" in msg:
                    if msg["type"] == "HumanMessage":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["type"] == "AIMessage":
                        messages.append(AIMessage(content=msg["content"]))
                    elif msg["type"] == "SystemMessage":
                        messages.append(SystemMessage(content=msg["content"]))
                    elif msg["type"] == "ToolMessage":
                        messages.append(
                            ToolMessage(
                                content=msg["content"],
                                tool_call_id=msg.get("tool_call_id", str(uuid.uuid4())),
                                name=msg.get("tool_name", "unknown"),
                            )
                        )
                else:
                    messages.append(msg)

            state_dict["messages"] = messages

        # Convert datetime strings back to datetime objects
        for key in ["created_at", "updated_at"]:
            if key in state_dict and isinstance(state_dict[key], str):
                state_dict[key] = datetime.fromisoformat(state_dict[key])

        # Create state instance
        return cls(**state_dict)


class EnhancedAgentState(AgentState):
    """
    Enhanced agent state with additional capabilities.

    This class extends the base AgentState with additional
    features like memory management, more advanced context tracking,
    and state persistence capabilities.
    """

    # Memory management
    memory: Dict[str, Any] = Field(default_factory=dict)
    memory_access_count: Dict[str, int] = Field(default_factory=dict)

    # Advanced context tracking
    session_context: Dict[str, Any] = Field(default_factory=dict)
    user_context: Dict[str, Any] = Field(default_factory=dict)

    # State persistence
    state_version: int = 1

    def memorize(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in memory with optional TTL.

        Args:
            key: Memory key
            value: Value to store
            ttl: Time to live in seconds (None for permanent)
        """
        self.memory[key] = {
            "value": value,
            "created_at": datetime.now(),
            "ttl": ttl,
            "last_accessed": datetime.now(),
        }
        self.memory_access_count[key] = 0
        self.updated_at = datetime.now()

    def recall(self, key: str, default: Any = None) -> Any:
        """
        Recall a value from memory.

        Args:
            key: Memory key
            default: Default value if key not found or expired

        Returns:
            Stored value or default
        """
        memory_entry = self.memory.get(key)

        if memory_entry is None:
            return default

        # Check if entry has expired
        if memory_entry["ttl"] is not None:
            created_at = memory_entry["created_at"]
            ttl = memory_entry["ttl"]

            if (datetime.now() - created_at).total_seconds() > ttl:
                return default

        # Update access stats
        memory_entry["last_accessed"] = datetime.now()
        self.memory_access_count[key] = self.memory_access_count.get(key, 0) + 1

        return memory_entry["value"]

    def forget(self, key: str) -> bool:
        """
        Remove a value from memory.

        Args:
            key: Memory key

        Returns:
            True if value was removed, False if key not found
        """
        if key in self.memory:
            del self.memory[key]
            if key in self.memory_access_count:
                del self.memory_access_count[key]
            self.updated_at = datetime.now()
            return True
        return False

    def clean_expired_memory(self) -> int:
        """
        Remove all expired memory entries.

        Returns:
            Number of entries removed
        """
        current_time = datetime.now()
        keys_to_remove = []

        for key, entry in self.memory.items():
            if entry["ttl"] is not None:
                created_at = entry["created_at"]
                ttl = entry["ttl"]

                if (current_time - created_at).total_seconds() > ttl:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.memory[key]
            if key in self.memory_access_count:
                del self.memory_access_count[key]

        if keys_to_remove:
            self.updated_at = datetime.now()

        return len(keys_to_remove)

    def add_user_context(self, key: str, value: Any) -> None:
        """
        Add a value to the user context.

        Args:
            key: Context key
            value: Context value
        """
        self.user_context[key] = value
        self.updated_at = datetime.now()

    def get_user_context(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the user context.

        Args:
            key: Context key
            default: Default value if key not found

        Returns:
            Context value or default
        """
        return self.user_context.get(key, default)

    def add_session_context(self, key: str, value: Any) -> None:
        """
        Add a value to the session context.

        Args:
            key: Context key
            value: Context value
        """
        self.session_context[key] = value
        self.updated_at = datetime.now()

    def get_session_context(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the session context.

        Args:
            key: Context key
            default: Default value if key not found

        Returns:
            Context value or default
        """
        return self.session_context.get(key, default)
