import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AgentState(BaseModel):
    """
    State management for AI agents.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identifiers
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None

    # Conversation context
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)

    # Execution tracking
    scratchpad: List[Dict[str, Any]] = Field(default_factory=list)
    tool_executions: List[Dict[str, Any]] = Field(default_factory=list)

    # Domain-specific management
    current_skill: Optional[str] = None
    available_skills: List[str] = Field(default_factory=list)
    actions: List[Dict[str, Any]] = Field(default_factory=list)

    # Performance and error tracking
    metrics: Dict[str, Any] = Field(default_factory=dict)
    execution_times: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    # Timestamp management
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the conversation with optional metadata."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
        }
        if metadata:
            message.update(metadata)
        self.messages.append(message)
        self.updated_at = datetime.now()

    def add_user_message(self, content: str):
        """Add a user message."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        self.add_message("assistant", content)

    def add_system_message(self, content: str):
        """Add a system message."""
        self.add_message("system", content)

    def add_to_scratchpad(self, content: Dict[str, Any]):
        """Add content to the scratchpad."""
        self.scratchpad.append(content)

    def clear_scratchpad(self):
        """Clear the scratchpad."""
        self.scratchpad = []

    def record_tool_execution(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Any,
        status: str = "success",
    ):
        """Record a tool execution with comprehensive details."""
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
    ):
        """Add an action with optional details."""
        self.actions.append(
            {
                "type": action_type,
                "description": description,
                "details": details or {},
                "timestamp": datetime.now(),
            }
        )
        self.updated_at = datetime.now()

    def add_error(self, error_type: str, details: Dict[str, Any]):
        """Add an error to the state's error log."""
        error_record = {
            "type": error_type,
            "details": details,
            "timestamp": datetime.now(),
        }
        self.errors.append(error_record)
        self.updated_at = datetime.now()

    def update_metric(self, metric_name: str, value: Any):
        """Update or add a metric."""
        self.metrics[metric_name] = value
        self.updated_at = datetime.now()

    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message from the conversation."""
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return message.get("content")
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message from the conversation."""
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                return message.get("content")
        return None

    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Safely retrieve context value."""
        return self.context.get(key, default)

    def update_context_value(self, key: str, value: Any):
        """Set a context value."""
        self.context[key] = value
        self.updated_at = datetime.now()

    def save_to_file(self, path: str):
        """Save the current state to a JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load_from_file(cls, path: str) -> "AgentState":
        """Load agent state from a JSON file."""
        return cls.model_validate_json(Path(path).read_text())
