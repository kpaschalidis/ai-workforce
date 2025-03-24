import uuid
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class AgentState(BaseModel):
    """
    Comprehensive state management for generic AI agents.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Unique identifiers
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None

    # Conversation context
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)

    # Reasoning and execution tracking
    scratchpad: List[Dict[str, Any]] = Field(default_factory=list)
    tool_executions: List[Dict[str, Any]] = Field(default_factory=list)

    # Skill and tool management
    current_skill: Optional[str] = None
    available_skills: List[str] = Field(default_factory=list)

    # Performance and metrics
    metrics: Dict[str, Any] = Field(default_factory=dict)
    execution_times: List[Dict[str, Any]] = Field(default_factory=list)

    # Error tracking
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

    def get_context(self, key: str, default: Any = None) -> Any:
        """Safely retrieve context value."""
        return self.context.get(key, default)

    def set_context(self, key: str, value: Any):
        """Set a context value."""
        self.context[key] = value
        self.updated_at = datetime.now()
