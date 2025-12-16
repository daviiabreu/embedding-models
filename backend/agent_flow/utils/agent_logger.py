"""
Structured logging for agent calls.

This module provides utilities for logging agent interactions with
structured data for debugging and monitoring.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AgentCallRecord:
    """Record of a single agent call."""

    agent_name: str
    operation: str
    input_summary: str
    output_summary: str = ""
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AgentCallLogger:
    """
    Logger for tracking agent calls with timing and context.

    Provides structured logging for:
    - Agent invocations
    - Sub-agent delegations
    - Tool calls
    - Error tracking

    Examples:
        >>> logger = AgentCallLogger("orchestrator")
        >>> with logger.track_call("process_message", input_summary="user query"):
        ...     result = process()
        >>> logger.log_delegation("knowledge_agent", "retrieve facts")
    """

    def __init__(self, agent_name: str):
        """
        Initialize the agent call logger.

        Args:
            agent_name: Name of the agent being logged
        """
        self.agent_name = agent_name
        self.call_stack: list[AgentCallRecord] = []
        self.current_request_id: str | None = None

    def set_request_id(self, request_id: str) -> None:
        """Set the current request ID for correlation."""
        self.current_request_id = request_id

    @contextmanager
    def track_call(
        self,
        operation: str,
        input_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        """
        Context manager to track an agent call with timing.

        Args:
            operation: Name of the operation being performed
            input_summary: Brief summary of the input
            metadata: Additional metadata to log

        Yields:
            AgentCallRecord: The call record being tracked

        Examples:
            >>> with logger.track_call("safety_check", input_summary="validate input"):
            ...     result = check_safety(input)
        """
        record = AgentCallRecord(
            agent_name=self.agent_name,
            operation=operation,
            input_summary=input_summary[:200] if input_summary else "",
            metadata=metadata or {},
        )

        start_time = time.time()
        self.call_stack.append(record)

        logger.info(
            "agent_call_started",
            agent=self.agent_name,
            operation=operation,
            request_id=self.current_request_id,
            input_preview=input_summary[:100] if input_summary else "",
        )

        try:
            yield record
            record.success = True
        except Exception as e:
            record.success = False
            record.error = str(e)
            logger.error(
                "agent_call_failed",
                agent=self.agent_name,
                operation=operation,
                request_id=self.current_request_id,
                error=str(e),
                duration_ms=record.duration_ms,
            )
            raise
        finally:
            record.duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "agent_call_completed",
                agent=self.agent_name,
                operation=operation,
                request_id=self.current_request_id,
                success=record.success,
                duration_ms=round(record.duration_ms, 2),
                output_preview=record.output_summary[:100]
                if record.output_summary
                else "",
            )

    def log_delegation(
        self,
        target_agent: str,
        purpose: str,
        input_preview: str = "",
    ) -> None:
        """
        Log delegation to a sub-agent.

        Args:
            target_agent: Name of the agent being delegated to
            purpose: Purpose of the delegation
            input_preview: Preview of the input being passed
        """
        logger.info(
            "agent_delegation",
            from_agent=self.agent_name,
            to_agent=target_agent,
            purpose=purpose,
            request_id=self.current_request_id,
            input_preview=input_preview[:100] if input_preview else "",
        )

    def log_tool_call(
        self,
        tool_name: str,
        input_summary: str = "",
        output_summary: str = "",
        duration_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """
        Log a tool invocation.

        Args:
            tool_name: Name of the tool called
            input_summary: Summary of tool input
            output_summary: Summary of tool output
            duration_ms: Execution time in milliseconds
            success: Whether the call succeeded
        """
        logger.info(
            "tool_call",
            agent=self.agent_name,
            tool=tool_name,
            request_id=self.current_request_id,
            success=success,
            duration_ms=round(duration_ms, 2),
            input_preview=input_summary[:100] if input_summary else "",
            output_preview=output_summary[:100] if output_summary else "",
        )

    def log_decision(
        self,
        decision: str,
        reason: str,
        alternatives: list[str] | None = None,
    ) -> None:
        """
        Log an agent decision point.

        Args:
            decision: The decision made
            reason: Why this decision was made
            alternatives: Other options considered
        """
        logger.info(
            "agent_decision",
            agent=self.agent_name,
            decision=decision,
            reason=reason,
            request_id=self.current_request_id,
            alternatives=alternatives or [],
        )

    def get_call_summary(self) -> dict[str, Any]:
        """
        Get summary of all calls in the current stack.

        Returns:
            dict: Summary including total calls, success rate, total time
        """
        if not self.call_stack:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_duration_ms": 0.0,
                "calls": [],
            }

        successful = sum(1 for c in self.call_stack if c.success)
        total_duration = sum(c.duration_ms for c in self.call_stack)

        return {
            "total_calls": len(self.call_stack),
            "successful_calls": successful,
            "failed_calls": len(self.call_stack) - successful,
            "total_duration_ms": round(total_duration, 2),
            "calls": [
                {
                    "operation": c.operation,
                    "success": c.success,
                    "duration_ms": round(c.duration_ms, 2),
                }
                for c in self.call_stack
            ],
        }

    def clear(self) -> None:
        """Clear the call stack."""
        self.call_stack.clear()
        self.current_request_id = None


# Pre-configured loggers for each agent
_agent_loggers: dict[str, AgentCallLogger] = {}


def get_agent_logger(agent_name: str) -> AgentCallLogger:
    """
    Get or create an agent logger.

    Args:
        agent_name: Name of the agent

    Returns:
        AgentCallLogger for the specified agent
    """
    if agent_name not in _agent_loggers:
        _agent_loggers[agent_name] = AgentCallLogger(agent_name)
    return _agent_loggers[agent_name]


__all__ = [
    "AgentCallLogger",
    "AgentCallRecord",
    "get_agent_logger",
]
