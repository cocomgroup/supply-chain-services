# Databricks notebook source
"""
agt16.agents.base
------------------
Abstract base class for all AGT-16 sub-agents.

Identical contract to agt_ss.agents.base:
  - can_run(state)  → bool           dependency check before invocation
  - run(state)      → partial dict   executes processes, returns state update

Agents must NOT mutate state in place; they return a partial dict merged
by the orchestrator. Tool failures raise AgentToolError, which the
orchestrator catches and routes to dead-letter.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class AgentToolError(Exception):
    """Raised when a tool call fails after all retries."""
    def __init__(self, tool_name: str, detail: str, attempt: int):
        self.tool_name = tool_name
        self.detail    = detail
        self.attempt   = attempt
        super().__init__(f"[{tool_name}] attempt {attempt}: {detail}")


class BaseAgent(ABC):
    """
    Contract every sub-agent must satisfy.

    Subclasses override `can_run` and `_execute`.
    The public `run` method wraps `_execute` with retry logic and audit logging.
    """

    name: str  # must be set by subclass (AgentName value)

    # ── Dependency check ─────────────────────────────────────────────────

    @abstractmethod
    def can_run(self, state: dict) -> bool:
        """
        Return True if this agent's preconditions are met in the current state.
        Called by the orchestrator's routing logic before scheduling this agent.
        """
        ...

    # ── Core execution ────────────────────────────────────────────────────

    @abstractmethod
    def _execute(self, state: dict) -> dict:
        """
        Execute this agent's processes.

        Receives the full WorkflowState.
        Must return a partial dict — only the keys this agent is responsible for.
        The orchestrator merges this into the main state.

        Raise AgentToolError on unrecoverable tool failures.
        """
        ...

    # ── Public entry point (called by orchestrator) ───────────────────────

    def run(self, state: dict) -> dict:
        """
        Execute with retry logic and audit trail.
        Returns a partial state update dict.
        """
        attempt = 0
        last_error: Exception | None = None

        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                logger.info("[%s] Starting (attempt %d)", self.name, attempt)
                result = self._execute(state)
                self._audit(state, "completed", f"attempt={attempt}")
                return result

            except AgentToolError as exc:
                last_error = exc
                logger.warning("[%s] Tool error attempt %d: %s", self.name, attempt, exc)
                if attempt >= MAX_RETRIES:
                    break

            except Exception as exc:
                last_error = exc
                logger.exception("[%s] Unexpected error attempt %d", self.name, attempt)
                if attempt >= MAX_RETRIES:
                    break

        # All retries exhausted — signal dead-letter to orchestrator
        self._audit(state, "dead_letter", str(last_error))
        raise AgentToolError(
            tool_name=self.name,
            detail=str(last_error),
            attempt=attempt,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _audit(self, state: dict, action: str, detail: str = "") -> None:
        """Append an entry to the in-memory audit log (orchestrator persists it)."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent":     self.name,
            "action":    action,
            "detail":    detail,
        }
        existing = state.get("audit_log", [])
        existing.append(entry)
        logger.debug("Audit: %s", entry)

    @staticmethod
    def _merge_list(state: dict, key: str, new_items: list) -> list:
        """Helper: extend an existing list in state without clobbering it."""
        return state.get(key, []) + new_items
