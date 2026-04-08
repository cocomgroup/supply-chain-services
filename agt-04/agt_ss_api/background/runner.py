"""
agt_ss_api.background.runner
-----------------------------
Async-safe background workflow execution.

When run_workflows_async=True (production default), POST /workflows returns
immediately with the workflow_id and a 202 Accepted response. The graph runs
in a ThreadPoolExecutor (LangGraph is synchronous Python) and writes state
updates to the checkpoint store after every node execution.

The polling pattern is:
  1. Client POST /workflows → 202 {workflow_id, status: "pending"}
  2. Client polls GET /workflows/{id} until status ∈ {completed, awaiting_human, failed}
  3. If awaiting_human → client POST /workflows/{id}/resume
  4. If completed → client reads final_output

State transitions written by the runner are visible to GET /workflows/{id}
as soon as the node's checkpoint is flushed (each node calls save_checkpoint).
The API reads from the same checkpoint store, so polling is consistent.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Global executor — one pool for all workflow runs
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agt-ss-workflow")

# In-memory future registry for tracking active runs
_active_futures: dict[str, Future] = {}
_registry_lock  = threading.Lock()


def submit_workflow(
    workflow_id: str,
    target_fn,           # run_workflow or resume_workflow from orchestrator
    *args,
    on_complete=None,    # optional callback(workflow_id, final_state)
    **kwargs,
) -> None:
    """
    Submit a workflow function to the background executor.

    target_fn is called with *args, **kwargs in a worker thread.
    on_complete(workflow_id, state) is called on the main thread via
    asyncio.get_event_loop().call_soon_threadsafe() when the run finishes.
    """

    def _run():
        try:
            logger.info("[Runner] Starting workflow %s in background thread", workflow_id)
            result = target_fn(*args, **kwargs)
            logger.info("[Runner] Workflow %s finished — status=%s",
                        workflow_id, result.get("status"))
            if on_complete:
                on_complete(workflow_id, result)
            return result
        except Exception as exc:
            logger.exception("[Runner] Workflow %s raised unhandled exception", workflow_id)
            if on_complete:
                on_complete(workflow_id, None)
            raise

    future = _executor.submit(_run)
    with _registry_lock:
        _active_futures[workflow_id] = future

    def _cleanup(f: Future):
        with _registry_lock:
            _active_futures.pop(workflow_id, None)

    future.add_done_callback(_cleanup)


def is_running(workflow_id: str) -> bool:
    """Return True if a background run is actively executing for this workflow."""
    with _registry_lock:
        future = _active_futures.get(workflow_id)
    return future is not None and not future.done()


def cancel_workflow(workflow_id: str) -> bool:
    """
    Attempt to cancel a pending (not yet started) workflow.
    Returns True if cancelled, False if already running or done.
    LangGraph runs cannot be interrupted mid-execution — only pending queue entries can be cancelled.
    """
    with _registry_lock:
        future = _active_futures.get(workflow_id)
    if future and not future.running():
        cancelled = future.cancel()
        if cancelled:
            logger.info("[Runner] Cancelled workflow %s", workflow_id)
        return cancelled
    return False


def active_workflow_ids() -> list[str]:
    """Return IDs of all currently executing workflows."""
    with _registry_lock:
        return [wid for wid, f in _active_futures.items() if not f.done()]
