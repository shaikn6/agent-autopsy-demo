"""AgentTracer — core interceptor for LLM agent execution."""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from .graph import ExecutionGraph, StepType
from .patterns import FailurePatternDetector
from .report import AutopsyReport, build_report

# Context variable so integrations can find the active tracer
_active_tracer: ContextVar[AgentTracer | None] = ContextVar("_active_tracer", default=None)


def get_active_tracer() -> AgentTracer | None:
    """Return the AgentTracer currently active in this context, or None."""
    return _active_tracer.get()


class AgentTracer:
    """Intercept and record every significant step an LLM agent takes.

    Usage — context manager::

        with AgentTracer(name="my-agent") as tracer:
            # your agent code here
            tracer.record_llm_call(...)
        report = tracer.get_report()

    Usage — decorator::

        @AgentTracer.trace(name="my-agent")
        def run_agent():
            ...
    """

    def __init__(
        self,
        name: str,
        max_steps: int = 50,
        *,
        auto_patch: bool = False,
    ) -> None:
        self.name = name
        self.max_steps = max_steps
        self.auto_patch = auto_patch

        self.run_id: str = str(uuid.uuid4())
        self._graph: ExecutionGraph = ExecutionGraph()
        self._start_time: float = 0.0
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None
        self._status: str = "SUCCESS"
        self._report: AutopsyReport | None = None
        self._token: Any = None  # ContextVar token for cleanup

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> AgentTracer:
        self._start_time = time.monotonic()
        self._started_at = datetime.now(tz=timezone.utc)
        self._token = _active_tracer.set(self)

        # START node
        self._graph.add_step(StepType.START, {"agent": self.name, "run_id": self.run_id})

        if self.auto_patch:
            self._apply_patches()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None:
            self._status = "FAILED"
            self.record_error(exc_val or exc_type, step_id=None)

        # END node
        self._graph.add_step(StepType.END, {"status": self._status})
        self._completed_at = datetime.now(tz=timezone.utc)

        # Build report
        # Wall-clock time is right for a live agent; a replayed trace finishes instantly, so never report less
        # than the latencies that were recorded on its steps.
        recorded_ms = sum(float(n["metadata"].get("latency_ms", 0) or 0) for n in self._graph.nodes())
        elapsed_ms = max((time.monotonic() - self._start_time) * 1000, recorded_ms)
        patterns = FailurePatternDetector().detect(self._graph)
        self._report = build_report(
            agent_name=self.name,
            started_at=self._started_at,  # type: ignore[arg-type]
            completed_at=self._completed_at,
            status=self._status,
            graph_dict=self._graph.to_dict(),
            failure_patterns=patterns,
            total_tokens=self._graph.total_tokens(),
            total_latency_ms=elapsed_ms,
            run_id=self.run_id,
        )

        if self._token is not None:
            _active_tracer.reset(self._token)

        if self.auto_patch:
            self._remove_patches()

        # Do not suppress exceptions
        return False

    # ------------------------------------------------------------------
    # Decorator factory
    # ------------------------------------------------------------------

    @classmethod
    def trace(cls, name: str | None = None, **kwargs: Any) -> Callable:
        """Class-method decorator factory.

        Can be used as::

            @AgentTracer.trace(name="my-agent")
            def run():
                ...

        or as a plain decorator (function name used as agent name)::

            @AgentTracer.trace
            def run():
                ...
        """

        def decorator(fn: Callable) -> Callable:
            agent_name = name or fn.__qualname__

            @functools.wraps(fn)
            def wrapper(*args: Any, **kw: Any) -> Any:
                with cls(name=agent_name, **kwargs):
                    result = fn(*args, **kw)
                return result

            return wrapper

        # Called as @AgentTracer.trace (no parentheses — name is the fn)
        if callable(name):
            fn = name
            name = fn.__qualname__  # type: ignore[assignment]
            return decorator(fn)  # type: ignore[arg-type]

        return decorator

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        latency_ms: float,
        tokens: int,
        *,
        context_window: int | None = None,
    ) -> str:
        """Record a single LLM call and return the node id."""
        metadata: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response": response,
            "latency_ms": round(latency_ms, 2),
        }
        if context_window:
            metadata["context_window"] = context_window

        return self._graph.add_step(StepType.LLM_CALL, metadata, tokens_used=tokens)

    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        latency_ms: float,
    ) -> str:
        """Record a single tool call and return the node id."""
        return self._graph.add_step(
            StepType.TOOL_CALL,
            {
                "tool_name": tool_name,
                "args": args,
                "result": result,
                "latency_ms": round(latency_ms, 2),
            },
        )

    def record_memory_read(self, key: str, value: Any) -> str:
        return self._graph.add_step(
            StepType.MEMORY_READ,
            {"key": key, "value": value},
        )

    def record_decision(self, description: str, metadata: dict[str, Any] | None = None) -> str:
        return self._graph.add_step(
            StepType.DECISION,
            {"description": description, **(metadata or {})},
        )

    def record_error(self, error: Any, step_id: str | None = None) -> str:
        self._status = "FAILED"
        node_id = self._graph.add_step(
            StepType.ERROR,
            {
                "error_type": type(error).__name__ if error else "Unknown",
                "error_message": str(error),
                "step_id": step_id,
            },
        )
        return node_id

    def mark_timeout(self) -> None:
        self._status = "TIMEOUT"

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_report(self) -> AutopsyReport:
        if self._report is None:
            raise RuntimeError("Report not yet available — tracer has not exited.")
        return self._report

    @property
    def graph(self) -> ExecutionGraph:
        return self._graph

    # ------------------------------------------------------------------
    # Optional monkey-patching
    # ------------------------------------------------------------------

    def _apply_patches(self) -> None:
        try:
            from .integrations.openai_patch import patch_openai

            patch_openai()
        except ImportError:
            pass
        try:
            from .integrations.anthropic_patch import patch_anthropic

            patch_anthropic()
        except ImportError:
            pass

    def _remove_patches(self) -> None:
        try:
            from .integrations.openai_patch import unpatch_openai

            unpatch_openai()
        except ImportError:
            pass
        try:
            from .integrations.anthropic_patch import unpatch_anthropic

            unpatch_anthropic()
        except ImportError:
            pass
