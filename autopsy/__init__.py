"""agent-autopsy — framework-agnostic LLM agent observability."""

from .graph import ExecutionGraph, StepType
from .patterns import FailurePattern, FailurePatternDetector
from .report import AutopsyReport, build_report
from .tracer import AgentTracer, get_active_tracer

__all__ = [
    "AgentTracer",
    "AutopsyReport",
    "ExecutionGraph",
    "FailurePattern",
    "FailurePatternDetector",
    "StepType",
    "build_report",
    "get_active_tracer",
]

__version__ = "0.1.0"
