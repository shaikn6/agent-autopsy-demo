"""The recorded run where the agent gave up after a tool timeout must be reported; recovered runs must not."""
import json
from pathlib import Path

import app
import examples_traces

TRACES = {t["id"]: t for t in json.loads((Path(__file__).parent.parent / "real_traces.json").read_text())}


def names(trace):
    return [p.name for p in app.replay_trace(trace).failure_patterns]


def test_agent_that_gave_up_after_a_timeout_is_reported():
    t = TRACES["flaky-api"]
    assert names({"name": t["name"], "steps": t["steps"]}) == ["UNRECOVERED_ERROR"]


def test_error_followed_by_a_successful_tool_call_is_not_reported():
    steps = [
        {"type": "ERROR", "error": "TimeoutError: api timed out"},
        {"type": "LLM_CALL", "model": "m", "prompt": "retry?", "response": "yes", "latency_ms": 1, "tokens": 1},
        {"type": "TOOL_CALL", "tool_name": "api", "args": {}, "result": "ok", "latency_ms": 1},
    ]
    assert "UNRECOVERED_ERROR" not in names({"name": "x", "steps": steps})


def test_a_cascade_is_not_reported_twice():
    found = names(examples_traces.EXAMPLES["Infinite loop + error cascade"])
    assert "ERROR_CASCADE" in found and "UNRECOVERED_ERROR" not in found


def test_runs_without_errors_are_unchanged():
    for key in ("Healthy run (no failures)", "Real run: population-sum", "Real run: percentage", "Real run: two-step"):
        assert "UNRECOVERED_ERROR" not in names(examples_traces.EXAMPLES[key])
