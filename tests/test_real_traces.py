"""Regression tests built from traces recorded off a real agent (see real_traces.json)."""
import json
from pathlib import Path

import pytest

import app

TRACES = {t["id"]: t for t in json.loads((Path(__file__).parent.parent / "real_traces.json").read_text())}


def names(trace_id):
    t = TRACES[trace_id]
    return [p.name for p in app.replay_trace({"name": t["name"], "steps": t["steps"]}).failure_patterns]


@pytest.mark.parametrize("trace_id", ["population-sum", "two-step", "percentage", "unknown-city"])
def test_successful_agent_with_a_system_prompt_is_not_a_loop(trace_id):
    # Every LLM call of these runs starts with the same system prompt; that is not a loop.
    assert "INFINITE_LOOP" not in names(trace_id)


def test_verbatim_repeated_prompt_is_still_a_loop():
    steps = [{"type": "LLM_CALL", "model": "m", "prompt": "same prompt", "response": "r", "latency_ms": 1, "tokens": 1}] * 4
    assert "INFINITE_LOOP" in [p.name for p in app.replay_trace({"name": "x", "steps": steps}).failure_patterns]


def test_failed_run_without_patterns_does_not_claim_success():
    t = TRACES["flaky-api"]
    report = app.replay_trace({"name": t["name"], "steps": t["steps"]})
    if report.status != "SUCCESS" and not report.failure_patterns:
        assert "completed successfully" not in report.summary


def test_replayed_latency_is_at_least_the_recorded_latency():
    t = TRACES["population-sum"]
    recorded = sum(s.get("latency_ms", 0) for s in t["steps"] if s["type"] in ("LLM_CALL", "TOOL_CALL"))
    assert app.replay_trace({"name": t["name"], "steps": t["steps"]}).total_latency_ms >= recorded
