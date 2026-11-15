"""REDUNDANT_TOOL_CALL and COST_RUNAWAY: positive cases fire, look-alikes stay silent, real runs stay clean."""
import json
from pathlib import Path

import app
import examples_traces

TRACES = {t["id"]: t for t in json.loads((Path(__file__).parent.parent / "real_traces.json").read_text())}


def names(steps):
    return [p.name for p in app.replay_trace({"name": "t", "steps": steps}).failure_patterns]


def tool(name, args, result):
    return {"type": "TOOL_CALL", "tool_name": name, "args": args, "result": result, "latency_ms": 10}


def llm(prompt, tokens, cw=8000):
    return {"type": "LLM_CALL", "model": "m", "prompt": prompt, "response": "ok", "latency_ms": 10,
            "tokens": tokens, "context_window": cw}


# --- REDUNDANT_TOOL_CALL ----------------------------------------------------

def test_identical_tool_call_with_usable_result_is_redundant():
    steps = [tool("get_price", {"item": "a"}, {"usd": 5}), tool("get_price", {"item": "a"}, {"usd": 5})]
    assert "REDUNDANT_TOOL_CALL" in names(steps)


def test_different_args_is_not_redundant():
    steps = [tool("get_price", {"item": "a"}, {"usd": 5}), tool("get_price", {"item": "b"}, {"usd": 9})]
    assert "REDUNDANT_TOOL_CALL" not in names(steps)


def test_retry_after_empty_result_is_not_redundant():
    steps = [tool("search", {"q": "x"}, None), tool("search", {"q": "x"}, {"hit": 1})]
    assert "REDUNDANT_TOOL_CALL" not in names(steps)


def test_retry_after_an_error_between_calls_is_not_redundant():
    steps = [
        tool("api", {"id": 1}, {"ok": True}),
        {"type": "ERROR", "error": "TimeoutError: transient"},
        tool("api", {"id": 1}, {"ok": True}),
    ]
    assert "REDUNDANT_TOOL_CALL" not in names(steps)


def test_loop_owned_by_infinite_loop_is_not_also_redundant():
    # Same tool + args 3 times -> INFINITE_LOOP (by name) owns it; REDUNDANT_TOOL_CALL defers.
    steps = [tool("get_price", {"item": "a"}, {"usd": 5})] * 3
    found = names(steps)
    assert "INFINITE_LOOP" in found and "REDUNDANT_TOOL_CALL" not in found


# --- COST_RUNAWAY -----------------------------------------------------------

def test_fast_growth_near_window_is_a_runaway():
    steps = [llm("compute totals", 2000), llm("compute totals again", 4000), llm("compute totals once more", 6000)]
    assert "COST_RUNAWAY" in names(steps)


def test_small_monotonic_growth_is_not_a_runaway():
    # grows, but the last call is a sliver of a 128k window -> projected to fill in far more than 3 calls
    steps = [llm("a", 180, cw=128000), llm("b", 260, cw=128000), llm("c", 540, cw=128000)]
    assert "COST_RUNAWAY" not in names(steps)


def test_token_explosion_is_not_double_reported_as_runaway():
    steps = [llm("a", 2000), llm("b", 4000), llm("c", 7900)]  # 7900/8000 = 98.75% -> TOKEN_EXPLOSION owns it
    found = names(steps)
    assert "TOKEN_EXPLOSION" in found and "COST_RUNAWAY" not in found


def test_two_llm_calls_are_too_few_to_project():
    steps = [llm("a", 2000), llm("b", 6000)]
    assert "COST_RUNAWAY" not in names(steps)


# --- scripted example + real runs stay as expected --------------------------

def test_scripted_example_fires_exactly_the_two_new_patterns():
    found = set(names(examples_traces.EXAMPLES["Redundant calls + cost runaway"]["steps"]))
    assert found == {"REDUNDANT_TOOL_CALL", "COST_RUNAWAY"}, found


def test_new_detectors_stay_silent_on_all_real_runs():
    for t in TRACES.values():
        found = names(t["steps"])
        assert "REDUNDANT_TOOL_CALL" not in found and "COST_RUNAWAY" not in found, (t["id"], found)


def test_healthy_run_stays_clean():
    found = names(examples_traces.EXAMPLES["Healthy run (no failures)"]["steps"])
    assert "REDUNDANT_TOOL_CALL" not in found and "COST_RUNAWAY" not in found
