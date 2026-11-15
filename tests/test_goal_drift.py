"""GOAL_DRIFT must flag sustained drift, not a single different-looking final prompt."""
import app
from examples_traces import EXAMPLES


def llm(prompt):
    return {"type": "LLM_CALL", "model": "m", "prompt": prompt, "response": "ok", "latency_ms": 10, "tokens": 10}


def names(steps):
    return [p.name for p in app.replay_trace({"name": "t", "steps": steps}).failure_patterns]


def test_two_step_run_with_a_different_final_prompt_is_not_drift():
    steps = [
        {"type": "DECISION", "description": "Goal: summarise the document"},
        llm("summarise the document"),
        llm("write the final answer using result A"),
    ]
    assert "GOAL_DRIFT" not in names(steps)


def test_one_off_topic_final_prompt_after_on_topic_steps_is_not_drift():
    steps = [
        {"type": "DECISION", "description": "Goal: research quarterly revenue for Acme Corp"},
        llm("Find the quarterly revenue numbers for Acme Corp from their latest filing."),
        llm("Extract the quarterly revenue figures for Acme Corp from the filing text."),
        llm("Format the answer as a short sentence."),
    ]
    assert "GOAL_DRIFT" not in names(steps)


def test_sustained_drift_is_still_flagged():
    for name, trace in EXAMPLES.items():
        if "drift" in name.lower():
            assert "GOAL_DRIFT" in names(trace["steps"])
            return
    raise AssertionError("no drift example found")


def test_example_buttons_do_not_add_api_endpoints():
    api = app.demo.get_api_info()["named_endpoints"]
    assert sorted(api) == ["/analyze"], sorted(api)
