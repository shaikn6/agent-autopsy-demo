"""The optional LLM root-cause layer explains patterns and never hard-fails."""
import json
from pathlib import Path

import app
from autopsy import llm_summary

TRACES = {t["id"]: t for t in json.loads((Path(__file__).parent.parent / "real_traces.json").read_text())}


def _report_dict(trace):
    return json.loads(app.replay_trace(trace).to_json())


def test_no_token_falls_back_to_template_naming_every_pattern(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    t = TRACES["flaky-api"]
    report = _report_dict({"name": t["name"], "steps": t["steps"]})
    text, source = llm_summary.generate_root_cause(report)
    assert source == "templated"
    for p in report["failure_patterns"]:
        assert p["name"] in text


def test_no_patterns_skips_the_model_entirely(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token-should-never-be-used")
    report = _report_dict({
        "name": "clean",
        "steps": [{"type": "LLM_CALL", "model": "m", "prompt": "hi", "response": "ok", "latency_ms": 1, "tokens": 1}],
    })
    text, source = llm_summary.generate_root_cause(report)
    assert source == "deterministic" and "nothing to explain" in text.lower()


def test_client_error_degrades_to_template_without_raising(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def chat_completion(self, *a, **k):
            raise RuntimeError("inference endpoint unavailable")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "InferenceClient", _Boom)
    t = TRACES["flaky-api"]
    report = _report_dict({"name": t["name"], "steps": t["steps"]})
    text, source = llm_summary.generate_root_cause(report)
    assert source == "templated" and "RuntimeError" in text


def test_app_explain_callback_handles_empty_and_populated_input(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert "Run an autopsy first" in app.explain_root_cause(None)
    t = TRACES["flaky-api"]
    report = _report_dict({"name": t["name"], "steps": t["steps"]})
    out = app.explain_root_cause(report)
    assert "Root-cause explanation" in out and "templated" in out
