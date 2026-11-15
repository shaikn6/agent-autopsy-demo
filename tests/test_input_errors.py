"""Bad traces must fail with a message that names the problem."""
import pytest

import app


def test_empty_trace_is_rejected_not_reported_as_success():
    with pytest.raises(ValueError, match="no steps"):
        app.replay_trace({"name": "x", "steps": []})


def test_non_numeric_latency_names_the_field():
    steps = [{"type": "LLM_CALL", "prompt": "p", "response": "r", "latency_ms": "x"}]
    with pytest.raises(ValueError, match="Step 0: `latency_ms` must be a number"):
        app.replay_trace({"name": "x", "steps": steps})
