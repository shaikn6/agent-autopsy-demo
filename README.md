# 🔬 Agent Autopsy

**Explains why an LLM agent run failed: loops, error cascades, unrecovered errors, goal drift, redundant tool calls, cost runaway. Deterministic detectors are the source of truth; an optional LLM layer explains them in plain language. Correct on all 5 recorded real-agent runs (see Limits).**

Interactive demo of **agent-autopsy** —
a framework-agnostic interceptor that turns an LLM agent's execution trace into a
structured post-mortem.

Paste an agent trace (or load a bundled example) and the app runs the **genuine**
analysis engine — the same `AgentTracer`, `ExecutionGraph`, and
`FailurePatternDetector` code from the repository, vendored into this Space — to
produce:

- **Detected failure patterns** with severity and remediation advice:
  - `INFINITE_LOOP` — the same prompt / tool repeats
  - `TOKEN_EXPLOSION` — a single call blows past the context window (token storm)
  - `GOAL_DRIFT` — the last two LLM prompts both diverge from the first one (needs 3+ LLM calls)
  - `ERROR_CASCADE` — consecutive failures pile up
  - `UNRECOVERED_ERROR` — the run ends after an error with no successful tool call afterwards
  - `REDUNDANT_TOOL_CALL` — the same tool is re-called with identical arguments after it already returned a usable result
  - `COST_RUNAWAY` — context usage is growing fast enough to overflow the window within a few more calls (projection, not a single oversized call)
  - `TOOL_STORM`, `EMPTY_TOOL_LOOP`
- A rule-based **post-mortem summary** (deterministic, no API keys)
- An **optional LLM root-cause explanation** — a button sends the detected patterns
  and a trace digest to an LLM that explains them in plain language. It never
  overrides the detectors and needs no paid key (see "Deterministic vs LLM" below)
- An **execution flame graph** rendered with networkx + matplotlib, with the
  steps implicated in each failure highlighted in red

## How it works

Your trace JSON is replayed through the real `AgentTracer` recording API. The
tracer builds the actual execution DAG, the real `FailurePatternDetector` runs
its heuristic battery, and a genuine `AutopsyReport` is produced. **The findings
are not fabricated** — the detected patterns are exactly what the library outputs.

## Deterministic vs LLM-generated

Be clear about which part is which:

- **Deterministic (source of truth).** Every failure pattern, severity, affected
  step and the post-mortem summary come from rule-based detectors in
  `autopsy/patterns.py`. `GOAL_DRIFT` uses CPU sentence embeddings (fastembed,
  ONNX, no torch) to compare meaning, with a bag-of-words cosine fallback if the
  model cannot load. These run with no network and no API key, and are what you
  should trust.
- **Generated (optional, explanatory only).** The "Explain root cause" button
  sends the already-detected patterns plus a short trace digest to an LLM (via the
  Hugging Face Inference API) and asks it to narrate the most likely root cause and
  the single most important fix. It is instructed to refer only to the findings,
  but like any LLM it can still be wrong, so it never overrides a detector. If no
  `HF_TOKEN` is configured — or the model is unavailable, times out, or the account
  is out of credits — it degrades gracefully to a templated recap of the same
  patterns (clearly labelled as templated). No paid key is required and no
  torch/transformers model is downloaded.

To enable live LLM explanations on your own deployment, set an `HF_TOKEN` secret
on the Space (a free token works for small models). Without it, every visitor
sees the templated summary. Override the model with the `AUTOPSY_LLM_MODEL`
environment variable (default: `meta-llama/Llama-3.1-8B-Instruct`).

## Trace schema

```json
{
  "name": "my-agent",
  "steps": [
    { "type": "DECISION", "description": "Goal: ..." },
    { "type": "LLM_CALL", "model": "gpt-4o", "prompt": "...", "response": "...",
      "latency_ms": 820, "tokens": 140, "context_window": 128000 },
    { "type": "TOOL_CALL", "tool_name": "web_search", "args": {}, "result": null,
      "latency_ms": 120 },
    { "type": "ERROR", "error": "TimeoutError: ..." }
  ]
}
```

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Runs on free CPU hardware — no GPU, no torch. The analysis itself needs no
external API keys; the optional LLM explanation uses the Hugging Face Inference
API only when an `HF_TOKEN` is set, and otherwise falls back to a templated recap.

---

- Source: the Files tab of this Space
- License: MIT

## Limits

- Traces are limited to 100 steps in this demo, because the flame graph's image size grows with the number of steps.
- Four example buttons load scripted traces written to show one failure pattern each. Five more, named "Real run",
  were recorded from a small tool-using agent (Qwen2.5-1.5B-Instruct with a lookup table, a calculator and an orders
  API whose first call times out). Four of those runs gave correct answers; in the fifth the agent gave up after the
  first timeout instead of retrying. Details are in `real_traces.json`.
- Recording those runs exposed four engine problems, fixed in this Space's copy (regression tests are in `tests/`):
  an agent with a system prompt was flagged as an infinite loop after three steps because only the first 120 characters
  of each prompt were compared; a failed run with no patterns was summarized as "completed successfully"; a
  replayed trace reported 0 ms latency instead of the recorded latencies; and an agent that gave up after a single
  tool error went unreported, because the error-cascade rule needs consecutive errors and a real agent has an LLM call
  between errors. A new `UNRECOVERED_ERROR` pattern now covers a run that ends after an error with no successful tool
  call afterwards.
- `UNRECOVERED_ERROR` is a heuristic: it also fires when an agent reasonably answers without an optional tool.
- Pattern detection is rule-based: it finds loops, error cascades and similar shapes, and can miss failures
  that do not match a known pattern.
- `GOAL_DRIFT` compares meaning via CPU sentence embeddings (fastembed), falling back to a bag-of-words cosine if the embedding model cannot load. A trace whose prompts are short and unrelated to each other can still trigger it; a real agent that reuses a system prompt did not in the 5 recorded runs.
- `REDUNDANT_TOOL_CALL` and `COST_RUNAWAY` are heuristics too. Redundant-call detection ignores retries after an empty result or an error (those are legitimate recovery); cost-runaway is a projection from recent per-call token growth toward the context window, not a measured overflow.
- The optional LLM explanation is generative and can be wrong; it never changes the detected patterns, which remain the source of truth.
- The Space exposes one API endpoint, `/analyze`.
