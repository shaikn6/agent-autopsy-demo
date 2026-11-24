"""Bundled, realistic example agent traces for the demo.

Each trace is a plain dict that is replayed through the REAL AgentTracer
recording API (see app.replay_trace), so the failure patterns and post-mortem
shown in the UI come from genuine agent-autopsy logic — nothing is faked.
"""

from __future__ import annotations

import json as _json
from pathlib import Path as _Path

# 1) An agent stuck repeating the same reasoning, then drowning in errors.
LOOP_AND_CASCADE = {
    "name": "flight-booking-agent",
    "steps": [
        {
            "type": "DECISION",
            "description": "Goal: book the cheapest flight NYC -> London",
        },
        {
            "type": "LLM_CALL",
            "model": "gpt-4o",
            "prompt": "What should I do to find a flight from NYC to London?",
            "response": "I should search the flights API.",
            "latency_ms": 820,
            "tokens": 140,
        },
        {
            "type": "LLM_CALL",
            "model": "gpt-4o",
            "prompt": "What should I do to find a flight from NYC to London?",
            "response": "I should search the flights API.",
            "latency_ms": 790,
            "tokens": 140,
        },
        {
            "type": "LLM_CALL",
            "model": "gpt-4o",
            "prompt": "What should I do to find a flight from NYC to London?",
            "response": "I should search the flights API.",
            "latency_ms": 810,
            "tokens": 140,
        },
        {
            "type": "LLM_CALL",
            "model": "gpt-4o",
            "prompt": "What should I do to find a flight from NYC to London?",
            "response": "I should search the flights API.",
            "latency_ms": 805,
            "tokens": 140,
        },
        {"type": "ERROR", "error": "TimeoutError: flights API timed out (attempt 1)"},
        {"type": "ERROR", "error": "TimeoutError: flights API timed out (attempt 2)"},
        {"type": "ERROR", "error": "TimeoutError: flights API timed out (attempt 3)"},
    ],
}

# 2) A single oversized context call — classic token storm.
TOKEN_STORM = {
    "name": "doc-summarizer-agent",
    "steps": [
        {
            "type": "DECISION",
            "description": "Goal: summarize the uploaded 400-page contract",
        },
        {
            "type": "LLM_CALL",
            "model": "gpt-4o",
            "prompt": "Summarize the following contract for the legal team.",
            "response": "Pulling the full document into context...",
            "latency_ms": 1200,
            "tokens": 4_000,
            "context_window": 128_000,
        },
        {
            "type": "LLM_CALL",
            "model": "gpt-4o",
            "prompt": "Now summarize the ENTIRE contract verbatim in one pass.",
            "response": "(truncated)",
            "latency_ms": 9_400,
            "tokens": 119_000,
            "context_window": 128_000,
        },
        {
            "type": "LLM_CALL",
            "model": "gpt-4o",
            "prompt": "Produce the final executive summary for the legal team.",
            "response": "Here is the summary...",
            "latency_ms": 1500,
            "tokens": 900,
            "context_window": 128_000,
        },
    ],
}

# 3) An agent that wanders off-objective — goal drift.
GOAL_DRIFT = {
    "name": "research-assistant-agent",
    "steps": [
        {
            "type": "DECISION",
            "description": "Goal: research quarterly revenue for Acme Corp",
        },
        {
            "type": "LLM_CALL",
            "model": "claude-3-5-sonnet",
            "prompt": "Find the quarterly revenue numbers for Acme Corp from their latest filing.",
            "response": "Let me look up the 10-Q filing.",
            "latency_ms": 640,
            "tokens": 180,
        },
        {
            "type": "TOOL_CALL",
            "tool_name": "web_search",
            "args": {"q": "Acme Corp 10-Q revenue"},
            "result": "Found 5 articles about Acme office culture and hiring.",
            "latency_ms": 320,
        },
        {
            "type": "LLM_CALL",
            "model": "claude-3-5-sonnet",
            "prompt": "Interesting — tell me about the company culture and employee reviews instead.",
            "response": "Acme has a 4-day work week and...",
            "latency_ms": 700,
            "tokens": 260,
        },
        {
            "type": "LLM_CALL",
            "model": "claude-3-5-sonnet",
            "prompt": "Write a blog post praising remote work based on what you found.",
            "response": "Remote work is the future...",
            "latency_ms": 880,
            "tokens": 540,
        },
    ],
}

# 3b) Scripted: an agent that re-fetches data it already had and lets its context balloon toward the window.
REDUNDANT_AND_COST = {
    "name": "quarterly-report-agent",
    "steps": [
        {"type": "DECISION", "description": "Goal: compute the quarterly report totals"},
        {
            "type": "LLM_CALL",
            "model": "gpt-4o",
            "prompt": "Compute the quarterly report totals.",
            "response": "I'll fetch the revenue first.",
            "latency_ms": 500,
            "tokens": 2_000,
            "context_window": 8_000,
        },
        {
            "type": "TOOL_CALL",
            "tool_name": "get_revenue",
            "args": {"quarter": 3},
            "result": {"revenue": 500},
            "latency_ms": 120,
        },
        {
            "type": "LLM_CALL",
            "model": "gpt-4o",
            "prompt": "Compute the quarterly report totals using the revenue figure.",
            "response": "Let me double-check the revenue.",
            "latency_ms": 900,
            "tokens": 4_000,
            "context_window": 8_000,
        },
        {
            "type": "TOOL_CALL",
            "tool_name": "get_revenue",
            "args": {"quarter": 3},
            "result": {"revenue": 500},
            "latency_ms": 118,
        },
        {
            "type": "LLM_CALL",
            "model": "gpt-4o",
            "prompt": "Compute the quarterly report totals and format the output.",
            "response": "Here are the totals...",
            "latency_ms": 1_100,
            "tokens": 6_000,
            "context_window": 8_000,
        },
    ],
}

# 4) A healthy run with no failures — useful as a contrast / control.
HEALTHY = {
    "name": "weather-lookup-agent",
    "steps": [
        {
            "type": "DECISION",
            "description": "Goal: report tomorrow's weather for Dayton, OH",
        },
        {
            "type": "LLM_CALL",
            "model": "gpt-4o-mini",
            "prompt": "What is tomorrow's forecast for Dayton, Ohio?",
            "response": "I'll call the weather tool.",
            "latency_ms": 410,
            "tokens": 90,
        },
        {
            "type": "TOOL_CALL",
            "tool_name": "get_weather",
            "args": {"city": "Dayton, OH", "day": "tomorrow"},
            "result": {"high_f": 72, "low_f": 55, "summary": "Partly cloudy"},
            "latency_ms": 130,
        },
        {
            "type": "LLM_CALL",
            "model": "gpt-4o-mini",
            "prompt": "Summarize tomorrow's forecast for Dayton, Ohio for the user.",
            "response": "Tomorrow in Dayton: partly cloudy, high 72F.",
            "latency_ms": 380,
            "tokens": 110,
        },
    ],
}


# Recorded from a real run of a small tool-using agent (Qwen2.5-1.5B-Instruct, three tools, one deliberately flaky),
# not written by hand. See real_traces.json for the task, final answer and recording date of each.
_REAL = _json.loads((_Path(__file__).parent / "real_traces.json").read_text())
REAL_RUNS = {f"Real run: {t['id']}": {"name": t["name"], "steps": t["steps"]} for t in _REAL}

EXAMPLES: dict[str, dict] = {
    "Infinite loop + error cascade": LOOP_AND_CASCADE,
    "Token storm (context blowout)": TOKEN_STORM,
    "Goal drift (off-objective)": GOAL_DRIFT,
    "Redundant calls + cost runaway": REDUNDANT_AND_COST,
    "Healthy run (no failures)": HEALTHY,
    **REAL_RUNS,
}
