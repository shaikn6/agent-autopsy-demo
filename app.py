"""Agent Autopsy — interactive Gradio demo.

Paste an agent execution trace (or load a bundled example) and get a real
post-mortem: detected failure patterns + execution flame graph, computed by the
genuine `agent-autopsy` analysis engine vendored in `./autopsy`.

Source: the Files tab of this Space
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))

from autopsy import AgentTracer  # noqa: E402  (genuine analysis engine)
from autopsy.report import AutopsyReport  # noqa: E402
from autopsy.llm_summary import generate_root_cause  # noqa: E402  (optional explain-only LLM layer)

from examples_traces import EXAMPLES  # noqa: E402

# --------------------------------------------------------------------------- #
# Trace replay — drives the REAL AgentTracer recording API.
# --------------------------------------------------------------------------- #

_NODE_COLORS = {
    "START": "#16a34a",
    "END": "#0f766e",
    "LLM_CALL": "#2563eb",
    "TOOL_CALL": "#7c3aed",
    "MEMORY_READ": "#0891b2",
    "DECISION": "#ca8a04",
    "ERROR": "#dc2626",
}

_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


# The flame graph is 1.25 inches wide per step at 120 dpi; past ~109 steps it exceeds the 16,383 px image limit.
MAX_STEPS = 100


def replay_trace(trace: dict[str, Any]) -> AutopsyReport:
    """Replay a trace dict through the real AgentTracer and return its report.

    The tracer builds the genuine ExecutionGraph, runs the real
    FailurePatternDetector and produces the real AutopsyReport on __exit__.
    """
    name = str(trace.get("name", "untitled-agent"))
    steps = trace.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("`steps` must be a list of step objects.")
    if not steps:
        raise ValueError("The trace has no steps, so there is nothing to analyse.")
    if len(steps) > MAX_STEPS:
        raise ValueError(f"Traces are limited to {MAX_STEPS:,} steps in this demo ({len(steps):,} given).")

    with AgentTracer(name=name) as tracer:
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i} is not an object.")
            step_type = str(step.get("type", "")).upper()
            for field in ("latency_ms", "tokens", "context_window"):
                value = step.get(field)
                if value is None:
                    continue
                try:
                    float(value)
                except (TypeError, ValueError):
                    raise ValueError(f"Step {i}: `{field}` must be a number, got {step.get(field)!r}.")

            if step_type == "LLM_CALL":
                tracer.record_llm_call(
                    model=str(step.get("model", "unknown")),
                    prompt=str(step.get("prompt", "")),
                    response=str(step.get("response", "")),
                    latency_ms=float(step.get("latency_ms", 0) or 0),
                    tokens=int(float(step.get("tokens", 0) or 0)),
                    context_window=step.get("context_window"),
                )
            elif step_type == "TOOL_CALL":
                tracer.record_tool_call(
                    tool_name=str(step.get("tool_name", "unknown")),
                    args=step.get("args", {}) or {},
                    result=step.get("result"),
                    latency_ms=float(step.get("latency_ms", 0) or 0),
                )
            elif step_type == "MEMORY_READ":
                tracer.record_memory_read(str(step.get("key", "")), step.get("value"))
            elif step_type == "DECISION":
                tracer.record_decision(str(step.get("description", "")))
            elif step_type == "ERROR":
                tracer.record_error(Exception(str(step.get("error", "error"))))
            else:
                raise ValueError(
                    f"Step {i}: unknown type {step_type!r}. "
                    "Use one of LLM_CALL, TOOL_CALL, MEMORY_READ, DECISION, ERROR."
                )

    return tracer.get_report()


# --------------------------------------------------------------------------- #
# Visualisation — draw the execution graph with networkx + matplotlib.
# --------------------------------------------------------------------------- #


def render_graph(report: AutopsyReport):
    """Render the execution DAG as a left-to-right flame graph image."""
    graph = report.graph
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    g = nx.DiGraph()
    flagged: set[str] = {sid for p in report.failure_patterns for sid in p.step_ids}

    for node in nodes:
        g.add_node(node["id"], type=node["type"], tokens=node.get("tokens_used", 0))
    for edge in edges:
        g.add_edge(edge["from"], edge["to"])

    # Linear left-to-right layout (execution order is sequential).
    pos = {node["id"]: (idx, 0) for idx, node in enumerate(nodes)}

    # Build the figure via the object API (not plt.subplots) so it is never
    # registered in pyplot's global state; otherwise each run leaks a figure.
    fig = Figure(figsize=(max(8, len(nodes) * 1.25), 3.2), dpi=120)
    ax = fig.subplots()
    fig.patch.set_facecolor("#ffffff")

    node_colors = [_NODE_COLORS.get(g.nodes[n]["type"], "#6b7280") for n in g.nodes]
    edge_lines = ["#dc2626" if (n in flagged) else "#cbd5e1" for n in g.nodes]
    sizes = [1500 if g.nodes[n]["type"] in ("START", "END") else 1900 for n in g.nodes]

    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        edge_color="#94a3b8",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        width=1.6,
        node_size=2000,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=sizes,
        edgecolors=edge_lines,
        linewidths=[3 if c == "#dc2626" else 1 for c in edge_lines],
    )
    labels = {n: g.nodes[n]["type"].replace("_", "\n") for n in g.nodes}
    nx.draw_networkx_labels(
        g, pos, labels, ax=ax, font_size=7, font_color="#ffffff", font_weight="bold"
    )

    ax.set_title(
        f"Execution flame graph — {report.agent_name}  "
        f"({report.total_steps} steps, status: {report.status})",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
        pad=14,
    )
    ax.margins(0.12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _post_mortem_markdown(report: AutopsyReport) -> str:
    """Compose the post-mortem view from the report's real summary + patterns."""
    icon = {"SUCCESS": "🟢", "FAILED": "🔴", "TIMEOUT": "🟡"}.get(report.status, "⚪")
    lines = [
        f"## {icon} Post-Mortem — `{report.agent_name}`",
        "",
        f"> {report.summary}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Status | **{report.status}** |",
        f"| Steps | {report.total_steps} |",
        f"| Tokens | {report.total_tokens:,} |",
        f"| Duration | {report.total_latency_ms:.0f} ms |",
        f"| Failure patterns | **{len(report.failure_patterns)}** |",
        "",
        "### Detected Failure Patterns",
        "",
    ]
    if not report.failure_patterns:
        lines.append("✅ No failure patterns detected — the agent completed cleanly.")
    else:
        emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
        ordered = sorted(
            report.failure_patterns, key=lambda p: _SEVERITY_ORDER.get(p.severity, 9)
        )
        for p in ordered:
            lines += [
                f"#### {emoji.get(p.severity, '⚪')} {p.name} — `{p.severity}`",
                "",
                f"{p.description}",
                "",
                f"**Affected steps:** {', '.join(f'`{s}`' for s in p.step_ids) or '—'}",
                "",
                f"**Recommendation:** {p.recommendation}",
                "",
            ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Gradio callbacks
# --------------------------------------------------------------------------- #


def analyze(trace_text: str):
    """Parse pasted JSON, run the real autopsy, return (markdown, figure, json)."""
    if not trace_text or not trace_text.strip():
        raise gr.Error("Paste a trace JSON or click an example button above.")
    try:
        trace = json.loads(trace_text)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Invalid JSON: {exc}") from exc
    if not isinstance(trace, dict):
        raise gr.Error("Trace must be a JSON object with `name` and `steps`.")

    try:
        report = replay_trace(trace)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    return (
        _post_mortem_markdown(report),
        render_graph(report),
        json.loads(report.to_json()),
    )


def load_example(name: str) -> str:
    return json.dumps(EXAMPLES[name], indent=2)


def explain_root_cause(report_dict: dict[str, Any] | None) -> str:
    """Optional LLM layer: explain the deterministic findings in plain language.

    Takes the raw report JSON already shown in the UI. The deterministic detectors stay authoritative; this only
    narrates them, and degrades to a templated summary when no model is reachable.
    """
    if not report_dict:
        return "_Run an autopsy first, then click this to get a plain-language root-cause explanation._"
    text, source = generate_root_cause(report_dict)
    if source == "deterministic":
        note = "_The deterministic detectors found nothing to explain._"
    elif source == "templated":
        note = (
            "_Source: **templated** (deterministic). No LLM was reachable, so this is a plain recap of the findings "
            "above._"
        )
    else:
        note = (
            f"_Source: **{source}** — explanatory only. The detected patterns above are the authoritative source of "
            "truth; this generated text can be wrong._"
        )
    return f"### 🧠 Root-cause explanation (optional)\n\n{note}\n\n{text}"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

_CSS = """
.gradio-container { max-width: 1180px !important; }
#hero h1 { font-size: 2.1rem; margin-bottom: .25rem; letter-spacing: -.02em; }
#hero p { color: #475569; margin-top: 0; }
.gr-button-primary { font-weight: 600 !important; }
footer { visibility: hidden; }
"""

_INTRO = """
<div id="hero">
  <h1>🔬 Agent Autopsy</h1>
  <p>Framework-agnostic post-mortems for LLM agents. Paste an execution trace and
  get the <b>real</b> failure-pattern analysis — infinite loops, token storms,
  goal drift, error cascades — plus an execution flame graph.</p>
  <p style="font-size:.9rem">
    Powered by the agent-autopsy analysis engine (source in the Files tab).
    Deterministic detectors + CPU sentence embeddings · networkx · no GPU.
  </p>
  <p style="font-size:.8rem;opacity:.75">
    The detectors are rule-based and are the source of truth. An optional
    LLM layer can then <i>explain</i> the findings in plain language (it never
    overrides them, and degrades to a templated recap when no model is reachable).
    Verified on the bundled example traces; goal-drift detection can
    false-positive on long, legitimately evolving tasks. Treat findings as
    signals, not verdicts.
  </p>
</div>
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue"), css=_CSS, title="Agent Autopsy"
) as demo:
    gr.HTML(_INTRO)

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 1 · Load an example trace")
            with gr.Row():
                ex_buttons = [gr.Button(name, size="sm") for name in EXAMPLES]

            gr.Markdown("### 2 · Or paste your own trace JSON")
            trace_box = gr.Textbox(
                label="Agent trace (JSON)",
                value=json.dumps(next(iter(EXAMPLES.values())), indent=2),
                lines=18,
                max_lines=28,
                show_copy_button=True,
            )
            analyze_btn = gr.Button("🔍 Run Autopsy", variant="primary", size="lg")
            gr.Markdown(
                '**Trace schema:** `{ "name": str, "steps": [ { "type": '
                '"LLM_CALL|TOOL_CALL|DECISION|MEMORY_READ|ERROR", ... } ] }`.  '
                "LLM steps accept `model, prompt, response, latency_ms, tokens, context_window`."
            )

        with gr.Column(scale=7):
            gr.Markdown("### 3 · Autopsy report")
            graph_plot = gr.Plot(label="Execution flame graph")
            post_mortem = gr.Markdown()
            explain_btn = gr.Button(
                "🧠 Explain root cause (optional LLM)", size="sm"
            )
            explain_md = gr.Markdown()
            with gr.Accordion("Raw report JSON", open=False):
                raw_json = gr.JSON(label="AutopsyReport")

    explain_btn.click(
        explain_root_cause, inputs=raw_json, outputs=explain_md, api_name=False
    )

    analyze_btn.click(
        analyze,
        inputs=trace_box,
        outputs=[post_mortem, graph_plot, raw_json],
        api_name="analyze",
    ).then(lambda: "", None, explain_md, api_name=False)
    for btn, name in zip(ex_buttons, EXAMPLES):
        btn.click(
            (lambda n=name: load_example(n)),
            inputs=None,
            outputs=trace_box,
            api_name=False,
        ).then(
            analyze,
            inputs=trace_box,
            outputs=[post_mortem, graph_plot, raw_json],
            api_name=False,
        ).then(lambda: "", None, explain_md, api_name=False)

    demo.load(
        analyze,
        inputs=trace_box,
        outputs=[post_mortem, graph_plot, raw_json],
        api_name=False,
    )

if __name__ == "__main__":
    demo.launch()
