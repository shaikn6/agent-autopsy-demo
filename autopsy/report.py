"""AutopsyReport — structured output from a completed agent trace."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .patterns import FailurePattern


@dataclass
class AutopsyReport:
    agent_name: str
    run_id: str
    started_at: datetime
    completed_at: datetime
    status: str  # SUCCESS | FAILED | TIMEOUT
    total_steps: int
    total_tokens: int
    total_latency_ms: float
    failure_patterns: list[FailurePattern]
    graph: dict[str, Any]
    summary: str = ""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _as_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "status": self.status,
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "failure_patterns": [
                {
                    "name": p.name,
                    "severity": p.severity,
                    "description": p.description,
                    "step_ids": p.step_ids,
                    "recommendation": p.recommendation,
                }
                for p in self.failure_patterns
            ],
            "graph": self.graph,
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self._as_dict(), indent=indent)

    def to_markdown(self) -> str:
        severity_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
        lines: list[str] = [
            f"# Autopsy Report — {self.agent_name}",
            "",
            f"**Run ID:** `{self.run_id}`  ",
            f"**Status:** `{self.status}`  ",
            f"**Started:** {self.started_at.isoformat()}  ",
            f"**Completed:** {self.completed_at.isoformat()}  ",
            f"**Duration:** {self.total_latency_ms:.0f} ms  ",
            f"**Total Steps:** {self.total_steps}  ",
            f"**Total Tokens:** {self.total_tokens:,}  ",
            "",
            "## Summary",
            "",
            self.summary or "_No summary generated._",
            "",
            "## Failure Patterns",
            "",
        ]

        if not self.failure_patterns:
            lines.append("_No failure patterns detected. Agent completed cleanly._")
        else:
            for p in self.failure_patterns:
                icon = severity_emoji.get(p.severity, "⚪")
                lines += [
                    f"### {icon} {p.name} ({p.severity})",
                    "",
                    f"**Description:** {p.description}  ",
                    f"**Affected Steps:** {', '.join(f'`{s}`' for s in p.step_ids)}  ",
                    f"**Recommendation:** {p.recommendation}",
                    "",
                ]

        lines += [
            "## Execution Graph",
            "",
            f"Nodes: {len(self.graph.get('nodes', []))} | "
            f"Edges: {len(self.graph.get('edges', []))}",
            "",
            "| Step ID | Type | Tokens | Timestamp |",
            "|---------|------|--------|-----------|",
        ]
        for node in self.graph.get("nodes", []):
            lines.append(
                f"| `{node['id']}` | {node['type']} "
                f"| {node.get('tokens_used', 0):,} | {node['timestamp']} |"
            )

        return "\n".join(lines)

    def to_html(self) -> str:
        severity_color = {
            "CRITICAL": "#dc2626",
            "HIGH": "#ea580c",
            "MEDIUM": "#ca8a04",
            "LOW": "#2563eb",
        }
        pattern_html = ""
        if not self.failure_patterns:
            pattern_html = "<p><em>No failure patterns detected.</em></p>"
        else:
            for p in self.failure_patterns:
                color = severity_color.get(p.severity, "#6b7280")
                pattern_html += f"""
                <div style="border-left:4px solid {color};padding:8px 16px;margin:12px 0;background:#fafafa;">
                  <strong style="color:{color}">[{p.severity}] {p.name}</strong>
                  <p>{p.description}</p>
                  <p><em>Recommendation:</em> {p.recommendation}</p>
                  <p><small>Steps: {", ".join(p.step_ids)}</small></p>
                </div>"""

        node_rows = "".join(
            f"<tr><td><code>{n['id']}</code></td><td>{n['type']}</td>"
            f"<td>{n.get('tokens_used', 0):,}</td><td>{n['timestamp']}</td></tr>"
            for n in self.graph.get("nodes", [])
        )

        status_color = {"SUCCESS": "#16a34a", "FAILED": "#dc2626", "TIMEOUT": "#ca8a04"}
        sc = status_color.get(self.status, "#6b7280")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Autopsy — {self.agent_name}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #111; }}
    h1 {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 12px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .badge {{ display:inline-block; padding:2px 8px; border-radius:9999px; font-size:.8rem; font-weight:600; color:#fff; background:{sc}; }}
  </style>
</head>
<body>
  <h1>Autopsy Report — {self.agent_name}</h1>
  <p>
    <span class="badge">{self.status}</span>
    &nbsp; Run <code>{self.run_id}</code>
    &nbsp;|&nbsp; {self.total_steps} steps
    &nbsp;|&nbsp; {self.total_tokens:,} tokens
    &nbsp;|&nbsp; {self.total_latency_ms:.0f} ms
  </p>
  <h2>Summary</h2>
  <p>{self.summary or "<em>No summary generated.</em>"}</p>
  <h2>Failure Patterns</h2>
  {pattern_html}
  <h2>Execution Graph</h2>
  <table>
    <thead><tr><th>Step ID</th><th>Type</th><th>Tokens</th><th>Timestamp</th></tr></thead>
    <tbody>{node_rows}</tbody>
  </table>
</body>
</html>"""

    def save(self, path: str) -> None:
        """Persist the report to *path*.

        Extension determines format: .json, .md/.markdown, .html.
        Defaults to JSON for unknown extensions.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        suffix = p.suffix.lower()
        if suffix in (".md", ".markdown"):
            p.write_text(self.to_markdown(), encoding="utf-8")
        elif suffix in (".html", ".htm"):
            p.write_text(self.to_html(), encoding="utf-8")
        else:
            p.write_text(self.to_json(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_report(
    *,
    agent_name: str,
    started_at: datetime,
    completed_at: datetime,
    status: str,
    graph_dict: dict[str, Any],
    failure_patterns: list[FailurePattern],
    total_tokens: int,
    total_latency_ms: float,
    run_id: str | None = None,
) -> AutopsyReport:
    """Convenience factory used by AgentTracer."""
    rid = run_id or str(uuid.uuid4())
    total_steps = len(graph_dict.get("nodes", []))

    # Rule-based summary (no LLM dependency)
    if not failure_patterns:
        outcome = "completed successfully" if status == "SUCCESS" else status.lower()
        summary = (
            f"Agent '{agent_name}' {outcome} in {total_steps} steps "
            f"using {total_tokens:,} tokens ({total_latency_ms:.0f} ms). "
            "No failure patterns detected."
        )
    else:
        criticals = [p for p in failure_patterns if p.severity == "CRITICAL"]
        summary = (
            f"Agent '{agent_name}' {status.lower()} after {total_steps} steps "
            f"({total_tokens:,} tokens, {total_latency_ms:.0f} ms). "
            f"Detected {len(failure_patterns)} failure pattern(s)"
        )
        if criticals:
            names = ", ".join(p.name for p in criticals)
            summary += f", including {len(criticals)} CRITICAL issue(s): {names}."
        else:
            summary += "."

    return AutopsyReport(
        agent_name=agent_name,
        run_id=rid,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        total_steps=total_steps,
        total_tokens=total_tokens,
        total_latency_ms=total_latency_ms,
        failure_patterns=failure_patterns,
        graph=graph_dict,
        summary=summary,
    )
