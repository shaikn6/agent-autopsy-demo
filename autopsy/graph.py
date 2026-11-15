"""ExecutionGraph — networkx DAG representing every step an agent takes."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import networkx as nx


class StepType(str, Enum):
    START = "START"
    LLM_CALL = "LLM_CALL"
    TOOL_CALL = "TOOL_CALL"
    MEMORY_READ = "MEMORY_READ"
    DECISION = "DECISION"
    END = "END"
    ERROR = "ERROR"


class ExecutionGraph:
    """Directed Acyclic Graph (DAG) of agent execution steps.

    Each node represents a discrete step; edges represent sequential or
    branching control flow.
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._last_node: str | None = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_step(
        self,
        step_type: StepType | str,
        metadata: dict[str, Any] | None = None,
        *,
        tokens_used: int = 0,
    ) -> str:
        """Add a step node and wire it to the previous step.

        Returns the new node id.
        """
        node_id = str(uuid.uuid4())[:8]
        self._g.add_node(
            node_id,
            type=StepType(step_type) if isinstance(step_type, str) else step_type,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            metadata=metadata or {},
            tokens_used=tokens_used,
        )
        if self._last_node is not None:
            self.add_edge(self._last_node, node_id)
        self._last_node = node_id
        return node_id

    def add_edge(self, from_id: str, to_id: str, label: str = "") -> None:
        """Explicitly add a directed edge between two existing nodes."""
        self._g.add_edge(from_id, to_id, label=label)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def nodes(self) -> list[dict[str, Any]]:
        return [{"id": n, **self._g.nodes[n]} for n in self._g.nodes]

    def edges(self) -> list[dict[str, Any]]:
        return [
            {"from": u, "to": v, "label": self._g.edges[u, v].get("label", "")}
            for u, v in self._g.edges
        ]

    def steps_of_type(self, step_type: StepType) -> list[dict[str, Any]]:
        return [n for n in self.nodes() if n["type"] == step_type]

    def detect_cycles(self) -> list[list[str]]:
        """Return all cycles in the graph (empty list if DAG is clean)."""
        try:
            return list(nx.simple_cycles(self._g))
        except nx.NetworkXNoCycle:
            return []

    def total_tokens(self) -> int:
        return sum(data.get("tokens_used", 0) for _, data in self._g.nodes(data=True))

    def step_count(self) -> int:
        return self._g.number_of_nodes()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": n,
                    "type": self._g.nodes[n]["type"].value
                    if isinstance(self._g.nodes[n]["type"], StepType)
                    else str(self._g.nodes[n]["type"]),
                    "timestamp": self._g.nodes[n]["timestamp"],
                    "metadata": self._g.nodes[n]["metadata"],
                    "tokens_used": self._g.nodes[n].get("tokens_used", 0),
                }
                for n in self._g.nodes
            ],
            "edges": self.edges(),
        }

    def to_dot(self) -> str:
        """Export as Graphviz DOT string."""
        lines = ["digraph agent_execution {", '  rankdir="LR";']
        for n in self._g.nodes:
            node_type = self._g.nodes[n].get("type", "")
            if isinstance(node_type, StepType):
                node_type = node_type.value
            lines.append(f'  "{n}" [label="{node_type}\\n{n}"];')
        for u, v in self._g.edges:
            label = self._g.edges[u, v].get("label", "")
            lines.append(f'  "{u}" -> "{v}" [label="{label}"];')
        lines.append("}")
        return "\n".join(lines)
