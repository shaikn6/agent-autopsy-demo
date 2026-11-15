"""FailurePatternDetector — identifies common agent failure modes."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

from .graph import ExecutionGraph, StepType


@dataclass
class FailurePattern:
    name: str
    severity: str  # CRITICAL | HIGH | MEDIUM | LOW
    description: str
    step_ids: list[str]
    recommendation: str


class FailurePatternDetector:
    """Runs a battery of heuristic checks against an ExecutionGraph.

    Each detector method returns a list of FailurePattern objects (possibly
    empty).  The public ``detect`` method aggregates all results.
    """

    # Tuneable thresholds ------------------------------------------------
    LOOP_REPEAT_THRESHOLD: int = 3
    TOOL_STORM_CALLS: int = 10
    TOOL_STORM_STEPS: int = 5
    TOKEN_EXPLOSION_FRACTION: float = 0.80
    DEFAULT_CONTEXT_WINDOW: int = 128_000
    GOAL_DRIFT_THRESHOLD: float = 0.32  # embedding-cosine scale; see _semantic_similarity
    EMPTY_RETRY_THRESHOLD: int = 3
    ERROR_CASCADE_THRESHOLD: int = 3
    REDUNDANT_CALL_THRESHOLD: int = 2  # identical (tool, args) calls with a usable result
    COST_RUNAWAY_MIN_LLM: int = 3
    COST_RUNAWAY_WINDOW_FRACTION: float = 0.25  # last call must already fill this much of the window
    COST_RUNAWAY_MAX_STEPS_TO_FULL: float = 3.0  # and the window must fill within this many more calls

    _EMPTY_RESULTS = (None, "", [], {}, "null", "[]", "{}")

    # ------------------------------------------------------------------

    def detect(self, graph: ExecutionGraph) -> list[FailurePattern]:
        patterns: list[FailurePattern] = []
        patterns.extend(self._detect_infinite_loop(graph))
        patterns.extend(self._detect_tool_storm(graph))
        patterns.extend(self._detect_token_explosion(graph))
        patterns.extend(self._detect_goal_drift(graph))
        patterns.extend(self._detect_empty_tool_loop(graph))
        patterns.extend(self._detect_redundant_tool_calls(graph))
        # TOKEN_EXPLOSION already owns the last oversized call; do not double-report it as a cost runaway.
        if not any(p.name == "TOKEN_EXPLOSION" for p in patterns):
            patterns.extend(self._detect_cost_runaway(graph))
        patterns.extend(self._detect_error_cascade(graph))
        if not any(p.name == "ERROR_CASCADE" for p in patterns):  # a cascade already says the run ended in errors
            patterns.extend(self._detect_unrecovered_error(graph))
        return patterns

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def _detect_infinite_loop(self, graph: ExecutionGraph) -> list[FailurePattern]:
        """INFINITE_LOOP: the same prompt or tool name repeats >= 3 times."""
        patterns: list[FailurePattern] = []

        # Check LLM calls — identical full prompts. A prefix is not enough: every call of an agent with a
        # system prompt starts the same way, so comparing 120 characters flags every agent after 3 steps.
        llm_nodes = graph.steps_of_type(StepType.LLM_CALL)
        prompt_counter: Counter[str] = Counter()
        prompt_node_map: dict[str, list[str]] = {}
        for node in llm_nodes:
            prompt = str(node["metadata"].get("prompt", ""))
            prompt_counter[prompt] += 1
            prompt_node_map.setdefault(prompt, []).append(node["id"])

        for prompt, count in prompt_counter.items():
            if count >= self.LOOP_REPEAT_THRESHOLD:
                patterns.append(
                    FailurePattern(
                        name="INFINITE_LOOP",
                        severity="CRITICAL",
                        description=(
                            f"Same LLM prompt repeated {count} times (prefix: {repr(prompt[:60])})"
                        ),
                        step_ids=prompt_node_map[prompt],
                        recommendation=(
                            "Add a step counter / visited-state check. "
                            "Break on repeated prompts or inject a diversity constraint."
                        ),
                    )
                )

        # Check tool calls — same tool name repeated
        tool_nodes = graph.steps_of_type(StepType.TOOL_CALL)
        tool_counter: Counter[str] = Counter()
        tool_node_map: dict[str, list[str]] = {}
        for node in tool_nodes:
            tool = str(node["metadata"].get("tool_name", "unknown"))
            tool_counter[tool] += 1
            tool_node_map.setdefault(tool, []).append(node["id"])

        for tool, count in tool_counter.items():
            if count >= self.LOOP_REPEAT_THRESHOLD:
                patterns.append(
                    FailurePattern(
                        name="INFINITE_LOOP",
                        severity="HIGH",
                        description=f"Tool '{tool}' called {count} times — possible loop.",
                        step_ids=tool_node_map[tool],
                        recommendation=(
                            "Guard tool re-invocation with a result-equality check. "
                            "Return early if consecutive results are identical."
                        ),
                    )
                )

        return patterns

    def _detect_tool_storm(self, graph: ExecutionGraph) -> list[FailurePattern]:
        """TOOL_STORM: more than TOOL_STORM_CALLS tool calls within TOOL_STORM_STEPS LLM steps.

        Strategy: slide a window of TOOL_STORM_STEPS *non-tool* nodes and count the
        tool calls that fall between adjacent non-tool checkpoints.  If the total
        tool-call count across the whole trace already exceeds the threshold we also
        flag it so that graphs with no LLM/decision interleaving are still caught.
        """
        nodes = graph.nodes()
        tool_nodes = [n for n in nodes if n["type"] == StepType.TOOL_CALL]

        if len(tool_nodes) < self.TOOL_STORM_CALLS:
            return []

        # Simple global check: if there are >= TOOL_STORM_CALLS tool nodes in
        # a run with fewer than TOOL_STORM_STEPS non-tool nodes, it's a storm.
        non_tool_nodes = [n for n in nodes if n["type"] != StepType.TOOL_CALL]
        storm_ids = [n["id"] for n in tool_nodes]

        if len(non_tool_nodes) <= self.TOOL_STORM_STEPS:
            # Very dense tool use — flag the whole sequence
            return [
                FailurePattern(
                    name="TOOL_STORM",
                    severity="HIGH",
                    description=(
                        f"{len(storm_ids)} tool calls with only "
                        f"{len(non_tool_nodes)} non-tool steps in between."
                    ),
                    step_ids=storm_ids,
                    recommendation=(
                        "Rate-limit tool invocations. "
                        "Collapse redundant calls or add a planning step before execution."
                    ),
                )
            ]

        # Sliding window: count tool calls between consecutive non-tool nodes
        non_tool_indices = [i for i, n in enumerate(nodes) if n["type"] != StepType.TOOL_CALL]
        for win_start in range(len(non_tool_indices) - self.TOOL_STORM_STEPS + 1):
            idx_lo = non_tool_indices[win_start]
            idx_hi = non_tool_indices[win_start + self.TOOL_STORM_STEPS - 1]
            window_tool_ids = [
                n["id"] for n in nodes[idx_lo : idx_hi + 1] if n["type"] == StepType.TOOL_CALL
            ]
            if len(window_tool_ids) >= self.TOOL_STORM_CALLS:
                return [
                    FailurePattern(
                        name="TOOL_STORM",
                        severity="HIGH",
                        description=(
                            f"{len(window_tool_ids)} tool calls in "
                            f"{self.TOOL_STORM_STEPS} consecutive steps."
                        ),
                        step_ids=window_tool_ids,
                        recommendation=(
                            "Rate-limit tool invocations. "
                            "Collapse redundant calls or add a planning step before execution."
                        ),
                    )
                ]
        return []

    def _detect_token_explosion(self, graph: ExecutionGraph) -> list[FailurePattern]:
        """TOKEN_EXPLOSION: a single step consumes >80 % of the context window."""
        patterns: list[FailurePattern] = []
        for node in graph.nodes():
            tokens = node.get("tokens_used", 0)
            raw_cw = node["metadata"].get("context_window")
            if raw_cw is None:
                raw_cw = self.DEFAULT_CONTEXT_WINDOW
            context_window = int(float(raw_cw))
            if context_window > 0 and tokens / context_window >= self.TOKEN_EXPLOSION_FRACTION:
                patterns.append(
                    FailurePattern(
                        name="TOKEN_EXPLOSION",
                        severity="CRITICAL",
                        description=(
                            f"Step {node['id']} used {tokens:,} tokens "
                            f"({tokens / context_window:.0%} of {context_window:,} context)."
                        ),
                        step_ids=[node["id"]],
                        recommendation=(
                            "Truncate or summarise context before this call. "
                            "Consider chunking or windowed memory strategies."
                        ),
                    )
                )
        return patterns

    def _detect_goal_drift(self, graph: ExecutionGraph) -> list[FailurePattern]:
        """GOAL_DRIFT: the last two LLM prompts are both semantically unlike the first.

        Similarity is a sentence-embedding cosine (``_semantic_similarity``), not a
        bag-of-words count, so a prompt that restates the objective in different
        words is recognised as on-goal instead of being mistaken for drift.

        Needs three LLM calls and two off-goal prompts in a row. A single different
        final prompt (a formatting or wrap-up step) is normal agent behaviour.
        """
        llm_nodes = graph.steps_of_type(StepType.LLM_CALL)
        if len(llm_nodes) < 3:
            return []

        # Embedding model when available (threshold GOAL_DRIFT_THRESHOLD); if it
        # cannot load, degrade to the bag-of-words cosine on its own threshold.
        similarity, threshold = _goal_similarity_backend(self.GOAL_DRIFT_THRESHOLD)

        first_prompt = str(llm_nodes[0]["metadata"].get("prompt", ""))
        tail = llm_nodes[-2:]
        sims = [
            similarity(first_prompt, str(n["metadata"].get("prompt", "")))
            for n in tail
        ]
        if all(sim < threshold for sim in sims):
            return [
                FailurePattern(
                    name="GOAL_DRIFT",
                    severity="HIGH",
                    description=(
                        f"The last two LLM prompts diverged from the initial objective "
                        f"(similarity={max(sims):.2f} or lower, threshold={threshold})."
                    ),
                    step_ids=[llm_nodes[0]["id"], *[n["id"] for n in tail]],
                    recommendation=(
                        "Pin the original objective in a system prompt. "
                        "Re-inject the goal into the context at regular intervals."
                    ),
                )
            ]
        return []

    def _detect_empty_tool_loop(self, graph: ExecutionGraph) -> list[FailurePattern]:
        """EMPTY_TOOL_LOOP: same tool returns empty/None 3+ consecutive times."""
        tool_nodes = graph.steps_of_type(StepType.TOOL_CALL)
        if not tool_nodes:
            return []

        patterns: list[FailurePattern] = []
        run_tool: str | None = None
        run_count = 0
        run_ids: list[str] = []

        for node in tool_nodes:
            tool = str(node["metadata"].get("tool_name", ""))
            result = node["metadata"].get("result")
            is_empty = result in (None, "", [], {}, "null", "[]", "{}")

            if is_empty and tool == run_tool:
                run_count += 1
                run_ids.append(node["id"])
            elif is_empty:
                run_tool = tool
                run_count = 1
                run_ids = [node["id"]]
            else:
                run_tool = None
                run_count = 0
                run_ids = []

            if run_count >= self.EMPTY_RETRY_THRESHOLD:
                patterns.append(
                    FailurePattern(
                        name="EMPTY_TOOL_LOOP",
                        severity="MEDIUM",
                        description=(
                            f"Tool '{tool}' returned empty result {run_count} consecutive times."
                        ),
                        step_ids=list(run_ids),
                        recommendation=(
                            "Add a fallback strategy when a tool returns empty. "
                            "Abort after 2 empty retries and report to the user."
                        ),
                    )
                )
                # Reset to avoid duplicate reports for the same run
                run_tool = None
                run_count = 0
                run_ids = []

        return patterns

    def _detect_redundant_tool_calls(self, graph: ExecutionGraph) -> list[FailurePattern]:
        """REDUNDANT_TOOL_CALL: the same tool is called again with identical arguments.

        Only fires when the agent already had a usable answer: every occurrence returned a non-empty result and no
        ERROR step sits between the duplicates. A retry after an empty result or an error is legitimate recovery
        (UNRECOVERED_ERROR / EMPTY_TOOL_LOOP cover those), not a redundant re-fetch. A tool already flagged by
        INFINITE_LOOP (same name 3+ times) is left to that detector.
        """
        nodes = graph.nodes()
        tool_nodes = [n for n in nodes if n["type"] == StepType.TOOL_CALL]
        if len(tool_nodes) < self.REDUNDANT_CALL_THRESHOLD:
            return []

        name_counts = Counter(str(n["metadata"].get("tool_name", "")) for n in tool_nodes)
        index = {n["id"]: i for i, n in enumerate(nodes)}
        error_indices = [i for i, n in enumerate(nodes) if n["type"] == StepType.ERROR]

        groups: dict[tuple[str, str], list[dict]] = {}
        for node in tool_nodes:
            tool = str(node["metadata"].get("tool_name", ""))
            args_key = json.dumps(node["metadata"].get("args", {}), sort_keys=True, default=str)
            groups.setdefault((tool, args_key), []).append(node)

        patterns: list[FailurePattern] = []
        for (tool, _args_key), occ in groups.items():
            if len(occ) < self.REDUNDANT_CALL_THRESHOLD:
                continue
            if name_counts[tool] >= self.LOOP_REPEAT_THRESHOLD:
                continue  # INFINITE_LOOP already owns this tool
            if any(x["metadata"].get("result") in self._EMPTY_RESULTS for x in occ):
                continue  # an empty result makes the re-call a legitimate retry
            lo, hi = index[occ[0]["id"]], index[occ[-1]["id"]]
            if any(lo < ei < hi for ei in error_indices):
                continue  # an error between the calls makes the re-call a legitimate retry
            patterns.append(
                FailurePattern(
                    name="REDUNDANT_TOOL_CALL",
                    severity="MEDIUM",
                    description=(
                        f"Tool '{tool}' was called {len(occ)} times with identical arguments and returned a "
                        "non-empty result each time — the later call(s) re-fetch data the agent already had."
                    ),
                    step_ids=[x["id"] for x in occ[1:]],
                    recommendation=(
                        "Cache or reuse the first result for identical arguments instead of re-invoking the tool. "
                        "Memoise idempotent tool calls within a run."
                    ),
                )
            )
        return patterns

    def _detect_cost_runaway(self, graph: ExecutionGraph) -> list[FailurePattern]:
        """COST_RUNAWAY: context usage is growing fast enough to overflow the window within a few more calls.

        Projection, not an absolute total: nearly every ReAct agent grows monotonically, so growth alone is normal.
        This fires only when the last call already fills a sizeable fraction of the window AND the recent per-call
        growth would exhaust the remaining window within COST_RUNAWAY_MAX_STEPS_TO_FULL more calls. TOKEN_EXPLOSION
        owns the already-oversized single call (suppressed in ``detect``).
        """
        llm_nodes = graph.steps_of_type(StepType.LLM_CALL)
        if len(llm_nodes) < self.COST_RUNAWAY_MIN_LLM:
            return []

        tokens = [int(n.get("tokens_used", 0) or 0) for n in llm_nodes]
        raw_cw = llm_nodes[-1]["metadata"].get("context_window")
        window = int(float(raw_cw)) if raw_cw else self.DEFAULT_CONTEXT_WINDOW
        if window <= 0:
            return []

        last_tokens = tokens[-1]
        if last_tokens / window >= self.TOKEN_EXPLOSION_FRACTION:
            return []  # that is a TOKEN_EXPLOSION, not a projected runaway
        if last_tokens / window < self.COST_RUNAWAY_WINDOW_FRACTION:
            return []

        recent = tokens[-3:]
        deltas = [b - a for a, b in zip(recent, recent[1:])]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
        if avg_delta <= 0:
            return []

        steps_to_full = (window - last_tokens) / avg_delta
        if steps_to_full > self.COST_RUNAWAY_MAX_STEPS_TO_FULL:
            return []

        return [
            FailurePattern(
                name="COST_RUNAWAY",
                severity="HIGH",
                description=(
                    f"Context usage is growing ~{avg_delta:,.0f} tokens per LLM call and the last call already used "
                    f"{last_tokens:,} tokens ({last_tokens / window:.0%} of the {window:,}-token window); at this "
                    f"rate the window fills in about {steps_to_full:.1f} more call(s)."
                ),
                step_ids=[n["id"] for n in llm_nodes[-len(recent):]],
                recommendation=(
                    "Summarise or window the context between calls, or cap the number of turns, "
                    "before the window overflows and every call pays for the full history."
                ),
            )
        ]

    def _detect_error_cascade(self, graph: ExecutionGraph) -> list[FailurePattern]:
        """ERROR_CASCADE: 3+ consecutive ERROR nodes."""
        nodes = graph.nodes()
        patterns: list[FailurePattern] = []
        run: list[str] = []

        for node in nodes:
            if node["type"] == StepType.ERROR:
                run.append(node["id"])
            else:
                if len(run) >= self.ERROR_CASCADE_THRESHOLD:
                    patterns.append(
                        FailurePattern(
                            name="ERROR_CASCADE",
                            severity="CRITICAL",
                            description=f"{len(run)} consecutive errors detected.",
                            step_ids=list(run),
                            recommendation=(
                                "Implement exponential back-off and circuit-breaker logic. "
                                "Surface errors to a human escalation path after 3 failures."
                            ),
                        )
                    )
                run = []

        # Trailing run
        if len(run) >= self.ERROR_CASCADE_THRESHOLD:
            patterns.append(
                FailurePattern(
                    name="ERROR_CASCADE",
                    severity="CRITICAL",
                    description=f"{len(run)} consecutive errors at end of trace.",
                    step_ids=list(run),
                    recommendation=(
                        "Implement exponential back-off and circuit-breaker logic. "
                        "Surface errors to a human escalation path after 3 failures."
                    ),
                )
            )

        return patterns

    def _detect_unrecovered_error(self, graph: ExecutionGraph) -> list[FailurePattern]:
        """UNRECOVERED_ERROR: an error occurred and no tool call succeeded afterwards.

        ERROR_CASCADE needs consecutive errors, but in a real agent every error is followed by an LLM call that
        decides what to do next, so errors are never adjacent. This catches the agent that gives up (or answers
        without the data it needed) after a failure instead of retrying or escalating.
        """
        pending: str | None = None
        for node in graph.nodes():
            if node["type"] == StepType.ERROR:
                pending = node["id"]
            elif node["type"] == StepType.TOOL_CALL:
                pending = None
        if pending is None:
            return []
        return [
            FailurePattern(
                name="UNRECOVERED_ERROR",
                severity="HIGH",
                description="The run ended after an error with no successful tool call afterwards.",
                step_ids=[pending],
                recommendation=(
                    "Retry the failed call with back-off, or escalate to a human, before giving up or answering "
                    "without the missing data."
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or that the this to was with you your".split()
)


def _cosine_similarity_bow(text_a: str, text_b: str) -> float:
    """Bag-of-words cosine similarity — zero-dependency approximation."""
    if not text_a or not text_b:
        return 0.0

    def tokenise(t: str) -> Counter[str]:
        return Counter(
            w.lower() for w in t.split() if w.isalpha() and w.lower() not in _STOPWORDS
        )

    vec_a = tokenise(text_a)
    vec_b = tokenise(text_b)
    vocab = set(vec_a) | set(vec_b)

    dot = sum(vec_a[w] * vec_b[w] for w in vocab)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --- Semantic similarity (sentence embeddings) ------------------------------
#
# Bag-of-words cosine false-positives on healthy runs: a step that restates the
# goal in different words ("quarterly revenue" -> "Q3 earnings figures") shares
# almost no tokens with the first prompt, so BoW scores it near zero and the run
# is wrongly flagged as GOAL_DRIFT. A sentence embedding compares meaning instead
# of word overlap, so paraphrases stay close and only a genuine topic change
# drops below threshold.
#
# fastembed runs the model through ONNX runtime on CPU (no torch), ~90 MB, and
# the model loads once per process. If fastembed or the model is unavailable the
# detector degrades to the bag-of-words cosine on its original threshold rather
# than crashing.

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_BOW_FALLBACK_THRESHOLD = 0.30  # the original BoW cosine threshold, kept for degraded mode

_embedder: object | None = None
_embedder_loaded = False
_embed_cache: dict[str, tuple[float, ...]] = {}


def _get_embedder() -> object | None:
    """Return a cached TextEmbedding, or None if embeddings are unavailable."""
    global _embedder, _embedder_loaded
    if not _embedder_loaded:
        _embedder_loaded = True
        try:
            from fastembed import TextEmbedding

            _embedder = TextEmbedding(model_name=_EMBED_MODEL_NAME)
        except Exception:
            _embedder = None  # degrade to bag-of-words
    return _embedder


def _embed_unit(text: str) -> tuple[float, ...] | None:
    """L2-normalised embedding of *text*, cached; None when no model is loaded."""
    model = _get_embedder()
    if model is None:
        return None
    if text not in _embed_cache:
        import numpy as np  # pulled in by fastembed; only imported when a model exists

        vec = next(iter(model.embed([text])))  # type: ignore[attr-defined]
        norm = float(np.linalg.norm(vec)) or 1.0
        _embed_cache[text] = tuple(float(x) / norm for x in vec)
    return _embed_cache[text]


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """Sentence-embedding cosine similarity, with a bag-of-words fallback."""
    if not text_a or not text_b:
        return 0.0
    vec_a = _embed_unit(text_a)
    vec_b = _embed_unit(text_b)
    if vec_a is None or vec_b is None:
        return _cosine_similarity_bow(text_a, text_b)
    return sum(a * b for a, b in zip(vec_a, vec_b))


def _goal_similarity_backend(embedding_threshold: float):
    """Pick the similarity function and matching threshold for goal-drift.

    Returns the embedding similarity on *embedding_threshold* when a model is
    loaded, otherwise the bag-of-words cosine on its original threshold.
    """
    if _get_embedder() is not None:
        return _semantic_similarity, embedding_threshold
    return _cosine_similarity_bow, _BOW_FALLBACK_THRESHOLD
