"""Run investigations and model reviews from JSON requests."""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Literal

sys.path.insert(0, "/app/src")
from first_investigation import InvestigationState, investigate, reflect, validate
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, StrictInt

TAG = "AGENT_RESULT_JSON:"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute(evidence: dict, *, procedural_memories: list[dict] | None = None) -> dict:
    events: list[dict] = []
    outputs: dict = {}

    def tracked(name, fn):
        def call(state):
            began = now()
            try:
                result = fn(state)
                if not isinstance(result, dict):
                    raise TypeError(f"Stage {name} returned non-dict output")
                outputs.update(result)
                events.append({
                    "stage": name, "started_at_utc": began,
                    "completed_at_utc": now(), "status": "COMPLETED",
                    "output_fields": list(result),
                })
                return result
            except Exception as exc:
                events.append({
                    "stage": name, "started_at_utc": began,
                    "completed_at_utc": now(), "status": "FAILED",
                    "error_type": type(exc).__name__,
                })
                raise
        return call

    graph = StateGraph(InvestigationState)
    graph.add_node("investigate", tracked("investigate", investigate))
    graph.add_node("reflect", tracked("reflect", reflect))
    graph.add_node("validate", tracked("validate", validate))
    graph.add_edge(START, "investigate")
    graph.add_edge("investigate", "reflect")
    graph.add_edge("reflect", "validate")
    graph.add_edge("validate", END)

    began = now()
    failure = None
    status = "COMPLETED"
    try:
        initial_state = {"evidence": evidence}
        if procedural_memories:
            initial_state["procedural_memories"] = procedural_memories
        result = graph.compile().invoke(initial_state)
        outputs.update({key: value for key, value in result.items() if key != "evidence"})
    except Exception as exc:
        status = "FAILED"
        # Do not echo detailed errors, which can contain model context or tokens.
        failure = {"type": type(exc).__name__, "message": "Agent stage failed; see trace error_type"}

    return {
        "started_at_utc": began,
        "completed_at_utc": now(),
        "execution_status": status,
        "trace": events,
        "report": outputs.get("report"),
        "reflection": outputs.get("reflection"),
        "validation": outputs.get("validation"),
        "failure": failure,
    }


class LearningDecision(BaseModel):
    """Recommendation only; the caller retains all governance decisions."""

    decision: Literal["NEW", "REINFORCE", "REFINE", "SUPERSEDE", "INVALIDATE", "NO_CHANGE"]
    memory_id: StrictInt | None
    rationale: str
    proposed_revised_lesson: str | None
    supporting_references: list[str]
    limitations: list[str]
    unresolved_contradictions: list[str]


def learning_decision(request: dict) -> dict:
    """Compare a proposed lesson with supplied memories and scientific provenance."""
    for field in ("investigation_report", "proposed_lesson"):
        value = request.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid {field}")
    memories = request.get("existing_memories")
    if not isinstance(memories, list):
        raise ValueError("Existing memories must be a list")
    ids = set()
    for memory in memories:
        if (not isinstance(memory, dict) or type(memory.get("memory_id")) is not int or
                memory["memory_id"] <= 0 or memory["memory_id"] in ids):
            raise ValueError("Invalid or duplicate existing memory ID")
        ids.add(memory["memory_id"])
    if "evidence" not in request or "provenance" not in request:
        raise ValueError("Scientific evidence and provenance are required")
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_MODEL"):
        raise RuntimeError("Main agent credentials/model are not configured")

    model = ChatOpenAI(model=os.environ["OPENAI_MODEL"])
    reviewer = model.with_structured_output(LearningDecision, method="function_calling")
    result = reviewer.invoke([
        ("system", "Recommend exactly one learning action based on semantic meaning and "
         "the supplied evidence. Never select NEW purely because similarity is low or "
         "REFINE purely because similarity is high. Existing procedural memories are "
         "review methods, not scientific evidence. Ground rationale and references in "
         "the supplied scientific evidence and provenance. Do not introduce scientific "
         "conclusions. State limitations and unresolved contradictions. This is a "
         "recommendation only; do not approve, promote, create or modify memory."),
        ("human", json.dumps({key: request[key] for key in (
            "investigation_report", "proposed_lesson", "existing_memories", "evidence", "provenance"
        )}, ensure_ascii=False)),
    ])
    # - Revalidate even when the model adapter returns a dictionary or model object.
    if isinstance(result, BaseModel):
        result = result.model_dump()
    decision = LearningDecision.model_validate(result)
    if not decision.rationale.strip():
        raise ValueError("Learning rationale must not be blank")
    if decision.memory_id is not None and decision.memory_id not in ids:
        raise ValueError("Recommendation references an unknown memory ID")
    if decision.decision in ("REINFORCE", "REFINE", "SUPERSEDE", "INVALIDATE") and decision.memory_id is None:
        raise ValueError("Recommendation requires an existing memory ID")
    if decision.decision in ("REFINE", "SUPERSEDE") and not (
        decision.proposed_revised_lesson and decision.proposed_revised_lesson.strip()
    ):
        raise ValueError("Recommendation requires a revised lesson")
    return decision.model_dump()


class ApplicabilityDecision(BaseModel):
    memory_id: StrictInt
    version_id: StrictInt
    decision: Literal["APPLY", "SKIP", "UNCERTAIN"]
    reason: str
    source_field_references: list[str]


class ApplicabilityMemoryVersion(BaseModel):
    memory_id: StrictInt
    version_id: StrictInt


class ApplicabilityConflict(BaseModel):
    memory_versions: list[ApplicabilityMemoryVersion]
    reason: str


class MemoryApplicabilityReview(BaseModel):
    decisions: list[ApplicabilityDecision]
    conflicts: list[ApplicabilityConflict]


def memory_applicability(request: dict) -> dict:
    """Assess supplied procedures against bounded caller-provided case context."""
    context = request.get("case_context")
    paths = request.get("source_field_paths")
    candidates = request.get("candidates")
    if not isinstance(context, str) or not context.strip() or len(context) > 4000:
        raise ValueError("Invalid or oversized case context")
    if (not isinstance(paths, list) or not paths or len(paths) > 50 or
            any(not isinstance(p, str) or not p.strip() or len(p) > 256 for p in paths) or
            len(set(paths)) != len(paths)):
        raise ValueError("Invalid case-context source field paths")
    if not isinstance(candidates, list) or len(candidates) > 3:
        raise ValueError("Expected at most three memory candidates")
    pairs = set()
    for candidate in candidates:
        if (not isinstance(candidate, dict) or
                set(candidate) != {"memory_id", "version_id", "title", "content"} or
                any(type(candidate[k]) is not int or candidate[k] <= 0
                    for k in ("memory_id", "version_id")) or
                any(not isinstance(candidate[k], str) or not candidate[k].strip() or
                    len(candidate[k]) > limit
                    for k, limit in (("title", 300), ("content", 7000)))):
            raise ValueError("Invalid procedural memory candidate")
        pair = (candidate["memory_id"], candidate["version_id"])
        if pair in pairs:
            raise ValueError("Duplicate procedural memory candidate")
        pairs.add(pair)
    if not candidates:
        return {"decisions": [], "conflicts": []}
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_MODEL"):
        raise RuntimeError("Main agent credentials/model are not configured")

    model = ChatOpenAI(model=os.environ["OPENAI_MODEL"])
    reviewer = model.with_structured_output(MemoryApplicabilityReview, method="function_calling")
    result = reviewer.invoke([
        ("system", "Assess each candidate's stated conditions for reuse against only the supplied "
         "case context. Return one APPLY, SKIP, or UNCERTAIN decision per supplied memory/version. "
         "UNCERTAIN never means automatic application. Use only exact supplied source_field_paths "
         "as references; cite applicable case-context fields for APPLY. Procedural memories are "
         "review methods, never scientific evidence. Do not invent case facts or clinical conclusions. "
         "Identify material conflicts between applicable procedures, rather than combining "
         "incompatible instructions. Explain decisions and conflicts concisely. This is a "
         "recommendation only, not memory injection or a governance decision."),
        ("human", json.dumps({"case_context": context, "source_field_paths": paths,
                              "candidates": candidates}, ensure_ascii=False)),
    ])
    if isinstance(result, BaseModel):
        result = result.model_dump()
    review = MemoryApplicabilityReview.model_validate(result)
    decisions = {}
    for item in review.decisions:
        pair = (item.memory_id, item.version_id)
        if pair not in pairs or pair in decisions:
            raise ValueError("Unknown or duplicate memory decision")
        if not item.reason.strip() or any(ref not in paths for ref in item.source_field_references):
            raise ValueError("Invalid decision reason or case-context reference")
        if item.decision == "APPLY" and not item.source_field_references:
            raise ValueError("APPLY requires a case-context source reference")
        decisions[pair] = item.decision
    if set(decisions) != pairs:
        raise ValueError("Missing memory decision")
    seen_conflicts = set()
    for conflict in review.conflicts:
        conflict_pairs = [(v.memory_id, v.version_id) for v in conflict.memory_versions]
        if (len(conflict_pairs) < 2 or len(set(conflict_pairs)) != len(conflict_pairs) or
                not conflict.reason.strip() or
                any(decisions.get(pair) != "APPLY" for pair in conflict_pairs)):
            raise ValueError("Invalid conflict between applicable memories")
        conflict_key = frozenset(conflict_pairs)
        if conflict_key in seen_conflicts:
            raise ValueError("Duplicate procedural conflict")
        seen_conflicts.add(conflict_key)
    return review.model_dump()


def main() -> int:
    # Input never comes from a local run file and is not printed/logged.
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    if not isinstance(request, dict):
        raise ValueError("Worker request must be an object")
    mode = request.get("mode")
    if mode in ("memory_applicability", "learning_decision"):
        review = memory_applicability if mode == "memory_applicability" else learning_decision
        sys.stdout.write(TAG + json.dumps(review(request), ensure_ascii=True) + "\n")
        return 0
    if mode is not None:
        raise ValueError("Unknown worker mode")
    if "procedural_memory" in request or "procedural_memory_support" in request:
        raise ValueError("Legacy memory inputs are retired; use procedural_memories")
    evidence = request["evidence"]
    required = {"interpretation.json", "guideline_v2.json"}
    if not isinstance(evidence, dict) or not required.issubset(evidence):
        raise ValueError("Missing required approved evidence documents")
    for name, document in evidence.items():
        if not isinstance(name, str) or not isinstance(document, dict):
            raise ValueError("Invalid approved evidence document")
        if name not in required and name != "criterion_rationale.json" and not (
            re.fullmatch(r"clinvar/SCV[0-9]+\.[0-9]+\.json", name) or
            re.fullmatch(r"publications/[0-9]+\.json", name)
        ):
            raise ValueError("Unexpected approved evidence path")
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_MODEL"):
        raise RuntimeError("Main agent credentials/model are not configured")
    memories = request.get("procedural_memories", [])
    if (not isinstance(memories, list) or len(memories) > 3 or
            any(not isinstance(item, dict) or
                any(not isinstance(item.get(key), str) or not item[key].strip()
                    for key in ("title", "content")) for item in memories)):
        raise ValueError("Invalid procedural memories")
    result = execute(evidence, procedural_memories=memories) if memories else execute(evidence)
    # ASCII JSON avoids Windows code-page/console encoding issues.
    sys.stdout.write(TAG + json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Worker startup failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2)
