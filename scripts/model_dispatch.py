"""Dispatch authorized model requests and assemble complete run records."""

import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from episode_persistence import TRACE_LEVEL
from evidence_access import check_standalone, need, revalidate_release, source_provenance
from procedural_retrieval import (DECISION_FIELDS, INVESTIGATION_QUESTION, retrieval_context,
                                 select_applicable_memories, validate_applicability, validate_candidates)

SCRIPT_DIR = Path(__file__).resolve().parent
TAG = "AGENT_RESULT_JSON:"


def authorize_model(model: str, allow_external_model: bool) -> str:
    need(isinstance(model, str) and bool(re.fullmatch(r"[a-zA-Z0-9_.:-]+", model)), "Invalid or missing model name")
    url = os.environ.get("OPENAI_BASE_URL", "").strip()
    endpoint = urlsplit(url)
    if endpoint.scheme == "http" and (endpoint.hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}:
        return "configured_local"
    need(allow_external_model is True, "External agent endpoint requires explicit --allow-external-model")
    return "configured_external" if url else "configured_external_openai"


def call_worker(request: dict, model: str, *, allow_external_model: bool) -> dict:
    authorize_model(model, allow_external_model)
    worker, src = SCRIPT_DIR / "agent_langgraph_worker.py", SCRIPT_DIR.parent / "src"
    need(worker.is_file() and src.is_dir(), "Docker worker or original src directory is missing")
    need(bool(os.environ.get("OPENAI_API_KEY")), "OPENAI_API_KEY is not configured")
    command = [
        "docker", "run", "--rm", "-i", "--name", f"agent-memory-run-{uuid4().hex[:12]}",
        "--mount", f"type=bind,source={src},target=/app/src,readonly",
        "--mount", f"type=bind,source={worker},target=/app/worker.py,readonly",
        "--entrypoint", "python", "-e", "OPENAI_API_KEY", "-e", f"OPENAI_MODEL={model}",
    ]
    if os.environ.get("OPENAI_BASE_URL"):
        command.extend(["-e", "OPENAI_BASE_URL"])
    command.extend(["agent-memory-lab:dev", "/app/worker.py"])
    try:
        result = subprocess.run(command, input=json.dumps(request, ensure_ascii=False), text=True,
                                encoding="utf-8", capture_output=True, check=False)
    except OSError:
        raise RuntimeError("Could not start Docker agent worker") from None
    if result.returncode:
        raise RuntimeError(f"Docker agent worker failed (exit code {result.returncode})")
    messages = [line[len(TAG):] for line in result.stdout.splitlines() if line.startswith(TAG)]
    need(len(messages) == 1, "Expected exactly one structured Docker agent result")
    try:
        reply = json.loads(messages[0])
    except ValueError:
        raise ValueError("Invalid worker result JSON") from None
    need(isinstance(reply, dict), "Worker result must be an object")
    return reply


def validate_investigation_result(result: dict) -> None:
    need(result.get("execution_status") in ("COMPLETED", "FAILED"), "Invalid worker completion status")
    need(all(isinstance(result.get(key), str) and result[key].strip()
             for key in ("started_at_utc", "completed_at_utc")), "Missing worker timestamps")
    trace = result.get("trace")
    need(isinstance(trace, list) and all(isinstance(event, dict) for event in trace), "Missing LangGraph stage trace")
    if result["execution_status"] == "COMPLETED":
        need([(event.get("stage"), event.get("status")) for event in trace] ==
             [(name, "COMPLETED") for name in ("investigate", "reflect", "validate")] and
             all(isinstance(result.get(key), str) and result[key].strip() for key in ("report", "reflection")) and
             isinstance(result.get("validation"), dict) and
             result["validation"].get("decision") in ("APPROVE", "REJECT", "DEFER"),
             "Incomplete successful investigation result")


def run_agent(release: dict, model: str, endpoint: str | None = None,
              procedural_memories: list | None = None, selection_report: dict | None = None,
              *, allow_external_model: bool = False) -> dict:
    """Run the approved snapshot with optional, freshly checked guidance."""
    actual_endpoint = authorize_model(model, allow_external_model)
    need(endpoint is None or endpoint == actual_endpoint, "Agent endpoint does not match the configured endpoint")
    endpoint = actual_endpoint
    release = revalidate_release(release)
    check_standalone(release)
    # Re-run selection immediately before transfer using the full ranked input.
    multi_requested = procedural_memories is not None or selection_report is not None
    if multi_requested:
        need(isinstance(procedural_memories, list) and len(procedural_memories) <= 3 and
             isinstance(selection_report, dict) and isinstance(selection_report.get("ranked_candidates"), list) and
             isinstance(selection_report.get("decisions"), list) and isinstance(selection_report.get("conflicts"), list),
             "Multi-memory selection and report are required")
        decision_fields = DECISION_FIELDS
        need(all(isinstance(item, dict) and all(key in item for key in decision_fields)
                 for item in selection_report["decisions"]), "Invalid selection decisions")
        review = {"decisions": [{key: item[key] for key in decision_fields}
                                for item in selection_report["decisions"]],
                  "conflicts": selection_report["conflicts"]}
        checked = select_applicable_memories(release, selection_report["ranked_candidates"], review)
        need(checked["selection_report"] == selection_report and
             checked["selected_memories"] == procedural_memories,
             "Selection report or selected memories changed")
    payload = {"evidence": release["evidence"]}
    if multi_requested and procedural_memories:
        payload["procedural_memories"] = [
            {"title": item["title"], "content": item["content"]} for item in procedural_memories
        ]
    worker_result = call_worker(payload, model, allow_external_model=allow_external_model)
    validate_investigation_result(worker_result)

    finished = worker_result["completed_at_utc"]
    interpretation = release["evidence"]["interpretation.json"]
    record = {
        "run_id": str(uuid4()), "timestamp_utc": finished,
        "started_at_utc": worker_result["started_at_utc"],
        "completed_at_utc": finished,
        "experiment_id": release["experiment_id"], "case_id": release["case_id"],
        "stage": release["stage"],
        "question": INVESTIGATION_QUESTION,
        "evidence": release["manifest"]["evidence"],
        "sources_consulted": source_provenance(release),
        "known_uncertainties": interpretation.get("known_uncertainties", []),
        "report": worker_result.get("report"),
        "reflection": worker_result.get("reflection"),
        "validation": worker_result.get("validation"),
        "validation_status": "model_review_only" if worker_result.get("validation") else "not_recorded",
        "memory_status": "not_persisted",
        "procedural_memory_injection_checked": True,
        "trace": worker_result["trace"], "trace_level": TRACE_LEVEL,
        "model": model, "model_endpoint": endpoint,
        "execution_status": worker_result["execution_status"],
        "failure": worker_result.get("failure"),
    }
    if multi_requested:
        ranked = [{key: value for key, value in item.items() if key != "content"}
                  for item in selection_report["ranked_candidates"]]
        supplied = [{"memory_id": item["memory_id"], "version_id": item["version_id"],
                     "version_number": item["version_number"], "content_sha256": item["content_sha256"],
                     "title": item["title"], "content": item["content"]}
                    for item in procedural_memories]
        record["procedural_multi_memory_selection"] = {
            "policy": selection_report["policy"], "case_id": selection_report["case_id"],
            "stage": selection_report["stage"], "release_id": selection_report["release_id"],
            "ranked_candidates": ranked, "decisions": selection_report["decisions"],
            "conflicts": selection_report["conflicts"],
            "selected_memories": [{key: item[key] for key in
                                   ("memory_id", "version_id", "version_number", "content_sha256")}
                                  for item in procedural_memories],
            "supplied_memories": supplied, "guidance_supplied_to_worker": bool(supplied),
            "application_status": "not_verified",
        }
        if supplied:
            # Existing governance treats this marker as proof of injection, not independent support.
            record["procedural_memory_supplied"] = {
                "memory_id": supplied[0]["memory_id"], "version_id": supplied[0]["version_id"],
                "version_number": supplied[0]["version_number"], "content_sha256": supplied[0]["content_sha256"],
                "guidance_supplied_to_worker": True, "multiple_memories": True,
            }
    return record


def call_memory_applicability(release: dict, candidates: list, model: str, *,
                              allow_external_model: bool = False) -> dict:
    verified = revalidate_release(release)
    context = retrieval_context(verified) if candidates else {}
    pairs = validate_candidates(candidates, context)
    if not candidates:
        return {"decisions": [], "conflicts": []}
    paths = context["query_fields"]
    request = {
        "mode": "memory_applicability", "case_context": context["query"], "source_field_paths": paths,
        "candidates": [{k: item[k] for k in ("memory_id", "version_id", "title", "content")} for item in candidates],
    }
    review = call_worker(request, model, allow_external_model=allow_external_model)
    validate_applicability(review, pairs, paths)
    return review


def call_learning_decision(request: dict, model: str, *, allow_external_model: bool = False) -> dict:
    result = call_worker({**request, "mode": "learning_decision"}, model,
                         allow_external_model=allow_external_model)
    decision = result.get("decision")
    need(decision in ("NEW", "REFINE", "REINFORCE", "SUPERSEDE", "INVALIDATE", "NO_CHANGE"),
         "Invalid learning decision")
    target = result.get("memory_id")
    memories = {item["memory_id"] for item in request["existing_memories"]}
    need((target is None or (type(target) is int and target in memories)) and
         (decision != "NEW" or target is None) and
         (decision not in ("REFINE", "REINFORCE", "SUPERSEDE", "INVALIDATE") or target is not None),
         "Invalid learning decision target")
    revised = result.get("proposed_revised_lesson")
    need(revised is None or isinstance(revised, str), "Invalid proposed lesson")
    if decision in ("REFINE", "SUPERSEDE"):
        need(isinstance(revised, str) and revised.strip(), "Recommendation requires a revised lesson")
    need(isinstance(result.get("rationale"), str) and result["rationale"].strip(), "Missing learning rationale")
    for name in ("supporting_references", "limitations", "unresolved_contradictions"):
        need(isinstance(result.get(name), list) and all(isinstance(item, str) for item in result[name]),
             f"Invalid learning decision {name}")
    return result
