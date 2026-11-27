"""Rank approved procedural guidance and enforce applicability selection."""

import hashlib
import json
import re
import sys

from evidence_access import (InvalidProvenance, load_episode, need, need_provenance, revalidate_release,
                             rows, sql_text, verified_episode)
from scientific_semantic_retrieval import MODEL as EMBEDDING_MODEL, embed, sql_vector

INVESTIGATION_QUESTION = "Investigate the approved scientific evidence; identify criteria, limitations and uncertainties"
POLICY = "CONTEXT_AWARE_PROCEDURAL_RETRIEVAL_V2"
IDENTITY = ("memory_id", "version_id")
DECISION_FIELDS = (*IDENTITY, "decision", "reason", "source_field_references")


def memory_pair(item: dict) -> tuple[int, int]:
    need(isinstance(item, dict) and all(type(item.get(key)) is int and item[key] > 0 for key in IDENTITY),
         "Invalid memory/version identity")
    return item["memory_id"], item["version_id"]


def retrieval_context(release: dict) -> dict:
    evidence = release["evidence"]
    case = release["case_id"]
    need(isinstance(evidence, dict), "Approved evidence must be an object")
    interpretation = evidence.get("interpretation.json")
    need(isinstance(interpretation, dict) and interpretation.get("caid") == case,
         "Approved interpretation case mismatch")

    parts = [INVESTIGATION_QUESTION]
    used_fields: list[str] = []

    def add(path: str, value: object, max_chars: int) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        text = " ".join(value.split())
        remaining = 2800 - sum(len(part) + 1 for part in parts) - len(path) - 2
        max_chars = min(max_chars, remaining)
        if max_chars < 20:
            return
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0]
            if not text:
                return
            text += " [...]"
        parts.append(path + ": " + text)
        used_fields.append(path)

    add("interpretation.json#/caid", interpretation.get("caid"), 40)
    add("interpretation.json#/classification", interpretation.get("classification"), 100)
    criteria = interpretation.get("criteria")
    if isinstance(criteria, list):
        for index, item in enumerate(criteria[:12]):
            if isinstance(item, dict):
                add(f"interpretation.json#/criteria/{index}/label", item.get("label"), 80)
                add(f"interpretation.json#/criteria/{index}/outcome", item.get("outcome"), 80)
    add("interpretation.json#/narrative", interpretation.get("narrative"), 650)
    uncertainties = interpretation.get("known_uncertainties")
    if isinstance(uncertainties, list):
        for index, value in enumerate(uncertainties[:3]):
            add(f"interpretation.json#/known_uncertainties/{index}", value, 180)

    rationale = evidence.get("criterion_rationale.json")
    if isinstance(rationale, dict) and isinstance(rationale.get("excerpts"), list):
        for index, item in enumerate(rationale["excerpts"][:2]):
            if isinstance(item, dict):
                add(f"criterion_rationale.json#/excerpts/{index}/criterion", item.get("criterion"), 80)
                add(f"criterion_rationale.json#/excerpts/{index}/source_text", item.get("source_text"), 700)

    clinvar = sorted(k for k in evidence if isinstance(k, str) and
                     re.fullmatch(r"clinvar/SCV[0-9]+\.[0-9]+\.json", k))
    for name in clinvar[:2]:
        document = evidence[name]
        if isinstance(document, dict):
            add(f"{name}#/classification", document.get("classification"), 120)
            add(f"{name}#/note", document.get("note"), 180)

    publications = sorted(k for k in evidence if isinstance(k, str) and
                          re.fullmatch(r"publications/[0-9]+\.json", k))
    for name in publications[:2]:
        document = evidence[name]
        if not isinstance(document, dict):
            continue
        add(f"{name}#/title", document.get("title"), 180)
        sections = document.get("abstract_sections")
        if isinstance(sections, list):
            for index, section in enumerate(sections[:2]):
                value = section.get("text") if isinstance(section, dict) else section
                suffix = "/text" if isinstance(section, dict) else ""
                add(f"{name}#/abstract_sections/{index}{suffix}", value, 250)

    query = "\n".join(parts)
    used_names = {field.split("#", 1)[0] for field in used_fields}
    sources = [{"file": item["file"], "sha256": item["sha256"]} for item in release["manifest"]["evidence"]
               if item["file"].removeprefix("evidence/") in used_names]
    return {
        "policy": POLICY, "query": query, "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "query_source": "fixed_investigation_question_and_approved_case_evidence",
        "query_fields": used_fields, "sources": sources, "case_id": case, "stage": release["stage"],
        "release_id": release["release_id"], "ranking_only": True, "selected_for_injection": False,
    }


def procedural_version_origin(memory: dict, case: str) -> dict | None:
    """Resolve a version's approved origin without borrowing a parent version's evidence."""
    reason = memory["change_reason"]
    need_provenance(isinstance(reason, str) and bool(reason.strip()), "Missing memory version provenance")
    episode_id = memory["original_episode_id"]
    expected_run = expected_sha = None
    if reason.startswith("DRAFT_LEARNING"):
        draft = re.fullmatch(r"DRAFT_LEARNING:([1-9][0-9]*):(NEW|REFINE):(.+)", reason, re.DOTALL)
        need_provenance(draft is not None, "Invalid memory version provenance")
        origin, decision, encoded = draft.groups()
        try:
            metadata = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise InvalidProvenance(f"Invalid memory version provenance JSON: {exc}") from exc
        need_provenance(isinstance(metadata, dict) and type(metadata.get("episode_id")) is int and
             metadata["episode_id"] == int(origin) and metadata.get("decision") == decision and
             isinstance(metadata.get("run_id"), str) and bool(metadata["run_id"]) and
             isinstance(metadata.get("record_sha256"), str) and
             re.fullmatch(r"[0-9a-f]{64}", metadata["record_sha256"]) is not None,
             "Conflicting or incomplete memory version provenance")
        if decision == "REFINE":
            base_version = metadata.get("active_version_id")
            need_provenance(memory["version_number"] > 1 and type(base_version) is int and base_version > 0 and
                 base_version != memory["version_id"] and type(metadata.get("target_memory_id")) is int and
                 metadata["target_memory_id"] == memory["memory_id"],
                 "REFINE provenance targets a different memory or invalid version")
        else:
            need_provenance(memory["version_number"] == 1 and metadata["episode_id"] == episode_id and
                 "target_memory_id" in metadata and metadata["target_memory_id"] is None,
                 "NEW provenance conflicts with the original episode or version")
        episode_id = metadata["episode_id"]
        expected_run, expected_sha = metadata["run_id"], metadata["record_sha256"]
    else:
        # The memory-level FK identifies the original version; a manual revision has no such guarantee.
        need_provenance(memory["version_number"] == 1, "Revised memory version lacks its own source episode")
        prefix = "Proposed in origin episode "
        if reason.startswith(prefix):
            expected_run = reason.removeprefix(prefix)
    need_provenance(type(episode_id) is int and episode_id > 0, "Missing memory version source episode")

    ep = load_episode(episode_id)
    if ep["case_id"] == case:
        return None
    release = verified_episode(ep, ep["record"], learning=True)
    need_provenance((expected_run is None or expected_run == ep["run_id"]) and
                    (expected_sha is None or expected_sha == ep["record_sha256"]),
                    "Memory version source episode identity or recorded origin hash mismatch")
    return {
        "source_episode_id": episode_id, "source_case_id": ep["case_id"], "source_stage": ep["stage"],
        "source_release_id": ep["release_id"], "source_record_sha256": ep["record_sha256"],
        "origin_evidence": [{"canonical_key": source["canonical_key"], "sha256": source["sha256"]}
                            for source in release["sources"]],
    }

def retrieve_procedural_memories(case: str, release: dict, limit: int = 3) -> list[dict]:
    need(type(limit) is int and 1 <= limit <= 3, "Memory limit must be between 1 and 3")
    verified = revalidate_release(release)
    need(case == verified["case_id"], "Supplied release case mismatch")
    context = retrieval_context(verified)
    query = context["query"]
    vector = sql_vector(embed(query))
    candidates = rows(f"""
SELECT json_build_object('memory_id',m.memory_id,'version_id',v.version_id,
 'version_number',v.version_number,'title',v.title,'content',v.content,
 'original_episode_id',m.source_episode_id,'change_reason',v.change_reason,'embedding_model',em.embedding_model,
 'similarity',1-(em.embedding <=> {vector}))::text
FROM agent_memory.memories m JOIN agent_memory.memory_versions v
 ON v.memory_id=m.memory_id AND v.version_id=m.active_version_id
JOIN agent_memory.embeddings em ON em.version_id=v.version_id
WHERE m.status='VALIDATED' AND m.memory_type='PROCEDURE'
 AND em.embedding_model={sql_text(EMBEDDING_MODEL)} AND em.embedding IS NOT NULL
ORDER BY em.embedding <=> {vector},m.memory_id,v.version_id;
""")
    eligible = []
    for candidate in candidates:
        need(type(candidate.get("version_id")) is int and candidate["version_id"] > 0 and
             type(candidate.get("memory_id")) is int and candidate["memory_id"] > 0 and
             type(candidate.get("version_number")) is int and candidate["version_number"] > 0 and
             all(isinstance(candidate.get(k), str) and candidate[k].strip() for k in ("title", "content")),
             "Invalid procedural memory candidate")
        try:
            origin = procedural_version_origin(candidate, case)
        except InvalidProvenance as exc:
            print(f"WARNING: Excluded procedural memory | memory_id={candidate['memory_id']} | "
                  f"version_id={candidate['version_id']} | {exc}", file=sys.stderr)
            continue
        if origin is not None:
            candidate.update(origin)
            del candidate["original_episode_id"], candidate["change_reason"]
            eligible.append(candidate)
    ranked = sorted(eligible, key=lambda item: (-item["similarity"], *memory_pair(item)))[:limit]
    for rank, candidate in enumerate(ranked, 1):
        candidate["score"] = candidate["similarity"]
        candidate["content_sha256"] = hashlib.sha256(candidate["content"].encode("utf-8")).hexdigest()
        candidate["retrieval_provenance"] = dict(context, rank=rank)
    return ranked


def validate_candidates(candidates: list, context: dict) -> set[tuple[int, int]]:
    need(isinstance(candidates, list) and len(candidates) <= 3, "Invalid ranked candidates")
    pairs = set()
    for rank, candidate in enumerate(candidates, 1):
        pair = memory_pair(candidate)
        need(pair not in pairs and
             type(candidate.get("version_number")) is int and candidate["version_number"] > 0 and
             all(isinstance(candidate.get(k), str) and candidate[k].strip() and len(candidate[k]) <= limit
                 for k, limit in (("title", 300), ("content", 7000))), "Invalid or duplicate memory candidate")
        need(candidate.get("content_sha256") == hashlib.sha256(candidate["content"].encode("utf-8")).hexdigest(),
             "Memory content hash mismatch")
        need(candidate.get("retrieval_provenance") == dict(context, rank=rank), "Invalid retrieval provenance")
        pairs.add(pair)
    return pairs


def validate_applicability(review: dict, pairs: set[tuple[int, int]], paths: list[str]) -> dict:
    need(isinstance(review, dict) and set(review) == {"decisions", "conflicts"} and
         all(isinstance(review[key], list) for key in review), "Invalid applicability review")
    decisions = {}
    for item in review["decisions"]:
        pair = memory_pair(item)
        need(set(item) == set(DECISION_FIELDS) and pair in pairs and pair not in decisions,
             "Unknown, duplicate or invalid applicability decision")
        refs = item["source_field_references"]
        need(item["decision"] in ("APPLY", "SKIP", "UNCERTAIN") and
             isinstance(item["reason"], str) and item["reason"].strip() and
             isinstance(refs, list) and all(ref in paths for ref in refs) and
             (item["decision"] != "APPLY" or bool(refs)), "Invalid applicability decision or source reference")
        decisions[pair] = item
    need(set(decisions) == pairs, "Missing applicability decision")
    seen = set()
    for conflict in review["conflicts"]:
        need(isinstance(conflict, dict) and set(conflict) == {"memory_versions", "reason"} and
             isinstance(conflict["reason"], str) and conflict["reason"].strip() and
             isinstance(conflict["memory_versions"], list), "Invalid applicability conflict")
        versions = [memory_pair(item) for item in conflict["memory_versions"]]
        key = frozenset(versions)
        need(len(key) == len(versions) and len(key) >= 2 and key.issubset(pairs) and key not in seen,
             "Invalid applicability conflict versions")
        seen.add(key)
    return decisions


def select_applicable_memories(release: dict, candidates: list, review: dict) -> dict:
    verified = revalidate_release(release)
    context = retrieval_context(verified) if candidates else {}
    pairs = validate_candidates(candidates, context)
    decisions = validate_applicability(review, pairs, context.get("query_fields", []))
    selected = [candidate for candidate in candidates if decisions[memory_pair(candidate)]["decision"] == "APPLY"]
    applying = {memory_pair(item) for item in selected}
    need(all(not applying.intersection(memory_pair(v) for v in conflict["memory_versions"])
             for conflict in review["conflicts"]), "Applicable procedural memories have a reported conflict")
    if selected:
        pairs_sql = " OR ".join(f"(m.memory_id={m['memory_id']} AND v.version_id={m['version_id']})" for m in selected)
        current = rows(f"""
SELECT json_build_object('memory_id',m.memory_id,'version_id',v.version_id,
 'version_number',v.version_number,'title',v.title,'content',v.content,
 'original_episode_id',m.source_episode_id,'change_reason',v.change_reason)::text
FROM agent_memory.memories m JOIN agent_memory.memory_versions v
 ON v.memory_id=m.memory_id AND v.version_id=m.active_version_id
WHERE m.status='VALIDATED' AND m.memory_type='PROCEDURE' AND ({pairs_sql});
""")
        by_pair = {memory_pair(item): item for item in current}
        need(len(current) == len(selected) and set(by_pair) == applying, "Selected memories are stale or ineligible")
        for candidate in selected:
            item = by_pair[memory_pair(candidate)]
            need(all(item[k] == candidate[k] for k in ("version_number", "title", "content")),
                 "Selected memory changed since retrieval")
            origin = procedural_version_origin(item, verified["case_id"])
            need(origin is not None and all(candidate.get(k) == v for k, v in origin.items()),
                 "Selected memory version source changed or is ineligible")
    entries = []
    for candidate in candidates:
        pair = memory_pair(candidate)
        decision = decisions[pair]
        entries.append({**decision, "version_number": candidate["version_number"],
                        "content_sha256": candidate["content_sha256"], "selected": pair in applying,
                        "exclusion_reason": None if pair in applying else decision["decision"]})
    return {"selected_memories": selected, "selection_report": {
        "policy": "APPLICABILITY_SELECTION_V1", "case_id": verified["case_id"],
        "stage": verified["stage"], "release_id": verified["release_id"],
        "decisions": entries, "conflicts": review["conflicts"], "ranked_candidates": candidates,
    }}
