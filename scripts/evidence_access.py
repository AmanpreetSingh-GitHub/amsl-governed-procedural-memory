"""Shared database transport and approved scientific evidence access."""

import base64
import hashlib
import json
import re
import subprocess

APPROVED = "approved_for_first_investigation"
DB_COMMAND = ["docker", "exec", "-i", "agent-memory-postgres", "psql", "-X", "-q",
              "-t", "-A", "-U", "agent_memory", "-d", "agent_memory_lab",
              "-v", "ON_ERROR_STOP=1"]


def need(test: bool, explanation: str) -> None:
    if not test:
        raise ValueError(explanation)


def sql_text(text: str | None) -> str:
    """ASCII-safe transport prevents SQL quoting or Windows UTF-8 corruption."""
    if text is None:
        return "NULL"
    need(isinstance(text, str), "Expected string")
    data = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"convert_from(decode('{data}', 'base64'), 'UTF8')"


def sql_json(value: object) -> str:
    return sql_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"))) + "::jsonb"


def db(sql: str) -> list[str]:
    result = subprocess.run(DB_COMMAND, input=sql, text=True, encoding="utf-8",
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "PostgreSQL query failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def rows(sql: str) -> list[dict]:
    return [json.loads(line) for line in db(sql)]


def scope(case: str, stage: int) -> None:
    need(bool(re.fullmatch(r"CA[0-9]+", case)), "Invalid case ID")
    need(stage >= 0, "Stage must not be negative")


class InvalidProvenance(ValueError):
    """Stored episode or release provenance cannot establish eligibility."""


def need_provenance(test: bool, explanation: str) -> None:
    if not test:
        raise InvalidProvenance(explanation)


def approved_release(case: str, stage: int) -> dict:
    """Load exactly the approved snapshot from DB, never a local run or sealed file."""
    records = rows(f"""
SELECT json_build_object(
 'release_id', er.release_id, 'experiment_id', er.experiment_id,
 'case_id', er.case_id, 'stage', er.stage,
 'manifest', er.manifest_json, 'receipt', er.receipt_json,
 'sources', COALESCE(json_agg(json_build_object(
   'source_id', s.source_id, 'source_type', s.source_type,
   'canonical_key', s.canonical_key, 'version', s.version_label,
   'sha256', s.content_sha256, 'declared_sha256', re.declared_sha256, 'snapshot_path', re.snapshot_path,
   'document', s.document_json) ORDER BY re.snapshot_path), '[]'::json)
)::text
FROM scientific.evidence_releases er
JOIN scientific.release_evidence re ON re.release_id = er.release_id
JOIN scientific.sources s ON s.source_id = re.source_id
WHERE er.case_id = {sql_text(case)} AND er.stage = {stage}
  AND er.release_status = {sql_text(APPROVED)}
  AND er.case_inventory_json->>'caid' = er.case_id
  AND re.declared_sha256 = s.content_sha256
GROUP BY er.release_id;
""")
    need_provenance(len(records) == 1, "Exactly one approved case/stage release is required")
    release = records[0]
    manifest = release["manifest"]
    need_provenance(manifest.get("experiment_id") == release["experiment_id"],
                "Manifest experiment mismatch")
    need_provenance(manifest.get("case_id") == case and manifest.get("stage") == stage,
                "Manifest case/stage mismatch")
    receipt = release["receipt"]
    need_provenance(receipt.get("status") == APPROVED,
                "Receipt has not approved this investigation")
    need_provenance(all(receipt.get(key) == manifest.get(key)
                        for key in ("experiment_id", "case_id", "stage", "evidence")),
                    "Approved receipt and manifest differ")
    evidence_items = manifest.get("evidence", [])
    need_provenance(isinstance(evidence_items, list) and len(evidence_items) == len(release["sources"]),
                "Approved source count does not match manifest")
    need_provenance(all(isinstance(item, dict) and
                        all(isinstance(item.get(field), str) for field in ("file", "sha256"))
                        for item in evidence_items), "Invalid approved manifest evidence")
    declared = {(x["file"], x["sha256"]) for x in evidence_items}
    actual = {(x["snapshot_path"], x["sha256"]) for x in release["sources"]}
    need_provenance(declared == actual and len(declared) == len(evidence_items),
                "Approved source paths/hashes do not match manifest")
    files = {}
    for src in release["sources"]:
        need_provenance(src["declared_sha256"] == src["sha256"],
                        "Release link hash does not match its scientific source")
        path = src["snapshot_path"].replace("\\", "/")
        need_provenance(path.startswith("evidence/"), "Unexpected scientific evidence path")
        filename = path.removeprefix("evidence/")
        if filename == "interpretation.json":
            expected_type = "INTERPRETATION"
        elif filename == "guideline_v2.json":
            expected_type = "GUIDELINE"
        elif filename == "criterion_rationale.json":
            expected_type = "CRITERION_RATIONALE"
        elif re.fullmatch(r"clinvar/SCV[0-9]+\.[0-9]+\.json", filename):
            expected_type = "CLINVAR"
        elif re.fullmatch(r"publications/[0-9]+\.json", filename):
            expected_type = "PUBLICATION"
        else:
            raise InvalidProvenance("Unexpected scientific evidence path")
        need_provenance(filename not in files and isinstance(src["document"], dict),
                    "Duplicate evidence path or invalid source JSON")
        need_provenance(src["source_type"] == expected_type, "Unexpected scientific evidence type")
        files[filename] = src["document"]
    need_provenance({"interpretation.json", "guideline_v2.json"}.issubset(files),
                "Missing required scientific evidence files")
    release["evidence"] = files
    return release


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def revalidate_release(release: dict) -> dict:
    need(isinstance(release, dict) and isinstance(release.get("case_id"), str) and
         type(release.get("stage")) is int, "Invalid supplied release")
    scope(release["case_id"], release["stage"])
    current = approved_release(release["case_id"], release["stage"])
    need(canonical_bytes(current) == canonical_bytes(release),
         "Supplied release does not match the approved case release")
    return current


def check_standalone(release: dict) -> None:
    need(release["manifest"].get("experiment_protocol") != "PAIRED_EXPERIMENT_V1",
         "Paired experiment releases cannot be run as standalone investigations")


def load_episode(episode_id: int) -> dict:
    need(type(episode_id) is int and episode_id > 0, "Invalid episode ID")
    found = rows(f"""
SELECT json_build_object('episode_id',e.episode_id,'run_id',e.run_id,'case_id',e.case_id,
 'stage',e.stage,'experiment_id',e.experiment_id,'release_id',e.release_id,'status',e.status,
 'record_sha256',e.record_sha256,'record',e.run_record)::text
FROM agent_memory.episodes e WHERE e.episode_id={episode_id};
""")
    need_provenance(len(found) == 1, "Memory version source episode not found or ambiguous")
    need_provenance(found[0]["episode_id"] == episode_id, "Memory version source episode ID mismatch")
    return found[0]


SOURCE_FIELDS = ("source_id", "canonical_key", "version", "sha256")


def source_provenance(release: dict) -> list[dict]:
    return [{key: source[key] for key in SOURCE_FIELDS} for source in release["sources"]]


def verified_episode(ep: dict, record: dict, *, learning: bool = False, allow_abstain: bool = False) -> dict:
    need_provenance(isinstance(record, dict) and hashlib.sha256(canonical_bytes(record)).hexdigest() ==
                    ep["record_sha256"], "Episode record SHA-256 mismatch")
    need_provenance(isinstance(ep["case_id"], str) and re.fullmatch(r"CA[0-9]+", ep["case_id"]) is not None and
                    type(ep["stage"]) is int and ep["stage"] >= 0, "Invalid episode case or stage")
    need_provenance(all(record.get(key) == ep[key] for key in ("case_id", "stage", "experiment_id")) and
                    record.get("execution_status") == ep["status"] and
                    record.get("run_id") == str(ep.get("run_id") or ""),
                    "Episode identity mismatch")
    if learning:
        validation = record.get("validation")
        decision = validation.get("decision") if isinstance(validation, dict) else None
        reflection = record.get("reflection")
        abstained = (allow_abstain and isinstance(reflection, str) and
                     reflection.partition("\n")[0].strip() == "DECISION: ABSTAIN")
        need_provenance(ep["status"] == "COMPLETED" and
                        (decision == "APPROVE" or (abstained and decision == "DEFER")),
                        "Source episode did not complete with an approved investigation")
    release = approved_release(ep["case_id"], ep["stage"])
    need_provenance(release["release_id"] == ep["release_id"] and release["experiment_id"] == ep["experiment_id"] and
                    record.get("evidence") == release["manifest"].get("evidence"),
                    "Episode approved release or manifest mismatch")
    expected = source_provenance(release)
    consulted = record.get("sources_consulted")
    need_provenance(isinstance(consulted, list) and len(consulted) == len(expected) and
                    all(item in consulted for item in expected), "Episode source provenance mismatch")
    return release
