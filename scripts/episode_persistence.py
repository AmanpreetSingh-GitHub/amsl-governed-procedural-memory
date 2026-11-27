"""Persist, index and replay investigation episodes."""

import hashlib
import json
import re
from uuid import UUID

from evidence_access import APPROVED, canonical_bytes, db, need, rows, sql_json, sql_text, verified_episode
from scientific_semantic_retrieval import MODEL as EMBEDDING_MODEL, embed, sql_vector

TRACE_LEVEL = "STAGE_OUTPUTS_ONLY"

def searchable(record: dict) -> str:
    review = record.get("validation") or {}
    text = "\n".join([
        f"Agent investigation of case {record['case_id']} stage {record['stage']}.",
        "Investigation: " + str(record.get("report") or "Not recorded")[:1200],
        "Reflection: " + str(record.get("reflection") or "Not recorded")[:1200],
        "Validation: " + str(review.get("decision") or "Not recorded"),
        "Uncertainties: " + json.dumps(record["known_uncertainties"], ensure_ascii=False)[:500],
    ])
    return text


def save_episode(record: dict) -> int:
    """Save run atomically, with release-scope checked again at write time."""
    digest = hashlib.sha256(canonical_bytes(record)).hexdigest()
    case, stage = record["case_id"], int(record["stage"])
    rid = str(UUID(record["run_id"]))
    start = record["started_at_utc"]
    end = record["completed_at_utc"]
    status = record["execution_status"]
    title = f"Investigation {case} / {rid}"
    summary = (f"Recorded local LangGraph run, status {status}; "
               "model review only, not independently validated scientific truth")
    approved = f"""
 SELECT release_id FROM scientific.evidence_releases
 WHERE experiment_id = {sql_text(record['experiment_id'])}
   AND case_id = {sql_text(case)} AND stage = {stage}
   AND release_status = {sql_text(APPROVED)}
   AND receipt_json->>'status' = {sql_text(APPROVED)}
   AND manifest_json->'evidence' = {sql_json(record['evidence'])}
"""
    sql = f"""
BEGIN;
WITH approved AS ({approved} FOR SHARE), inserted AS (
 INSERT INTO agent_memory.episodes (
  title, objective, status, summary, started_at, completed_at, run_id,
  experiment_id, case_id, stage, release_id, record_sha256,
  run_record, trace_level, embedding_model, episode_embedding
 ) SELECT
  {sql_text(title)},
  {sql_text('Investigate, reflect and validate approved evidence')},
  {sql_text(status)}, {sql_text(summary)},
  {sql_text(start)}::timestamptz, {sql_text(end)}::timestamptz,
  {sql_text(rid)}::uuid,
  {sql_text(record['experiment_id'])}, {sql_text(case)}, {stage},
  approved.release_id, {sql_text(digest)}, {sql_json(record)},
  {sql_text(TRACE_LEVEL)}, NULL, NULL
 FROM approved WHERE TRUE
 ON CONFLICT (run_id) DO NOTHING
 RETURNING episode_id
)
SELECT episode_id FROM inserted;
COMMIT;
"""
    reconcile = f"""
SELECT CASE WHEN e.release_id = approved.release_id
             AND e.record_sha256 = {sql_text(digest)}
             AND e.run_record = {sql_json(record)}
             AND e.trace_level = {sql_text(TRACE_LEVEL)}
             AND e.case_id = {sql_text(case)} AND e.stage = {stage}
             AND e.experiment_id = {sql_text(record['experiment_id'])}
             AND e.status = {sql_text(status)}
            THEN e.episode_id::text ELSE 'conflict' END
FROM agent_memory.episodes e
LEFT JOIN ({approved}) approved ON approved.release_id = e.release_id
WHERE e.run_id = {sql_text(rid)}::uuid;
"""
    for attempt in range(2):
        # A lost acknowledgement can follow a commit. Check this run ID before resending.
        try:
            result = db(sql)
        except (RuntimeError, OSError, UnicodeError):
            result = None
        if result and len(result) == 1 and re.fullmatch(r"[1-9][0-9]*", result[0]):
            return int(result[0])
        existing = db(reconcile)
        if existing:
            need(len(existing) == 1 and re.fullmatch(r"[1-9][0-9]*", existing[0]),
                 "Run ID already exists with conflicting episode data or an unapproved release")
            return int(existing[0])
        if result == [] or attempt == 1:
            break
    raise ValueError(f"Episode save could not be confirmed for run_id={rid}; check the release and database")


def retry_episode_embedding(run_id: str) -> tuple[int, bool]:
    """Index one persisted episode by run ID without running an investigation."""
    rid = str(UUID(run_id))
    found = rows(f"""
SELECT json_build_object(
 'episode_id', e.episode_id, 'run_id', e.run_id, 'case_id', e.case_id, 'stage', e.stage,
 'experiment_id', e.experiment_id, 'release_id', e.release_id, 'status', e.status,
 'trace_level', e.trace_level, 'record_sha256', e.record_sha256, 'run_record', e.run_record,
 'embedding_model', e.embedding_model, 'has_embedding', e.episode_embedding IS NOT NULL)::text
FROM agent_memory.episodes e WHERE e.run_id = {sql_text(rid)}::uuid;
""")
    need(len(found) == 1, "No episode found for that run ID")
    episode = found[0]
    record = episode["run_record"]
    release = verified_episode(episode, record)
    digest = episode["record_sha256"]
    case, stage = episode["case_id"], episode["stage"]
    need(record.get("trace_level") == episode["trace_level"] == TRACE_LEVEL,
         "Persisted episode trace level mismatch")
    need(episode["embedding_model"] == (EMBEDDING_MODEL if episode["has_embedding"] else None),
         "Existing episode embedding conflicts with the configured model or vector state")

    episode_id = int(episode["episode_id"])
    if episode["has_embedding"]:
        return episode_id, False

    vector = sql_vector(embed(searchable(record)))
    approval = f"""
SELECT release_id FROM scientific.evidence_releases
WHERE release_id = {int(release['release_id'])}
  AND release_status = {sql_text(APPROVED)}
  AND manifest_json = {sql_json(release['manifest'])}
  AND receipt_json = {sql_json(release['receipt'])}
  AND case_id = {sql_text(case)} AND stage = {stage}
  AND experiment_id = {sql_text(episode['experiment_id'])}
"""
    match = f"""e.episode_id = {episode_id} AND e.run_id = {sql_text(rid)}::uuid
      AND e.release_id = approved.release_id AND e.record_sha256 = {sql_text(digest)}
      AND e.run_record = {sql_json(record)} AND e.trace_level = {sql_text(TRACE_LEVEL)}
      AND e.case_id = {sql_text(case)} AND e.stage = {stage}
      AND e.experiment_id = {sql_text(episode['experiment_id'])}
      AND e.status = {sql_text(episode['status'])}"""
    try:
        result = db(f"""
WITH approved AS ({approval} FOR SHARE)
UPDATE agent_memory.episodes e
SET embedding_model = {sql_text(EMBEDDING_MODEL)}, episode_embedding = {vector}
FROM approved
WHERE {match} AND e.embedding_model IS NULL AND e.episode_embedding IS NULL
RETURNING e.episode_id;
""")
    except (RuntimeError, OSError, UnicodeError):
        result = None
    if result == [str(episode_id)]:
        return episode_id, True

    # Reconcile a lost acknowledgement or a concurrent retry without overwriting the vector.
    state = db(f"""
WITH approved AS ({approval})
SELECT CASE WHEN e.embedding_model = {sql_text(EMBEDDING_MODEL)}
                 AND e.episode_embedding = {vector} THEN 'matching'
            WHEN e.embedding_model IS NULL AND e.episode_embedding IS NULL THEN 'pending'
            ELSE 'conflict' END
FROM agent_memory.episodes e JOIN approved ON e.release_id = approved.release_id
WHERE {match};
""")
    need(state != ["conflict"], "Existing episode embedding conflicts with the generated embedding")
    need(state == ["matching"], "Embedding update could not be confirmed for the approved episode")
    return episode_id, False


def scoped_filter(case: str, stage: int) -> str:
    return f"""e.case_id = {sql_text(case)} AND e.stage = {stage}
    AND er.release_id = e.release_id AND er.case_id = e.case_id
    AND er.stage = e.stage AND er.release_status = {sql_text(APPROVED)}"""


def fetch_episode(case: str, stage: int, run_id: str) -> list[dict]:
    rid = str(UUID(run_id))
    return rows(f"""
SELECT json_build_object(
 'episode_id', e.episode_id, 'run_id', e.run_id,
 'case_id', e.case_id, 'stage', e.stage, 'trace_level', e.trace_level,
 'record_sha256', e.record_sha256, 'run_record', e.run_record)::text
FROM agent_memory.episodes e
JOIN scientific.evidence_releases er ON er.release_id = e.release_id
WHERE {scoped_filter(case, stage)} AND e.run_id = {sql_text(rid)}::uuid;
""")


def verify(case: str, stage: int, run_id: str) -> None:
    records = fetch_episode(case, stage, run_id)
    need(len(records) == 1, "No approved episode for that run")
    episode = records[0]
    digest = hashlib.sha256(canonical_bytes(episode["run_record"])).hexdigest()
    need(digest == episode["record_sha256"], "Persisted episode JSON hash mismatch")
    record = episode["run_record"]
    need(record["run_id"] == episode["run_id"], "Run ID mismatch")
    need(record["case_id"] == case and record["stage"] == stage,
         "Case/stage mismatch")
    need(record["trace_level"] == TRACE_LEVEL, "Trace fidelity mismatch")
    print(f"RESULT: VERIFY OK | run_id={run_id} | case={case} | stage={stage}")
    print(f"NOTE  status={record['execution_status']}; stages={len(record['trace'])}; stored JSON SHA-256 verified")


def replay(case: str, stage: int, run_id: str) -> None:
    data = fetch_episode(case, stage, run_id)
    need(len(data) == 1, "No approved episode for that run")
    print("RESULT: REPLAY OK | DB record readback only, not an LLM rerun")
    print(json.dumps(data[0], ensure_ascii=True, indent=2))


def search(case: str, stage: int, question: str, top: int) -> None:
    vec = sql_vector(embed(question))
    result = rows(f"""
WITH permitted AS MATERIALIZED (
 SELECT e.episode_id, e.run_id, e.run_record, e.episode_embedding
 FROM agent_memory.episodes e
 JOIN scientific.evidence_releases er ON er.release_id = e.release_id
 WHERE {scoped_filter(case, stage)}
   AND e.embedding_model = {sql_text(EMBEDDING_MODEL)}
   AND e.episode_embedding IS NOT NULL
)
SELECT json_build_object('episode_id', p.episode_id, 'run_id', p.run_id,
 'decision', p.run_record #>> '{{validation,decision}}',
 'similarity', ROUND((1 - (p.episode_embedding <=> {vec}))::numeric, 4))::text
FROM permitted p
ORDER BY p.episode_embedding <=> {vec}, p.run_id
LIMIT {top};
""")
    if not result:
        print(f"RESULT: NO APPROVED EPISODES | case={case} | stage={stage}")
    else:
        print(f"RESULT: SEARCH OK | case={case} | stage={stage} | matches={len(result)}")
        for item in result:
            print(json.dumps(item, ensure_ascii=False, indent=2))
