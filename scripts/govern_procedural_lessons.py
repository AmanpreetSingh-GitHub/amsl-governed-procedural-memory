"""Propose, revise and activate versioned procedures with explicit operator approval."""

import argparse
import json
import os
import re
import sys

from evidence_access import (APPROVED, SOURCE_FIELDS, db, load_episode, need, rows, sql_json, sql_text,
                             verified_episode)
from model_dispatch import call_learning_decision
from scientific_semantic_retrieval import MODEL as EMBEDDING_MODEL, embed, sql_vector


def text(value: str, field: str) -> str:
    need(isinstance(value, str) and value.strip(), f"{field} must not be blank")
    return value.strip()


def extract(reflection: str) -> str:
    need(isinstance(reflection, str) and reflection.partition("\n")[0].strip() == "DECISION: PROPOSE",
         "Episode reflection does not propose a lesson")
    parts = []
    for name in ("Candidate lesson", "Conditions for reuse", "Limitations"):
        match = re.search(r"(?ims)^###\s+" + re.escape(name) + r"\s*\n(.*?)(?=^###\s+|\Z)", reflection)
        need(match is not None and match[1].strip(), f"Reflection lacks {name}")
        parts.append(match[1].strip())
    return f"{parts[0]}\n\nConditions for reuse:\n{parts[1]}\n\nLimitations:\n{parts[2]}"


def outcome(sql: str) -> dict:
    result = [json.loads(line) for line in db(sql) if line.startswith("{")]
    need(len(result) == 1, "Database operation did not return a result; check approval or stale state")
    return result[0]


def episode(episode_id: int, *, allow_abstain: bool = False) -> tuple[dict, dict]:
    ep = load_episode(episode_id)
    return ep, verified_episode(ep, ep["record"], learning=True, allow_abstain=allow_abstain)


def record(memory_id: int) -> dict:
    need(type(memory_id) is int and memory_id > 0, "Invalid memory ID")
    result = rows(f"""
SELECT json_build_object('memory_id',m.memory_id,'episode_id',m.source_episode_id,
 'title',m.title,'status',m.status,'active_version_id',m.active_version_id,
 'versions',(SELECT json_agg(json_build_object('version_id',v.version_id,'n',v.version_number,
  'title',v.title,'content',v.content,'change_reason',v.change_reason) ORDER BY v.version_number)
  FROM agent_memory.memory_versions v WHERE v.memory_id=m.memory_id))::text
FROM agent_memory.memories m WHERE m.memory_id={memory_id} AND m.memory_type='PROCEDURE';
""")
    need(len(result) == 1, "Procedure not found")
    return result[0]


def origin_guard(ep: dict, release: dict) -> str:
    # Bind the write to the record and approved sources verified on read.
    sources = [[s["snapshot_path"], *(s[k] for k in SOURCE_FIELDS), s["sha256"]] for s in release["sources"]]
    return f"""EXISTS (
 SELECT 1 FROM agent_memory.episodes e JOIN scientific.evidence_releases r ON r.release_id=e.release_id
 WHERE e.episode_id={ep['episode_id']} AND e.status='COMPLETED' AND e.release_id={release['release_id']}
   AND e.record_sha256={sql_text(ep['record_sha256'])} AND e.run_id::text={sql_text(ep['run_id'])}
   AND e.run_record::text=({sql_json(ep['record'])})::text
   AND e.case_id=r.case_id AND e.stage=r.stage AND e.experiment_id=r.experiment_id
   AND r.release_status={sql_text(APPROVED)} AND r.manifest_json={sql_json(release['manifest'])}
   AND r.receipt_json={sql_json(release['receipt'])}
   AND (SELECT jsonb_agg(jsonb_build_array(re.snapshot_path,s.source_id,s.canonical_key,
                         s.version_label,s.content_sha256,re.declared_sha256) ORDER BY re.snapshot_path)
        FROM scientific.release_evidence re JOIN scientific.sources s ON s.source_id=re.source_id
        WHERE re.release_id=r.release_id)={sql_json(sources)}
 FOR SHARE OF e,r
)"""


def assess(episode_id: int, *, decide: bool = False, allow_external_model: bool = False):
    ep, release = episode(episode_id, allow_abstain=True)
    rec = ep["record"]
    reflection = text(rec.get("reflection"), "Reflection")
    if reflection.partition("\n")[0].strip() == "DECISION: ABSTAIN":
        print("RESULT: NO_CHANGE | Reflection abstained; no lesson was proposed")
        return None
    proposal = extract(reflection)
    vector = sql_vector(embed(proposal))
    matches = rows(f"""
SELECT json_build_object('memory_id',m.memory_id,'version_id',v.version_id,'version_number',v.version_number,
 'title',v.title,'content',v.content,'similarity',1-(em.embedding <=> {vector}))::text
FROM agent_memory.embeddings em JOIN agent_memory.memory_versions v ON v.version_id=em.version_id
JOIN agent_memory.memories m ON m.memory_id=v.memory_id AND m.active_version_id=v.version_id
WHERE m.status='VALIDATED' AND m.memory_type='PROCEDURE' AND em.embedding_model={sql_text(EMBEDDING_MODEL)}
ORDER BY em.embedding <=> {vector},m.memory_id,v.version_id LIMIT 3;
""")
    if not decide:
        print(f"RESULT: ASSESS | episode_id={episode_id} | matches={len(matches)}")
        for match in matches:
            print(json.dumps(match, ensure_ascii=True))
        return None
    provenance = {key: release[key] for key in ("release_id", "experiment_id", "case_id", "stage", "manifest")}
    provenance["sources"] = [{k: item[k] for k in (*SOURCE_FIELDS, "snapshot_path")} for item in release["sources"]]
    recommendation = call_learning_decision({
        "investigation_report": text(rec.get("report"), "Report"), "proposed_lesson": proposal,
        "existing_memories": matches, "evidence": release["evidence"], "provenance": provenance,
    }, os.environ.get("OPENAI_MODEL", "").strip(), allow_external_model=allow_external_model)
    print("RESULT: LEARNING RECOMMENDATION | recommendation only")
    print(json.dumps(recommendation, ensure_ascii=True))
    return recommendation, ep, matches


def store_version(ep: dict, release: dict, title: str, content: str, draft: dict | None = None) -> dict:
    episode_id = ep["episode_id"]
    marker = f"DRAFT_LEARNING:{episode_id}:{draft['decision']}:" if draft else ""
    reason = marker + json.dumps(draft, sort_keys=True) if draft else "Proposed in origin episode " + ep["run_id"]
    duplicate = (f"AND NOT EXISTS (SELECT 1 FROM agent_memory.memory_versions "
                 f"WHERE change_reason LIKE {sql_text(marker + '%')})") if marker else ""
    vector = sql_vector(embed(title + "\n" + content))
    origin = origin_guard(ep, release)
    if draft and draft["decision"] == "REFINE":
        target, base = draft["target_memory_id"], draft["active_version_id"]
        lock = f"SELECT 1 FROM agent_memory.memories WHERE memory_id={target} FOR UPDATE;"
        made = f"""SELECT m.memory_id,1+(SELECT MAX(version_number) FROM agent_memory.memory_versions
 WHERE memory_id=m.memory_id) AS n FROM agent_memory.memories m
 WHERE m.memory_id={target} AND m.status='VALIDATED' AND m.active_version_id={base}
 AND {origin} {duplicate}"""
    else:
        lock = ""
        made = f"""INSERT INTO agent_memory.memories(source_episode_id,memory_type,title,content,status)
 SELECT {episode_id},'PROCEDURE',{sql_text(title)},{sql_text(content)},'CANDIDATE'
 WHERE {origin} {duplicate} AND NOT EXISTS (
 SELECT 1 FROM agent_memory.memories WHERE source_episode_id={episode_id}
 AND memory_type='PROCEDURE' AND title={sql_text(title)})
 RETURNING memory_id,1 AS n"""
    return outcome(f"""BEGIN;
{lock}
SELECT 1 FROM agent_memory.episodes WHERE episode_id={episode_id} FOR UPDATE;
WITH made AS ({made}), version AS (
 INSERT INTO agent_memory.memory_versions(memory_id,version_number,title,content,change_reason)
 SELECT memory_id,n,{sql_text(title)},{sql_text(content)},{sql_text(reason)} FROM made
 RETURNING memory_id,version_id,version_number
), embedded AS (
 INSERT INTO agent_memory.embeddings(version_id,embedding_model,embedding)
 SELECT version_id,{sql_text(EMBEDDING_MODEL)},{vector} FROM version RETURNING version_id
)
SELECT row_to_json(v)::text FROM version v JOIN embedded USING(version_id);
COMMIT;""")


def candidate(episode_id: int, title: str) -> None:
    ep, release = episode(episode_id)
    made = store_version(ep, release, text(title, "Title"), extract(ep["record"]["reflection"]))
    print(f"RESULT: CANDIDATE STORED | memory_id={made['memory_id']} | version=1 | status=CANDIDATE")


def draft_learning(episode_id: int, *, allow_external_model: bool = False) -> None:
    result = assess(episode_id, decide=True, allow_external_model=allow_external_model)
    if result is None or result[0]["decision"] not in ("NEW", "REFINE"):
        print("NOTE  No draft created for this recommendation")
        return
    recommendation, assessed, matches = result
    ep, release = episode(episode_id)
    need(ep["record_sha256"] == assessed["record_sha256"] and ep["run_id"] == assessed["run_id"],
         "Episode changed after learning assessment")
    decision = recommendation["decision"]
    draft = {key: ep[key] for key in ("episode_id", "run_id", "record_sha256")}
    draft.update(decision=decision, target_memory_id=recommendation.get("memory_id"),
                 model_rationale=recommendation["rationale"],
                 supporting_references=recommendation["supporting_references"],
                 limitations=recommendation["limitations"],
                 unresolved_contradictions=recommendation["unresolved_contradictions"])
    lesson = extract(ep["record"]["reflection"])
    title = lesson.splitlines()[0][:150]
    if decision == "REFINE":
        target = next(item for item in matches if item["memory_id"] == recommendation["memory_id"])
        draft["active_version_id"] = target["version_id"]
        title, lesson = target["title"], recommendation["proposed_revised_lesson"]
    made = store_version(ep, release, title, lesson, draft)
    print(f"RESULT: DRAFT STORED | memory_id={made['memory_id']} | version={made['version_number']} | inactive")


def approve(memory_id: int, n: int, actor: str, reason: str, action: str) -> None:
    need(action in ("PROMOTE", "ROLLBACK"), "Invalid activation action")
    m = record(memory_id)
    v = next((item for item in m["versions"] if item["n"] == n), None)
    need(v is not None and m["status"] in ("CANDIDATE", "VALIDATED"), "Version not found or memory inactive")
    previous, version = m["active_version_id"], v["version_id"]
    need(previous != version, "This version is already active")
    guard = ""
    if action == "ROLLBACK":
        need(m["status"] == "VALIDATED", "Rollback requires an active version")
        guard = f"""AND EXISTS (SELECT 1 FROM agent_memory.validations
 WHERE version_id={version} AND validation_type='OPERATOR_DECISION' AND outcome='PASS'
 AND rationale LIKE 'PROMOTE:%')"""
    else:
        reason_stored = v["change_reason"]
        if reason_stored.startswith("DRAFT_LEARNING"):
            match = re.fullmatch(r"DRAFT_LEARNING:([1-9][0-9]*):(NEW|REFINE):(.+)", reason_stored, re.DOTALL)
            need(match is not None, "Invalid learning draft metadata")
            metadata = json.loads(match[3])
            ep, release = episode(int(match[1]))
            need(isinstance(metadata, dict) and
                 all(metadata.get(k) == ep[k] for k in ("episode_id", "run_id", "record_sha256")) and
                 metadata.get("decision") == match[2], "Draft source metadata mismatch")
            if match[2] == "REFINE":
                need(m["status"] == "VALIDATED" and metadata.get("target_memory_id") == memory_id and
                     metadata.get("active_version_id") == previous, "REFINE base version is stale")
            else:
                need(m["status"] == "CANDIDATE" and previous is None and n == 1 and
                     ep["episode_id"] == m["episode_id"] and metadata.get("target_memory_id") is None,
                     "NEW draft is no longer an inactive candidate")
        else:
            need(n == 1 and type(m["episode_id"]) is int,
                 "Manual revision has no verifiable episode; use a REFINE draft")
            ep, release = episode(m["episode_id"])
            prefix = "Proposed in origin episode "
            need(not reason_stored.startswith(prefix) or reason_stored.removeprefix(prefix) == ep["run_id"],
                 "Candidate source run mismatch")
        guard = "AND " + origin_guard(ep, release)
    previous_sql = str(previous) if previous is not None else "NULL"
    notes = {"previous_active_version_id": previous, "new_active_version_id": version,
             "scope": "Operator authorization for procedural reuse"}
    result = outcome(f"""BEGIN;
SELECT 1 FROM agent_memory.memories WHERE memory_id={memory_id} FOR UPDATE;
WITH changed AS (
 UPDATE agent_memory.memories m SET status='VALIDATED',active_version_id=v.version_id,
  title=v.title,content=v.content,updated_at=NOW()
 FROM agent_memory.memory_versions v WHERE m.memory_id={memory_id} AND v.memory_id=m.memory_id
 AND v.version_id={version} AND v.change_reason={sql_text(v['change_reason'])}
 AND v.content={sql_text(v['content'])} AND m.status={sql_text(m['status'])}
 AND m.source_episode_id IS NOT DISTINCT FROM {m['episode_id'] if m['episode_id'] is not None else 'NULL'}
 AND m.active_version_id IS NOT DISTINCT FROM {previous_sql} {guard}
 RETURNING m.active_version_id
)
INSERT INTO agent_memory.validations(version_id,validation_type,outcome,rationale,validator,evidence_notes)
SELECT active_version_id,'OPERATOR_DECISION','PASS',{sql_text(action + ': ' + text(reason,'Reason'))},
 {sql_text(text(actor,'Actor'))},{sql_text(json.dumps(notes))} FROM changed
RETURNING json_build_object('audit_id',validation_id)::text;
COMMIT;""")
    print(f"RESULT: {action} | memory_id={memory_id} | active_version={n} | audit_id={result['audit_id']}")


def reinforce(episode_id: int, memory_id: int, actor: str, reason: str) -> None:
    ep, release = episode(episode_id)
    memory = record(memory_id)
    active = memory["active_version_id"]
    need(memory["status"] == "VALIDATED" and type(active) is int, "Reinforcement requires a validated version")
    notes = {key: ep[key] for key in ("episode_id", "run_id", "record_sha256", "release_id")}
    notes.update(scope="Operator observation only", memory_injection=ep["record"].get("procedural_memory_supplied"))
    result = outcome(f"""BEGIN;
SELECT 1 FROM agent_memory.memories WHERE memory_id={memory_id} FOR UPDATE;
INSERT INTO agent_memory.validations(version_id,validation_type,outcome,rationale,validator,evidence_notes)
SELECT m.active_version_id,'OPERATOR_DECISION','PASS',{sql_text('REINFORCE: ' + text(reason,'Reason'))},
 {sql_text(text(actor,'Actor'))},{sql_text(json.dumps(notes))}
FROM agent_memory.memories m WHERE m.memory_id={memory_id} AND m.status='VALIDATED' AND m.active_version_id={active}
 AND {origin_guard(ep, release)}
 AND NOT EXISTS (SELECT 1 FROM agent_memory.validations a WHERE a.version_id=m.active_version_id
     AND a.validation_type='OPERATOR_DECISION' AND a.rationale LIKE 'REINFORCE:%'
     AND a.evidence_notes::jsonb->>'episode_id'={sql_text(str(episode_id))})
RETURNING json_build_object('audit_id',validation_id)::text;
COMMIT;""")
    print(f"RESULT: REINFORCED | memory_id={memory_id} | version_id={active} | audit_id={result['audit_id']}")


def retire(memory_id: int, actor: str, reason: str) -> None:
    record(memory_id)  # Restrict operator commands to procedural memories.
    result = outcome(f"""BEGIN;
SELECT 1 FROM agent_memory.memories WHERE memory_id={memory_id} FOR UPDATE;
WITH target AS (
 SELECT m.memory_id,COALESCE(m.active_version_id,(SELECT v.version_id FROM agent_memory.memory_versions v
  WHERE v.memory_id=m.memory_id ORDER BY v.version_number DESC LIMIT 1)) AS version_id
 FROM agent_memory.memories m WHERE m.memory_id={memory_id} AND m.status IN ('CANDIDATE','VALIDATED')
), changed AS (
 UPDATE agent_memory.memories m SET status='RETIRED',active_version_id=NULL,updated_at=NOW()
 FROM target t WHERE m.memory_id=t.memory_id AND t.version_id IS NOT NULL RETURNING t.version_id
)
INSERT INTO agent_memory.validations(version_id,validation_type,outcome,rationale,validator)
SELECT version_id,'OPERATOR_DECISION','FAIL',{sql_text('RETIRE: ' + text(reason,'Reason'))},
 {sql_text(text(actor,'Actor'))} FROM changed RETURNING json_build_object('audit_id',validation_id)::text;
COMMIT;""")
    print(f"RESULT: RETIRED | memory_id={memory_id} | audit_id={result['audit_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("assess", "draft-learning", "candidate", "reinforce", "promote", "rollback", "retire", "inspect"):
        cmd = commands.add_parser(name)
        if name in ("assess", "draft-learning", "candidate", "reinforce"):
            cmd.add_argument("--episode-id", type=int, required=True)
        if name in ("reinforce", "promote", "rollback", "retire", "inspect"):
            cmd.add_argument("--memory-id", type=int, required=True)
        if name in ("reinforce", "promote", "rollback", "retire"):
            cmd.add_argument("--actor", required=True)
            cmd.add_argument("--reason", required=True)
        if name in ("promote", "rollback"):
            cmd.add_argument("--version", type=int, required=True)
        if name in ("assess", "draft-learning"):
            cmd.add_argument("--allow-external-model", action="store_true")
        if name == "assess":
            cmd.add_argument("--decide", action="store_true")
        if name == "candidate":
            cmd.add_argument("--title", required=True)
    args = parser.parse_args()
    try:
        name = args.command
        if name == "assess":
            assess(args.episode_id, decide=args.decide, allow_external_model=args.allow_external_model)
        elif name == "draft-learning":
            draft_learning(args.episode_id, allow_external_model=args.allow_external_model)
        elif name == "candidate":
            candidate(args.episode_id, args.title)
        elif name in ("promote", "rollback"):
            approve(args.memory_id, args.version, args.actor, args.reason, name.upper())
        elif name == "retire":
            retire(args.memory_id, args.actor, args.reason)
        elif name == "reinforce":
            reinforce(args.episode_id, args.memory_id, args.actor, args.reason)
        else:
            memory = record(args.memory_id)
            active = memory["active_version_id"]
            print(f"RESULT: MEMORY | memory_id={args.memory_id} | status={memory['status']} | active_version_id={active}")
            print(f"NOTE  origin_episode_id={memory['episode_id']}; versions={len(memory['versions'])}")
            for item in memory["versions"]:
                print(f"VERSION {item['n']} | active={item['version_id'] == active}")
        return 0
    except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
        print(f"RESULT: FAIL - {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
