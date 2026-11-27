r"""Ingest one verified Stage-0 evidence snapshot into the local lab PostgreSQL.

Semantic indexing of criterion rationales requires a separate curated_fields update.

Usage (from repo root):
    python .\scripts\ingest_approved_evidence.py .\experiments\EXP-001_CA1585329339_stage0

Requires only Python standard library and the existing Docker/PostgreSQL container.
No embeddings, agent memories, network fetches or unrestricted bulk ingestion."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import UUID

from evidence_access import db, sql_json, sql_text
from verify_evidence_release import PROJECT_ROOT, verify


def sql_bool(value: bool) -> str:
    if not isinstance(value, bool):
        raise ValueError("Release receipt agent_execution_verified must be boolean")
    return "TRUE" if value else "FALSE"


def source_statement(doc: dict, key: str, kind: str, title: str,
                     version: str, digest: str, source_url: str | None,
                     snapshot: str, original: str) -> str:
    """Upsert one canonical document and link it to this specific release."""
    return f"""
    INSERT INTO scientific.sources
        (title, source_type, source_url, version_label, canonical_key,
         content_sha256, document_json)
    VALUES ({sql_text(title)}, {sql_text(kind)}, {sql_text(source_url)},
            {sql_text(version)}, {sql_text(key)}, {sql_text(digest)}, {sql_json(doc)})
    ON CONFLICT (canonical_key, version_label, content_sha256)
        WHERE canonical_key IS NOT NULL DO NOTHING;

    SELECT source_id INTO STRICT v_source_id
    FROM scientific.sources
    WHERE canonical_key = {sql_text(key)}
      AND version_label = {sql_text(version)}
      AND content_sha256 = {sql_text(digest)};

    IF NOT EXISTS (
        SELECT 1 FROM scientific.sources WHERE source_id = v_source_id
          AND document_json = {sql_json(doc)}
          AND source_type = {sql_text(kind)}
    ) THEN
        RAISE EXCEPTION 'Existing canonical document conflicts with incoming JSON/type';
    END IF;

    INSERT INTO scientific.release_evidence
        (release_id, source_id, snapshot_path, original_source_path, declared_sha256)
    VALUES (v_release_id, v_source_id, {sql_text(snapshot)},
            {sql_text(original)}, {sql_text(digest)})
    ON CONFLICT ON CONSTRAINT uq_release_snapshot DO NOTHING;

    IF NOT EXISTS (
        SELECT 1 FROM scientific.release_evidence
        WHERE release_id = v_release_id AND source_id = v_source_id
          AND snapshot_path = {sql_text(snapshot)}
          AND original_source_path = {sql_text(original)}
          AND declared_sha256 = {sql_text(digest)}
    ) THEN
        RAISE EXCEPTION 'Existing release evidence conflicts with incoming document';
    END IF;
    """


def build_sql(experiment: Path) -> str:
    manifest, receipt, inventory, documents, hashes = verify(str(experiment.resolve()), project_root=PROJECT_ROOT)
    case = manifest["case_id"]
    interpretation = documents["interpretation.json"]
    guideline = documents["guideline_v2.json"]
    references = {item["file"].replace("\\", "/"): item for item in manifest["evidence"]}
    try:
        interpretation_key = f"INTERPRETATION:{UUID(interpretation['interpretation_uuid'])}"
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError("Interpretation UUID is missing or invalid") from error
    guideline_key = f"GUIDELINE:{guideline['guideline_id']}"
    guideline_version = str(guideline["version"])
    interpretation_version = str(interpretation["version"])
    source_url = interpretation["narrative_provenance"].get("source_url")

    sources = [
        (interpretation, interpretation_key, "INTERPRETATION",
         f"{guideline['gene']} {case} interpretation v{interpretation_version}",
         interpretation_version, references["evidence/interpretation.json"], source_url),
        (guideline, guideline_key, "GUIDELINE",
         f"{guideline['guideline_id']} v{guideline_version} ({guideline['gene']})",
         guideline_version, references["evidence/guideline_v2.json"], None),
    ]
    for path, ref in references.items():
        if path in ("evidence/interpretation.json", "evidence/guideline_v2.json"):
            continue
        doc = documents[path.removeprefix("evidence/")]
        if path == "evidence/criterion_rationale.json":
            sources.append((doc, f"CRITERION_RATIONALE:{case}:{interpretation_key}",
                            "CRITERION_RATIONALE", f"{case} criterion rationale v{doc['source_version']}",
                            doc["source_version"], ref, doc["source_url"]))
        elif path.startswith("evidence/clinvar/"):
            accession = doc["scv_accession_version"]
            sources.append((doc, f"CLINVAR:{accession}", "CLINVAR",
                            f"ClinVar {accession}", accession.rsplit(".", 1)[1],
                            ref, doc.get("source_url")))
        elif path.startswith("evidence/publications/"):
            pmid = str(doc["pmid"])
            sources.append((doc, f"PUBMED:{pmid}", "PUBLICATION", doc["title"],
                            "snapshot", ref, doc.get("source_url")))
        else:
            raise ValueError(f"Unsupported evidence path: {path}")

    source_sql = "\n".join(source_statement(doc, key, kind, title, version,
                    ref["sha256"].lower(), url, ref["file"], ref["source"])
                    for doc, key, kind, title, version, ref, url in sources)

    exp_id = sql_text(manifest["experiment_id"])
    case_id = sql_text(case)
    stage = int(manifest["stage"])
    manifest_digest, receipt_digest, inventory_digest = (
        sql_text(hashes[key]) for key in ("manifest", "receipt", "inventory")
    )

    return f"""
BEGIN;
DO $task8$
DECLARE
    v_release_id BIGINT;
    v_source_id BIGINT;
BEGIN
    INSERT INTO scientific.evidence_releases
        (experiment_id, case_id, stage, release_status, released_at_utc,
         agent_execution_verified, access_policy, manifest_json, receipt_json,
         case_inventory_json, manifest_sha256, receipt_sha256, case_inventory_sha256)
    VALUES ({exp_id}, {case_id}, {stage}, {sql_text(receipt['status'])},
            {sql_text(receipt['released_at_utc'])}::timestamptz,
            {sql_bool(receipt['agent_execution_verified'])},
            {sql_text(inventory['policy'])}, {sql_json(manifest)},
            {sql_json(receipt)}, {sql_json(inventory)},
            {manifest_digest}, {receipt_digest}, {inventory_digest})
    ON CONFLICT ON CONSTRAINT uq_evidence_release DO NOTHING;

    SELECT release_id INTO STRICT v_release_id
    FROM scientific.evidence_releases
    WHERE experiment_id = {exp_id} AND case_id = {case_id} AND stage = {stage};

    IF NOT EXISTS (
        SELECT 1 FROM scientific.evidence_releases
        WHERE release_id = v_release_id
          AND release_status = {sql_text(receipt['status'])}
          AND agent_execution_verified = {sql_bool(receipt['agent_execution_verified'])}
          AND access_policy = {sql_text(inventory['policy'])}
          AND manifest_sha256 = {manifest_digest}
          AND receipt_sha256 = {receipt_digest}
          AND case_inventory_sha256 = {inventory_digest}
    ) THEN
        RAISE EXCEPTION 'Existing release conflicts with approved snapshot metadata';
    END IF;

    {source_sql}

    IF (SELECT count(*) FROM scientific.release_evidence
        WHERE release_id = v_release_id) <> {len(manifest["evidence"])} THEN
        RAISE EXCEPTION 'Release must reference exactly the declared evidence files';
    END IF;
END;
$task8$;
COMMIT;
SELECT 'RESULT: DB OK | sources=' || COUNT(DISTINCT re.source_id)
       || ' | releases=1 | links=' || COUNT(*)
FROM scientific.evidence_releases er
JOIN scientific.release_evidence re ON re.release_id = er.release_id
WHERE er.experiment_id = {exp_id} AND er.case_id = {case_id} AND er.stage = {stage};
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Stage-0 experiment folder")
    args = parser.parse_args()
    try:
        experiment = (Path.cwd() / args.experiment).resolve()
        sql = build_sql(experiment)
        print("\n".join(db(sql)))
        print("NOTE  No agent memories or investigation runs were modified")
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(f"RESULT: FAIL - {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
