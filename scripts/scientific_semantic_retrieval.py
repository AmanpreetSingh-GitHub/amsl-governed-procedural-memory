r"""Index and search approved scientific evidence using local Ollama and pgvector.

Requires the Python standard library, local Ollama and the Docker PostgreSQL container.

From the repository root:
    python .\scripts\scientific_semantic_retrieval.py index --case CA1585329339 --stage 0
    python .\scripts\scientific_semantic_retrieval.py search --case CA1585329339 --stage 0 --question "Which criteria were met?"

Reads approved JSON from PostgreSQL, not experiment run files or sealed files.
Source text remains in PostgreSQL; embeddings are an index, not evidence."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from urllib.error import URLError
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

from evidence_access import APPROVED as APPROVED_STATUS, db, sql_text

MODEL = "embeddinggemma:latest"
DIMENSIONS = 768
OLLAMA_URL = "http://127.0.0.1:11434/api/embed"  # Loopback ONLY


class NoRedirect(HTTPRedirectHandler):
    """Never follow a redirect away from the approved local endpoint."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Local embedding endpoint unexpectedly redirected")


def sql_path(path: tuple[str, ...]) -> str:
    if not path or len(path) > 8 or any(not part or not isinstance(part, str) for part in path):
        raise ValueError("Invalid JSON path")
    return "ARRAY[" + ",".join(sql_text(part) for part in path) + "]::text[]"


def sql_vector(values: list[float]) -> str:
    if len(values) != DIMENSIONS or any(not isinstance(x, (float, int))
                                       or isinstance(x, bool) or not math.isfinite(x) for x in values):
        raise ValueError(f"Embedding must contain {DIMENSIONS} finite numbers")
    return "'[" + ",".join(format(x, ".9g") for x in values) + "]'::vector(768)"


def embed(text: str) -> list[float]:
    """Only sends content to Ollama on 127.0.0.1. No external API involved."""
    payload = json.dumps({"model": MODEL, "input": text}).encode("utf-8")
    request = Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"},
                      method="POST")
    try:
        # Ignore machine proxy configuration and refuse HTTP redirects.
        opener = build_opener(ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=120) as response:
            result = json.load(response)
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Local Ollama is unavailable: {exc}") from exc
    vectors = result.get("embeddings", [])
    if not isinstance(vectors, list) or len(vectors) != 1:
        raise ValueError("Unexpected Ollama embedding response")
    vector = vectors[0]
    sql_vector(vector)  # Validate dimension, values and finite numbers.
    return vector


def authorized_source_sql(case: str, stage: int) -> str:
    """Select ONLY canonical documents joined to a scoped approved release."""
    return f"""
SELECT json_build_object(
    'source_id', s.source_id,
    'source_type', s.source_type,
    'canonical_key', s.canonical_key,
    'version_label', s.version_label,
    'content_sha256', s.content_sha256,
    'document_json', s.document_json
)::text
FROM scientific.evidence_releases er
JOIN scientific.release_evidence re ON re.release_id = er.release_id
JOIN scientific.sources s ON s.source_id = re.source_id
WHERE er.case_id = {sql_text(case)}
  AND er.stage = {stage}
  AND er.release_status = {sql_text(APPROVED_STATUS)}
  AND er.case_inventory_json->>'caid' = er.case_id
  AND re.declared_sha256 = s.content_sha256
ORDER BY s.source_id;
"""


def curated_fields(source: dict) -> list[tuple[tuple[str, ...], str]]:
    """Select explicit paths, not every key in scientific JSON."""
    document = source["document_json"]
    if not isinstance(document, dict):
        raise ValueError("Source JSON must be an object")
    fields: list[tuple[tuple[str, ...], str]] = []

    def add(path: tuple[str, ...], value: object) -> None:
        if isinstance(value, str):
            content = value.strip()
        elif isinstance(value, dict):
            content = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            return
        if content:
            if len(content) > 7000:
                raise ValueError(f"Source {source['source_id']} path {path} exceeds 7000 characters")
            fields.append((path, content))

    kind = source["source_type"]
    if kind == "INTERPRETATION":
        for key in ("narrative", "caveat", "narrative_scope"):
            add((key,), document.get(key))
        for key in ("criteria", "known_uncertainties"):
            values = document.get(key, [])
            if not isinstance(values, list):
                raise ValueError(f"Interpretation {key} is not a list")
            for index, value in enumerate(values):
                add((key, str(index)), value)
    elif kind == "GUIDELINE":
        add(("note",), document.get("note"))
        criteria = document.get("criteria", [])
        if isinstance(criteria, list):
            for index, value in enumerate(criteria):
                add(("criteria", str(index)), value)
        elif isinstance(criteria, dict):
            for key, value in criteria.items():
                add(("criteria", str(key)), value)
        else:
            raise ValueError("Guideline criteria must be an object or list")
    elif kind == "CLINVAR":
        for key in ("classification", "review_status", "note"):
            add((key,), document.get(key))
    elif kind == "PUBLICATION":
        for key in ("title", "eligibility"):
            add((key,), document.get(key))
        sections = document.get("abstract_sections")
        if isinstance(sections, list):
            for index, value in enumerate(sections):
                add(("abstract_sections", str(index)), value)
        else:
            add(("abstract_sections",), sections)
    else:
        raise ValueError(f"Unrecognized scientific source type {kind!r}")
    if not fields:
        raise ValueError(f"Source {source['source_id']} has no curated textual fields")
    return fields


def existing_index_paths(case: str, stage: int) -> set[tuple[int, tuple[str, ...]]]:
    """Avoid re-embedding unchanged paths on repeat runs."""
    rows = db(f"""
SELECT json_build_object('source_id', se.source_id, 'json_path', se.json_path)::text
FROM scientific.source_embeddings se
JOIN scientific.sources s ON s.source_id = se.source_id
JOIN scientific.release_evidence re ON re.source_id = s.source_id
JOIN scientific.evidence_releases er ON er.release_id = re.release_id
WHERE er.case_id = {sql_text(case)} AND er.stage = {stage}
  AND er.release_status = {sql_text(APPROVED_STATUS)}
  AND er.case_inventory_json->>'caid' = er.case_id
  AND re.declared_sha256 = s.content_sha256
  AND se.embedding_model = {sql_text(MODEL)};
""")
    return {(int(item["source_id"]), tuple(item["json_path"]))
            for item in (json.loads(row) for row in rows)}


def run_index(case: str, stage: int) -> None:
    source_rows = [json.loads(row) for row in db(authorized_source_sql(case, stage))]
    if not source_rows:
        raise ValueError(f"No approved sources for {case}, stage {stage}; nothing indexed")
    # Confirm there is no conflicting canonical source metadata in the scoped set.
    seen = set()
    for src in source_rows:
        if src["source_id"] in seen:
            raise ValueError("Duplicate source association in an approved release")
        seen.add(src["source_id"])

    existing = existing_index_paths(case, stage)
    parts: list[str] = []
    for src in source_rows:
        for path, content in curated_fields(src):
            if (int(src["source_id"]), path) in existing:
                continue
            prefix = f"{src['source_type']} {src['canonical_key']} " + "/".join(path)
            vector = embed(prefix + ": " + content)
            path_sql = sql_path(path)
            source_id = int(src["source_id"])
            # Recheck authorization at write time. An unapproved record cannot
            # obtain a vector through a stale initial source selection.
            parts.append(f"""
INSERT INTO scientific.source_embeddings (source_id, json_path, embedding_model, embedding)
SELECT s.source_id, {path_sql}, {sql_text(MODEL)}, {sql_vector(vector)}
FROM scientific.sources s
WHERE s.source_id = {source_id}
  AND s.document_json #> {path_sql} IS NOT NULL
  AND EXISTS (
      SELECT 1 FROM scientific.evidence_releases er
      JOIN scientific.release_evidence re ON re.release_id = er.release_id
      WHERE re.source_id = s.source_id
        AND er.case_id = {sql_text(case)} AND er.stage = {stage}
        AND er.release_status = {sql_text(APPROVED_STATUS)}
        AND er.case_inventory_json->>'caid' = er.case_id
        AND re.declared_sha256 = s.content_sha256
  )
ON CONFLICT (source_id, embedding_model, json_path) DO NOTHING;""")
    # - Index new paths in one transaction; existing paths are skipped.
    count_sql = f"""
SELECT 'RESULT: INDEX OK | case={case} | stage={stage} | vectors=' || COUNT(DISTINCT se.embedding_id)
FROM scientific.source_embeddings se
JOIN scientific.release_evidence re ON re.source_id = se.source_id
JOIN scientific.evidence_releases er ON er.release_id = re.release_id
JOIN scientific.sources s ON s.source_id = se.source_id
WHERE er.case_id = {sql_text(case)} AND er.stage = {stage}
  AND er.release_status = {sql_text(APPROVED_STATUS)}
  AND er.case_inventory_json->>'caid' = er.case_id
  AND re.declared_sha256 = s.content_sha256
  AND se.embedding_model = {sql_text(MODEL)};
"""
    result = db(("BEGIN;\n" + "\n".join(parts) + "\nCOMMIT;\n" if parts else "") + count_sql)
    if not any(line.startswith("RESULT: INDEX OK") for line in result):
        raise RuntimeError("Database indexing count verification did not complete")
    for line in result:
        if line.startswith("RESULT:"):
            print(line)
    print(f"NOTE  Added {len(parts)} new vectors using LOCAL Ollama; {len(existing)} existing skipped; source text not duplicated")


def search_sql(case: str, stage: int, vector: list[float], limit: int) -> str:
    """MATERIALIZED authorization filter is evaluated before vector ranking."""
    qvec = sql_vector(vector)
    return f"""
WITH permitted AS MATERIALIZED (
    SELECT s.source_id, s.canonical_key, s.version_label, s.content_sha256,
           se.json_path, se.embedding
    FROM scientific.evidence_releases er
    JOIN scientific.release_evidence re ON re.release_id = er.release_id
    JOIN scientific.sources s ON s.source_id = re.source_id
    JOIN scientific.source_embeddings se ON se.source_id = s.source_id
    WHERE er.case_id = {sql_text(case)}
      AND er.stage = {stage}
      AND er.release_status = {sql_text(APPROVED_STATUS)}
      AND er.case_inventory_json->>'caid' = er.case_id
      AND re.declared_sha256 = s.content_sha256
      AND se.embedding_model = {sql_text(MODEL)}
)
SELECT json_build_object(
    'source_id', p.source_id,
    'canonical_key', p.canonical_key,
    'version', p.version_label,
    'source_sha256', p.content_sha256,
    'case_id', {sql_text(case)}, 'stage', {stage},
    'json_path', p.json_path,
    'similarity', ROUND((1 - (p.embedding <=> {qvec}))::numeric, 4),
    'original_text', CASE
        WHEN jsonb_typeof(s.document_json #> p.json_path) = 'string'
        THEN s.document_json #>> p.json_path
        ELSE (s.document_json #> p.json_path)::text
    END
)::text
FROM permitted p
JOIN scientific.sources s ON s.source_id = p.source_id
WHERE s.document_json #> p.json_path IS NOT NULL
ORDER BY p.embedding <=> {qvec}, p.source_id, p.json_path
LIMIT {limit};
"""


def run_search(case: str, stage: int, question: str, top: int) -> None:
    vector = embed(question)
    results = [json.loads(row) for row in db(search_sql(case, stage, vector, top))]
    if not results:
        print(f"RESULT: NO APPROVED MATCHES | case={case} | stage={stage}")
        return
    print(f"RESULT: SEARCH OK | case={case} | stage={stage} | matches={len(results)}")
    for result in results:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("index", "search"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--case", required=True, help="Approved case ID (e.g. CA1585329339)")
        sub.add_argument("--stage", type=int, required=True, help="Approved release stage")
        if command == "search":
            sub.add_argument("--question", required=True, help="Text to search by meaning")
            sub.add_argument("--top", type=int, default=3, help="Maximum number of matches (1-10)")
    args = parser.parse_args()
    try:
        if not re.fullmatch(r"CA[0-9]+", args.case):
            raise ValueError("Case must look like CA1585329339")
        if args.stage < 0:
            raise ValueError("Stage must not be negative")
        if args.command == "index":
            run_index(args.case, args.stage)
        else:
            if not args.question.strip():
                raise ValueError("Question must not be empty")
            if not 1 <= args.top <= 10:
                raise ValueError("--top must be between 1 and 10")
            run_search(args.case, args.stage, args.question, args.top)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"RESULT: FAIL - {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
