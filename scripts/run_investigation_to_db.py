"""Run and audit approved investigations in PostgreSQL."""

import argparse
import hashlib
import json
import os
import re
import sys

from episode_persistence import replay, retry_episode_embedding, save_episode, search, verify
from evidence_access import approved_release, check_standalone, need, scope
from model_dispatch import authorize_model, call_memory_applicability, run_agent
from procedural_retrieval import retrieve_procedural_memories, select_applicable_memories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_subparsers(dest="command", required=True)
    run = choices.add_parser("run", description="Investigate an approved release with procedural memory OFF by default.")
    run.add_argument("--case", required=True)
    run.add_argument("--stage", required=True, type=int)
    run.add_argument("--model", default=None,
                     help="Optional override of OPENAI_MODEL; original configuration by default")
    memory_mode = run.add_mutually_exclusive_group()
    memory_mode.add_argument("--use-multi-memory", action="store_true",
                             help="Select up to three applicable procedures for this investigation")
    memory_mode.add_argument("--preview-multi-memory", action="store_true",
                             help="Review procedural applicability without investigation or persistence")
    run.add_argument("--expected-memory-id", type=int, default=None)
    run.add_argument("--expected-version-id", type=int, default=None)
    run.add_argument("--expected-content-sha256", default=None)
    run.add_argument("--allow-external-model", action="store_true",
                     help="Only after explicit approval to send approved evidence to the configured external agent endpoint")
    for cmd in ("replay", "verify", "search"):
        p = choices.add_parser(cmd)
        p.add_argument("--case", required=True)
        p.add_argument("--stage", required=True, type=int)
        if cmd == "search":
            p.add_argument("--question", required=True)
            p.add_argument("--top", default=3, type=int)
        else:
            p.add_argument("--run-id", required=True)
    retry = choices.add_parser("retry-embedding", description="Index an existing approved episode without rerunning it.")
    retry.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        if args.command == "retry-embedding":
            episode_id, updated = retry_episode_embedding(args.run_id)
            status = "STORED" if updated else "ALREADY STORED"
            print(f"RESULT: EMBEDDING {status} | episode_id={episode_id} | run_id={args.run_id}")
            return 0
        scope(args.case, args.stage)
        if args.command == "run":
            expected = (args.expected_memory_id, args.expected_version_id, args.expected_content_sha256)
            supplied = [value is not None for value in expected]
            need(not any(supplied) or (all(supplied) and args.use_multi_memory),
                 "Expected memory ID, version ID and content SHA-256 must be supplied together with --use-multi-memory")
            if all(supplied):
                need(args.expected_memory_id > 0 and args.expected_version_id > 0 and
                     re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_content_sha256) is not None,
                     "Expected memory IDs must be positive and content SHA-256 must be 64 hexadecimal characters")
            model = args.model or os.environ.get("OPENAI_MODEL", "").strip()
            endpoint = authorize_model(model, args.allow_external_model)
            release = approved_release(args.case, args.stage)
            check_standalone(release)
            if args.use_multi_memory or args.preview_multi_memory:
                candidates = retrieve_procedural_memories(args.case, release)
                review = call_memory_applicability(
                    release, candidates, model, allow_external_model=args.allow_external_model)
                selection = select_applicable_memories(release, candidates, review)
                if args.preview_multi_memory:
                    print(json.dumps({"case_id": args.case, "stage": args.stage,
                                      "decisions": [{key: item[key] for key in
                                                     ("memory_id", "version_id", "decision",
                                                      "selected", "content_sha256", "reason")}
                                                    for item in selection["selection_report"]["decisions"]],
                                      "conflicts": selection["selection_report"]["conflicts"]},
                                     ensure_ascii=True))
                    return 0
                selected = selection["selected_memories"]
                if all(supplied):
                    need(len(selected) == 1, "An expected memory pin requires exactly one selected memory")
                    memory = selected[0]
                    actual_hash = hashlib.sha256(memory["content"].encode("utf-8")).hexdigest()
                    need(memory["memory_id"] == args.expected_memory_id and
                         memory["version_id"] == args.expected_version_id and
                         actual_hash == args.expected_content_sha256.lower() and
                         memory["content_sha256"] == actual_hash,
                         "Selected procedural memory does not match the expected ID, version or content hash")
                record = run_agent(release, model, endpoint,
                                   procedural_memories=selected,
                                   selection_report=selection["selection_report"],
                                   allow_external_model=args.allow_external_model)
            else:
                record = run_agent(release, model, endpoint, allow_external_model=args.allow_external_model)
            try:
                episode_id = save_episode(record)
            except (ValueError, RuntimeError, OSError, UnicodeError) as exc:
                raise ValueError(f"Episode save unconfirmed | run_id={record['run_id']} | {exc}") from exc
            print(f"RESULT: RUN STORED | episode_id={episode_id} | run_id={record['run_id']}", flush=True)
            print(f"NOTE  status={record['execution_status']}; "+
                  f"stages={len(record['trace'])}; no run files written; no memory promotion")
            try:
                _, updated = retry_episode_embedding(record["run_id"])
            except (ValueError, RuntimeError, TypeError, KeyError, AttributeError, OSError, UnicodeError) as exc:
                print(f"RESULT: EMBEDDING PENDING | run_id={record['run_id']} | {exc}", file=sys.stderr)
            else:
                status = "STORED" if updated else "ALREADY STORED"
                print(f"RESULT: EMBEDDING {status} | episode_id={episode_id} | run_id={record['run_id']}")
            if record["execution_status"] != "COMPLETED":
                print("ERROR: A LangGraph stage failed. See DB replay for recorded error.", file=sys.stderr)
                return 1
        elif args.command == "verify":
            verify(args.case, args.stage, args.run_id)
        elif args.command == "replay":
            replay(args.case, args.stage, args.run_id)
        else:
            need(args.question.strip() != "", "Question cannot be empty")
            need(1 <= args.top <= 10, "Top must be 1-10")
            search(args.case, args.stage, args.question, args.top)
        return 0
    except (ValueError, RuntimeError, TypeError, KeyError, AttributeError, OSError, UnicodeError) as exc:
        print(f"RESULT: FAIL - {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
