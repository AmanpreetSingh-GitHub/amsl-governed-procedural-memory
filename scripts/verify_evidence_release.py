"""Verify one explicitly released genomics evidence snapshot without modifying data.

Usage: python scripts/verify_evidence_release.py experiments/EXP-001_CA1585329339_stage0"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from functools import cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NAMES = {"interpretation.json", "guideline_v2.json"}
EXPECTED_OUTCOMES = {
    "PVS1": "Met",
    "PM2_Supporting": "Met",
    "PM3": "Not Met",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def contained_path(parent: Path, relative: str) -> Path:
    require(isinstance(relative, str) and relative, "Missing path string")
    candidate = (parent / relative).resolve()
    require(candidate.is_relative_to(parent.resolve()), f"Path escapes allowed directory: {relative}")
    require(candidate.is_file(), f"Missing declared evidence file: {relative}")
    return candidate


def verify(experiment_argument: str, *, project_root: Path | None = None) -> tuple[dict, dict, dict, dict, dict]:
    root = project_root or PROJECT_ROOT
    experiments_root = (root / "experiments").resolve()
    scientific_root = (root / "scientific_data").resolve()

    @cache
    def read(path: Path) -> bytes:
        return path.read_bytes()

    def digest(path: Path) -> str:
        return hashlib.sha256(read(path)).hexdigest()

    def load_json(path: Path):
        try:
            return json.loads(read(path).decode("utf-8-sig"))
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"Invalid JSON: {path}: {error}") from error

    def validate_hash(path: Path, expected: str) -> None:
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-fA-F]{64}", expected) is not None,
                f"Invalid declared SHA-256 for {path.name}")
        require(digest(path).casefold() == expected.casefold(), f"SHA-256 mismatch: {path}")
    experiment = (Path.cwd() / experiment_argument).resolve()
    require(experiment.is_relative_to(experiments_root) and experiment.is_dir(),
            "Experiment must be an existing directory inside project experiments/")

    manifest = load_json(experiment / "manifest.json")
    receipt = load_json(experiment / "release_receipt.json")
    require(isinstance(manifest, dict) and isinstance(receipt, dict), "Manifest/receipt must be JSON objects")

    for field in ("experiment_id", "case_id", "stage"):
        require(manifest.get(field) == receipt.get(field), f"Manifest/receipt mismatch: {field}")
    case = manifest.get("case_id")
    require(isinstance(case, str) and re.fullmatch(r"CA[0-9]+", case) is not None,
            "Invalid case ID")
    require(manifest.get("stage") == 0, "This verifier accepts only Stage-0 releases")
    require(receipt.get("status") == "approved_for_first_investigation",
            "This snapshot is not approved for a first investigation")
    require(isinstance(receipt.get("released_at_utc"), str) and receipt["released_at_utc"],
            "Missing release timestamp")
    print("PASS  Release: matching case/stage and scoped approval")

    evidence = manifest.get("evidence")
    require(isinstance(evidence, list) and evidence == receipt.get("evidence"),
            "Manifest and receipt evidence lists differ")

    case_root = contained_path(scientific_root, f"active/variants/{case}/case_inventory.json").parent
    inventory = load_json(case_root / "case_inventory.json")
    require(isinstance(inventory, dict) and inventory.get("caid") == case,
            "Case inventory ID mismatch")
    require(isinstance(inventory.get("files"), list), "Invalid case inventory file list")
    policy = inventory.get("policy")
    require(isinstance(policy, str) and policy.strip(), "Missing inventory access policy")

    indexed = {}
    for item in inventory["files"]:
        require(isinstance(item, dict), "Invalid inventory entry")
        name = item.get("path")
        require(isinstance(name, str) and name and name.casefold() not in indexed,
                "Duplicate/invalid inventory path")
        indexed[name.casefold()] = (name, item.get("sha256"))
    require(all(name.casefold() in indexed for name in EXPECTED_NAMES),
            "Case inventory missing required Stage-0 files")

    documents = {}
    seen = set()
    for item in evidence:
        require(isinstance(item, dict), "Invalid manifest evidence entry")
        relative_snapshot = item.get("file")
        source = item.get("source")
        require(isinstance(relative_snapshot, str), "Invalid snapshot file")
        # - Restrict paths before resolving them so aliases cannot bypass the inventory.
        relative_snapshot = relative_snapshot.replace("\\", "/")
        require(relative_snapshot.startswith("evidence/"), "Unexpected snapshot evidence path")
        name = relative_snapshot.removeprefix("evidence/")
        allowed = (name in EXPECTED_NAMES or name == "criterion_rationale.json" or
                   re.fullmatch(r"clinvar/SCV[0-9]+\.[0-9]+\.json", name) is not None or
                   re.fullmatch(r"publications/[0-9]+\.json", name) is not None)
        require(allowed and name.casefold() not in seen, "Unexpected or duplicate evidence path")
        require(source == f"active/variants/{case}/{name}",
                "Source path exceeds the approved case scope")
        declared = item.get("sha256")
        inventory_entry = indexed.get(name.casefold())
        require(inventory_entry is not None and inventory_entry[0] == name and
                inventory_entry[1] == declared, f"Inventory path/hash differs for {name}")

        source_path = contained_path(scientific_root, source)
        snapshot_path = contained_path(experiment, relative_snapshot)
        validate_hash(source_path, declared)
        validate_hash(snapshot_path, declared)
        documents[name] = load_json(snapshot_path)
        seen.add(name.casefold())
    require({name.casefold() for name in EXPECTED_NAMES}.issubset(seen),
            "Missing expected evidence file")

    # - Declared files are not sufficient if the snapshot also contains undeclared material.
    evidence_root = experiment / "evidence"
    require(evidence_root.is_dir() and not evidence_root.is_symlink(), "Invalid evidence directory")
    actual = set()
    for path in evidence_root.rglob("*"):
        require(not path.is_symlink(), f"Symlink in evidence directory: {path}")
        relative = path.relative_to(evidence_root).as_posix()
        if path.is_dir():
            require(relative in ("clinvar", "publications"), f"Unexpected evidence directory: {relative}")
        else:
            require(path.is_file(), f"Invalid evidence file: {relative}")
            require(relative.casefold() not in actual, f"Duplicate evidence file: {relative}")
            actual.add(relative.casefold())
    require(actual == seen, "Evidence directory contains undeclared or missing files")
    print("PASS  Evidence: inventory, manifest, receipt, active and snapshot SHA-256")
    print("PASS  Scope: only declared, case-scoped evidence files")

    interpretation = documents["interpretation.json"]
    guideline = documents["guideline_v2.json"]
    require(isinstance(interpretation, dict) and isinstance(guideline, dict),
            "Interpretation and guideline must be JSON objects")
    require(interpretation.get("caid") == case, "Interpretation case ID mismatch")
    for field in ("interpretation_uuid", "version", "classification", "narrative_scope",
                  "narrative_provenance",
                  "guideline_assignment_provenance", "source_sepio_sha256",
                  "source_expert_page_sha256"):
        require(field in interpretation, f"Missing interpretation field: {field}")
    if case == "CA1585329339" or "known_uncertainties" in interpretation:
        require(isinstance(interpretation.get("known_uncertainties"), list),
                "Uncertainties must remain a list")
    criteria = interpretation.get("criteria")
    require(isinstance(criteria, list), "Interpretation criteria must be a list")
    outcomes = {}
    for item in criteria:
        require(isinstance(item, dict) and "label" in item and "outcome" in item,
                "Invalid criterion item")
        require(item["label"] not in outcomes, "Duplicate interpretation criterion")
        outcomes[item["label"]] = item["outcome"]
    if case == "CA1585329339":
        for label, expected in EXPECTED_OUTCOMES.items():
            require(outcomes.get(label) == expected, f"Unexpected source-reported outcome: {label}")
    require(guideline.get("guideline_id") == "GN010" and str(guideline.get("version")) == "2.0",
            "Not the expected GN010 v2.0 guideline")
    require(str(interpretation.get("guideline_version")) in ("2", "2.0"),
            "Interpretation does not reference guideline version 2")
    if "criterion_rationale.json" in documents:
        # - Excerpt review and the owner's release authorization are separate decisions.
        authorization = receipt.get("owner_authorization")
        require(isinstance(authorization, dict), "Missing owner authorization")
        for field in ("authority", "statement", "scope"):
            require(isinstance(authorization.get(field), str) and authorization[field].strip(),
                    f"Missing owner authorization {field}")
        require(authorization["authority"].strip().casefold() == "project owner",
                "Authorization must identify the project owner")
        require(re.fullmatch(r"agent memory (?:&|and) self-learning lab controlled poc(?: only)?",
                             authorization["scope"].strip(), re.IGNORECASE) is not None,
                "Authorization must identify the controlled Agent Memory & Self-Learning Lab POC")
        require(authorization.get("independent_clinical_review") is False,
                "Owner authorization must retain the independent clinical review limitation")
        validate_hash(experiment / "manifest.json", authorization.get("manifest_sha256"))
        rationale = documents["criterion_rationale.json"]
        require(isinstance(rationale, dict), "Criterion rationale must be a JSON object")
        require(rationale.get("status") == "REVIEWED_FOR_POC", "Criterion rationale is not reviewed for POC")
        require(rationale.get("case_id") == case, "Criterion rationale case ID mismatch")
        require(rationale.get("source_version") == interpretation["version"], "Criterion rationale version mismatch")
        provenance = interpretation["narrative_provenance"]
        require(isinstance(provenance, dict) and isinstance(provenance.get("source_url"), str)
                and provenance["source_url"].strip() and rationale.get("source_url") == provenance["source_url"],
                "Criterion rationale source URL mismatch")
        source = rationale.get("source_provenance")
        require(isinstance(source, dict) and isinstance(source.get("source_sha256"), str)
                and re.fullmatch(r"[0-9a-fA-F]{64}", source["source_sha256"]) is not None
                and source["source_sha256"] == interpretation["source_expert_page_sha256"]
                and source["source_sha256"] == provenance.get("source_sha256"),
                "Criterion rationale source provenance mismatch")
        review = rationale.get("review")
        require(isinstance(review, dict) and all(isinstance(review.get(k), str) and review[k].strip()
                for k in ("reviewer", "method")) and review.get("exact_excerpts_verified") is True,
                "Missing exact-excerpt review metadata")
        require(isinstance(rationale.get("independent_clinical_review"), bool)
                and isinstance(rationale.get("historical_availability_verified"), bool),
                "Missing review or historical-availability limitation")
        limitations = rationale.get("limitations")
        require(isinstance(limitations, list) and limitations and all(isinstance(x, str) and x.strip()
                for x in limitations), "Missing rationale limitations")
        excerpts = rationale.get("excerpts")
        require(isinstance(excerpts, list) and excerpts, "Missing rationale excerpts")
        labels = set()
        for excerpt in excerpts:
            require(isinstance(excerpt, dict), "Invalid rationale excerpt")
            label = excerpt.get("criterion")
            require(isinstance(label, str) and re.fullmatch(r"(?:PVS|PS|PM|PP|BA|BS|BP)[0-9]+(?:_[A-Za-z]+)?", label)
                    and label not in labels, "Invalid or duplicate rationale criterion")
            require(isinstance(excerpt.get("source_text"), str) and excerpt["source_text"].strip(),
                    "Missing rationale source text")
            labels.add(label)

    for name, document in documents.items():
        if not name.startswith(("clinvar/", "publications/")):
            continue
        require(isinstance(document, dict), f"Evidence must be a JSON object: {name}")
        if "caid" in document:
            require(document["caid"] == case, f"Evidence case ID mismatch: {name}")
        if name.startswith("clinvar/"):
            require(document.get("scv_accession_version") == Path(name).stem,
                    f"ClinVar accession/filename mismatch: {name}")
        else:
            require(str(document.get("pmid")) == Path(name).stem,
                    f"Publication PMID/filename mismatch: {name}")
            for field, interpretation_field in (("initial_interpretation_uuid", "interpretation_uuid"),
                                                ("initial_interpretation_version", "version")):
                if field in document:
                    require(document[field] == interpretation[interpretation_field],
                            f"Publication initial interpretation mismatch: {name}")
    print("PASS  Fidelity: GN010 v2.0 and preserved source fields")
    if case == "CA1585329339":
        print("PASS  Criteria: PVS1=Met, PM2_Supporting=Met, PM3=Not Met")
    if any(name.startswith("publications/") for name in documents):
        print("NOTE  Publication historical availability is not established")
    print(f"NOTE  agent_execution_verified={receipt.get('agent_execution_verified')!r} "
          "(separate from release approval)")
    print("RESULT: PASS (read-only; no database writes)")
    hashes = {
        "manifest": digest(experiment / "manifest.json"),
        "receipt": digest(experiment / "release_receipt.json"),
        "inventory": digest(case_root / "case_inventory.json"),
    }
    return manifest, receipt, inventory, documents, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Experiment directory relative to current working directory")
    args = parser.parse_args()
    try:
        verify(args.experiment)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"RESULT: FAIL - {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
