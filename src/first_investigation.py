"""Investigation stages for the approved PostgreSQL runner."""

import json
import os
import re
from typing import Literal, NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class InvestigationState(TypedDict):
    """Working state shared across the agent's stages."""

    evidence: dict
    procedural_memories: NotRequired[list[dict[str, str]]]
    report: str
    reflection: str
    validation: dict


class ValidationResult(BaseModel):
    """Structured outcome of the lesson-review stage."""

    decision: Literal["APPROVE", "REJECT", "DEFER"]
    grounded: bool
    reusable: bool
    evidence_references: list[str]
    contradictions: list[str]
    rationale: str
    conditions_for_reuse: str


def investigate(state: InvestigationState) -> dict:
    """Investigate the approved scientific evidence."""

    if "procedural_memories" in state:
        lessons = state["procedural_memories"]
        if (not isinstance(lessons, list) or len(lessons) > 3 or
                any(not isinstance(lesson, dict) or
                    any(not isinstance(lesson.get(key), str) or not lesson[key].strip()
                        for key in ("title", "content")) for lesson in lessons)):
            raise ValueError("Invalid procedural memories")

    model = ChatOpenAI(model=os.environ["OPENAI_MODEL"])

    system_instructions = (
        "You are a scientific evidence investigation agent. "
        "Work exclusively from the supplied records. "
        "Distinguish original interpretations from your observations. "
        "Identify criteria, limitations, uncertainties and questions. "
        "Cite source JSON filenames and fields. "
        "Do not invent findings or use outside scientific knowledge. "
        "Do not offer clinical decisions."
    )
    if state.get("procedural_memories"):
        system_instructions += (
            "\n\nProcedural guidance (review methods, NOT scientific evidence):"
        )
        for index, lesson in enumerate(state["procedural_memories"], start=1):
            system_instructions += (
                f"\n\nProcedural guidance {index}:\n"
                + json.dumps({"title": lesson["title"], "content": lesson["content"]},
                             ensure_ascii=False)
            )
        system_instructions += (
            "\nAll scientific findings must be grounded in and cited from "
            "the approved scientific JSON records."
        )

    response = model.invoke([
        (
            "system",
            system_instructions
        ),
        (
            "human",
            "Investigate the approved evidence and produce "
            "a structured report:\n\n"
            + json.dumps(state["evidence"], ensure_ascii=False)
        ),
    ])

    return {"report": str(response.content)}


def reflect(state: InvestigationState) -> dict:
    """Propose a reusable procedural lesson without hardcoding it."""

    model = ChatOpenAI(model=os.environ["OPENAI_MODEL"])

    response = model.invoke([
        (
            "system",
            "You are continuing your own investigation in a "
            "reflection stage. Examine your completed report and "
            "independently propose at most ONE reusable procedural "
            "lesson that could improve future investigations. "
            "Ground it in actual observations. "
            "Do not invent experiences, evidence or findings. "
            "Do not propose a new clinical classification. "
            "If no useful generalizable lesson is justified, "
            "explicitly ABSTAIN. "
            "A proposal is not automatically validated. "

            # Keep the first line machine-readable.
            "Your response MUST start with exactly one of these lines: "
            "'DECISION: PROPOSE' or 'DECISION: ABSTAIN'. "
            "Put nothing before that line, including Markdown headings. "

            "Then use these headings: "
            "Candidate lesson, "
            "Origin in this investigation, "
            "Conditions for reuse, "
            "Limitations, "
            "Validation required. "
            "If you abstain, briefly explain why no reusable "
            "lesson was justified."
        ),
        (
            "human",
            "Reflect on your completed investigation report. "
            "Do not conduct a new investigation:\n\n"
            + state["report"]
        ),
    ])

    return {"reflection": str(response.content)}


def reference_exists(reference: str, evidence: dict) -> bool:
    """Resolve an exact evidence key and a JSON Pointer to a field."""
    if not isinstance(reference, str):
        return False
    name, separator, pointer = reference.partition("#/")
    if not separator or name not in evidence or re.search(r"~(?![01])", pointer):
        return False
    current = evidence[name]
    for token in pointer.split("/"):
        part = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif (isinstance(current, list) and len(part) <= len(str(len(current))) and
              re.fullmatch(r"0|[1-9][0-9]*", part) and int(part) < len(current)):
            current = current[int(part)]
        else:
            return False
    return True


def validate(state: InvestigationState) -> dict:
    """Review lesson grounding, reusability and contradictions."""

    # Read the first line of the reflection.
    # Unexpected or missing decisions must not enter validation.
    first_line = state["reflection"].splitlines()
    decision = first_line[0].strip() if first_line else ""

    if decision != "DECISION: PROPOSE":
        return {
            "validation": {
                "decision": "DEFER",
                "grounded": False,
                "reusable": False,
                "evidence_references": [],
                "contradictions": [],
                "rationale": (
                    "Reflection abstained or returned an invalid decision."
                ),
                "conditions_for_reuse": "None",
            }
        }

    model = ChatOpenAI(model=os.environ["OPENAI_MODEL"])

    # Structured validation allows us to inspect each decision field.
    reviewer = model.with_structured_output(
        ValidationResult,
        method="function_calling",
    )

    review = reviewer.invoke([
        (
            "system",
            "You are reviewing a proposed procedural lesson. "
            "Be skeptical and evaluate it using ONLY the supplied "
            "source records and investigation report. "
            "Do not assume the proposing agent is correct. "
            "Check whether its claimed origin is supported by "
            "specific source fields, whether the procedure could "
            "generalize to other investigations, and whether "
            "contradictions or missing context undermine it. "
            "APPROVE only when the procedural lesson is grounded, "
            "reusable and free of unresolved contradictions that "
            "invalidate the procedure. "
            "For evidence_references, provide exact JSON Pointer paths to fields in the supplied "
            "scientific evidence, for example interpretation.json#/criteria/0/outcome. "
            "Use a filename that exactly matches a key in evidence, followed by #/ and the field path. "
            "Use zero-based array indices and JSON Pointer escapes (~0 for ~ and ~1 for /). "
            "Do not cite bare filenames, descriptive report headings, prose summaries, or invented paths. "
            "REJECT unsupported or misleading lessons. "
            "DEFER when important information is insufficient. "
            "A discrepancy between records is not proof that "
            "either scientific record is wrong. "
            "Do not make clinical decisions."
        ),
        (
            "human",
            "Review the proposed lesson against these records:\n\n"
            + json.dumps(
                {
                    "evidence_source_filenames": sorted(state["evidence"]),
                    "evidence": state["evidence"],
                    "investigation_report": state["report"],
                    "candidate_reflection": state["reflection"],
                },
                ensure_ascii=False,
            )
        ),
    ])

    validation = review.model_dump()

    # Never accept an internally inconsistent approval.
    if validation["decision"] == "APPROVE":
        if (
            not validation["grounded"]
            or not validation["reusable"]
            or validation["contradictions"]
            or not validation["evidence_references"]
            or not all(reference_exists(ref, state["evidence"]) for ref in validation["evidence_references"])
        ):
            validation["decision"] = "DEFER"
            validation["rationale"] += (
                " Approval withheld because required checks "
                "were not all satisfied."
            )

    return {"validation": validation}
