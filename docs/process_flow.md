# AMSL | Process Flow

```mermaid
flowchart TB
    PACK["Approved evidence pack<br/>manifest / receipt / evidence JSON"]
    VERIFY["Verify release"]
    INGEST["Ingest approved evidence"]
    SCI[("scientific schema<br/>sources / releases / release_evidence")]

    PACK --> VERIFY --> INGEST --> SCI

    RUN["run_investigation_to_db.py<br/>Load approved release"]
    SCI --> RUN
    CHOICE{"Memory selection mode?"}
    RUN --> CHOICE

    RETR["procedural_retrieval.py<br/>pgvector + approved origin checks"]
    APP["LLM applicability + conflicts<br/>APPLY / SKIP / UNCERTAIN"]
    SEL["Python selection<br/>0–3 compatible memories"]
    PREV["Preview result<br/>no investigation / no episode write"]
    WORKER["model_dispatch.py<br/>Docker agent worker"]

    CHOICE -->|OFF| WORKER
    CHOICE -->|ON or preview| RETR --> APP --> SEL
    SEL -->|preview| PREV
    SEL -->|ON run| WORKER

    I["Investigate"] --> R["Reflect"] --> V["Validate<br/>exact scientific JSON Pointers"]
    WORKER --> I
    V --> SAVE["episode_persistence.py<br/>Save full run_record + trace"]
    SAVE --> EP[("agent_memory.episodes<br/>full JSON and record hash")]
    EP --> EVEC["Ollama: embed saved episode"]
    EVEC --> EP

    OP["Operator"] --> GOV["govern_procedural_lessons.py<br/>Assess saved eligible episode"]
    EP -. "operator selects episode" .-> GOV
    GOV --> DEC["LLM recommendation<br/>NEW / REFINE / REINFORCE /<br/>SUPERSEDE / INVALIDATE / NO_CHANGE"]
    DEC -->|NEW or REFINE via draft-learning| DRAFT["Store inactive candidate/version<br/>and procedural-memory embedding"]
    DEC -->|Other recommendations| REVIEW["Operator decides next action<br/>no automatic draft"]
    DRAFT --> PROM["Explicit operator promotion"]
    OP --> PROM
    PROM --> ACTIVE[("Active approved memory version")]
    OP -->|rollback to earlier approved version| ACTIVE
    OP -->|retire| RETIRED["Retired memory"]
    OP -->|reinforce observation| AUDIT[("Operator audit records")]
    PROM --> AUDIT
    ACTIVE --> RETR

    MVEC[("agent_memory.embeddings<br/>procedural version vectors")]
    DRAFT --> MVEC
    MVEC --> RETR

    SCI_VEC["Explicit scientific indexing"] --> SVEC[("scientific.source_embeddings")]
    SCI -. "approved sources" .-> SCI_VEC
    EP -. "explicit semantic episode search" .-> ES["Episode search"]

    classDef data fill:#eef7f1,stroke:#25794b,color:#153d2e;
    classDef compute fill:#edf3ff,stroke:#2c5dab,color:#193865;
    classDef governance fill:#f5efff,stroke:#7951b7,color:#433164;
    class SCI,EP,ACTIVE,MVEC,SVEC,AUDIT data;
    class VERIFY,INGEST,RUN,RETR,APP,SEL,WORKER,I,R,V,SAVE,EVEC compute;
    class OP,GOV,DEC,DRAFT,PROM,REVIEW,RETIRED governance;
```

## Execution boundaries

- The runner uses the **approved scientific release in PostgreSQL**, not unchecked local evidence files.
- Memory applicability is an **LLM call outside LangGraph**; the selected procedures are guidance, not scientific evidence.
- `--preview-multi-memory` runs retrieval and applicability but stops before investigation and persistence.
- A completed investigation is **saved before** its episode embedding is attempted; embedding can be retried.
- Learning assessment and memory governance are **operator-initiated**, not invoked automatically by the runner. Only `NEW` and `REFINE` recommendations can create inactive drafts through `draft-learning`.
- Procedural-memory embeddings are stored with candidate versions; **active approved versions** are eligible for reuse.
