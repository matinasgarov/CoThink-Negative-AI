"""HTTP service exposing the moderation gate.

This is the roadmap's "wrap the baseline model as a small API" step. Any
backend calls POST /moderate before showing user text to another student or
sending it to the LLM.

Design notes:

  The gate is loaded once at startup, not per request. Loading the classifier
  artifacts takes seconds; a request must not pay that.

  Non-allow decisions are appended to a JSONL audit log. The roadmap asks for
  blocked attempts to be logged "without storing them as normal content", so
  they go to this separate file rather than into the comment/question tables.

  The service never throws away the reason for a decision. Every response says
  which signal fired and, for lexicon hits, exactly which word matched -- so a
  moderator reviewing a queue can act without re-running anything.

Run:
    uvicorn api:app --reload
    curl -X POST localhost:8000/moderate -H "Content-Type: application/json" \
         -d '{"text": "s3n w3r3fs1z", "surface": "public"}'
"""

import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from moderation_gate import ALLOW, PRIVATE, PUBLIC, ModerationGate, apply_policy

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT_LOG = os.path.join(HERE, "moderation_audit.jsonl")
MAX_TEXT_LENGTH = 10_000
MAX_BATCH_SIZE = 100

state = {"gate": None, "loaded_at": None}


@asynccontextmanager
async def lifespan(_app):
    started = time.perf_counter()
    state["gate"] = ModerationGate.load()
    state["loaded_at"] = datetime.now(timezone.utc).isoformat()
    print(f"moderation gate ready in {time.perf_counter() - started:.1f}s "
          f"({len(state['gate'].lexicon)} lexicon forms, "
          f"classifier={'yes' if state['gate'].has_classifier else 'no'})")
    yield
    state["gate"] = None


app = FastAPI(
    title="CoThink Moderation API",
    description="Lexicon + classifier moderation gate for Azerbaijani text.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ModerateRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    surface: Literal["private", "public"] = Field(
        PUBLIC,
        description="private = seen only by its author (a question to the AI); "
                    "public = seen by other students. Defaults to the stricter "
                    "of the two.",
    )
    context: str | None = Field(
        None, max_length=200,
        description="Optional caller-supplied tag recorded in the audit log, "
                    "e.g. 'comment:1234'.",
    )


class LexiconMatch(BaseModel):
    token: str
    word: str
    category: str
    severity: int


class ModerateResponse(BaseModel):
    action: Literal["block", "review", "allow"]
    proceed: bool = Field(..., description="Whether the caller may show or use this text.")
    outcome: Literal["blocked", "held_for_review", "allowed_flagged", "allowed"]
    reason: str
    toxicity_score: float
    lexicon_severity: int
    lexicon_matches: list[LexiconMatch]
    flagged_categories: dict[str, float]
    advisory_categories: list[str] = Field(
        ..., description="Sparse categories shown to reviewers but never acted on."
    )
    latency_ms: float


class BatchRequest(BaseModel):
    items: list[ModerateRequest] = Field(..., max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    results: list[ModerateResponse]


class HealthResponse(BaseModel):
    status: str
    lexicon_forms: int
    classifier_loaded: bool
    loaded_at: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _gate():
    gate = state["gate"]
    if gate is None:
        raise HTTPException(status_code=503, detail="moderation gate not loaded")
    return gate


def _record(request, response):
    """Append non-allow decisions to the audit log.

    Kept deliberately separate from normal content storage: a blocked comment
    should be reviewable without ever having been published.
    """
    if response["action"] == ALLOW:
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "surface": request.surface,
        "context": request.context,
        "action": response["action"],
        "outcome": response["outcome"],
        "reason": response["reason"],
        "toxicity_score": response["toxicity_score"],
        "matched_words": sorted({m["word"] for m in response["lexicon_matches"]}),
        "text": request.text,
    }
    try:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        # An unwritable log must never block moderation itself.
        print(f"warning: could not write audit log: {exc}")


def _moderate_one(request):
    started = time.perf_counter()
    decision = _gate().moderate(request.text)
    proceed, outcome = apply_policy(decision, request.surface)
    response = {
        **decision,
        "proceed": proceed,
        "outcome": outcome,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    response.pop("allowed", None)
    _record(request, response)
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    gate = state["gate"]
    return {
        "status": "ok" if gate else "loading",
        "lexicon_forms": len(gate.lexicon) if gate else 0,
        "classifier_loaded": bool(gate and gate.has_classifier),
        "loaded_at": state["loaded_at"],
    }


@app.post("/moderate", response_model=ModerateResponse)
def moderate(request: ModerateRequest):
    """Moderate a single piece of user text."""
    return _moderate_one(request)


@app.post("/moderate/batch", response_model=BatchResponse)
def moderate_batch(request: BatchRequest):
    """Moderate many items at once, e.g. re-scanning an existing archive."""
    return {"results": [_moderate_one(item) for item in request.items]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
