"""
DEMO: end-to-end shape of the moderated video-Q&A pipeline.

This is a walkthrough, not production code. It shows the full chain:

    lexicon + classifier -> moderation gate -> RAG-style Q&A

Two things are deliberately simplified for the demo (called out inline):
  1. Retrieval uses naive keyword overlap instead of real embeddings/vector DB.
  2. The "LLM answer" step just prints the prompt that would be sent to
     Claude (or another LLM) instead of making a real API call.

Everything else -- the moderation gate and how it sits in front of the Q&A
call -- is the real integration pattern.

The gate returns block / review / allow rather than a boolean (see
moderation_gate.py for why). What an app does with `review` depends on who
sees the text, which is the surface distinction below:

  private surface (a student's question to the AI, seen by nobody else)
      -> answer it, but log the flag. Blocking here on a 0.796-precision
         signal would refuse a lot of legitimate questions for no safety gain.
  public surface (comments, chat, anything other students read)
      -> hold for a human. The cost of showing abuse to another student is
         much higher than the cost of a short delay.

Run: python demo_moderated_qa_pipeline.py
"""

from moderation_gate import PRIVATE, PUBLIC, ModerationGate, apply_policy


# ---------------------------------------------------------------------------
# 2. Mock "video knowledge base" + naive retrieval.
#    In the real system this is: transcribe videos -> chunk -> embed ->
#    store in a vector DB (pgvector/Pinecone/etc) -> similarity search.
# ---------------------------------------------------------------------------

MOCK_VIDEO_CHUNKS = [
    "In this lesson the mentor explains that a Python list is mutable, "
    "meaning its contents can change after creation, unlike a tuple.",
    "The mentor walks through binary search: repeatedly halving a sorted "
    "array to find a target value in O(log n) time.",
    "This section covers gradient descent: an optimization algorithm that "
    "updates model weights in the direction that reduces the loss function.",
]


def retrieve_relevant_chunks(question, chunks=MOCK_VIDEO_CHUNKS, top_k=1):
    question_words = set(question.lower().split())
    scored = [
        (len(question_words & set(chunk.lower().split())), chunk)
        for chunk in chunks
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k] if score > 0] or [chunks[0]]


# ---------------------------------------------------------------------------
# 3. The gated Q&A entry point: moderate first, call the LLM only if the text
#    clears the gate for this surface.
# ---------------------------------------------------------------------------

def ask_question(question, gate, surface=PRIVATE):
    decision = gate.moderate(question)
    proceed, outcome = apply_policy(decision, surface)

    if not proceed:
        message = (
            "Your question was blocked by our content policy."
            if outcome == "blocked"
            else "Your question is being reviewed by a moderator."
        )
        return {"answered": False, "outcome": outcome,
                "message": message, "moderation": decision}

    context_chunks = retrieve_relevant_chunks(question)
    prompt = (
        "You are a helpful teaching assistant. Answer the student's "
        "question using only the video excerpt below.\n\n"
        f"Video excerpt:\n{context_chunks[0]}\n\n"
        f"Student question: {question}\n\nAnswer:"
    )

    # In production this is where you'd call the LLM, e.g.:
    #   response = anthropic_client.messages.create(
    #       model="claude-sonnet-5", messages=[{"role": "user", "content": prompt}]
    #   )
    #   answer = response.content[0].text
    return {"answered": True, "outcome": outcome,
            "prompt_sent_to_llm": prompt, "moderation": decision}


def post_comment(text, gate):
    """Comments are a public surface, so `review` holds instead of publishing."""
    decision = gate.moderate(text)
    proceed, outcome = apply_policy(decision, PUBLIC)
    return {"published": proceed, "outcome": outcome, "moderation": decision}


# ---------------------------------------------------------------------------
# 4. Demo run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading moderation gate (lexicon + trained classifier) ...")
    gate = ModerationGate.load()
    print(f"Ready: {len(gate.lexicon)} lexicon forms, "
          f"classifier={'yes' if gate.has_classifier else 'no'}\n")

    print("=" * 78)
    print("STUDENT QUESTIONS TO THE AI (private surface)")
    print("=" * 78)
    for question in [
        "Can you explain what binary search is?",
        "Mən erməni dilini öyrənirəm, bu barədə video varmı?",  # benign
        "Sen kimsen? Pay atonan!!!",
        "s3n w3r3fs1z",                                         # obfuscated
    ]:
        result = ask_question(question, gate, surface=PRIVATE)
        print(f"\n--- {question!r}  [{result['outcome']}]")
        if result["answered"]:
            print("    passed the gate; prompt would go to the LLM")
        else:
            print(f"    {result['message']}")
        m = result["moderation"]
        print(f"    reason: {m['reason']}  (toxicity={m['toxicity_score']})")

    print("\n" + "=" * 78)
    print("THE SAME TEXT AS A PUBLIC COMMENT")
    print("=" * 78)
    for text in ["Bu dərs çox faydalı idi", "Sen kimsen? Pay atonan!!!", "s3n w3r3fs1z"]:
        result = post_comment(text, gate)
        state = "published" if result["published"] else "NOT published"
        print(f"\n--- {text!r}\n    {state}  [{result['outcome']}]  "
              f"{result['moderation']['reason']}")
