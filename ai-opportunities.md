# Where AI Could Fit in CoThink

CoThink connects students and mentors for local-language learning: Q&A, course/resource discovery, and mentor-guided collaborative projects. This document surveys where AI is a genuine fit — versus where a simpler, non-AI mechanism (search, filters, forms) already solves the problem just as well.

For each candidate, the question asked is: **does this need AI, or would a database query do?** Only ideas that clear that bar are worth building.

---

## 1. Mentor–Student Matching

**Problem:** Students need to find the right mentor for their goals; manually browsing mentor lists doesn't scale as the platform grows.

**Why AI (not just filters):** If mentor/student data is just structured tags (subject, level, availability), a filter/search UI is enough — no AI needed. AI earns its place only once matching depends on *unstructured* signal: a student's free-text goal description, a mentor's free-text bio/experience, and the fact that "good fit" is fuzzy (teaching style, pacing, prior similar students) rather than a clean tag match.

**Approach:** Embed student goal text and mentor bio/expertise text, rank mentors by similarity, optionally re-rank using structured filters (availability, price, language) as hard constraints.

**Data needed:** Mentor bios/expertise write-ups, student goal statements, ideally historical match outcomes (accepted/rejected, session ratings) to eventually train a better ranker.

**Complexity:** Medium. Cold-start problem — no historical outcome data on day one, so v1 is pure text-similarity, not learned ranking.

**Priority:** High — this is the platform's core value prop (right mentor, not just any mentor), so it's the most defensible use of AI.

---

## 2. Question Triage / Routing

**Problem:** Student questions come in unstructured; someone (mentor or admin) currently has to read and route each one to the right person or resource.

**Why AI:** Classifying free-text questions by topic/subject/difficulty is a genuine NLP task — can't be done with a dropdown unless you force students to self-categorize (which they often get wrong or skip).

**Approach:** Lightweight text classifier (or LLM call) tags incoming questions by subject/topic/urgency, routes to matching mentor or suggests existing FAQ/resource before a human is looped in.

**Data needed:** Historical questions + their correct category/mentor (for supervised training), or just a topic taxonomy if using zero-shot LLM classification.

**Complexity:** Low–Medium. Can start as a thin LLM-prompt classifier with no training data required.

**Priority:** Medium — valuable once question volume is high enough that manual triage is a bottleneck; premature if the platform has low traffic.

---

## 3. Resource/Course Recommendation

**Problem:** Students may not know which existing course/resource already answers their question.

**Why AI:** Basic recommendation ("students who viewed X also viewed Y") doesn't need AI — that's a co-occurrence query. AI adds value only if recommending based on the *content* of a student's question/goal against the *content* of resources (semantic search), not just click history.

**Approach:** Semantic search (embeddings) over course/resource content, surfaced when a student asks a question or browses.

**Data needed:** A real content library of courses/resources — this is a prerequisite, not something AI can bootstrap. If the resource catalog is thin, this feature has nothing to recommend.

**Complexity:** Low (semantic search is well-trodden), but **blocked on content existing first**.

**Priority:** Medium, and sequenced *after* the platform has enough resources to make search meaningful.

---

## 4. RAG Assistant Over Platform Content ("ask before you ask a mentor")

**Problem:** Mentors repeatedly answer the same beginner questions that are already covered in existing courses/FAQs.

**Why AI:** This is the most "obviously AI" item on the list — a retrieval-augmented chat assistant that answers from your own content, deflecting repetitive questions before they reach a mentor.

**Approach:** RAG pipeline — embed course/FAQ content, retrieve relevant chunks per student question, generate an answer grounded in that content (with citation back to the source), fall back to human mentor if confidence is low or content doesn't cover it.

**Data needed:** Same prerequisite as #3 — a real content library. Also needs a fallback/escalation path so wrong or missing answers don't erode trust.

**Complexity:** Medium–High. Retrieval quality, hallucination control, and knowing when to escalate to a human are the hard parts, not the LLM call itself.

**Priority:** High in impact but **should follow #1 or #3** — it needs both content to retrieve from and, ideally, validated matching so escalation has somewhere good to go.

---

## Candidates Considered and Rejected (for now)

- **Generic chatbot wrapper** ("talk to an AI tutor") — rejected as the *first* feature. It's a thin LLM wrapper with no platform-specific data behind it; any competitor can copy it in a weekend, and it doesn't use anything unique about CoThink (your mentors, your content, your matching). Worth revisiting later as an add-on, not a v1 feature.
- **AI-generated course content** — rejected for now. Editorial/quality risk is high, and it competes with the human-mentor value prop rather than supporting it.
- **Sentiment/engagement monitoring on mentor sessions** — plausible later (flagging disengaged students), but needs session transcript/interaction data that won't exist until the platform has real usage.

---

## Suggested Sequencing

1. **Mentor–student matching** — core differentiator, works from day one with just profile text, no historical data required.
2. **Question triage/routing** — once question volume creates a real bottleneck.
3. **Resource/content library reaches critical mass** — non-AI prerequisite work.
4. **Semantic resource recommendation** and **RAG assistant** — once #3 is true.

## Open Questions to Resolve Before Committing to a Build

- Is there a rough estimate of expected mentor/student volume at launch? (Affects whether matching or triage is more urgent.)
- Will mentor bios and student goals be free-text, structured tags, or both?
- Is there any existing content library (courses/FAQs), or does that need to be built first?
- Who owns "wrong AI answer" risk — is a human-in-the-loop fallback acceptable, or does every AI-touched interaction need a human review step at launch?
