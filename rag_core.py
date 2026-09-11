"""
Pure RAG/triage logic for MediBot, deliberately decoupled from Streamlit,
FAISS, PyTorch, transformers, and PyPDF2.

Why this file exists: app.py previously imported all of those heavy
libraries at module level, which meant the actual decision logic (confidence
scoring, the in-scope/out-of-scope gate, risk fallback direction) could not
be unit tested without installing the full ML stack and standing up a
Streamlit session. That made "does this behave correctly" something you
could only check by manually clicking through the app.

Everything in this module takes its model-dependent pieces (an embedding
function, a text-generation function, a vector index) as PLAIN ARGUMENTS
(dependency injection), instead of importing and calling specific libraries
directly. app.py wires in the real ones (FAISS, HF Inference API / local
flan-t5). Tests wire in cheap fakes. The control-flow logic being tested is
identical in both cases.

Only faiss is still imported, and only lazily inside FaissIndexBackend, so
importing this module never requires faiss (or torch, or transformers) to be
installed at all.
"""

import hashlib
import numpy as np


# ---------------------------------------------------------------------------
# Vector index backends
# ---------------------------------------------------------------------------
# Both classes expose the same four operations, so retrieve_context() below
# can be written once against this interface and work with either backend.


class FaissIndexBackend:
    """Thin wrapper around a real FAISS IndexIDMap2(IndexFlatIP). Used in
    production. faiss is imported lazily here, not at module import time, so
    rag_core.py can be imported (and its pure functions tested) even in an
    environment where faiss isn't installed."""

    def __init__(self, dim: int):
        import faiss  # noqa: local import by design, see module docstring

        # macOS-specific fix, not a workaround for anything in this codebase:
        # PyTorch and FAISS each bundle their own copy of the OpenMP runtime
        # (libomp.dylib). When both are loaded in the same process and FAISS
        # enters a multi-threaded region (add/search), the two runtimes
        # collide and the process segfaults -- a long-standing, widely
        # documented issue in the FAISS/PyTorch macOS wheels, independent of
        # anything specific to this app. Restricting FAISS to one thread
        # means it never enters that parallel code path, avoiding the
        # collision entirely. At this project's scale (a handful of PDFs,
        # thousands of chunks at most) the cost is negligible --
        # sub-millisecond to low-millisecond per query, single-threaded.
        faiss.omp_set_num_threads(1)

        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def add_with_ids(self, vecs: np.ndarray, ids: np.ndarray):
        self._index.add_with_ids(vecs, ids)

    def search(self, query_vecs: np.ndarray, k: int):
        return self._index.search(query_vecs, k)

    def remove_ids(self, ids: np.ndarray):
        self._index.remove_ids(ids)


class NumpyBruteForceIndex:
    """Pure-numpy exact cosine-similarity search, with the exact same
    external shape as FaissIndexBackend. Zero dependency on faiss -- this is
    what test code (and this sandbox) uses to exercise the real retrieval
    control flow without needing FAISS installed."""

    def __init__(self, dim: int):
        self.dim = dim
        self._ids = np.empty((0,), dtype=np.int64)
        self._vecs = np.empty((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self._ids)

    def add_with_ids(self, vecs: np.ndarray, ids: np.ndarray):
        vecs = vecs.astype(np.float32)
        self._ids = np.concatenate([self._ids, ids.astype(np.int64)])
        self._vecs = (
            np.vstack([self._vecs, vecs]) if self._vecs.size else vecs
        )

    def search(self, query_vecs: np.ndarray, k: int):
        n = self.ntotal
        if n == 0:
            empty_scores = np.zeros((query_vecs.shape[0], 0), dtype=np.float32)
            empty_ids = np.full((query_vecs.shape[0], 0), -1, dtype=np.int64)
            return empty_scores, empty_ids

        sims = self._vecs @ query_vecs[0]
        k_eff = min(k, n)
        top_idx = np.argsort(-sims)[:k_eff]
        scores = sims[top_idx].reshape(1, -1).astype(np.float32)
        ids = self._ids[top_idx].reshape(1, -1)
        return scores, ids

    def remove_ids(self, ids: np.ndarray):
        mask = ~np.isin(self._ids, ids)
        self._ids = self._ids[mask]
        self._vecs = self._vecs[mask]


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Hashing & chunking (pure string/bytes logic, no ML dependency at all)
# ---------------------------------------------------------------------------


def compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def chunk_document(
    page_texts: list[str],
    filename: str,
    doc_hash: str,
    chunk_size: int = 150,
    overlap: int = 20,
) -> list[dict]:
    words: list[str] = []
    word_page_map: list[int] = []
    for page_num, text in enumerate(page_texts, start=1):
        page_words = text.split()
        words.extend(page_words)
        word_page_map.extend([page_num] * len(page_words))

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        page_num = word_page_map[start] if word_page_map else 1
        chunks.append(
            {"text": chunk_text, "source": filename, "page": page_num, "doc_hash": doc_hash}
        )
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Retrieval + confidence gate
# ---------------------------------------------------------------------------

NOT_COVERED_MSG = (
    "This specific topic is not covered in the current knowledge base. "
    "Please consult a qualified healthcare professional for guidance."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a careful medical information assistant. Answer ONLY using "
    "the provided context. If the context does not contain the answer, "
    "respond with exactly: UNAVAILABLE. Do not use outside knowledge."
)

CONFIDENCE_THRESHOLD = 30.0


def compute_confidence(best_score: float) -> float:
    """Maps a raw cosine similarity (roughly -1..1) onto a 10..98 display
    scale. Extracted as its own function specifically so its boundary
    behavior can be unit tested without needing a real index at all."""
    return round(max(10.0, min(98.0, best_score * 100)), 1)


def should_answer_from_context(
    confidence: float, has_chunks: bool, threshold: float = CONFIDENCE_THRESHOLD
) -> bool:
    """The single gate that decides 'answer from context' vs 'say we don't
    know'. Kept as one pure function so both retrieve-time and generate-time
    logic stay in sync, and so it's trivially testable in isolation."""
    return has_chunks and confidence >= threshold


def retrieve_context(
    index_backend, chunks_by_id: dict, embed_fn, query: str, k: int = 4, min_relative_score: float = 0.5
):
    """embed_fn(list[str]) -> np.ndarray of L2-normalized float32 vectors,
    one row per input string. index_backend is a FaissIndexBackend or
    NumpyBruteForceIndex (or anything with the same four methods).

    k=4 (raised from an original k=2) exists to avoid one document starving
    out another once the KB holds multiple documents. But k alone is a hard
    slot count with no relevance awareness -- on a SMALL knowledge base
    (e.g. a handful of chunks total), k=4 can mean "return literally every
    chunk in the KB" regardless of whether it has anything to do with the
    question, padding the generator's context with unrelated topics and
    measurably degrading answer quality (this reproduced in testing: asking
    about one condition pulled in three unrelated conditions' text purely
    because k allowed it).

    min_relative_score fixes this without undoing the k=4 change: a chunk is
    only kept if its score is at least min_relative_score times the TOP
    chunk's score for this query. A genuinely relevant document's chunks
    cluster near the top score; an unrelated document's chunks score
    meaningfully lower and get dropped here, even though k technically had
    room for them. This keeps k=4's benefit (room for multiple relevant
    documents) while removing its cost (padding with irrelevant ones).
    """
    if index_backend is None or index_backend.ntotal == 0:
        return [], 0.0

    q_vec = embed_fn([query])
    k_eff = min(k, index_backend.ntotal)
    scores, indices = index_backend.search(q_vec, k_eff)

    score_list = scores[0].tolist() if scores.shape[1] else []
    best_score = float(score_list[0]) if score_list else 0.0
    score_cutoff = best_score * min_relative_score if best_score > 0 else float("-inf")

    results = []
    seen_text = set()
    for score, idx in zip(score_list, indices[0]):
        if idx < 0:
            continue
        if score < score_cutoff:
            continue
        chunk = chunks_by_id.get(int(idx))
        if chunk is None or chunk["text"] in seen_text:
            continue
        seen_text.add(chunk["text"])
        results.append(chunk)

    confidence = compute_confidence(best_score)
    return results, confidence


def _looks_like_refusal(answer: str) -> bool:
    """True only if the model's ENTIRE response is (essentially) just the
    UNAVAILABLE token we instructed it to use for "can't answer from
    context" -- not merely CONTAINS that word somewhere in a longer
    response. A bare substring check ("UNAVAILABLE" in answer) caused false
    positives with reasoning-capable models (e.g. openai/gpt-oss-20b) that
    can surface visible chain-of-thought text discussing the possibility of
    refusing, even when their actual final verdict was to answer normally --
    reproduced in testing on a genuinely in-scope question, where a 56%
    confidence retrieval (clearly above threshold) still got discarded
    because the word appeared somewhere in the model's reasoning trace.
    A well-behaved model following the "respond with exactly: UNAVAILABLE"
    instruction should produce close to just that token when it truly can't
    answer, which this still catches correctly.
    """
    stripped = answer.strip().strip("\"'. ").strip()
    return stripped.upper() == "UNAVAILABLE"


def generate_answer(
    index_backend,
    chunks_by_id: dict,
    embed_fn,
    generate_fn,
    question: str,
    patient_info: str = "",
    k: int = 4,
    threshold: float = CONFIDENCE_THRESHOLD,
    max_new_tokens: int = 500,
) -> dict:
    """generate_fn(system_prompt, user_prompt, max_new_tokens) -> str.

    max_new_tokens defaults to 500. It was originally 300 (fine-seeming for
    plain factual Q&A) but reproduced truncated mid-sentence answers on
    BOTH the plain Q&A path and the symptom-assessment path with
    openai/gpt-oss-20b -- this model spends part of its budget on internal
    reasoning before the visible answer, so 300 was tight regardless of
    which call site was asking. Raising the shared default (rather than
    overriding it per call site) means any future call site gets adequate
    headroom automatically instead of needing to remember this.

    Critically: when should_answer_from_context() is False, generate_fn is
    NEVER CALLED. The "I don't know" path does not go anywhere near the
    language model -- it can't hallucinate an answer for a question that
    didn't retrieve anything relevant, because the model is never invoked
    for it. This is the property the test suite checks directly (by making
    the fake generate_fn raise if called, in the out-of-scope test case).
    """
    context_chunks, confidence = retrieve_context(index_backend, chunks_by_id, embed_fn, question, k=k)

    if not should_answer_from_context(confidence, bool(context_chunks), threshold):
        return {"answer": NOT_COVERED_MSG, "confidence": confidence, "sources": [], "raw_generation": None}

    clean_texts = [c["text"].replace('"""', "").replace('"', "").strip() for c in context_chunks]
    context = " ".join(clean_texts)
    patient_part = f" Patient details: {patient_info}." if patient_info else ""
    user_prompt = f"Context: {context}{patient_part}\n\nQuestion: {question}"

    answer = generate_fn(ANSWER_SYSTEM_PROMPT, user_prompt, max_new_tokens=max_new_tokens)

    # raw_generation carries the model's ACTUAL, UNMODIFIED output through to
    # the caller regardless of which branch fires below -- added specifically
    # because "the model said something that made our gate say not-covered"
    # was, before this, undiagnosable without re-running with print statements
    # scattered in ad hoc. The caller decides what to do with it (e.g. app.py
    # surfaces it in a debug expander only when the not-covered path fires).
    if not answer or len(answer) < 5 or _looks_like_refusal(answer):
        return {"answer": NOT_COVERED_MSG, "confidence": confidence, "sources": [], "raw_generation": answer}

    return {"answer": answer, "confidence": confidence, "sources": context_chunks, "raw_generation": answer}


def condense_question(history: list[dict], follow_up: str, generate_fn) -> str:
    """generate_fn(system_prompt, user_prompt, max_new_tokens) -> str.
    Fails open to the raw follow_up on any exception or malformed output --
    this is an optional quality enhancement, never a hard dependency for the
    turn to proceed."""
    if not history:
        return follow_up

    convo_lines = []
    for h in history[-6:]:
        role = "User" if h["role"] == "user" else "Assistant"
        text = h.get("plain_text", h["content"])
        convo_lines.append(f"{role}: {text}")
    convo_text = "\n".join(convo_lines)

    system_prompt = (
        "Rewrite the follow-up question as a single standalone question that "
        "makes sense without the conversation history. If it is already "
        "standalone, repeat it unchanged. Respond with ONLY the rewritten "
        "question, nothing else."
    )
    user_prompt = f"Conversation history:\n{convo_text}\n\nFollow-up: {follow_up}"

    try:
        # temperature=0.0: this is a rewrite task with one correct-ish
        # output, not a creative one -- sampling variance here previously
        # caused the SAME follow-up to occasionally get rewritten into
        # something that retrieved worse than the original, an intermittent
        # bug that was hard to pin down because it wasn't reproducible on
        # demand.
        rewritten = generate_fn(system_prompt, user_prompt, max_new_tokens=60, temperature=0.0).strip()
        if not rewritten or len(rewritten) > 300:
            return follow_up
        return rewritten
    except Exception:
        return follow_up


# ---------------------------------------------------------------------------
# Risk triage
# ---------------------------------------------------------------------------

HIGH_RISK = [
    "chest pain", "heart attack", "stroke", "difficulty breathing",
    "severe shortness of breath", "unconscious", "seizure", "coughing blood",
    "suicidal", "overdose", "severe bleeding",
]
MODERATE_RISK = [
    "shortness of breath", "high blood pressure", "elevated sugar", "fever",
    "dizziness", "headache", "nausea", "fatigue", "vomiting", "swelling",
    "blurred vision", "wheezing",
]
SYMPTOM_TRIGGERS = [
    "i have", "i feel", "i am feeling", "i've been", "i've had", "i'm having",
    "i'm feeling", "suffering from", "experiencing", "my chest", "my head",
    "my stomach", "my back", "my leg", "pain", "ache", "hurts", "breathless",
    "coughing", "dizzy", "tired", "nausea", "vomiting", "swollen", "blurred",
    "burning", "itching", "rash",
]

RISK_TAXONOMY_SYSTEM_PROMPT = (
    "You triage the urgency of a patient's described symptoms into exactly "
    "one of three levels.\n"
    "HIGH: signs of a possible medical emergency (e.g. cardiac, respiratory, "
    "or neurological emergency, severe bleeding, suicidal ideation).\n"
    "MODERATE: symptoms warranting medical evaluation soon but not "
    "immediately life-threatening.\n"
    "LOW: mild or non-urgent symptoms.\n"
    "If you are uncertain between two levels, choose the HIGHER risk level. "
    "Respond with exactly one word: HIGH, MODERATE, or LOW."
)

_RISK_META = {
    "HIGH": (
        "#ff4444",
        "You have described symptoms that could indicate a medical emergency. "
        "Please contact emergency services (111 / 999) or go to the nearest "
        "hospital immediately.",
    ),
    "MODERATE": (
        "#ffaa00",
        "You have described symptoms that require medical evaluation. Please "
        "consult a doctor if these symptoms persist or worsen.",
    ),
    "LOW": (
        "#00aa44",
        "No urgent risk indicators detected based on your description.",
    ),
}


def _risk_result(level: str) -> dict:
    border, explanation = _RISK_META[level]
    return {"level": level, "border": border, "explanation": explanation}


def detect_intent(message: str) -> str:
    if any(kw in message.lower() for kw in SYMPTOM_TRIGGERS):
        return "symptom_query"
    return "info_query"


def assess_risk(text: str, classify_fn=None) -> dict:
    """classify_fn(text) -> "HIGH"|"MODERATE"|"LOW"|None, or may raise.

    Layering, in order:
      1. Deterministic keyword scan for HIGH risk. Fires before anything
         else and never depends on classify_fn -- an emergency detector
         must not depend on a network call succeeding.
      2. classify_fn (LLM second pass), only reached if step 1 didn't fire.
         Any exception or unparseable result is treated as unavailable.
      3. Keyword scan for MODERATE risk, as the fail-safe when classify_fn
         is unavailable. Never silently falls all the way to LOW without
         checking this list first.
    """
    lower = text.lower()

    for kw in HIGH_RISK:
        if kw in lower:
            return _risk_result("HIGH")

    llm_level = None
    if classify_fn is not None:
        try:
            llm_level = classify_fn(text)
        except Exception:
            llm_level = None

    if llm_level in ("HIGH", "MODERATE", "LOW"):
        return _risk_result(llm_level)

    for kw in MODERATE_RISK:
        if kw in lower:
            return _risk_result("MODERATE")

    return _risk_result("LOW")