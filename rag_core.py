"""Pure RAG/triage logic, decoupled from Streamlit/FAISS/torch/transformers so
it can be unit tested without the ML stack. Models are passed in as arguments."""

import re
import hashlib
import numpy as np


# ---------------------------------------------------------------------------
# Vector index backends (same four methods, so retrieve_context works with either)
# ---------------------------------------------------------------------------


class FaissIndexBackend:
    """Production FAISS IndexIDMap2(IndexFlatIP); faiss imported lazily."""

    def __init__(self, dim: int):
        import faiss  # noqa: local import by design (see module docstring)

        # torch and faiss each bundle their own libomp; letting both spin up
        # OpenMP worker threads segfaults on macOS, so pin faiss to one thread.
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
    """Pure-numpy cosine search with the same shape as FaissIndexBackend; used
    by tests so retrieval control flow runs without faiss installed."""

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
# Hashing & chunking (pure string/bytes logic)
# ---------------------------------------------------------------------------


def compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


# Headers used to split a KB into focused chunks: a quoted topic title
# ("Asthma":) and, within a topic, its subsection labels. KBs without these
# fall back to fixed-size word chunks.
_HEADER_TOKEN_RE = re.compile(r'^"[^"]+":?$')
_SUBSECTION_RE = re.compile(
    r"(?=(?:Causes / Risk Factors:|Symptoms:|Diagnosis:|Treatment / Management:|Prevention:))"
)
_NAME_RE = re.compile(r'^"?([A-Za-z][A-Za-z /]*?)"?\s*:')


def chunk_document(
    page_texts: list[str],
    filename: str,
    doc_hash: str,
    chunk_size: int = 150,
    overlap: int = 20,
) -> list[dict]:
    # Split on quoted topic headers, then on subsection labels, so a terse query
    # ("I have a cough") matches a focused "Symptoms" chunk rather than a whole
    # condition (measured: lifts that query's similarity ~29% -> ~38%). The
    # condition name is prefixed so each piece keeps its context.
    step = max(1, chunk_size - overlap)
    chunks = []

    words, word_page = [], []
    for page_num, text in enumerate(page_texts, start=1):
        pw = text.split()
        words.extend(pw)
        word_page.extend([page_num] * len(pw))

    def add(text: str, page: int):
        w = text.split()
        s = 0
        while s < len(w):
            ct = " ".join(w[s:s + chunk_size]).strip()
            if ct:
                chunks.append({"text": ct, "source": filename, "page": page, "doc_hash": doc_hash})
            s += step

    header_idxs = [i for i, w in enumerate(words) if _HEADER_TOKEN_RE.match(w)]
    if len(header_idxs) >= 2:
        bounds = header_idxs + [len(words)]
        spans = ([(0, header_idxs[0])] if header_idxs[0] > 0 else []) + list(zip(bounds, bounds[1:]))
        for lo, hi in spans:
            page = word_page[lo] if lo < len(word_page) else 1
            section = " ".join(words[lo:hi]).strip()
            if not section:
                continue
            nm = _NAME_RE.match(section)
            name = nm.group(1).strip() if nm else ""
            for part in _SUBSECTION_RE.split(section):
                part = part.strip()
                if not part:
                    continue
                keep = part if (not name or part.lower().startswith(name.lower())) else f"{name} - {part}"
                add(keep, page)
    else:
        # Fallback: word-chunk each page independently.
        for page_num, text in enumerate(page_texts, start=1):
            add(text, page_num)

    return chunks


# ---------------------------------------------------------------------------
# Retrieval + confidence gate
# ---------------------------------------------------------------------------

NOT_COVERED_MSG = (
    "I don't have enough knowledge about this in my current knowledge base "
    "to answer reliably. Please consult a qualified healthcare professional "
    "for guidance."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a careful medical information assistant. Answer ONLY using "
    "the provided context. If the context does not contain the answer, "
    "respond with exactly: UNAVAILABLE. Do not use outside knowledge."
)

# Gate on the compute_confidence (cosine x 100) scale. 35 (= cosine 0.35):
# MiniLM scores run low on this short/dense KB (a clearly-relevant multi-symptom
# query measured ~0.41), so 40 clipped legitimate terse symptoms like "I have a
# cough". Off-KB terms still score ~0 and stay rejected. Tunable via a golden
# eval; lower = more recall, higher = stricter "I don't have enough knowledge".
CONFIDENCE_THRESHOLD = 35.0


def compute_confidence(best_score: float) -> float:
    """Raw cosine -> 10..98 scale. This is the GATE value; move the threshold
    too if you rescale it."""
    return round(max(10.0, min(98.0, best_score * 100)), 1)


def display_confidence(best_score: float) -> float:
    """UI-only calibration: MiniLM's strong matches sit ~0.5-0.65, so map the
    [0.30, 0.70] band onto [55, 90] for display. Modest, not inflated; gate unchanged."""
    lo_c, hi_c, lo_p, hi_p = 0.30, 0.70, 55.0, 90.0
    frac = (best_score - lo_c) / (hi_c - lo_c)
    return round(max(lo_p, min(hi_p, lo_p + frac * (hi_p - lo_p))), 1)


def should_answer_from_context(
    confidence: float, has_chunks: bool, threshold: float = CONFIDENCE_THRESHOLD
) -> bool:
    """Single gate for 'answer from context' vs 'say we don't know'."""
    return has_chunks and confidence >= threshold


def retrieve_context(
    index_backend, chunks_by_id: dict, embed_fn, query: str, k: int = 4, min_relative_score: float = 0.5
):
    # k=4 leaves room for multiple relevant docs; min_relative_score then drops
    # any chunk scoring below half the top chunk, so k's slack can't pad context
    # with unrelated topics on a small KB.
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
    """True only when the whole reply is essentially just the UNAVAILABLE token
    (a bare substring check false-positived on models that reason out loud)."""
    stripped = answer.strip().strip("\"'. ").strip()
    return stripped.upper() == "UNAVAILABLE"


def extractive_answer(context_chunks: list[dict], max_chunks: int = 2, max_chars: int = 600) -> str:
    """Grounded answer built straight from retrieved passages (no model, so no
    hallucination), used when retrieval succeeded but the generator failed/refused."""
    texts = []
    for chunk in context_chunks[:max_chunks]:
        cleaned = chunk["text"].replace('"""', "").replace('"', "").strip()
        if cleaned:
            texts.append(cleaned)
    combined = " ".join(texts).strip()
    if not combined:
        return NOT_COVERED_MSG
    if len(combined) > max_chars:
        truncated = combined[:max_chars]
        cut = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))  # trim to a sentence boundary
        combined = (truncated[: cut + 1] if cut > 0 else truncated).strip() + " …"
    return combined


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
    retrieval_query: str | None = None,
) -> dict:
    """generate_fn(system_prompt, user_prompt, max_new_tokens) -> str. Returns a
    `mode`: not_covered | extractive | llm. When the gate fails, generate_fn is
    never called. retrieval_query (defaults to question) retrieves on the
    complaint alone so a patient profile can't skew which topic is retrieved."""
    rquery = retrieval_query if retrieval_query is not None else question
    context_chunks, confidence = retrieve_context(index_backend, chunks_by_id, embed_fn, rquery, k=k)

    # True out-of-scope: nothing relevant retrieved. Only path that says "not
    # covered"; generator never called; no relevance figure shown.
    if not should_answer_from_context(confidence, bool(context_chunks), threshold):
        return {
            "answer": NOT_COVERED_MSG, "confidence": confidence, "display_confidence": 0.0,
            "sources": [], "raw_generation": None, "mode": "not_covered",
        }

    disp = display_confidence(confidence / 100.0)

    clean_texts = [c["text"].replace('"""', "").replace('"', "").strip() for c in context_chunks]
    context = " ".join(clean_texts)
    patient_part = f" Patient details: {patient_info}." if patient_info else ""
    user_prompt = f"Context: {context}{patient_part}\n\nQuestion: {question}"

    gen_error = None
    try:
        answer = generate_fn(ANSWER_SYSTEM_PROMPT, user_prompt, max_new_tokens=max_new_tokens)
    except Exception as e:
        # Both backends failed; don't crash the turn -- treat as a refusal and
        # fall back to the retrieved passage below.
        answer = None
        gen_error = f"{type(e).__name__}: {e}"

    # Retrieval found relevant content but generation came back empty/short/
    # refusing: show the passage (grounded) rather than a false "not covered".
    if not answer or len(answer) < 5 or _looks_like_refusal(answer):
        return {
            "answer": extractive_answer(context_chunks), "confidence": confidence,
            "display_confidence": disp, "sources": context_chunks,
            "raw_generation": answer if answer else gen_error, "mode": "extractive",
        }

    return {
        "answer": answer, "confidence": confidence, "display_confidence": disp,
        "sources": context_chunks, "raw_generation": answer, "mode": "llm",
    }


def condense_question(history: list[dict], follow_up: str, generate_fn) -> str:
    """Rewrite a follow-up into a standalone question; fails open to the raw
    follow-up on any error (optional quality step, never a hard dependency)."""
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
        # temperature=0.0: deterministic rewrite; sampling here caused flaky retrieval.
        rewritten = generate_fn(system_prompt, user_prompt, max_new_tokens=60, temperature=0.0).strip()
        if not rewritten or len(rewritten) > 300:
            return follow_up
        return rewritten
    except Exception:
        return follow_up


# ---------------------------------------------------------------------------
# Risk triage
#
# These lists are GENERAL medical triage/intent signals, NOT tied to which KB
# PDF is loaded -- an emergency ("chest pain") and a self-report ("i have ...")
# mean the same thing regardless of the knowledge base. So swapping the KB does
# not require editing them. (Retrieval/grounding above is already KB-agnostic.)
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

# First-person self-report -> symptom flow; checked before the info test so it
# wins even when phrased as a question.
SELF_REPORT_TRIGGERS = [
    "i have", "i've got", "i have got", "i feel", "i am feeling", "i'm feeling",
    "i've been", "i've had", "i'm having", "i am having", "suffering from",
    "experiencing", "my chest", "my head", "my stomach", "my back", "my leg",
    "been feeling",
]

# Informational-question shapes -> grounded info answer, even with a symptom
# noun present (so "What is chest pain?" / "chest pain treatment" don't trip the
# emergency path; a bare "chest pain" still routes to symptom as a safe default).
INFO_QUESTION_PREFIXES = (
    "what", "how", "why", "when", "where", "which", "who", "is ", "are ",
    "can ", "could ", "does ", "do ", "should ", "define", "tell me", "explain",
    "list ",
)
INFO_QUESTION_MARKERS = (
    "what is", "what are", "symptoms of", "treatment for", "causes of",
    "cause of", "difference between",
    "treatment", "treatments", "remedy", "remedies", "medication",
    "medications", "medicine", "symptom", "symptoms", "prevention",
    "diagnosis", "how to treat", "how to manage", "how to prevent",
)

# LOW < MODERATE < HIGH, for taking a floor without ever downgrading.
_LEVEL_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2}

# General comorbidities that warrant a MODERATE floor when symptoms are present.
# General medical triage, not tied to the loaded KB; the LLM classifier handles
# any condition outside this set (e.g. free-text "Other").
SERIOUS_CONDITIONS = {
    "diabetes", "hypertension", "asthma", "copd", "heart disease",
    "kidney disease", "cancer", "pregnancy", "immunocompromised",
}


def _max_level(a: str, b: str) -> str:
    return a if _LEVEL_ORDER[a] >= _LEVEL_ORDER[b] else b

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
    lower = message.strip().lower()

    if any(kw in lower for kw in SELF_REPORT_TRIGGERS):  # self-report wins
        return "symptom_query"

    if (  # informational question
        lower.endswith("?")
        or lower.startswith(INFO_QUESTION_PREFIXES)
        or any(m in lower for m in INFO_QUESTION_MARKERS)
    ):
        return "info_query"

    if any(kw in lower for kw in SYMPTOM_TRIGGERS):  # bare symptom-keyword fallback
        return "symptom_query"
    return "info_query"


def assess_risk(text: str, severity=None, conditions=None, classify_fn=None) -> dict:
    """Triage to HIGH/MODERATE/LOW. Order: HIGH keyword scan -> LLM classifier
    (given severity/conditions too) -> MODERATE keyword scan -> severity floor
    (>=7 never LOW) -> condition floor (serious comorbidity never LOW). Floors
    only raise, and stop at MODERATE."""
    lower = text.lower()

    for kw in HIGH_RISK:
        if kw in lower:
            return _risk_result("HIGH")

    # Feed severity/conditions to the classifier (the keyword lists don't name
    # specific conditions); keyword scans stay on the raw text.
    classify_input = text
    annotations = []
    if isinstance(severity, (int, float)):
        annotations.append(f"self-rated severity {int(severity)}/10")
    if conditions:
        named = ", ".join(str(c) for c in conditions if str(c).strip().lower() != "none")
        if named:
            annotations.append(f"existing conditions: {named}")
    if annotations:
        classify_input = f"{text} ({'; '.join(annotations)})"

    llm_level = None
    if classify_fn is not None:
        try:
            llm_level = classify_fn(classify_input)
        except Exception:
            llm_level = None

    if llm_level in ("HIGH", "MODERATE", "LOW"):
        level = llm_level
    else:
        level = "LOW"
        for kw in MODERATE_RISK:
            if kw in lower:
                level = "MODERATE"
                break

    if isinstance(severity, (int, float)) and severity >= 7:
        level = _max_level(level, "MODERATE")

    if conditions:
        cond_lower = {str(c).strip().lower() for c in conditions}
        if cond_lower & SERIOUS_CONDITIONS:
            level = _max_level(level, "MODERATE")

    return _risk_result(level)
