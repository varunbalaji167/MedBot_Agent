import faulthandler
faulthandler.enable()

import os

# Must be set before torch/faiss/numpy import: both bundle libomp and a dual
# OpenMP pool segfaults on macOS; one thread avoids it.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
import html
import pickle
from io import BytesIO
from pathlib import Path

import streamlit as st
import numpy as np
import requests
import PyPDF2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import rag_core

# Anchor .env to this file's dir so it loads regardless of the launch cwd.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
_dotenv_loaded = load_dotenv(dotenv_path=_ENV_PATH)


# 1. PAGE CONFIG & SESSION STATE

st.set_page_config(page_title="MediBot", page_icon="🩺", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "phase" not in st.session_state:
    st.session_state.phase = "idle"
if "initial_msg" not in st.session_state:
    st.session_state.initial_msg = ""
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {}
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None  # a rag_core.FaissIndexBackend once created
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = {}  # {chunk_id: {"text","source","page","doc_hash"}}
if "next_chunk_id" not in st.session_state:
    st.session_state.next_chunk_id = 0
if "kb_hashes" not in st.session_state:
    st.session_state.kb_hashes = set()
if "removed_hashes" not in st.session_state:
    st.session_state.removed_hashes = set()
if "doc_registry" not in st.session_state:
    st.session_state.doc_registry = []


def reset_chat():
    st.session_state.phase = "idle"
    st.session_state.initial_msg = ""
    st.session_state.patient_info = {}
    st.session_state.messages = []


def reset_knowledge_base():
    st.session_state.faiss_index = None
    st.session_state.text_chunks = {}
    st.session_state.next_chunk_id = 0
    st.session_state.kb_hashes = set()
    st.session_state.removed_hashes = set()
    st.session_state.doc_registry = []


# 2. MODEL LOADING
# Decision logic lives in rag_core.py; this file supplies the real
# embed_fn / generate_fn / index backend it's called with.

# Generation backend: ANY OpenAI-compatible chat-completions endpoint (HF
# router, Groq, OpenRouter, Google Gemini, local Ollama, ...). Set all three in
# .env to switch providers -- no code change. Defaults keep the HF router.
# (Old HUGGINGFACE_HUB_TOKEN / HF_GENERATION_MODEL names still work.)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://router.huggingface.co/v1/chat/completions").strip()
LLM_API_KEY = (os.environ.get("LLM_API_KEY") or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")).strip()
LLM_MODEL = (os.environ.get("LLM_MODEL") or os.environ.get("HF_GENERATION_MODEL") or "openai/gpt-oss-20b").strip()

# Off by default; the debug expanders expose internal error/prompt text that a
# shared instance shouldn't show. Set MEDIBOT_DEBUG=1 in .env to enable.
DEBUG_MODE = os.environ.get("MEDIBOT_DEBUG", "").strip() == "1"


@st.cache_resource
def load_embed_model():
    # device="cpu" on purpose: MPS auto-select segfaults under Streamlit's
    # worker thread, and MiniLM is cheap on CPU anyway.
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


@st.cache_resource
def load_local_generator():
    # Import here so plain `import app`/tests never need torch/transformers.
    import torch  # noqa
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    MODEL_ID = "google/flan-t5-large"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    mdl.to("cpu")  # avoid MPS auto-select under Streamlit's worker thread
    mdl.eval()
    return tok, mdl


embed_model = load_embed_model()
EMBED_DIM = embed_model.get_sentence_embedding_dimension()

CACHE_DIR = ".kb_cache"
# Bump when cached content changes (chunking/embedding) so old pickles are
# ignored and documents re-embed. v4: subsection-level chunking.
_CACHE_VERSION = "v4"
os.makedirs(CACHE_DIR, exist_ok=True)


def embed_fn(texts: list[str]) -> np.ndarray:
    raw = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return rag_core.l2_normalize(raw.astype(np.float32))


# 3. GENERATION BACKEND (OpenAI-compatible API with local fallback)


def _call_chat_api(system_prompt: str, user_prompt: str, max_new_tokens: int, temperature: float = 0.3) -> str:
    """Call any OpenAI-compatible chat-completions endpoint (LLM_BASE_URL)."""
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }

    last_error = None
    for attempt in range(2):  # one retry for transient 5xx
        resp = requests.post(LLM_BASE_URL, headers=headers, json=payload, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected LLM API response shape: {data}") from e
        last_error = f"LLM API error {resp.status_code}: {resp.text[:300]}"
        if resp.status_code >= 500 and attempt == 0:
            time.sleep(2)
            continue
        break

    raise RuntimeError(last_error)


def _generate_with_local_flan(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    import torch

    tokenizer, model = load_local_generator()
    # flan-t5 expects instruction->completion framing; the "Answer:" cue pulls
    # it toward completing rather than echoing its own instructions.
    full_prompt = f"{system_prompt}\n{user_prompt}\n\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=900)
    with torch.no_grad():
        output_ids = model.generate(
            # num_beams=2: fallback path, so trade a little search quality for CPU latency.
            **inputs, max_new_tokens=max_new_tokens, num_beams=2,
            length_penalty=2.0, early_stopping=True, no_repeat_ngram_size=3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def llm_generate(system_prompt: str, user_prompt: str, max_new_tokens: int = 300, temperature: float = 0.3) -> str:
    """temperature=0.3 for answers; pass 0.0 for deterministic work (query
    rewrite, risk classify). No effect on the beam-search local fallback."""
    if LLM_API_KEY:
        try:
            return _call_chat_api(system_prompt, user_prompt, max_new_tokens, temperature=temperature)
        except Exception as e:
            # Record the failure (don't swallow it) so "slow and odd" is diagnosable.
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[MediBot] LLM API call failed, falling back to local model: {err_msg}")
            # User-visible signal (read by _backup_notice); caller resets it first.
            st.session_state["hf_fallback_used"] = True
            if DEBUG_MODE:
                st.session_state.setdefault("hf_api_errors", [])
                st.session_state["hf_api_errors"].append(err_msg)
                st.session_state["hf_api_errors"] = st.session_state["hf_api_errors"][-5:]
    return _generate_with_local_flan(system_prompt, user_prompt, max_new_tokens)


def classify_risk_llm(symptom_text: str) -> str | None:
    try:
        result = llm_generate(
            rag_core.RISK_TAXONOMY_SYSTEM_PROMPT,
            f"Patient description: {symptom_text}",
            max_new_tokens=10,
            temperature=0.0,  # a safety classification must be stable per input
        ).strip().upper()
        for level in ("HIGH", "MODERATE", "LOW"):
            if level in result:
                return level
        return None
    except Exception:
        return None


# 4. PDF PROCESSING & KB MANAGEMENT (I/O layer; chunking math is in rag_core)


def extract_pages(file_bytes: bytes) -> list[str]:
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        pages.append(extracted.replace('"""', "").replace('""', "").strip())
    return pages


def load_cached_doc(doc_hash: str):
    path = os.path.join(CACHE_DIR, f"{doc_hash}.{_CACHE_VERSION}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"], data["embeddings"]
    except Exception:
        return None


def save_cached_doc(doc_hash: str, chunks: list[dict], embeddings: np.ndarray):
    path = os.path.join(CACHE_DIR, f"{doc_hash}.{_CACHE_VERSION}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)


def add_document_to_kb(file_bytes: bytes, filename: str, persist: bool) -> dict:
    doc_hash = rag_core.compute_hash(file_bytes)

    if doc_hash in st.session_state.removed_hashes:
        return {"status": "skipped_removed", "filename": filename}
    if doc_hash in st.session_state.kb_hashes:
        return {"status": "skipped_duplicate", "filename": filename}

    cached = load_cached_doc(doc_hash) if persist else None
    if cached is not None:
        chunk_dicts, embeddings = cached
    else:
        page_texts = extract_pages(file_bytes)
        if not any(t.strip() for t in page_texts):
            return {
                "status": "error",
                "filename": filename,
                "message": f"No extractable text found in '{filename}'. It may be a scanned/image-only PDF -- OCR isn't supported yet.",
            }
        chunk_dicts = rag_core.chunk_document(page_texts, filename, doc_hash)
        embeddings = embed_fn([c["text"] for c in chunk_dicts])
        if persist:
            save_cached_doc(doc_hash, chunk_dicts, embeddings)

    if st.session_state.faiss_index is None:
        st.session_state.faiss_index = rag_core.FaissIndexBackend(EMBED_DIM)

    n = len(chunk_dicts)
    ids = np.arange(st.session_state.next_chunk_id, st.session_state.next_chunk_id + n)
    st.session_state.faiss_index.add_with_ids(embeddings, ids)

    for cid, chunk in zip(ids.tolist(), chunk_dicts):
        st.session_state.text_chunks[cid] = chunk
    st.session_state.next_chunk_id += n

    st.session_state.kb_hashes.add(doc_hash)
    st.session_state.doc_registry.append(
        {
            "filename": filename,
            "hash": doc_hash,
            "pages": len({c["page"] for c in chunk_dicts}),
            "chunks": n,
            "scope": "shared, cached on disk" if persist else "this session only",
            "chunk_ids": ids.tolist(),
        }
    )
    return {"status": "added", "filename": filename, "chunks": n}


def remove_document(doc_hash: str):
    entry = next((d for d in st.session_state.doc_registry if d["hash"] == doc_hash), None)
    if entry is None:
        return
    ids = np.array(entry["chunk_ids"], dtype="int64")
    if st.session_state.faiss_index is not None and len(ids) > 0:
        st.session_state.faiss_index.remove_ids(ids)
    for cid in entry["chunk_ids"]:
        st.session_state.text_chunks.pop(cid, None)
    st.session_state.kb_hashes.discard(doc_hash)
    st.session_state.removed_hashes.add(doc_hash)
    st.session_state.doc_registry = [d for d in st.session_state.doc_registry if d["hash"] != doc_hash]


# 5. CONVERSATIONAL LOGIC

# Controlled form inputs (no free-text parsing). Conditions are GENERAL
# comorbidities, not tied to the loaded KB, plus a free-text "Other".
DURATION_OPTIONS = ["Less than a day", "1-2 days", "3-7 days", "1-2 weeks", "More than 2 weeks"]
CONDITION_OPTIONS = [
    "None", "Diabetes", "Hypertension", "Asthma", "COPD", "Heart disease",
    "Kidney disease", "Cancer", "Pregnancy", "Immunocompromised", "Other",
]


def _record_debug_generation(result: dict):
    """Keep the raw model output visible in the sidebar when it didn't become
    the shown answer (out-of-scope or extractive fallback). DEBUG_MODE only."""
    if not DEBUG_MODE:
        return
    raw = result.get("raw_generation")
    if raw and result.get("mode") in ("not_covered", "extractive"):
        st.session_state.setdefault("debug_generations", [])
        st.session_state["debug_generations"].append(raw)
        st.session_state["debug_generations"] = st.session_state["debug_generations"][-5:]


def _backup_notice() -> str:
    """Notice when this turn fell back to the local backup model (HF API
    failed). Empty otherwise; callers reset the flag before generating."""
    if st.session_state.get("hf_fallback_used"):
        return (
            "⚠️ The primary model was unavailable, so this was generated by the "
            "local backup model — quality may be lower. "
        )
    return ""


def format_patient_summary() -> str:
    p = st.session_state.patient_info
    parts = []
    if p.get("profile"):
        parts.append(f"patient: {p['profile']}")
    if p.get("duration"):
        parts.append(f"duration: {p['duration']}")
    if p.get("severity"):
        parts.append(f"severity: {p['severity']}/10")
    return ", ".join(parts) if parts else "not provided"


def build_final_response(initial_msg: str) -> dict:
    patient_summary = format_patient_summary()
    p = st.session_state.patient_info

    st.session_state["hf_fallback_used"] = False  # reset before any LLM call this turn
    # Retrieve on the complaint only; the profile (esp. conditions) would
    # otherwise skew retrieval toward the wrong topic. The LLM still gets the
    # profile via patient_info.
    gen_question = f"{initial_msg}. Explain possible causes and recommended next steps."
    result = rag_core.generate_answer(
        st.session_state.faiss_index, st.session_state.text_chunks, embed_fn, llm_generate,
        gen_question, patient_info=patient_summary, retrieval_query=initial_msg,
    )
    _record_debug_generation(result)
    # Pass structured form signals so triage reflects severity/conditions.
    risk = rag_core.assess_risk(
        initial_msg,
        severity=p.get("severity"),
        conditions=p.get("conditions"),
        classify_fn=classify_risk_llm,
    )

    sev = p.get("severity", "not provided")
    dur = html.escape(str(p.get("duration", "not provided")))
    pro = html.escape(str(p.get("profile", "not provided")))

    action = "Monitor your symptoms. Maintain a healthy lifestyle and stay hydrated."
    if "HIGH" in risk["level"]:
        action = "**Seek emergency care immediately.** Do not delay."
    elif "MODERATE" in risk["level"]:
        sev_num = int(sev) if isinstance(sev, int) else 5  # slider guarantees an int
        action = "Visit a doctor or urgent care **today**." if sev_num >= 7 else "Schedule a doctor appointment **within 48 hours**."

    # Escape model/KB text before it enters the unsafe_allow_html card.
    safe_answer = html.escape(result["answer"])
    if result.get("mode") == "extractive":
        assessment = (
            "<i>I couldn't generate a polished summary, so here is the most "
            "relevant information directly from the knowledge base:</i><br>"
            + safe_answer
        )
    else:
        assessment = safe_answer

    # No relevance figure next to an "I don't have enough knowledge" answer.
    relevance_line = ""
    if result.get("mode") != "not_covered":
        relevance_line = f"Source relevance: {result.get('display_confidence', 0)}% &nbsp;|&nbsp; "

    card_html = f"""
<div style='background-color: #f8f9fa; color: #1e1e1e; border-radius: 8px; border-left:5px solid {risk["border"]}; padding: 15px; margin-bottom: 10px;'>
<b style='color: #000;'>Patient Summary</b><br>
Profile: <b>{pro}</b> | Duration: <b>{dur}</b> | Severity: <b>{sev}/10</b><br><br>

<b style='color: #000;'>Medical Assessment</b><br>
{assessment}<br><br>

<b style='color: #000;'>Risk Level: {risk["level"]}</b><br>
{risk["explanation"]}<br><br>

<b style='color: #000;'>Recommended Action</b><br>
{action}<br><br>

<small style='color:#555;'>{_backup_notice()}{relevance_line}This is not a medical diagnosis. Always consult a qualified doctor.</small>
</div>
"""
    return {"content": card_html, "sources": result["sources"], "plain_text": result["answer"]}


def process_message(user_message: str) -> dict:
    """Handles chat_input turns; the clarifying phase is driven by the form below."""
    user_message = user_message.strip()
    st.session_state["hf_fallback_used"] = False  # reset before any LLM call this turn

    if st.session_state.faiss_index is None or st.session_state.faiss_index.ntotal == 0:
        msg = "**Please add a medical knowledge PDF in the sidebar first.**"
        return {"content": msg, "sources": None, "plain_text": "Please add a medical knowledge PDF in the sidebar first."}

    intent = rag_core.detect_intent(user_message)

    if intent == "symptom_query":
        risk = rag_core.assess_risk(user_message, classify_fn=classify_risk_llm)
        if "HIGH" in risk["level"]:
            content = f"<div style='background-color: #f8f9fa; color: #1e1e1e; border-left:4px solid #ff4444;padding:10px;'><b>Emergency Detected</b><br>{risk['explanation']}<br><br><b>Do not wait. Call 111 / 999 now.</b></div>"
            return {"content": content, "sources": None, "plain_text": "Emergency detected -- advised immediate emergency care."}

        st.session_state.phase = "clarifying"
        st.session_state.initial_msg = user_message
        st.session_state.patient_info = {}

        content = "Thanks for sharing that. A few quick details to personalize this -- please fill in the form below."
        return {"content": content, "sources": None, "plain_text": content}

    else:
        prior_history = st.session_state.messages[:-1]
        query_for_rag = (
            rag_core.condense_question(prior_history, user_message, llm_generate)
            if prior_history else user_message
        )
        if query_for_rag != user_message:
            # Make the silent query rewrite inspectable.
            print(f"[MediBot] condensed query: {user_message!r} -> {query_for_rag!r}")
            if DEBUG_MODE:
                st.session_state.setdefault("condensed_queries", [])
                st.session_state["condensed_queries"].append((user_message, query_for_rag))
                st.session_state["condensed_queries"] = st.session_state["condensed_queries"][-5:]
        result = rag_core.generate_answer(
            st.session_state.faiss_index, st.session_state.text_chunks, embed_fn, llm_generate, query_for_rag,
        )
        _record_debug_generation(result)
        answer_text = html.escape(result["answer"])
        if result.get("mode") == "extractive":
            answer_text = (
                "*I couldn't generate a polished summary, so here is the most "
                "relevant information directly from the knowledge base:*\n\n"
                + answer_text
            )
        if result.get("mode") == "not_covered":
            content = f"**Medical Information**\n\n{answer_text}"  # no relevance figure
        else:
            content = (
                f"**Medical Information**\n\n{answer_text}\n\n"
                f"<small style='color:#888;'>{_backup_notice()}Source relevance: {result.get('display_confidence', 0)}% | Always verify with a healthcare professional.</small>"
            )
        return {"content": content, "sources": result["sources"], "plain_text": result["answer"]}


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 View {len(sources)} source passage(s) used for this answer"):
        for i, src in enumerate(sources, 1):
            preview = src["text"][:400] + ("…" if len(src["text"]) > 400 else "")
            st.markdown(f"**Source {i}** — *{src['source']}*, page {src['page']}")
            st.markdown(f"> {preview}")


# 6. STREAMLIT UI

with st.sidebar:
    st.header("Knowledge Base")

    local_pdf_path = "Rag_pdf.pdf"
    if os.path.exists(local_pdf_path):
        with open(local_pdf_path, "rb") as f:
            default_bytes = f.read()
        default_hash = rag_core.compute_hash(default_bytes)
        if default_hash not in st.session_state.removed_hashes:
            result = add_document_to_kb(default_bytes, local_pdf_path, persist=True)
            if result["status"] == "error":
                st.warning(result["message"])

    uploaded_files = st.file_uploader(
        "Add PDF(s) to the knowledge base",
        type="pdf",
        accept_multiple_files=True,
        help="Uploaded files are embedded for THIS session only and are never written to shared storage.",
    )
    if uploaded_files:
        for uf in uploaded_files:
            result = add_document_to_kb(uf.read(), uf.name, persist=False)
            if result["status"] == "added":
                st.success(f"Added '{result['filename']}' ({result['chunks']} chunks).")
            elif result["status"] == "error":
                st.error(result["message"])

    if st.session_state.doc_registry:
        st.caption("Currently loaded:")
        for doc in st.session_state.doc_registry:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{doc['filename']}** — {doc['pages']} pages, {doc['chunks']} chunks  \n_{doc['scope']}_")
            with c2:
                if st.button("✕", key=f"remove_{doc['hash']}", help="Remove this document"):
                    remove_document(doc["hash"])
                    st.rerun()
    else:
        st.warning("No documents loaded. Add a PDF above.")

    st.divider()
    _host = LLM_BASE_URL.split("/v1")[0].replace("https://", "")
    st.caption(f"Generation backend: {_host} — {LLM_MODEL}" if LLM_API_KEY else "local flan-t5-large (no API key set)")
    if not LLM_API_KEY:
        st.caption(f"⚠️ No LLM API key set (LLM_API_KEY / HUGGINGFACE_HUB_TOKEN). Looked for .env at: `{_ENV_PATH}` (found: {_dotenv_loaded}). Using the local flan-t5 fallback.")
    if st.session_state.get("hf_api_errors"):
        with st.expander(f"⚠️ HF API call(s) failed, used local fallback ({len(st.session_state['hf_api_errors'])} recent)"):
            for err in reversed(st.session_state["hf_api_errors"]):
                st.code(err)
    if st.session_state.get("debug_generations"):
        with st.expander(f"🔍 Raw model output for 'not covered' answers ({len(st.session_state['debug_generations'])} recent)"):
            for gen_text in reversed(st.session_state["debug_generations"]):
                st.code(gen_text)
    if st.session_state.get("condensed_queries"):
        with st.expander(f"🔍 Query condensation ({len(st.session_state['condensed_queries'])} recent)"):
            for original, condensed in reversed(st.session_state["condensed_queries"]):
                st.code(f"original:  {original}\ncondensed: {condensed}")
    st.caption(f"Confidence threshold for answering: {rag_core.CONFIDENCE_THRESHOLD}% (below this, MediBot says the topic isn't covered instead of guessing)")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            reset_chat()
            st.rerun()
    with col2:
        if st.button("Clear KB", use_container_width=True):
            reset_knowledge_base()
            st.rerun()

st.title("MediBot")

_loaded_names = ", ".join(d["filename"] for d in st.session_state.doc_registry) or "no documents loaded yet"

st.markdown(
    f"""
<div style='background:linear-gradient(135deg,#e8f4fd,#f0fff4); color:#1e1e1e; padding:14px;border-radius:10px;margin-bottom:20px; border-left:4px solid #0078d4;'>
<h4 style='margin:0 0 8px 0;color:#0078d4;'>Welcome to your intelligent health assistant.</h4>
<b>General medical questions</b> → Try: <i>"What is COPD?"</i><br>
<b>Symptom assessment</b> → Try: <i>"I have been feeling dizzy and tired for 2 days"</i><br><br>
<b>Scope:</b> answers are grounded ONLY in the currently loaded document(s): <i>{_loaded_names}</i>. This is a portfolio/demo project, not a reviewed clinical guideline source -- verify the loaded PDF(s) yourself before trusting any answer.<br>
<small style='color:#555;'>MediBot provides general health information only. It is NOT a substitute for professional medical advice.</small>
</div>
""",
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        render_sources(msg.get("sources"))

if st.session_state.phase == "clarifying":
    # Controlled-input form: each widget constrains its own input, so there's
    # no free-text "idk" to validate after the fact.
    with st.chat_message("assistant"):
        with st.form("symptom_details_form"):
            duration = st.selectbox("Duration of symptoms", DURATION_OPTIONS, index=None, placeholder="Select one...")
            severity = st.slider("Severity (1 = very mild, 10 = unbearable)", min_value=1, max_value=10, value=5)
            age = st.number_input("Age", min_value=0, max_value=120, step=1, value=None, placeholder="Enter age")
            conditions = st.multiselect("Existing medical conditions (select all that apply)", CONDITION_OPTIONS)
            other_conditions = st.text_input("If you chose 'Other', list the condition(s)", placeholder="e.g. lupus, epilepsy")
            submitted = st.form_submit_button("Get my assessment")

        if submitted:
            missing = []
            if duration is None:
                missing.append("duration")
            if age is None:
                missing.append("age")
            if not conditions:
                missing.append("existing conditions (choose 'None' if not applicable)")
            if "Other" in conditions and not other_conditions.strip():
                missing.append("the 'Other' condition details")

            if missing:
                st.error(f"Please fill in: {', '.join(missing)}.")
            else:
                # Replace the "Other" placeholder with the typed condition(s).
                final_conditions = [c for c in conditions if c != "Other"]
                if other_conditions.strip():
                    final_conditions += [t.strip() for t in other_conditions.split(",") if t.strip()]
                condition_str = ", ".join(final_conditions) if final_conditions else "None"
                st.session_state.patient_info = {
                    "duration": duration,
                    "severity": severity,
                    "conditions": final_conditions,  # structured list, used by assess_risk
                    "profile": f"{int(age)} years old, conditions: {condition_str}",
                }
                with st.spinner("Analyzing..."):
                    response = build_final_response(st.session_state.initial_msg)
                st.session_state.phase = "idle"
                st.session_state.initial_msg = ""
                st.session_state.patient_info = {}
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["content"],
                        "sources": response.get("sources"),
                        "plain_text": response.get("plain_text", response["content"]),
                    }
                )
                st.rerun()

if st.session_state.phase != "clarifying":
    if prompt := st.chat_input("Type your question or describe your symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": None, "plain_text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = process_message(prompt)
                st.markdown(response["content"], unsafe_allow_html=True)
                render_sources(response.get("sources"))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["content"],
                        "sources": response.get("sources"),
                        "plain_text": response.get("plain_text", response["content"]),
                    }
                )
                if st.session_state.phase == "clarifying":
                    st.rerun()  # show the form immediately
