import faulthandler
faulthandler.enable()

import os

# MUST be set before torch/faiss/numpy are imported below, not after --
# these libraries read OMP_NUM_THREADS when their native OpenMP thread pool
# is first initialized. A macOS crash report (EXC_BAD_ACCESS/SIGSEGV inside
# libomp.dylib's __kmp_launch_worker/__kmp_fork_barrier, i.e. an OpenMP
# worker thread being spun up) confirmed this is the classic dual-OpenMP-
# runtime collision: torch and faiss each bundle their own copy of libomp,
# and when BOTH actually spin up worker threads in the same process, the two
# runtimes corrupt each other's thread-pool bookkeeping. Setting
# faiss.omp_set_num_threads(1) (done separately in rag_core.py) only
# constrained FAISS's side of this -- torch's own OpenMP pool (used inside
# MiniLM's forward pass during embed_fn) was still spinning up multiple
# worker threads and hitting the same collision. Forcing this globally,
# before import, is the fix that actually stops any of these libraries from
# creating a multi-thread OpenMP pool in the first place.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
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

# Anchored to this file's own directory, NOT the current working directory --
# load_dotenv() with no arguments depends on where the process was launched
# FROM, which silently finds nothing if you run `streamlit run app.py` from
# any directory other than this one. This makes .env loading independent of
# that.
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
#
# All decision logic (confidence gate, risk fallback, chunking) lives in
# rag_core.py and knows nothing about Streamlit/FAISS/PyTorch. This file's
# job is just to supply the real embed_fn / generate_fn / index backend that
# rag_core's functions are called with.

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
# NOTE ON MODEL CHOICE: Hugging Face's Inference Providers only serve a
# SUBSET of models on the Hub, through a SUBSET of partner providers, and
# which specific provider variant is "live" (serverless, pay-per-call) vs.
# requiring a paid dedicated endpoint varies per model and changes over
# time. Don't guess -- check directly before relying on a model:
#
#   curl -s "https://huggingface.co/api/models/<MODEL_ID>?expand[]=inferenceProviderMapping"
#
# Look for a provider entry with "status": "live".
#
# Qwen/Qwen2.5-7B-Instruct (the previous default) turned out to have exactly
# ONE provider mapped (Together AI), and that mapping resolved to a "Turbo"
# variant requiring a paid DEDICATED endpoint -- i.e. no free/serverless
# route existed for it at all, so every single call fell through to the
# slow local fallback. gpt-oss-20b is used here instead: OpenAI's own
# open-weight release is confirmed (via third-party integration docs dated
# the same day this was written) to have live multi-provider routing for
# its 120b sibling; the 20b variant is released as part of the same matched
# pair and providers hosting one typically host both, but that specific
# claim for 20b was NOT independently re-verified at the time this default
# was set (a live web check was unavailable) -- confirm with the curl
# command above before trusting this in anything beyond a demo.
HF_GENERATION_MODEL = os.environ.get("HF_GENERATION_MODEL", "openai/gpt-oss-20b:fastest")
# Hugging Face deprecated the old per-model "api-inference.huggingface.co"
# endpoint in favor of a single OpenAI-compatible router across all
# Inference Providers. This is the CURRENT endpoint as of this writing.
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

# Off by default. The two debug expanders (HF API failure details, raw model
# output for "not covered" answers) were essential during development but
# expose internal error text and prompt/response internals -- fine for you
# running this locally, not something a random visitor to a shared/deployed
# instance should see by default. Set MEDIBOT_DEBUG=1 in .env to turn them
# back on.
DEBUG_MODE = os.environ.get("MEDIBOT_DEBUG", "").strip() == "1"


@st.cache_resource
def load_embed_model():
    # device="cpu" is explicit and deliberate, not a default we left
    # unset. On Apple Silicon, sentence-transformers auto-selects the MPS
    # (Metal) backend when available -- but MPS has a known constraint
    # around being driven from the process's main thread, and Streamlit
    # runs the app script in a separate worker thread (ScriptRunner), not
    # the main thread. That mismatch is a plausible-fit explanation for a
    # native segfault that reproduces under `streamlit run` but not when
    # the same encode() call is run as a plain top-level script. MiniLM is
    # tiny enough that CPU-only has no meaningful performance cost here.
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


@st.cache_resource
def load_local_generator():
    # torch/transformers imported here, not at module top -- so simply
    # importing app.py (or rag_core.py, for tests) never requires them
    # unless the local fallback generator is actually instantiated.
    import torch  # noqa
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    MODEL_ID = "google/flan-t5-large"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    mdl.to("cpu")  # explicit for the same reason as load_embed_model() above -- avoid MPS auto-selection under Streamlit's worker thread
    mdl.eval()
    return tok, mdl


embed_model = load_embed_model()
EMBED_DIM = embed_model.get_sentence_embedding_dimension()

CACHE_DIR = ".kb_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def embed_fn(texts: list[str]) -> np.ndarray:
    raw = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return rag_core.l2_normalize(raw.astype(np.float32))


# 3. GENERATION BACKEND (HF Inference API with local fallback)


def _call_hf_inference_api(system_prompt: str, user_prompt: str, max_new_tokens: int, temperature: float = 0.3) -> str:
    """Calls Hugging Face's Inference Providers router -- an OpenAI-compatible
    chat completions endpoint that fans out to whichever partner (Together,
    Fireworks, Cerebras, etc.) currently serves HF_GENERATION_MODEL. We send
    plain system/user messages; the provider applies that model's own chat
    template server-side, so there's no hand-built prompt-template string to
    maintain here (unlike the old per-model text-generation endpoint)."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "model": HF_GENERATION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }

    last_error = None
    for attempt in range(2):  # one retry for transient 5xx from the routed provider
        resp = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected HF router response shape: {data}") from e
        last_error = f"HF router error {resp.status_code}: {resp.text[:300]}"
        if resp.status_code >= 500 and attempt == 0:
            time.sleep(2)
            continue
        break

    raise RuntimeError(last_error)


def _generate_with_local_flan(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    import torch

    tokenizer, model = load_local_generator()
    # flan-t5 was instruction-tuned on short, direct instruction->completion
    # pairs, not chat-style system/user framing. A plain concatenation of
    # the two (as used for the API model, which handles chat format
    # natively) left this weaker model prone to pattern-matching onto the
    # word "UNAVAILABLE" mentioned in its own instructions rather than
    # actually reasoning about the context -- reproduced in testing on a
    # clearly in-scope question. Appending an explicit "Answer:" completion
    # cue is a well-established way to pull a T5-family model toward
    # actually completing the answer instead of echoing back part of its
    # instructions.
    full_prompt = f"{system_prompt}\n{user_prompt}\n\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=900)
    with torch.no_grad():
        output_ids = model.generate(
            # num_beams reduced from 4 -> 2: this is a FALLBACK path, not the
            # primary one -- beam search cost scales roughly linearly with
            # beam count on CPU, so this materially cuts local-generation
            # latency. 2 beams still gives some search benefit over pure
            # greedy decoding (num_beams=1) without paying for 4.
            **inputs, max_new_tokens=max_new_tokens, num_beams=2,
            length_penalty=2.0, early_stopping=True, no_repeat_ngram_size=3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def llm_generate(system_prompt: str, user_prompt: str, max_new_tokens: int = 300, temperature: float = 0.3) -> str:
    """temperature defaults to 0.3 (a bit of natural fluency for the main
    answer-generation task) but callers doing near-deterministic work --
    rewriting a query, classifying into one of three fixed risk labels --
    should pass temperature=0.0. Using 0.3 unconditionally everywhere was a
    real bug: the SAME input to condense_question could get rewritten
    differently across runs purely from sampling noise, which is exactly
    what reproduced as an intermittent "this worked a minute ago" failure
    (a previously-fine standalone question occasionally retrieving poorly
    after being needlessly reworded). The local flan-t5 fallback already
    uses beam search (no sampling), so temperature has no effect there --
    it's only meaningful on the HF API path."""
    if HF_TOKEN:
        try:
            return _call_hf_inference_api(system_prompt, user_prompt, max_new_tokens, temperature=temperature)
        except Exception as e:
            # Previously this was a bare `except Exception: pass` -- silent
            # by design for graceful fallback, but that also meant genuine
            # API failures (bad model id, auth issue, rate limit, malformed
            # response) were indistinguishable from "no token configured" in
            # the UI, and the only symptom was "it's slow and answers oddly"
            # with zero way to find out why. Recording it (without breaking
            # the fallback behavior itself) is what actually makes this
            # debuggable instead of another guessing exercise.
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[MediBot] HF Inference API call failed, falling back to local model: {err_msg}")
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
            temperature=0.0,  # a safety classification should not vary run-to-run for identical input
        ).strip().upper()
        for level in ("HIGH", "MODERATE", "LOW"):
            if level in result:
                return level
        return None
    except Exception:
        return None


# 4. PDF PROCESSING & KB MANAGEMENT (I/O layer; chunking math itself is in rag_core)


def extract_pages(file_bytes: bytes) -> list[str]:
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        pages.append(extracted.replace('"""', "").replace('""', "").strip())
    return pages


def load_cached_doc(doc_hash: str):
    path = os.path.join(CACHE_DIR, f"{doc_hash}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"], data["embeddings"]
    except Exception:
        return None


def save_cached_doc(doc_hash: str, chunks: list[dict], embeddings: np.ndarray):
    path = os.path.join(CACHE_DIR, f"{doc_hash}.pkl")
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

# Controlled-input options for the symptom-detail form, replacing free-text
# answers to "duration/severity/profile" questions. Free text (e.g. "idk")
# used to flow straight through into the final assessment with no real
# validation -- a slider literally cannot produce an out-of-range severity,
# a selectbox literally cannot produce an unparseable duration, so this
# closes that gap structurally rather than by adding string-parsing checks
# after the fact.
DURATION_OPTIONS = ["Less than a day", "1-2 days", "3-7 days", "1-2 weeks", "More than 2 weeks"]
CONDITION_OPTIONS = ["None", "Diabetes", "Hypertension", "Asthma", "COPD", "Other"]


def _record_debug_generation(result: dict):
    """When the not-covered message fires but the model actually produced
    non-trivial output, keep that raw output visible in the sidebar so
    "why did this get refused" is answerable by looking, not by guessing and
    re-running with print statements each time. Only active when DEBUG_MODE
    is on -- see the DEBUG_MODE definition for why this is opt-in."""
    if not DEBUG_MODE:
        return
    raw = result.get("raw_generation")
    if result["answer"] == rag_core.NOT_COVERED_MSG and raw:
        st.session_state.setdefault("debug_generations", [])
        st.session_state["debug_generations"].append(raw)
        st.session_state["debug_generations"] = st.session_state["debug_generations"][-5:]


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

    enriched_query = f"{initial_msg}. Patient profile: {patient_summary}. Explain possible causes and recommended next steps."
    result = rag_core.generate_answer(
        st.session_state.faiss_index, st.session_state.text_chunks, embed_fn, llm_generate,
        enriched_query, patient_info=patient_summary,
    )
    _record_debug_generation(result)
    risk = rag_core.assess_risk(initial_msg, classify_fn=classify_risk_llm)

    sev = p.get("severity", "not provided")
    dur = p.get("duration", "not provided")
    pro = p.get("profile", "not provided")

    action = "Monitor your symptoms. Maintain a healthy lifestyle and stay hydrated."
    if "HIGH" in risk["level"]:
        action = "**Seek emergency care immediately.** Do not delay."
    elif "MODERATE" in risk["level"]:
        # severity now always comes from a 1-10 slider (never free text), so
        # this can be trusted as a real int directly -- no isdigit() fallback needed.
        sev_num = int(sev) if isinstance(sev, int) else 5
        action = "Visit a doctor or urgent care **today**." if sev_num >= 7 else "Schedule a doctor appointment **within 48 hours**."

    html = f"""
<div style='background-color: #f8f9fa; color: #1e1e1e; border-radius: 8px; border-left:5px solid {risk["border"]}; padding: 15px; margin-bottom: 10px;'>
<b style='color: #000;'>Patient Summary</b><br>
Profile: <b>{pro}</b> | Duration: <b>{dur}</b> | Severity: <b>{sev}/10</b><br><br>

<b style='color: #000;'>Medical Assessment</b><br>
{result["answer"]}<br><br>

<b style='color: #000;'>Risk Level: {risk["level"]}</b><br>
{risk["explanation"]}<br><br>

<b style='color: #000;'>Recommended Action</b><br>
{action}<br><br>

<small style='color:#555;'>Confidence: {result["confidence"]}% &nbsp;|&nbsp; This is not a medical diagnosis. Always consult a qualified doctor.</small>
</div>
"""
    return {"content": html, "sources": result["sources"], "plain_text": result["answer"]}


def process_message(user_message: str) -> dict:
    """Only called for chat_input turns now -- the clarifying phase is
    handled entirely by the form in the UI section below, not through this
    function, since it's no longer a sequence of free-text chat replies."""
    user_message = user_message.strip()

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
            # Same principle as the HF-error and raw-generation visibility
            # added earlier: a step that silently rewrites the user's actual
            # question is exactly the kind of thing that needs to be
            # inspectable, not inferred after the fact -- this is what let a
            # regression (a previously-working standalone question breaking
            # once prior chat history existed) actually get diagnosed instead
            # of guessed at.
            print(f"[MediBot] condensed query: {user_message!r} -> {query_for_rag!r}")
            if DEBUG_MODE:
                st.session_state.setdefault("condensed_queries", [])
                st.session_state["condensed_queries"].append((user_message, query_for_rag))
                st.session_state["condensed_queries"] = st.session_state["condensed_queries"][-5:]
        result = rag_core.generate_answer(
            st.session_state.faiss_index, st.session_state.text_chunks, embed_fn, llm_generate, query_for_rag,
        )
        _record_debug_generation(result)
        content = f"**Medical Information**\n\n{result['answer']}\n\n<small style='color:#888;'>Confidence: {result['confidence']}% | Always verify with a healthcare professional.</small>"
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
    st.caption(f"Generation backend: {'Hugging Face API (' + HF_GENERATION_MODEL + ')' if HF_TOKEN else 'local flan-t5-large (no HF token set)'}")
    if not HF_TOKEN:
        st.caption(f"⚠️ HUGGINGFACE_HUB_TOKEN not detected. Looked for a .env file at: `{_ENV_PATH}` (found: {_dotenv_loaded}).")
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
    # Controlled-input form instead of free-text chat replies. Each widget
    # constrains its own input by construction -- the slider cannot leave
    # 1-10, the selectbox cannot contain arbitrary text, age is a bounded
    # number -- so there is no "idk"-shaped input to defend against here,
    # rather than validating free text after the fact.
    with st.chat_message("assistant"):
        with st.form("symptom_details_form"):
            duration = st.selectbox("Duration of symptoms", DURATION_OPTIONS, index=None, placeholder="Select one...")
            severity = st.slider("Severity (1 = very mild, 10 = unbearable)", min_value=1, max_value=10, value=5)
            age = st.number_input("Age", min_value=0, max_value=120, step=1, value=None, placeholder="Enter age")
            conditions = st.multiselect("Existing medical conditions (select all that apply)", CONDITION_OPTIONS)
            submitted = st.form_submit_button("Get my assessment")

        if submitted:
            missing = []
            if duration is None:
                missing.append("duration")
            if age is None:
                missing.append("age")
            if not conditions:
                missing.append("existing conditions (choose 'None' if not applicable)")

            if missing:
                st.error(f"Please fill in: {', '.join(missing)}.")
            else:
                condition_str = ", ".join(conditions)
                st.session_state.patient_info = {
                    "duration": duration,
                    "severity": severity,
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
                    st.rerun()  # immediately show the form instead of waiting for the next interaction