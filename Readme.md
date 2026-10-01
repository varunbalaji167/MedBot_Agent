# MediBot — Intelligent Healthcare Assistant

**MediBot** is a conversational RAG (Retrieval-Augmented Generation) chatbot designed to provide general medical information and personalized symptom assessments. It uses an LLM (Hugging Face `openai/gpt-oss-20b` by default, with a local **FLAN-T5-Large** fallback) and **FAISS** vector search to deliver answers **grounded in a provided medical knowledge base** — and says "I don't have enough knowledge" for anything the knowledge base doesn't cover.

🔗 **Live Demo:** [https://medbotagent.streamlit.app/](https://medbotagent.streamlit.app/)

[![CI](https://github.com/varunbalaji167/MedBot_Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/varunbalaji167/MedBot_Agent/actions/workflows/ci.yml)

## Key Features

* **Retrieval-Augmented Generation (RAG):** Answers medical questions grounded in an uploaded knowledge base, with every answer's retrieved source passages (filename + page) shown in a collapsible panel for verification. Chunks that score well below the top match for a given question are pruned from context even when more retrieval slots are available, so a small/diverse knowledge base doesn't get dumped wholesale into every prompt.
* **Incremental, multi-document knowledge base:** Add multiple PDFs; each is embedded and merged into one FAISS index (keyed by content hash, so re-uploads and new files are detected correctly). Individual documents can be removed without rebuilding the whole index.
* **Conversational follow-ups:** Follow-up questions ("what about in children?") are rewritten into standalone queries using recent chat history before retrieval.
* **Symptom Assessment Flow:** Symptom-shaped messages trigger a structured details form (duration dropdown, 1-10 severity slider, age, a general existing-conditions multi-select with free-text "Other") instead of free-text follow-up questions -- each widget's type constrains what can be submitted, so there's no "idk"-shaped input to defend against after the fact.
* **Layered Risk Detection:** A deterministic keyword scan provides a fast, network-independent gate for emergency phrases; an LLM classifier adds a second pass for descriptions the keyword list misses, biased toward the higher risk level when uncertain. The reported **severity (1-10) and existing conditions** feed the classifier and apply a floor, so a severe case (e.g. 8/10) can never be reported as LOW.
* **Grounded answers + honest fallback:** Each answer shows a calibrated **source relevance** figure (retrieval similarity, not a diagnosis confidence). If retrieval finds relevant content but the LLM is unavailable or refuses, MediBot returns the matching knowledge-base passage verbatim rather than falsely claiming the topic isn't covered; the "I don't have enough knowledge" message is reserved for genuinely out-of-scope questions.
* **Streamlit UI:** A clean, user-friendly interface for real-time chatting and PDF knowledge base management.

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM (primary)** | **Any OpenAI-compatible chat-completions endpoint**, set via `LLM_BASE_URL` + `LLM_API_KEY` + `LLM_MODEL` in `.env` (used whenever `LLM_API_KEY` is set). Works with Hugging Face's router (default), **Groq**, **OpenRouter**, **Google Gemini**, or a local **Ollama** — see `.example.env` for ready-to-paste free-provider recipes. (The older `HUGGINGFACE_HUB_TOKEN` / `HF_GENERATION_MODEL` names still work as fallbacks.) |
| **LLM (fallback)** | `google/flan-t5-large`, run locally, lazy-loaded only if the API call fails or no token is configured |
| **Embeddings** | `all-MiniLM-L6-v2` |
| **Vector Database** | **FAISS** (`IndexIDMap2` over `IndexFlatIP`, for stable per-chunk ids and per-document removal) |
| **Frameworks** | Streamlit, PyTorch, Sentence-Transformers |

## Architecture: `app.py` vs `rag_core.py`

The RAG/triage decision logic (confidence scoring, the in-scope/out-of-scope
gate, chunking + page attribution, risk-detection fallback direction) lives
in `rag_core.py` and has **zero dependency on Streamlit, FAISS, PyTorch, or
transformers** -- it takes an embedding function, a generation function, and
a vector-index object as plain arguments instead of importing specific
libraries directly. `app.py` is the thin layer that wires in the real
FAISS index and real models; `test_rag_core.py` wires in cheap fakes
instead. This is what makes the logic testable without installing the full
ML stack, and is also just better separation of concerns.

## Using a different knowledge base

Retrieval and the in-scope/out-of-scope decision are embedding-based, so
**swapping the knowledge base PDF needs no code changes** — grounding works for
any medical text. The emergency keyword lists and intent triggers in
`rag_core.py` are general *medical/linguistic* signals (not tied to the loaded
PDF), and the symptom form's conditions are a **general comorbidity list +
free-text "Other"** rather than the KB's specific topics (the LLM risk
classifier handles conditions outside that list). The app does assume the KB is
*medical*. To change the default document, replace `Rag_pdf.pdf`; the on-disk
cache is versioned, so new chunking/embedding re-embeds automatically. Documents
are split on quoted topic headers (e.g. `"Asthma":`) and then on subsection
labels (Symptoms/Causes/Treatment/…), name-prefixed, so a terse query like "I
have a cough" matches a focused chunk; falls back to fixed-size word chunks when
no such headers are present.
For the symptom flow, retrieval runs on the complaint alone (the patient profile
is sent to the LLM but kept out of the retrieval query so it can't skew which
topic is retrieved).

## Testing

```
python -m unittest test_rag_core -v
```

This runs in **CI** on every push/PR ([`.github/workflows/ci.yml`](.github/workflows/ci.yml),
Python 3.10–3.12) — it installs only numpy and byte-compiles `app.py`/`rag_core.py`,
so the heavy ML stack isn't needed to gate merges.

Runs with only numpy and the standard library. It checks, among other
things:
- An in-scope question retrieves the right source and returns a confidence
  at or above the answering threshold.
- An out-of-scope question returns the "not covered" message with **zero
  sources**, and the text-generation function is provably never even
  called (so it's structurally impossible for that path to hallucinate an
  answer).
- When retrieval **succeeds** but the generator refuses or errors, the answer
  falls back to the retrieved knowledge-base passage (grounded), rather than a
  false "not covered".
- The reported severity (>= 7) and a serious existing condition each floor the
  risk level so it can never read LOW; emergency keywords still win outright.
- The HIGH-risk keyword scan fires before, and independent of, any LLM
  classification call.
- If the LLM risk classifier is unavailable, risk assessment falls back to
  the keyword-based MODERATE list -- never silently down to LOW.
- Removing one document from the knowledge base only removes that
  document's chunks from retrieval.
- A **golden set** of representative statements routes and triages correctly:
  intent routing (info vs symptom vs emergency), risk floors (severity ≥ 7 and
  serious conditions never read LOW), and in/out-of-scope decisions.

What this test suite does **not** verify: real answer quality from the
actual embedding/generation models. That needs the full dependency stack,
network access, and a real PDF -- see the comment at the bottom of
`test_rag_core.py` for how to build a small golden-question eval set once
you have both.

## Notes on data handling

* The bundled `Rag_pdf.pdf` is embedded once and cached to disk (`.kb_cache/`, gitignored) so restarting the app doesn't require re-embedding it.
* PDFs uploaded through the sidebar are embedded **in memory for that session only** and are never written to the shared disk cache -- this matters on a hosted, multi-user deployment, since anything written to `.kb_cache/` would otherwise be visible to every future visitor.

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/varunbalaji167/MedBot_Agent.git
   cd MedBot_Agent
   ```

2. **Install dependencies**
   Ensure you have a `requirements.txt` file, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Knowledge Base**
   A bundled `Rag_pdf.pdf` loads automatically on first run and is cached to disk so it isn't re-embedded on every restart. To use your own default document instead, replace `Rag_pdf.pdf` in the project root. You can also add further PDFs at runtime through the sidebar uploader -- these merge into the same knowledge base for that session (not persisted to disk) rather than replacing what's already loaded.

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

## How to Use

* **General Queries:** Ask questions like *"What is COPD?"* or *"What causes hypertension?"*.
* **Symptom Check:** Type how you are feeling, such as *"I have been feeling dizzy and tired."* -- this opens a details form (duration, severity, age, existing conditions) rather than a back-and-forth chat; submit it to get a personalized assessment.

## Known limitations

* **Small demo knowledge base.** The bundled `Rag_pdf.pdf` covers exactly four conditions (COPD, Diabetes, Hypertension, Asthma). Vague, cross-cutting symptoms that don't map cleanly onto one of those (e.g. general fatigue) may legitimately get pruned to "not covered" even when a KB with denser, more varied content would have something relevant -- this is the relevance filter doing its job on a narrow KB, not a bug, but it's worth understanding before treating a "not covered" answer as proof the topic can't be handled at all.
* **Semantically-adjacent out-of-scope terms.** The confidence gate is a cosine cutoff, and the embedding model can't always tell "related but not in the KB" from "in the KB" -- e.g. *pneumonia* scores about as high as *COPD* because they're close in meaning. Such a query may clear the gate and get an answer **grounded in the nearest KB content** (not a fabrication), rather than "I don't have enough knowledge". Calibrate `CONFIDENCE_THRESHOLD` against your own KB if this matters.
* **Model routing is a moving target.** Hugging Face's Inference Providers catalog changes over time -- a model that's servable today may not be next month, and vice versa. If the sidebar's "Generation backend" line shows a fallback error, re-run the curl check above against your configured `HF_GENERATION_MODEL` before assuming something else broke.
* **Local fallback is materially weaker and slower.** `flan-t5-large` running on CPU is a safety net for when the API is unreachable, not an equivalent substitute -- expect noticeably lower answer quality and higher latency on that path.
* **macOS + FAISS + PyTorch OpenMP crash.** On some macOS setups, importing both FAISS and PyTorch in the same process (as this app does) causes a segfault from two bundled OpenMP runtimes colliding. This is addressed via `faiss.omp_set_num_threads(1)` (in `rag_core.py`) and forcing `OMP_NUM_THREADS=1` before any heavy imports (top of `app.py`) -- if you ever see a `libomp.dylib`-related segfault after modifying imports, this is the first thing to check, and `diagnose_env.py` (run directly with `python3 diagnose_env.py`, not through Streamlit) isolates library-loading crashes from Streamlit-specific ones.
* **Debug expanders are off by default.** Set `MEDIBOT_DEBUG=1` in `.env` to see HF API failure details and raw model output behind "not covered" answers in the sidebar -- useful for development, not meant to be shown to an end user by default.

## Medical Disclaimer

> **MediBot provides general health information based on uploaded guidelines only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health providers with any questions you may have regarding a medical condition.