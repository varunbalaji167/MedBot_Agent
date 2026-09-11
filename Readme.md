# MediBot — Intelligent Healthcare Assistant

**MediBot** is a conversational RAG (Retrieval-Augmented Generation) chatbot designed to provide general medical information and personalized symptom assessments. It leverages the **FLAN-T5-Large** model and **FAISS** vector search to deliver context-aware health insights from a provided medical knowledge base.

🔗 **Live Demo:** [https://medbotagent.streamlit.app/](https://medbotagent.streamlit.app/)

## Key Features

* **Retrieval-Augmented Generation (RAG):** Answers medical questions grounded in an uploaded knowledge base, with every answer's retrieved source passages (filename + page) shown in a collapsible panel for verification. Chunks that score well below the top match for a given question are pruned from context even when more retrieval slots are available, so a small/diverse knowledge base doesn't get dumped wholesale into every prompt.
* **Incremental, multi-document knowledge base:** Add multiple PDFs; each is embedded and merged into one FAISS index (keyed by content hash, so re-uploads and new files are detected correctly). Individual documents can be removed without rebuilding the whole index.
* **Conversational follow-ups:** Follow-up questions ("what about in children?") are rewritten into standalone queries using recent chat history before retrieval.
* **Symptom Assessment Flow:** Symptom-shaped messages trigger a structured details form (duration dropdown, 1-10 severity slider, age, existing-conditions multi-select) instead of free-text follow-up questions -- each widget's type constrains what can be submitted, so there's no "idk"-shaped input to defend against after the fact.
* **Layered Risk Detection:** A deterministic keyword scan provides a fast, network-independent gate for emergency phrases; an LLM classifier adds a second pass for symptom descriptions the keyword list misses, biased toward the higher risk level when uncertain.
* **Confidence Scoring:** Provides a percentage-based confidence level (retrieval similarity) for every medical answer generated.
* **Streamlit UI:** A clean, user-friendly interface for real-time chatting and PDF knowledge base management.

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM (primary)** | Hugging Face Inference Providers router (OpenAI-compatible, `https://router.huggingface.co/v1/chat/completions`), `openai/gpt-oss-20b:fastest` by default (configurable via `HF_GENERATION_MODEL`), used when `HUGGINGFACE_HUB_TOKEN` is set. Not every model listed on a provider's page is actually servable this way -- some resolve to a paid dedicated-endpoint-only variant and return HTTP 400. Verify before changing the default: `curl -s "https://huggingface.co/api/models/<MODEL_ID>?expand[]=inferenceProviderMapping"` and look for `"status": "live"`. |
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

## Testing

```
python -m unittest test_rag_core -v
```

Runs with only numpy and the standard library. It checks, among other
things:
- An in-scope question retrieves the right source and returns a confidence
  at or above the answering threshold.
- An out-of-scope question returns the "not covered" message with **zero
  sources**, and the text-generation function is provably never even
  called (so it's structurally impossible for that path to hallucinate an
  answer).
- The HIGH-risk keyword scan fires before, and independent of, any LLM
  classification call.
- If the LLM risk classifier is unavailable, risk assessment falls back to
  the keyword-based MODERATE list -- never silently down to LOW.
- Removing one document from the knowledge base only removes that
  document's chunks from retrieval.

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
* **Model routing is a moving target.** Hugging Face's Inference Providers catalog changes over time -- a model that's servable today may not be next month, and vice versa. If the sidebar's "Generation backend" line shows a fallback error, re-run the curl check above against your configured `HF_GENERATION_MODEL` before assuming something else broke.
* **Local fallback is materially weaker and slower.** `flan-t5-large` running on CPU is a safety net for when the API is unreachable, not an equivalent substitute -- expect noticeably lower answer quality and higher latency on that path.
* **macOS + FAISS + PyTorch OpenMP crash.** On some macOS setups, importing both FAISS and PyTorch in the same process (as this app does) causes a segfault from two bundled OpenMP runtimes colliding. This is addressed via `faiss.omp_set_num_threads(1)` (in `rag_core.py`) and forcing `OMP_NUM_THREADS=1` before any heavy imports (top of `app.py`) -- if you ever see a `libomp.dylib`-related segfault after modifying imports, this is the first thing to check, and `diagnose_env.py` (run directly with `python3 diagnose_env.py`, not through Streamlit) isolates library-loading crashes from Streamlit-specific ones.
* **Debug expanders are off by default.** Set `MEDIBOT_DEBUG=1` in `.env` to see HF API failure details and raw model output behind "not covered" answers in the sidebar -- useful for development, not meant to be shown to an end user by default.

## Medical Disclaimer

> **MediBot provides general health information based on uploaded guidelines only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health providers with any questions you may have regarding a medical condition.