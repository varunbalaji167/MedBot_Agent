import os
import streamlit as st
import torch
import faiss
import numpy as np
import PyPDF2
from io import BytesIO
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 1. PAGE CONFIG & SESSION STATE

st.set_page_config(page_title="MediBot V3", page_icon="🩺", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "phase" not in st.session_state:
    st.session_state.phase = "idle"
if "ask_index" not in st.session_state:
    st.session_state.ask_index = 0
if "initial_msg" not in st.session_state:
    st.session_state.initial_msg = ""
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {}
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []


def reset_session():
    st.session_state.phase = "idle"
    st.session_state.ask_index = 0
    st.session_state.initial_msg = ""
    st.session_state.patient_info = {}
    st.session_state.messages = []


# 2. MODEL LOADING (CACHED)


@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    MODEL_ID = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()

    return embed_model, tokenizer, model


embed_model, tokenizer, model = load_models()


# 3. PDF PROCESSING & FAISS


def split_text(text, chunk_size=150, overlap=20):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def process_pdf(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    pdf_text = ""
    for page in reader.pages:
        extracted = page.extract_text() or ""
        pdf_text += extracted.replace('"""', "").replace('""', "").strip() + "\n"

    chunks = split_text(pdf_text)

    chunk_embeddings = embed_model.encode(
        chunks, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)
    faiss.normalize_L2(chunk_embeddings)

    faiss_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    faiss_index.add(chunk_embeddings)

    st.session_state.text_chunks = chunks
    st.session_state.faiss_index = faiss_index


# 4. RAG BACKEND


def retrieve_context(query: str, k: int = 2):
    if st.session_state.faiss_index is None:
        return [], 0.0

    q_vec = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)
    scores, indices = st.session_state.faiss_index.search(q_vec, k)
    chunks = [
        st.session_state.text_chunks[i]
        for i in indices[0]
        if i < len(st.session_state.text_chunks)
    ]

    best_score = float(scores[0][0])
    confidence = round(max(10.0, min(98.0, best_score * 100)), 1)
    return chunks, confidence


def generate_answer(question: str, patient_info: str = "") -> dict:
    context_chunks, confidence = retrieve_context(question)

    # If the semantic search confidence is very low, it's definitely out of scope
    if confidence < 30.0:
        return {
            "answer": "This specific topic is not covered in the current knowledge base. Please consult a qualified healthcare professional for guidance.",
            "confidence": confidence,
            "sources": [],
        }

    clean_chunks = [
        c.replace('"""', "").replace('"', "").strip() for c in context_chunks
    ]
    context = " ".join(clean_chunks)
    patient_part = f" Patient details: {patient_info}." if patient_info else ""

    # Strict out-of-scope restriction logic
    prompt = f"Read the following medical context carefully. Context: {context}{patient_part} Answer the question using ONLY the provided context. If the context does not contain the answer, reply exactly with 'UNAVAILABLE'. Question: {question}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Catch out-of-scope triggers
    if "UNAVAILABLE" in answer or not answer or len(answer) < 5:
        answer = "This specific topic is not covered in the current knowledge base. Please consult a qualified healthcare professional for guidance."

    return {"answer": answer, "confidence": confidence, "sources": context_chunks}


# 5. CONVERSATIONAL LOGIC

HIGH_RISK = [
    "chest pain",
    "heart attack",
    "stroke",
    "difficulty breathing",
    "severe shortness of breath",
    "unconscious",
    "seizure",
    "coughing blood",
    "suicidal",
    "overdose",
    "severe bleeding",
]
MODERATE_RISK = [
    "shortness of breath",
    "high blood pressure",
    "elevated sugar",
    "fever",
    "dizziness",
    "headache",
    "nausea",
    "fatigue",
    "vomiting",
    "swelling",
    "blurred vision",
    "wheezing",
]
SYMPTOM_TRIGGERS = [
    "i have",
    "i feel",
    "i am feeling",
    "i've been",
    "i've had",
    "i'm having",
    "i'm feeling",
    "suffering from",
    "experiencing",
    "my chest",
    "my head",
    "my stomach",
    "my back",
    "my leg",
    "pain",
    "ache",
    "hurts",
    "breathless",
    "coughing",
    "dizzy",
    "tired",
    "nausea",
    "vomiting",
    "swollen",
    "blurred",
    "burning",
    "itching",
    "rash",
]

CLARIFYING_QUESTIONS = [
    {
        "key": "duration",
        "text": "**Question 1 of 3 — Duration:** How long have you been experiencing these symptoms? (e.g. 2 days, 1 week)",
    },
    {
        "key": "severity",
        "text": "**Question 2 of 3 — Severity:** On a scale of 1 to 10, how severe is it? (1 = very mild, 10 = unbearable)",
    },
    {
        "key": "profile",
        "text": "**Question 3 of 3 — Profile:** What is your age and do you have any existing medical conditions? (e.g. 35, diabetic / 45, none)",
    },
]


def assess_risk(text: str) -> dict:
    lower = text.lower()
    for kw in HIGH_RISK:
        if kw in lower:
            return {
                "level": "HIGH",
                "border": "#ff4444",
                "explanation": "You have described symptoms that could indicate a medical emergency. Please contact emergency services (111 / 999) or go to the nearest hospital immediately.",
            }
    for kw in MODERATE_RISK:
        if kw in lower:
            return {
                "level": "MODERATE",
                "border": "#ffaa00",
                "explanation": "You have described symptoms that require medical evaluation. Please consult a doctor if these symptoms persist or worsen.",
            }
    return {
        "level": "LOW",
        "border": "#00aa44",
        "explanation": "No urgent risk indicators detected based on your description.",
    }


def detect_intent(message: str) -> str:
    if any(kw in message.lower() for kw in SYMPTOM_TRIGGERS):
        return "symptom_query"
    return "info_query"


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


def build_final_response(initial_msg: str):
    patient_summary = format_patient_summary()
    p = st.session_state.patient_info

    enriched_query = f"{initial_msg}. Patient profile: {patient_summary}. Explain possible causes and recommended next steps."
    result = generate_answer(enriched_query, patient_info=patient_summary)
    risk = assess_risk(initial_msg)

    sev = p.get("severity", "not provided")
    dur = p.get("duration", "not provided")
    pro = p.get("profile", "not provided")

    action = "Monitor your symptoms. Maintain a healthy lifestyle and stay hydrated."
    if "HIGH" in risk["level"]:
        action = "**Seek emergency care immediately.** Do not delay."
    elif "MODERATE" in risk["level"]:
        sev_num = int(sev) if str(sev).isdigit() else 5
        action = (
            "Visit a doctor or urgent care **today**."
            if sev_num >= 7
            else "Schedule a doctor appointment **within 48 hours**."
        )

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
    return html


def process_message(user_message: str):
    user_message = user_message.strip()

    if st.session_state.faiss_index is None:
        return "**Please upload a medical knowledge PDF in the sidebar first.**"

    if st.session_state.phase == "idle":
        intent = detect_intent(user_message)

        if intent == "symptom_query":
            risk = assess_risk(user_message)
            if "HIGH" in risk["level"]:
                return f"<div style='background-color: #f8f9fa; color: #1e1e1e; border-left:4px solid #ff4444;padding:10px;'><b>Emergency Detected</b><br>{risk['explanation']}<br><br><b>Do not wait. Call 111 / 999 now.</b></div>"

            st.session_state.phase = "clarifying"
            st.session_state.ask_index = 1
            st.session_state.initial_msg = user_message
            st.session_state.patient_info = {}

            return f"Thank you for sharing that. I have **3 quick questions** to give you a more personalised assessment.\n\n{CLARIFYING_QUESTIONS[0]['text']}"

        else:
            result = generate_answer(user_message)
            return f"**Medical Information**\n\n{result['answer']}\n\n<small style='color:#888;'>Confidence: {result['confidence']}% | Always verify with a healthcare professional.</small>"

    elif st.session_state.phase == "clarifying":
        store_index = st.session_state.ask_index - 1
        key = CLARIFYING_QUESTIONS[store_index]["key"]
        st.session_state.patient_info[key] = user_message

        if st.session_state.ask_index < len(CLARIFYING_QUESTIONS):
            q_text = CLARIFYING_QUESTIONS[st.session_state.ask_index]["text"]
            st.session_state.ask_index += 1
            return q_text
        else:
            st.session_state.phase = "done"
            html = build_final_response(st.session_state.initial_msg)

            st.session_state.phase = "idle"
            st.session_state.ask_index = 0
            st.session_state.initial_msg = ""
            st.session_state.patient_info = {}
            return html


# 6. STREAMLIT UI

with st.sidebar:
    st.header("Configuration")
    st.write(
        "MediBot will automatically load 'Rag_pdf.pdf' from the project root if available."
    )

    # 1. Check for the local file in the project root
    local_pdf_path = "Rag_pdf.pdf"
    default_file_exists = os.path.exists(local_pdf_path)

    # 2. Keep the uploader as an "Override" option
    uploaded_file = st.file_uploader("Override with a different PDF", type="pdf")

    # 3. Logic to process the PDF automatically
    if st.session_state.faiss_index is None:
        # Priority 1: Use the manually uploaded file
        if uploaded_file is not None:
            with st.spinner("Processing uploaded PDF..."):
                process_pdf(uploaded_file.read())
            st.success("Custom knowledge base ready!")

        # Priority 2: Use the local Rag_pdf.pdf file
        elif default_file_exists:
            with st.spinner("Loading local medical guidelines..."):
                with open(local_pdf_path, "rb") as f:
                    process_pdf(f.read())
            st.success("Default knowledge base loaded!")

        # Fallback: No file found
        else:
            st.warning(
                "No PDF found. Please upload a file or add 'Rag_pdf.pdf' to the root folder."
            )

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        reset_session()
        st.rerun()

st.title("MediBot")

st.markdown(
    """
<div style='background:linear-gradient(135deg,#e8f4fd,#f0fff4); color:#1e1e1e; padding:14px;border-radius:10px;margin-bottom:20px; border-left:4px solid #0078d4;'>
<h4 style='margin:0 0 8px 0;color:#0078d4;'>Welcome to your intelligent health assistant.</h4>
<b>General medical questions</b> → Try: <i>"What is COPD?"</i><br>
<b>Symptom assessment</b> → Try: <i>"I have been feeling dizzy and tired for 2 days"</i><br>
<small style='color:#555;'>MediBot provides general health information only. It is NOT a substitute for professional medical advice.</small>
</div>
""",
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Type your question or describe your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response_html = process_message(prompt)
            st.markdown(response_html, unsafe_allow_html=True)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_html}
            )
