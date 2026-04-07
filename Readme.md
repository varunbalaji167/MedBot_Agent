# MediBot — Intelligent Healthcare Assistant

**MediBot** is a conversational RAG (Retrieval-Augmented Generation) chatbot designed to provide general medical information and personalized symptom assessments. It leverages the **FLAN-T5-Large** model and **FAISS** vector search to deliver context-aware health insights from a provided medical knowledge base.

🔗 **Live Demo:** [https://medbotagent.streamlit.app/](https://medbotagent.streamlit.app/)

## Key Features

* **Retrieval-Augmented Generation (RAG):** Answers medical questions using a specific knowledge base (`Rag_pdf.pdf`) for higher accuracy.
* **Symptom Assessment Flow:** When symptoms are detected, the bot initiates a 3-question diagnostic flow (**Duration**, **Severity**, and **Patient Profile**) to provide a tailored assessment.
* **Emergency Risk Detection:** Automatically scans for high-risk keywords (e.g., *"chest pain," "difficulty breathing"*) and triggers immediate emergency care warnings.
* **Confidence Scoring:** Provides a percentage-based confidence level for every medical answer generated.
* **Streamlit UI:** A clean, user-friendly interface for real-time chatting and PDF knowledge base management.

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM** | `google/flan-t5-large` |
| **Embeddings** | `all-MiniLM-L6-v2` |
| **Vector Database** | **FAISS** (Facebook AI Similarity Search) |
| **Frameworks** | Streamlit, PyTorch, Sentence-Transformers |

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
   Place your medical PDF in the root directory and rename it to `Rag_pdf.pdf` for automatic loading.

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

## How to Use

* **General Queries:** Ask questions like *"What is COPD?"* or *"What causes hypertension?"*.
* **Symptom Check:** Type how you are feeling, such as *"I have been feeling dizzy and tired."*
* **Assessment:** Answer the follow-up questions regarding duration and severity to receive a recommended action.

## Medical Disclaimer

> **MediBot provides general health information based on uploaded guidelines only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health providers with any questions you may have regarding a medical condition.