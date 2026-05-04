import streamlit as st
import os

from src.ingestion import load_pdfs
from src.retriever import AdvancedRetriever
from src.rag_pipeline import generate_answer
from src.utils import highlight_text, best_sentence


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Advanced Multi-Document RAG",
    layout="wide"
)

st.title("📄 Advanced Multi-Document RAG System")
st.caption("Hybrid Retrieval + Reranking + Explainable Answers")


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------
files = st.file_uploader(
    "Upload PDF Documents",
    type="pdf",
    accept_multiple_files=True
)


# ---------------------------------------------------------
# CACHE PIPELINE (IMPORTANT FOR PERFORMANCE)
# ---------------------------------------------------------
# WHY?
# - Avoid recomputing embeddings every time
# - Speeds up repeated queries significantly
# ---------------------------------------------------------
@st.cache_resource
def build_pipeline(paths):
    docs = load_pdfs(paths)
    return AdvancedRetriever(docs)


# ---------------------------------------------------------
# MAIN FLOW
# ---------------------------------------------------------
if files:

    paths = []

    # Save uploaded files locally
    os.makedirs("data", exist_ok=True)

    for f in files:
        file_path = os.path.join("data", f.name)

        with open(file_path, "wb") as out:
            out.write(f.read())

        paths.append(file_path)

    st.success("✅ PDFs uploaded and processed successfully")

    # Build RAG pipeline
    retriever = build_pipeline(paths)

    # -----------------------------------------------------
    # QUERY INPUT
    # -----------------------------------------------------
    query = st.text_input("🔎 Ask a question from your documents")

    if query:

        with st.spinner("Searching documents and generating answer..."):

            # Step 1: Retrieve relevant chunks
            results, scores = retriever.search(query)

            # Step 2: Generate answer using LLM
            answer, confidence = generate_answer(query, results, scores)

        # -------------------------------------------------
        # ANSWER SECTION
        # -------------------------------------------------
        st.markdown("## 🧠 Generated Answer")
        st.write(answer)

        # -------------------------------------------------
        # CONFIDENCE VISUALIZATION
        # -------------------------------------------------
        st.markdown("## 📊 Confidence Score")

        confidence_clamped = min(max(confidence, 0.0), 1.0)

        st.progress(confidence_clamped)
        st.write(f"Score: **{confidence:.3f}**")

        # -------------------------------------------------
        # SOURCES + EXPLANATION (KEY FEATURE)
        # -------------------------------------------------
        st.markdown("## 🔍 Sources & Evidence")

        for i, r in enumerate(results):

            # Extract most relevant sentence instead of full chunk
            snippet = best_sentence(r["text"], query)

            # Highlight query words in snippet
            highlighted = highlight_text(snippet, query)

            st.markdown(f"""
### 📄 {r['source']} (Page {r['page']})
**Relevance Score:** `{round(float(scores[i]), 3)}`

{highlighted}

---
""")