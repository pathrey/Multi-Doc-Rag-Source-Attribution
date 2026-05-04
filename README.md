# Multi-Document RAG with Source Attribution

### 🔍 The Problem
Finding exact answers across multiple PDFs—research papers, reports, or notes—is often slow, manual, and unreliable. Standard search tools present several hurdles:

*   **Keyword search (BM25)** misses the underlying meaning.
*   **Semantic search (embeddings)** can sometimes retrieve irrelevant context.
*   **Lack of Evidence:** Most AI answers don’t specify where the information originated.
*   **Trust Issues:** There is often no way to judge the reliability of a generated answer.

**The real challenge:** Getting accurate, explainable, and trustworthy answers from a vast pool of documents.

---

### 💡 Our Approach
To solve this, we built a **Multi-Document RAG (Retrieval-Augmented Generation)** system that focuses on transparency and precision:

1.  **Hybrid Search:** Combines BM25 and Semantic Embeddings for superior retrieval.
2.  **Clear Attribution:** Maps every answer back to its specific document and page number.
3.  **Confidence Scoring:** Provides a metric to help users judge the reliability of the output.

#### Why Hybrid Search?
*   **BM25:** Excellent for exact term matches.
*   **Embeddings:** Understands context and intent even when wording differs.
By combining both, we create a more resilient retrieval system that captures what a single method might miss.

---

### ⚙️ How It Works
The system follows a structured pipeline:

1.  **Ingestion:** Extracts text and metadata (filename, page number) via `pdfplumber`.
2.  **Chunking:** Breaks text into smaller segments for more granular retrieval.
3.  **Indexing:** Stores data in a **BM25 index** and a **FAISS vector database**.
4.  **Retrieval:** Merges similarity scores from both indexes to find the top context chunks.
5.  **Generation:** Passes context to the **LLM** to generate a structured answer.
6.  **Trust Layer:** Appends citations and calculates a confidence score based on retrieval agreement.

---

### 🧰 Libraries & Tools
*   **LangChain:** Orchestrates the RAG pipeline.
*   **FAISS:** High-speed vector similarity search.
*   **sentence-transformers:** Generates semantic embeddings.
*   **rank_bm25:** Powers the keyword-based retrieval.
*   **OpenAI API:** High-quality text generation.
*   **Streamlit:** Provides a clean, functional UI.

---

### 🚀 Key Features
*   ✅ **Cross-Document Analysis:** Works seamlessly across multiple files.
*   ✅ **Hybrid Retrieval:** Better accuracy than vector-only search.
*   ✅ **Deep Citations:** Precise source tracking (File + Page).
*   ✅ **Explainability:** Focuses on trust rather than just "black box" answers.

---

### ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/multi-doc-rag-source-attribution.git
cd multi-doc-rag-source-attribution

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

**Example Query:**
> *“What are the main conclusions of the report?”*
>
> **Output:**
> * [Generated Answer]
> * **Sources:** `report.pdf` (Page 4)
> * **Confidence Score:** 0.89

---

### 🔮 Future Improvements
*   **OCR Support:** For scanned or image-heavy PDFs.
*   **Reranking Models:** To further refine retrieved results.
*   **Local LLM Deployment:** Support for privacy-focused local models.
*   **Evaluation Metrics:** Implementing precision and recall tracking.

---

**🧑‍💻 Author:** Tushar Pathak
*Focusing on making AI trustworthy, explainable, and usable in real-world scenarios.*
