# ⭐ Multi-Document RAG with Source Attribution

## 🔍 The Problem We Ran Into

While working with multiple PDFs—research papers, reports, documentation—we noticed something frustrating:

> Even when answers exist in the documents, finding them is slow, inconsistent, and often unreliable.

We tried traditional approaches:

* **Keyword search (BM25)**
  Works well when exact words match, but completely fails if the phrasing is different.

* **Semantic search (embeddings)**
  Understands meaning, but sometimes retrieves *loosely related* or irrelevant chunks.

* **LLM-based answers**
  Sound convincing, but often:

  * Don’t show where the answer came from
  * Can’t be verified
  * May hallucinate

So the real problem wasn’t just retrieval or generation—it was:

> **How do we build a system that is both accurate *and* trustworthy?**

---

## 💡 Our Approach

We built an **Advanced Multi-Document RAG system** focused on one idea:

> Not just getting answers — but making them **reliable, explainable, and verifiable**.

To achieve that, we designed a pipeline with three key layers:

1. **Hybrid Retrieval** → find good candidate chunks
2. **Reranking** → ensure the *best* chunks are actually on top
3. **Answer + Trust Layer** → generate + explain + quantify confidence

---

## 🧠 Why Hybrid Retrieval?

We didn’t rely on a single retrieval method because each has clear limitations:

### 🔍 BM25 (Keyword Search)

* Very strong when exact terms appear
* Fast and reliable
* But:

  > Fails when the query and document use different wording

---

### 🧬 Embeddings (Semantic Search)

* Understands meaning and context
* Can retrieve relevant content even with different phrasing
* But:

  > Sometimes retrieves “similar-sounding” but irrelevant chunks

---

### 👉 So we combine them:

```text
Final Score = 0.5 × BM25 + 0.5 × Embedding Similarity
```

This gives us:

* Better recall (we don’t miss relevant chunks)
* More balanced retrieval

But we quickly noticed a problem…

---

## ⚠️ The Problem with Hybrid Retrieval

Even after combining BM25 + embeddings:

> The **top results were not always the best results**

Why?

Because:

* Both BM25 and embeddings score **independently**
* They don’t deeply understand **query–document interaction**
* So ranking can still be noisy

---

## 🔁 Why We Added Reranking (Key Upgrade)

This is where **reranking** comes in—and it’s one of the most important improvements in the system.

### 🧠 What is Reranking?

After retrieving top candidates, we pass them through a **cross-encoder model** that:

* Looks at **(query, document)** together
* Scores their *true relevance*
* Reorders the results

---

### 🤔 Why is this better?

Earlier methods:

* Compare query and document **independently**
* Approximate similarity

Reranker:

* Looks at them **jointly**
* Understands:

  * intent
  * context
  * subtle relevance

---

### 📈 How It Improves the System

Without reranking:

* Top results can be noisy
* Important chunks may be ranked lower

With reranking:

* Better ordering of results
* Higher precision in final context
* More accurate answers

> In simple terms:
> **Retrieval finds candidates → Reranking finds the best ones**

---

## ⚙️ Full System Flow

```text
PDFs → Text Extraction → Chunking
        ↓
BM25 + Embeddings (Hybrid Retrieval)
        ↓
Top-K Candidate Chunks
        ↓
Cross-Encoder Reranking
        ↓
Best Context Selection
        ↓
LLM (Groq)
        ↓
Answer + Sources + Confidence
```

---

## ✂️ Why Chunking Matters

We split documents into chunks of:

* **300 words**
* **50-word overlap**

### Why not full documents?

* Too large → irrelevant information
* Poor retrieval quality

### Why not very small chunks?

* Lose context
* Answers become incomplete

### Why overlap?

* Prevents cutting important context at boundaries

---

## 🤖 Answer Generation

We use **Groq LLM API** for:

* Fast inference
* Low latency
* Good response quality

The model receives:

* Top-ranked chunks
* Structured prompt

And generates:

* Answer
* Source references

---

## 📊 Confidence Score (Trust Layer)

We don’t just give answers—we try to estimate **how reliable they are**.

### How it works:

* Take similarity scores of retrieved chunks
* Normalize them (0–1 range)
* Compute average

### Why this helps:

* High agreement → higher confidence
* Low agreement → uncertain answer

> Note: This is a heuristic, not a true probability

---

## 📊 Evaluation (Why This Matters)

Most RAG projects stop at “it works.”

We added **evaluation** using:

### Precision@K

Measures:

> How many retrieved documents are actually relevant?

This allows:

* Comparing retrieval strategies
* Validating improvements (e.g., reranking impact)

---

## 🧰 Tech Stack

* Python
* sentence-transformers (embeddings + reranking)
* FAISS (vector search)
* rank_bm25 (keyword search)
* Groq API (LLM)
* Streamlit (UI)
* pdfplumber (PDF parsing)

---

## 🚀 What Makes This Project Strong

* ✅ Not just retrieval — **retrieval + refinement**
* ✅ Not just answers — **answers + sources + confidence**
* ✅ Not just demo — **evaluation included**
* ✅ Designed with **trade-offs and reasoning in mind**

---

## ▶️ Run Locally

```bash
git clone https://github.com/<your-username>/multi-doc-rag-source-attribution.git
cd multi-doc-rag-source-attribution

pip install -r requirements.txt
streamlit run app.py
```

---

---

## ⚠️ Environment & Setup Notes (Important)

This project uses `sentence-transformers`, which depends on **PyTorch**.
On some systems, you may encounter errors like:

```
ModuleNotFoundError: No module named 'torchvision'
```

### ✅ Recommended Fix (CPU Systems)

If you're running on a CPU-only machine (e.g., 8GB RAM laptop), install dependencies using:

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

This ensures:

* Correct PyTorch (CPU) version
* Compatibility with `torchvision`
* Stable execution of embedding and reranking models

---

### 💡 Why This Is Needed

Even though this project is text-based:

* `sentence-transformers` internally depends on **PyTorch**
* Some configurations require `torchvision`, even if not directly used

---

### 🧠 System Recommendation

For smooth performance:

* RAM: **8GB+**
* CPU: Modern multi-core processor (i5 or equivalent)
* GPU: Not required

---

### 🛠️ If Installation Fails

Try:

```bash
pip cache purge
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

---

> These steps ensure a consistent setup across different environments and avoid common dependency issues.


## 🧑‍💻 Author

**Tushar Pathak**

> Focused on building systems that don’t just generate answers—but help users **trust them**.
