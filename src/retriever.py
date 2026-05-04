from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
from numpy.linalg import norm

class AdvancedRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.texts = [d["text"] for d in docs]

        # -------------------------------------------------
        # EMBEDDING MODEL
        # -------------------------------------------------
        # WHY MiniLM?
        # - Fast (CPU friendly)
        # - Good semantic understanding
        # -------------------------------------------------
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Precompute embeddings
        self.embeddings = self.embed_model.encode(self.texts)

        # -------------------------------------------------
        # BM25 (KEYWORD SEARCH)
        # -------------------------------------------------
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # -------------------------------------------------
        # CROSS-ENCODER (RERANKER)
        # -------------------------------------------------
        # WHY use reranker?
        # - Hybrid retrieval gives candidates
        # - Reranker improves ordering quality
        # -------------------------------------------------
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # -----------------------------------------------------
    # COSINE SIMILARITY
    # -----------------------------------------------------
    def cosine(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    # -----------------------------------------------------
    # HYBRID RETRIEVAL
    # -----------------------------------------------------
    def hybrid_search(self, query, top_k=10):
        query_embed = self.embed_model.encode([query])[0]

        bm25_scores = self.bm25.get_scores(query.split())
        embed_scores = np.array([
            self.cosine(e, query_embed) for e in self.embeddings
        ])

        # Combine both signals
        hybrid_scores = 0.5 * bm25_scores + 0.5 * embed_scores

        # Take top candidates BEFORE reranking
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]

        docs = [self.docs[i] for i in top_indices]

        return docs

    # -----------------------------------------------------
    # RERANKING STEP
    # -----------------------------------------------------
    def rerank(self, query, docs):
        pairs = [(query, d["text"]) for d in docs]

        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        final_docs = [d for d, _ in ranked]
        final_scores = np.array([s for _, s in ranked])

        return final_docs, final_scores

    # -----------------------------------------------------
    # FINAL SEARCH PIPELINE
    # -----------------------------------------------------
    def search(self, query, final_k=3):
        # Step 1: Get candidate docs
        candidates = self.hybrid_search(query, top_k=10)

        # Step 2: Rerank them
        reranked_docs, rerank_scores = self.rerank(query, candidates)

        return reranked_docs[:final_k], rerank_scores[:final_k]