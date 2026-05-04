# ---------------------------------------------------------
# PRECISION@K
# ---------------------------------------------------------
# Measures:
# How many retrieved docs are actually relevant
# ---------------------------------------------------------
def precision_at_k(retrieved_docs, ground_truth_sources, k=3):
    retrieved = retrieved_docs[:k]

    correct = 0
    for doc in retrieved:
        if doc["source"] in ground_truth_sources:
            correct += 1

    return correct / k


# ---------------------------------------------------------
# COMPARE RETRIEVAL METHODS
# ---------------------------------------------------------
def compare_methods(query, retriever):

    print("\n--- COMPARISON ---")

    # Hybrid + rerank
    docs, _ = retriever.search(query)
    print("\nHybrid + Rerank:")
    for d in docs:
        print(d["source"], "Page", d["page"])