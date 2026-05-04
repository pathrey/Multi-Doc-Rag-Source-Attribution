from groq import Groq
import os
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env file

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query, contexts, scores):

    # -----------------------------------------------------
    # CONTEXT BUILDING
    # -----------------------------------------------------
    # WHY limit context?
    # - Prevent token overflow
    # - Keep only relevant info
    # -----------------------------------------------------
    context_text = "\n\n".join([
        c["text"][:400] for c in contexts
    ])

    prompt = f"""
Answer the question using the context below.

Context:
{context_text}

Question:
{query}

Provide:
- Clear answer
- Mention sources
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    # -----------------------------------------------------
    # CONFIDENCE SCORE (IMPROVED)
    # -----------------------------------------------------
    # Normalize scores → 0 to 1 range
    # WHY?
    # - Raw scores are inconsistent
    # - Normalization makes interpretation easier
    # -----------------------------------------------------
    norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

    confidence = float(np.mean(norm_scores))

    return answer, round(confidence, 3)