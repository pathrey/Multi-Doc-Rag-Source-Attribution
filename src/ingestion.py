import pdfplumber

# ---------------------------------------------------------
# CHUNKING FUNCTION
# ---------------------------------------------------------
# WHY chunk_size = 300 words?
# - Small enough for precise retrieval
# - Large enough to retain semantic meaning
#
# WHY overlap = 50?
# - Prevents losing context between chunks
# - Helps continuity across boundaries
# ---------------------------------------------------------
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------
# LOAD AND PROCESS PDFs
# ---------------------------------------------------------
# Output:
# [
#   {
#     "text": "...",
#     "source": "file.pdf",
#     "page": 3
#   }
# ]
#
# WHY store metadata?
# - Enables source attribution
# - Improves explainability
# ---------------------------------------------------------
def load_pdfs(files):
    docs = []

    for file in files:
        with pdfplumber.open(file) as pdf:
            for page_index, page in enumerate(pdf.pages):
                text = page.extract_text()

                if text:
                    chunks = chunk_text(text)

                    for chunk in chunks:
                        docs.append({
                            "text": chunk,
                            "source": file,
                            "page": page_index + 1
                        })

    return docs