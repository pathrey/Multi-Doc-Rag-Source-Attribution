import re

# ---------------------------------------------------------
# HIGHLIGHT QUERY WORDS IN TEXT
# ---------------------------------------------------------
def highlight_text(text, query):
    keywords = query.lower().split()
    highlighted = text

    for word in keywords:
        pattern = re.compile(rf"({word})", re.IGNORECASE)
        highlighted = pattern.sub(r"**\1**", highlighted)

    return highlighted


# ---------------------------------------------------------
# GET BEST MATCHING SENTENCE
# ---------------------------------------------------------
def best_sentence(text, query):
    sentences = text.split(".")
    query_words = query.lower().split()

    def score(sentence):
        return sum(word in sentence.lower() for word in query_words)

    best = max(sentences, key=score)
    return best.strip()