import numpy as np

# ----------------------------- Common setup -----------------------------
embeddings = {
    "king": np.array([7, 5]),
    "queen": np.array([6, 4]),
    "man": np.array([5, 3]),
    "woman": np.array([4, 2]),
    "apple": np.array([2, 1])
}

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# =======================================================================
#     METHOD 1: Word2Vec-style (b - a + c)
# =======================================================================
def find_analogy_v1(embeddings, a, b, c):
    # Example: king - man + woman = ?
    vec_a = embeddings[a]
    vec_b = embeddings[b]
    vec_c = embeddings[c]

    target_vec = vec_b - vec_a + vec_c      # ← linear vector arithmetic
    similarities = {}

    for word, vec in embeddings.items():
        if word not in [a, b, c]:
            similarities[word] = cosine_similarity(target_vec, vec)

    best_match = max(similarities, key=similarities.get)
    return best_match, similarities[best_match]


# =======================================================================
#     METHOD 2: Relationship-comparison style
# =======================================================================
def find_analogy_v2(embeddings, a, b, c):
    # Example: king is to queen as man is to ?
    vec_a = embeddings[a]
    vec_b = embeddings[b]
    vec_c = embeddings[c]

    v1 = vec_b - vec_a                     # ← relationship between a → b
    similarities = {}

    for word, vec in embeddings.items():
        if word not in [a, b, c]:
            v2 = vec - vec_c               # ← relationship between c → word
            similarities[word] = cosine_similarity(v1, v2)

    best_match = max(similarities, key=similarities.get)
    return best_match, similarities[best_match]


# ----------------------------- Run both -----------------------------
print("Analogy V1:", find_analogy_v1(embeddings, "king", "queen", "man"))
print("Analogy V2:", find_analogy_v2(embeddings, "king", "queen", "man"))
