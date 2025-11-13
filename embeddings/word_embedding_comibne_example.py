from gensim.models import Word2Vec
import gensim.downloader as api
import numpy as np

# -------------------------
# 1️⃣ Sample sentences
# -------------------------
sentences = [
    ["king", "rules", "kingdom"],
    ["queen", "rules", "country"],
    ["man", "is", "strong"],
    ["woman", "is", "kind"],
    ["prince", "is", "son", "of", "king"],
    ["princess", "is", "daughter", "of", "queen"],
]

# -------------------------
# 2️⃣ Train CBOW model (sg=0)
# -------------------------
cbow_model = Word2Vec(sentences, vector_size=50, window=2, sg=0, min_count=1)
print("CBOW trained!")

# -------------------------
# 3️⃣ Train Skip-Gram model (sg=1)
# -------------------------
skip_model = Word2Vec(sentences, vector_size=50, window=2, sg=1, min_count=1)
print("Skip-Gram trained!")

# -------------------------
# 4️⃣ Load pre-trained GloVe
# -------------------------
glove_model = api.load("glove-wiki-gigaword-50")
print("GloVe loaded!")

# -------------------------
# 5️⃣ Cosine similarity function
# -------------------------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# -------------------------
# 6️⃣ Dot product function
# -------------------------
def dot_product(v1, v2):
    return np.dot(v1, v2)

# -------------------------
# 7️⃣ Analogy solver
# vec_b - vec_a + vec_c ≈ vec_result
# -------------------------
def solve_analogy(model, a, b, c, topn=1):
    try:
        result = model.wv.most_similar(
            positive=[b, c], negative=[a], topn=topn
        )
        return result[0][0], result[0][1]  # word, similarity score
    except KeyError:
        return None, None

# -------------------------
# 8️⃣ Example analogy queries
# -------------------------
queries = [
    ("king", "man", "queen"),       # king : man :: queen : ?
    ("prince", "king", "princess")  # prince : king :: princess : ?
]

# -------------------------
# 9️⃣ Run analogies on all embeddings
# -------------------------
for a, b, c in queries:
    print(f"\nQuery: {a}:{b} :: {c}:")
    
    # CBOW
    word, score = solve_analogy(cbow_model, a, b, c)
    print(f"CBOW → {word} (score={score:.3f})")
    
    # Skip-Gram
    word, score = solve_analogy(skip_model, a, b, c)
    print(f"Skip-Gram → {word} (score={score:.3f})")
    
    # GloVe
    try:
        vec_result = glove_model[b] - glove_model[a] + glove_model[c]
        # Compute cosine similarity with all words
        best_word = None
        best_score = -float('inf')
        for w in ["king", "queen", "man", "woman", "prince", "princess"]:
            sim = cosine_similarity(vec_result, glove_model[w])
            if sim > best_score and w not in {a, b, c}:
                best_score = sim
                best_word = w
        print(f"GloVe → {best_word} (cosine={best_score:.3f})")
    except KeyError:
        print("GloVe → Word not in vocabulary")

# -------------------------
# 10️⃣ Example cosine similarity & dot product
# -------------------------
vec_king_cbow = cbow_model.wv["king"]
vec_queen_cbow = cbow_model.wv["queen"]
print("\nCBOW cosine similarity (king, queen):", cosine_similarity(vec_king_cbow, vec_queen_cbow))
print("CBOW dot product (king, queen):", dot_product(vec_king_cbow, vec_queen_cbow))
