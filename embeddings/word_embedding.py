"""
| Technique            | Model Type                   | Learns From       | Gensim Param | Good For                       |
| -------------------- | ---------------------------- | ----------------- | ------------ | ------------------------------ |
| Word2Vec (CBOW)      | Predict target from context  | Local context     | `sg=0`       | Speed                          |
| Word2Vec (Skip-Gram) | Predict context from target  | Local context     | `sg=1`       | Quality                        |
| GloVe                | Co-occurrence matrix         | Global statistics | Pre-trained  | General semantic relationships |
| Cosine Similarity    | Measure closeness of vectors | Vector math       | —            | Comparison                     |
| Dot Product          | Raw alignment measure        | Vector math       | —            | Model scoring                  |

"""

""" ###################################################################################
Word2Vec — CBOW (Predict target word from context)

Use case: Use CBOW when you have a small corpus or want faster training. It predicts a word from its surrounding context.
"""
from gensim.models import Word2Vec

# Sample sentences
sentences = [
    ["king", "rules", "kingdom"],
    ["queen", "rules", "country"],
    ["man", "is", "strong"],
    ["woman", "is", "kind"],
]

# Train CBOW model (sg=0)
cbow_model = Word2Vec(sentences, vector_size=50, window=2, sg=0, min_count=1)

# Example: Find words similar to 'queen'
similar_words = cbow_model.wv.most_similar("queen")
print("CBOW - Similar to 'queen':", similar_words)


"""################################################################################################
Word2Vec — Skip-Gram (Predict context from target word)

Use case: Skip-Gram is better for rare words and captures fine-grained semantic relationships. Slower but higher quality embeddings.
"""

from gensim.models import Word2Vec

# Same sentences
sentences = [
    ["king", "rules", "kingdom"],
    ["queen", "rules", "country"],
    ["prince", "is", "son", "of", "king"],
    ["princess", "is", "daughter", "of", "queen"],
]

# Train Skip-Gram model (sg=1)
skip_model = Word2Vec(sentences, vector_size=50, window=2, sg=1, min_count=1)

# Example: Find words similar to 'king'
similar_words = skip_model.wv.most_similar("king")
print("Skip-Gram - Similar to 'king':", similar_words)

"""################################################################################
GloVe embeddings (pre-trained)

Use case: GloVe is great if you don’t want to train embeddings yourself or need general semantic meaning learned from huge corpora (Wikipedia + Gigaword).
"""

import gensim.downloader as api

# Load pre-trained GloVe embeddings
glove_model = api.load("glove-wiki-gigaword-50")

# Find most similar words
similar_words = glove_model.most_similar("king")
print("GloVe - Similar to 'king':", similar_words)


"""####################################################################################
Cosine similarity (measure closeness between two word vectors)

Use case: Cosine similarity is used to measure semantic closeness between words. E.g., “king” and “queen” should be high, “king” and “banana” low.
"""

import numpy as np

# Using CBOW vectors from above
vec_king = cbow_model.wv["king"]
vec_queen = cbow_model.wv["queen"]

# Cosine similarity formula
cos_sim = np.dot(vec_king, vec_queen) / (np.linalg.norm(vec_king) * np.linalg.norm(vec_queen))
print("Cosine similarity (king, queen):", cos_sim)


"""###################################################################################
Dot product (raw vector alignment)

Use case: Dot product is used internally in Word2Vec and GloVe training as a similarity measure. 
Cosine similarity is often preferred because it normalizes vector length, but dot product is faster.
"""

dot_product = np.dot(vec_king, vec_queen)
print("Dot product (king, queen):", dot_product)