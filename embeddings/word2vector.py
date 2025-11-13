from gensim.models import Word2Vec

# Define training data
sentences = [
    ['gensim', 'is', 'billed', 'as', 'a', 'natural', 'language', 'processing', 'package'],
    ['but', 'it', 'is', 'practically', 'much', 'more', 'than', 'that'],
    ['it', 'is', 'a', 'leading', 'and', 'a', 'state', 'of', 'the', 'art', 'package',
     'for', 'processing', 'texts', 'working', 'with', 'word', 'vector', 'models']
]

# Train model
model = Word2Vec(sentences, min_count=1, vector_size=10)

# Summarize the loaded model
print(model)

# Summarize vocabulary
words = list(model.wv.key_to_index)
print(words)

# Access vector for one word
print(model.wv['gensim'])

##====================================================================================================================
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define training data
sentences = [
    ['gensim', 'is', 'billed', 'as', 'a', 'natural', 'language', 'processing', 'package'],
    ['but', 'it', 'is', 'practically', 'much', 'more', 'than', 'that'],
    ['it', 'is', 'a', 'leading', 'and', 'a', 'state', 'of', 'the', 'art', 'package',
     'for', 'processing', 'texts', 'working', 'with', 'word', 'vector', 'models']
]

# Train model
model = Word2Vec(sentences, min_count=1, vector_size=10, window=3, sg=1)

# Extract word vectors
words = list(model.wv.key_to_index)
word_vectors = [model.wv[word] for word in words]

# Reduce dimensions from 10D → 2D using PCA
pca = PCA(n_components=2)
result = pca.fit_transform(word_vectors)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(result[:, 0], result[:, 1])

# Annotate each point with the corresponding word
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.title("2D Visualization of Word2Vec Embeddings (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
