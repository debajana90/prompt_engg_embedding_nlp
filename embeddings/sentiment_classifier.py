import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from nltk.corpus import stopwords
import re

# import nltk
# nltk.download('stopwords')

# ==============================================
#  Data Preprocessing-
# ==============================================
reviews = [
    "I loved this movie, it was fantastic!",
    "This movie was terrible and boring",
    "An awesome movie, I really enjoyed it",
    "I hated this movie, it was so bad"
]

labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

# Stop words removal using nltk
stop_words = set(stopwords.words('english'))

# Clean and preprocess reviews
clean_reviews = []
for review in reviews:
    # Convert to lowercase
    review = review.lower()
    # Remove non-alphabetic characters
    review = re.sub(r'[^a-z\s]', '', review)
    # Split into words
    words = review.split()
    # Remove stop words
    words = [w for w in words if w not in stop_words]
    clean_reviews.append(' '.join(words))

print("Cleaned Reviews:\n", clean_reviews)


# ==============================================
#  Tokenization - Convert words to integer indices
# ==============================================
# Create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_reviews)  # Build vocabulary from cleaned reviews

# Transform words into sequences of integer indices
sequences = tokenizer.texts_to_sequences(clean_reviews)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding (0)

print("Word Index:", tokenizer.word_index)
print("Sequences:", sequences)
print("Vocabulary Size:", vocab_size)

# ==============================================
# Padding sequences before encoding
# ==============================================
max_review_length = 500
padded_sequences = pad_sequences(sequences, maxlen=max_review_length, padding='post')

print("\n✅ Padded Sequences Shape:", padded_sequences.shape)

################--------------- One-Hot Encoding - manually create one-hot vectors -----------################################

one_hot_results = [to_categorical(seq, num_classes=vocab_size) for seq in padded_sequences]
one_hot_results = np.array(one_hot_results)
print(one_hot_results)

# ==============================================
# Build Sentiment Classifier (RNN)
# ==============================================
embedding_dim = 50

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_review_length))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ==============================================
# 9️⃣ Train the Model
# ==============================================
labels = np.array(labels)
model.fit(padded_sequences, labels, epochs=10, batch_size=2, verbose=1)

#==================================================================================================================================================

def predict_sentiment(review_text, model, tokenizer, max_review_length=500):
    """
    Takes raw review text and returns predicted sentiment using the trained model.
    """
    stop_words = set(stopwords.words('english'))
    
    # Clean the review (same as training)
    review = review_text.lower()
    review = re.sub(r'[^a-z\s]', '', review)
    words = review.split()
    words = [w for w in words if w not in stop_words]
    clean_review = ' '.join(words)
    
    # Convert to sequence using the same tokenizer
    sequence_review = tokenizer.texts_to_sequences([clean_review])
    
    # Pad the sequence
    padded_review = pad_sequences(sequence_review, maxlen=max_review_length, padding='post')
    
    # Predict
    prediction = model.predict(padded_review)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    
    return sentiment, prediction

new_review = "The movie was absolutely fantastic and enjoyable!"
sentiment, probability = predict_sentiment(new_review, model, tokenizer)
print(f"Predicted Sentiment: {sentiment} (Probability: {probability:.2f})")