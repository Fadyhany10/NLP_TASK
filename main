# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the IMDB dataset
num_words = 10000  # Keep the top 10,000 most frequent words
maxlen = 200  # Cut reviews after 200 words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences to ensure uniform input size
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Build the RNN (LSTM) model
model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=maxlen),  # Embedding layer
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),  # LSTM layer with dropout for regularization
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions on new data
def predict_sentiment(text):
    word_index = imdb.get_word_index()
    text = text.lower().split()
    text = [word_index[word] if word in word_index and word_index[word] < num_words else 0 for word in text]
    text = pad_sequences([text], maxlen=maxlen)
    prediction = model.predict(text)
    return "Positive" if prediction > 0.5 else "Negative"

# Test the function
sample_text = "This movie was fantastic and I really enjoyed it!"
print(f"Sentiment: {predict_sentiment(sample_text)}")
