Here’s a **README.md** file for your Sentiment Analysis project using RNN (LSTM). You can customize it further based on your specific needs.

---

# Sentiment Analysis using RNN (LSTM)

This project demonstrates how to perform **Sentiment Analysis** using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) units. The model is trained on the **IMDB Movie Reviews dataset** to classify text as either positive or negative sentiment.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Model Architecture](#model-architecture)
7. [Results](#results)
8. [License](#license)

---

## Project Overview
Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment expressed in a piece of text. In this project, we use an **LSTM-based RNN** to classify movie reviews from the IMDB dataset as either positive or negative.

---

## Requirements
To run this project, you need the following:
- Python 3.8–3.11
- TensorFlow 2.x
- NumPy
- scikit-learn (for evaluation)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-rnn.git
   cd sentiment-analysis-rnn
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install tensorflow numpy
   ```

---

## Usage
1. Train the model:
   ```bash
   python train.py
   ```

2. Evaluate the model on the test set:
   ```bash
   python evaluate.py
   ```

3. Predict sentiment for new text:
   ```bash
   python predict.py "This movie was fantastic!"
   ```

---

## Dataset
The **IMDB Movie Reviews dataset** is used for this project. It contains 50,000 movie reviews labeled as positive (1) or negative (0). The dataset is split into 25,000 reviews for training and 25,000 for testing.

- **Source**: [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Preprocessing**:
  - Reviews are truncated/padded to a fixed length of 200 words.
  - Only the top 10,000 most frequent words are used.

---

## Model Architecture
The model consists of the following layers:
1. **Embedding Layer**: Converts word indices into dense vectors of size 128.
2. **LSTM Layer**: A 64-unit LSTM layer with dropout for regularization.
3. **Dense Layer**: A single neuron with a sigmoid activation function for binary classification.

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
```

---

## Results
The model achieves the following performance:
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~85%
- **Test Accuracy**: ~85%

Example prediction:
```bash
Input: "This movie was fantastic and I really enjoyed it!"
Output: Positive
```

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- TensorFlow/Keras for providing the deep learning framework.
- Stanford AI for the IMDB dataset.
