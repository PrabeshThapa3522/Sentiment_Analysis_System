import joblib
import numpy as np
import joblib
import numpy as np
import tensorflow as tf

from utils.preprocessing import clean_text
from utils.embeddings import GloveEmbedding


class SentimentPredictor:

    def __init__(self):

        # Load tokenizer + embedding matrix
        self.embedding = GloveEmbedding()
        self.embedding.load()

        # Load label encoder
        self.label_encoder = joblib.load(
            "saved_model/label_encoder.pkl"
        )

        # Load RNN model
        self.rnn = tf.keras.models.load_model(
            "saved_model/rnn.keras"
        )

        # Load Logistic Regression model
        self.logistic = joblib.load(
            "saved_model/logistic_model.pkl"
        )

    # ==========================================
    # RNN Prediction
    # ==========================================

    def predict_rnn(self, text):

        text = clean_text(text)

        sequence = self.embedding.texts_to_sequences(
            [text]
        )

        prediction = self.rnn.predict(
            sequence,
            verbose=0
        )

        index = np.argmax(prediction)

        confidence = float(
            np.max(prediction)
        )

        label = self.label_encoder.inverse_transform(
            [index]
        )[0]

        return label, confidence

    # ==========================================
    # Logistic Regression Prediction
    # ==========================================

    def predict_logistic(self, text):

        text = clean_text(text)

        prediction = self.logistic.predict(
            [text]
        )[0]

        probability = self.logistic.predict_proba(
            [text]
        )

        confidence = float(
            np.max(probability)
        )

        return prediction, confidence


if __name__ == "__main__":

    predictor = SentimentPredictor()

    while True:

        sentence = input("\nEnter a sentence (or type 'exit'): ")

        if sentence.lower() == "exit":
            break

        print("\n========== RNN ==========")

        label, confidence = predictor.predict_rnn(
            sentence
        )

        print("Prediction :", label)

        print(f"Confidence : {confidence*100:.2f}%")

        print("\n===== Logistic Regression =====")

        label, confidence = predictor.predict_logistic(
            sentence
        )

        print("Prediction :", label)

        print(f"Confidence : {confidence*100:.2f}%")