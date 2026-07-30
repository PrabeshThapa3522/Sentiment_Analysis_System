# evaluate.py
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from utils.preprocessing import preprocess_dataframe
from utils.embeddings import WordEmbedding
from utils.metrics import evaluate_model

# ============================================
# Configuration
# ============================================

TEST_PATH = "dataset/test.csv"

TEXT_COLUMN = "text"

LABEL_COLUMN = "sentiment"

# ============================================
# Load Dataset
# ============================================

print("Loading Test Dataset...")

test_df = pd.read_csv(
    TEST_PATH,
    header=None,
    names=["id", "entity", "sentiment", "text"]
)

test_df = preprocess_dataframe(
    test_df,
    TEXT_COLUMN
)

# ============================================
# Load Label Encoder
# ============================================

label_encoder = joblib.load(
    "saved_model/label_encoder.pkl"
)

y_true = label_encoder.transform(
    test_df[LABEL_COLUMN]
)

class_names = list(
    label_encoder.classes_
)

# ============================================
# Load Tokenizer
# ============================================

embedding = WordEmbedding()

embedding.load()

X_test = embedding.texts_to_sequences(
    test_df[TEXT_COLUMN]
)

# ============================================
# Logistic Regression Evaluation
# ============================================

print("\nEvaluating Logistic Regression...\n")

logistic = joblib.load(
    "saved_model/logistic_model.pkl"
)

logistic_predictions = logistic.predict(
    test_df[TEXT_COLUMN]
)

logistic_predictions = label_encoder.transform(
    logistic_predictions
)

logistic_results = evaluate_model(

    y_true,

    logistic_predictions,

    class_names,

    "Logistic Regression"

)

# ============================================
# RNN Evaluation
# ============================================

print("\nEvaluating RNN...\n")

rnn = tf.keras.models.load_model(
    "saved_model/rnn.keras"
)

prediction = rnn.predict(
    X_test,
    verbose=0
)

prediction = np.argmax(
    prediction,
    axis=1
)

rnn_results = evaluate_model(

    y_true,

    prediction,

    class_names,

    "RNN"

)
# ============================================
# Comparison
# ============================================

comparison = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ],

    "Logistic Regression":[

        logistic_results["Accuracy"],
        logistic_results["Precision"],
        logistic_results["Recall"],
        logistic_results["F1-score"]

    ],

    "RNN":[

        rnn_results["Accuracy"],
        rnn_results["Precision"],
        rnn_results["Recall"],
        rnn_results["F1-score"]

    ]

})

comparison.to_csv(

    "results/model_comparison.csv",

    index=False

)

print("\n")

print("="*60)

print("MODEL COMPARISON")

print("="*60)

print(comparison)

print("\nComparison saved to:")

print("results/model_comparison.csv")
