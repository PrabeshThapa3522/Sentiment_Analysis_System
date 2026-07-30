# train.py
import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

from utils.preprocessing import preprocess_dataframe


from models.logistic_model import LogisticSentimentModel
from utils.embeddings import GloveEmbedding
from models.rnn_model import build_rnn_model

# ============================================
# Configuration
# ============================================

TRAIN_PATH = "dataset/train.csv"
VALID_PATH = "dataset/validation.csv"
TEST_PATH = "dataset/test.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"

EPOCHS = 15
BATCH_SIZE = 64

os.makedirs("saved_model", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================
# Load Dataset
# ============================================

print("Loading Dataset...")

COLUMN_NAMES = [
    "id",
    "entity",
    "sentiment",
    "text"
]

train_df = pd.read_csv(
    TRAIN_PATH,
    header=None,
    names=COLUMN_NAMES
)

valid_df = pd.read_csv(
    VALID_PATH,
    header=None,
    names=COLUMN_NAMES
)

test_df = pd.read_csv(
    TEST_PATH,
    header=None,
    names=COLUMN_NAMES
)

print("Train :", len(train_df))
print("Validation :", len(valid_df))
print("Test :", len(test_df))

train_df["sentiment"] = train_df["sentiment"].replace(
    "Irrelevant",
    "Neutral"
)

valid_df["sentiment"] = valid_df["sentiment"].replace(
    "Irrelevant",
    "Neutral"
)

test_df["sentiment"] = test_df["sentiment"].replace(
    "Irrelevant",
    "Neutral"
)

# ============================================
# Text Preprocessing
# ============================================

print("\nPreprocessing Text...")

train_df = preprocess_dataframe(train_df, TEXT_COLUMN)
valid_df = preprocess_dataframe(valid_df, TEXT_COLUMN)
test_df = preprocess_dataframe(test_df, TEXT_COLUMN)

# ============================================
# Label Encoding
# ============================================

encoder = LabelEncoder()

y_train = encoder.fit_transform(train_df[LABEL_COLUMN])
y_valid = encoder.transform(valid_df[LABEL_COLUMN])
y_test = encoder.transform(test_df[LABEL_COLUMN])

joblib.dump(
    encoder,
    "saved_model/label_encoder.pkl"
)

y_train_cat = to_categorical(y_train)
y_valid_cat = to_categorical(y_valid)
y_test_cat = to_categorical(y_test)

# ============================================
# Compute Class Weights
# ============================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(
    enumerate(class_weights)
)

print("\nClass Weights:")
print(class_weights)
# ============================================
# GloVe Embeddings
# ============================================
print("\nLoading GloVe Embeddings...")

embedding = GloveEmbedding()

embedding.fit_tokenizer(
    train_df[TEXT_COLUMN]
)

embedding.build_embedding_matrix()

embedding.save()

X_train = embedding.texts_to_sequences(
    train_df[TEXT_COLUMN]
)

X_valid = embedding.texts_to_sequences(
    valid_df[TEXT_COLUMN]
)

X_test = embedding.texts_to_sequences(
    test_df[TEXT_COLUMN]
)


# ============================================
# Logistic Regression
# ============================================

print("\nTraining Logistic Regression...")

logistic = LogisticSentimentModel()

logistic.train(
    train_df[TEXT_COLUMN],
    train_df[LABEL_COLUMN]
)

logistic.save()

print("Logistic Regression Saved.")

# ============================================
# LSTM
# ============================================

print("\nBuilding RNN...")

rnn = build_rnn_model(

    embedding_matrix=embedding.embedding_matrix,

    num_classes=len(encoder.classes_)

)

checkpoint = ModelCheckpoint(
    "saved_model/rnn.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1
)

print("\nTraining RNN...\n")

history = rnn.fit(

    X_train,
    y_train_cat,

    validation_data=(
        X_valid,
        y_valid_cat
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]

)

# ============================================
# Save Graph
# ============================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("RNN Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Train",
    "Validation"
])

plt.subplot(1,2,2)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("RNN Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend([
    "Train",
    "Validation"
])

plt.tight_layout()

plt.savefig(
    "results/rnn_training.png",
    dpi=300
)

plt.show()

print("\nTraining Completed Successfully!")

print("\nSaved Models:")

print("saved_model/rnn.keras")
print("saved_model/logistic_model.pkl")
print("saved_model/tokenizer.pkl")
print("saved_model/label_encoder.pkl")
print("saved_model/GloVe.model")

