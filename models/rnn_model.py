import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    GRU,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

MAX_WORDS = 30000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100


def build_rnn_model(embedding_matrix, num_classes):

    model = Sequential([

        Embedding(
            input_dim=MAX_WORDS,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False
        ),

        GRU(
            256,
            return_sequences=False,
            dropout=0.3,
            recurrent_dropout=0.3
        ),

        Dropout(0.5),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            num_classes,
            activation="softmax"
        )

    ])

    model.compile(

        optimizer=Adam(
            learning_rate=0.001
        ),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


if __name__ == "__main__":

    import numpy as np

    dummy_embedding = np.random.randn(
        MAX_WORDS,
        EMBEDDING_DIM
    )

    model = build_rnn_model(
        dummy_embedding,
        3
    )

    model.summary()
    
    