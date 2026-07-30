# lstm_model.py
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100


def build_lstm_model(num_classes=3):

    model = Sequential([

        Embedding(
            input_dim=MAX_WORDS,
            output_dim=EMBEDDING_DIM,
            input_shape=(MAX_SEQUENCE_LENGTH,),
            trainable=True
        ),

        Bidirectional(
            LSTM(
                128,
                return_sequences=True
            )
        ),

        Dropout(0.4),

        Bidirectional(
            LSTM(
                64
            )
        ),

        Dropout(0.4),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            num_classes,
            activation="softmax"
        )

    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_lstm_model()

    model.summary()
    
    