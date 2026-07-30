import os
import joblib
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# Configuration
# ==========================================

MAX_WORDS = 30000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100

GLOVE_PATH = "embeddings/glove.6B.100d.txt"


class GloveEmbedding:

    def __init__(self):

        self.tokenizer = Tokenizer(
            num_words=MAX_WORDS,
            oov_token="<OOV>"
        )

        self.embedding_matrix = None

    # ==========================================
    # Train Tokenizer
    # ==========================================

    def fit_tokenizer(self, texts):

        self.tokenizer.fit_on_texts(texts)

    # ==========================================
    # Text → Sequence
    # ==========================================

    def texts_to_sequences(self, texts):

        sequences = self.tokenizer.texts_to_sequences(texts)

        return pad_sequences(
            sequences,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post"
        )

    # ==========================================
    # Load GloVe
    # ==========================================

    def load_glove(self):

        embeddings_index = {}

        with open(GLOVE_PATH, encoding="utf8") as f:

            for line in f:

                values = line.split()

                word = values[0]

                vector = np.asarray(
                    values[1:],
                    dtype="float32"
                )

                embeddings_index[word] = vector

        print(f"Loaded {len(embeddings_index)} GloVe vectors.")

        return embeddings_index

    # ==========================================
    # Build Embedding Matrix
    # ==========================================

    def build_embedding_matrix(self):

        glove = self.load_glove()

        word_index = self.tokenizer.word_index

        self.embedding_matrix = np.zeros(

            (
                MAX_WORDS,
                EMBEDDING_DIM
            )

        )

        for word, index in word_index.items():

            if index >= MAX_WORDS:
                continue

            vector = glove.get(word)

            if vector is not None:

                self.embedding_matrix[index] = vector

        found = np.count_nonzero(np.any(self.embedding_matrix != 0, axis=1))

        print("Embedding Matrix Created.")
        print("Tokenizer vocabulary :", len(word_index))
        print("Matched GloVe words  :", found)

    # ==========================================
    # Save Tokenizer
    # ==========================================

    def save(self):

        os.makedirs(
            "saved_model",
            exist_ok=True
        )

        joblib.dump(
            self.tokenizer,
            "saved_model/tokenizer.pkl"
        )

        np.save(
            "saved_model/embedding_matrix.npy",
            self.embedding_matrix
        )

    # ==========================================
    # Load
    # ==========================================

    def load(self):

        self.tokenizer = joblib.load(
            "saved_model/tokenizer.pkl"
        )

        self.embedding_matrix = np.load(
            "saved_model/embedding_matrix.npy"
        )
        
        