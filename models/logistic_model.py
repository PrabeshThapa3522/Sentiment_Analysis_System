import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class LogisticSentimentModel:

    def __init__(self):

        self.pipeline = Pipeline(

            [

                (

                    "tfidf",

                    TfidfVectorizer(

                        max_features=10000,

                        ngram_range=(1, 2)

                    )

                ),

                (

                    "classifier",

                    LogisticRegression(

                        max_iter=1000,

                        random_state=42

                    )

                )

            ]

        )

    # ===========================
    # Train
    # ===========================

    def train(self, X_train, y_train):

        self.pipeline.fit(

            X_train,

            y_train

        )

    # ===========================
    # Predict
    # ===========================

    def predict(self, X):

        return self.pipeline.predict(X)

    # ===========================
    # Predict Probability
    # ===========================

    def predict_proba(self, X):

        return self.pipeline.predict_proba(X)

    # ===========================
    # Save Model
    # ===========================

    def save(self):

        joblib.dump(

            self.pipeline,

            "saved_model/logistic_model.pkl"

        )

    # ===========================
    # Load Model
    # ===========================

    def load(self):

        self.pipeline = joblib.load(

            "saved_model/logistic_model.pkl"

        )


# ===========================
# Testing
# ===========================

if __name__ == "__main__":

    X_train = [

        "i love this phone",

        "this is amazing",

        "worst product ever",

        "very bad experience",

        "it is okay",

        "average quality"

    ]

    y_train = [

        "positive",

        "positive",

        "negative",

        "negative",

        "neutral",

        "neutral"

    ]

    model = LogisticSentimentModel()

    model.train(

        X_train,

        y_train

    )

    prediction = model.predict(

        [

            "this phone is fantastic"

        ]

    )

    print("Prediction:")

    print(prediction[0])

    model.save()

    print()

    print("Model Saved Successfully.")