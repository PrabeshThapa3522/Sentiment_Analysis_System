import streamlit as st

from utils.predict import SentimentPredictor

# ======================================
# Page Configuration
# ======================================

st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="😊",
    layout="centered"
)

# ======================================
# Load Predictor
# ======================================

@st.cache_resource
def load_predictor():
    return SentimentPredictor()

predictor = load_predictor()

# ======================================
# Title
# ======================================

st.title("😊 Sentiment Analysis System")

st.write(
    """
This application classifies text into:

- 😊 Positive
- 😐 Neutral
- 😞 Negative

The project compares:

- Logistic Regression (Machine Learning)
- RNN (Deep Learning)
"""
)

# ======================================
# Model Selection
# ======================================

model_name = st.selectbox(

    "Choose Model",

    [

        "RNN",

        "Logistic Regression"

    ]

)

# ======================================
# Input Text
# ======================================

text = st.text_area(

    "Enter your review or social media post",

    height=150

)

# ======================================
# Prediction
# ======================================

if st.button("Analyze Sentiment"):

    if text.strip() == "":

        st.warning("Please enter some text.")

    else:

        if model_name == "RNN":

            label, confidence = predictor.predict_rnn(text)

        else:

            label, confidence = predictor.predict_logistic(text)

        # Color Output

        if label.lower() == "positive":

            st.success(f"Prediction: {label}")

        elif label.lower() == "negative":

            st.error(f"Prediction: {label}")

        else:

            st.info(f"Prediction: {label}")

        st.write(f"Confidence: **{confidence*100:.2f}%**")

        st.progress(float(confidence))

# ======================================
# Footer
# ======================================

st.markdown("---")

st.caption("Built using TensorFlow, RNN, GloVe Embeddings, Logistic Regression and Streamlit.")