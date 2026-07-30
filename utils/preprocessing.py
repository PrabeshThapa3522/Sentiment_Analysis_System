# preprocessing.py
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources (only first time)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Perform text preprocessing:
    - Lowercase
    - Remove URLs
    - Remove HTML tags
    - Remove punctuation
    - Remove numbers
    - Tokenization
    - Stopword removal
    - Lemmatization
    """

    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords + Lemmatization
    cleaned_words = []

    for word in words:

        if word not in stop_words:

            word = lemmatizer.lemmatize(word)

            cleaned_words.append(word)

    return " ".join(cleaned_words)


def preprocess_dataframe(df, text_column):
    """
    Clean an entire dataframe text column.
    """

    df[text_column] = df[text_column].apply(clean_text)

    return df


if __name__ == "__main__":

    sample = "I absolutely LOVE this mobile phone!! It costs $300 and works amazingly. https://example.com"

    print("Original:\n")
    print(sample)

    print("\nProcessed:\n")
    print(clean_text(sample))
    
    