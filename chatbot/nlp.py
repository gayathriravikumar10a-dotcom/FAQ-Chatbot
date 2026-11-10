
import re
import nltk
from nltk.tokenize import wordpunct_tokenize


def clean_text(text: str) -> str:
    """Convert text to lowercase and remove unnecessary punctuation."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> list[str]:
    """Split cleaned text into individual tokens."""

    cleaned_text = clean_text(text)
    tokens = wordpunct_tokenize(cleaned_text)

    return tokens


def preprocess_text(text: str) -> str:
    """Return cleaned text for the FAQ matching engine."""

    tokens = tokenize_text(text)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "HOW can I Reset My Password??"

    print("Original text:", sample)
    print("Cleaned text:", clean_text(sample))
    print("Tokens:", tokenize_text(sample))
    print("Preprocessed text:", preprocess_text(sample))
