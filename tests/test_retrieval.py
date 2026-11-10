
import json
from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer

from chatbot.rag_pipeline import FAQRAG


PROJECT_DIR = Path(__file__).resolve().parent.parent
FAQ_FILE = PROJECT_DIR / "data" / "faqs.json"


@pytest.fixture(scope="module")
def rag():
    """Create a retriever without initializing or calling Gemini."""

    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        faqs = json.load(file)

    # Create the object without running its normal initializer.
    # This prevents any Gemini API setup or calls.
    instance = FAQRAG.__new__(FAQRAG)

    instance.faqs = faqs
    instance.questions = [faq["question"] for faq in faqs]

    print("\nLoading embedding model for tests...")

    instance.embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    instance.embeddings = instance.embedding_model.encode(
        instance.questions,
        normalize_embeddings=True,
    )

    return instance


@pytest.mark.parametrize(
    ("question", "expected_keyword"),
    [
        ("I forgot my password", "password"),
        ("How can I register for an account?", "account"),
        ("How can I contact customer support?", "support"),
        ("How do I get my money back?", "refund"),
    ],
)
def test_retrieve_expected_faq(rag, question, expected_keyword):
    results = rag.retrieve(question, top_k=3)

    assert results, "The retriever returned no FAQ results."

    assert any(
        expected_keyword in item["question"].lower()
        for item in results
    ), f"Expected an FAQ containing '{expected_keyword}'."


def test_out_of_scope_query_has_low_similarity(rag):
    results = rag.retrieve("Explain the solar system", top_k=3)

    assert results, "The retriever returned no results."

    assert results[0]["score"] < 0.35, (
        "The unrelated question matched an FAQ too strongly. "
        f"Top similarity score: {results[0]['score']:.3f}"
    )
