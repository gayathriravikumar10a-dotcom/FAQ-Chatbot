
from unittest.mock import MagicMock

import pytest

from chatbot.rag_pipeline import FAQRAG


@pytest.fixture
def rag():
    """Create a lightweight RAG object without calling Gemini."""

    instance = FAQRAG.__new__(FAQRAG)

    instance.retrieve = MagicMock(
        return_value=[
            {
                "question": "How can I reset my password?",
                "answer": (
                    'To reset your password, click "Forgot Password" '
                    "on the login page."
                ),
                "score": 0.90,
            }
        ]
    )

    instance.client = MagicMock()
    instance.model_name = "test-model"

    return instance


def test_answer_uses_retrieved_context(rag):
    mock_response = MagicMock()
    mock_response.text = (
        'Click "Forgot Password" on the login page to reset your password.'
    )

    rag.client.models.generate_content.return_value = mock_response

    result = rag.answer_question("How do I reset my password?")

    assert "Forgot Password" in result["answer"]
    assert len(result["sources"]) == 1


def test_empty_question_is_handled(rag):
    result = rag.answer_question("   ")

    assert result["answer"] == "Please enter a question."
    assert result["sources"] == []

    rag.client.models.generate_content.assert_not_called()


def test_irrelevant_question_is_rejected(rag):
    rag.retrieve.return_value = [
        {
            "question": "How can I reset my password?",
            "answer": "Click Forgot Password.",
            "score": 0.10,
        }
    ]

    result = rag.answer_question("Explain the solar system")

    assert "couldn't find relevant information" in result["answer"]
    assert result["sources"] == []

    rag.client.models.generate_content.assert_not_called()


def test_gemini_failure_is_handled(rag):
    rag.client.models.generate_content.side_effect = RuntimeError(
        "Simulated Gemini failure"
    )

    result = rag.answer_question("How do I reset my password?")

    assert "temporarily unavailable" in result["answer"]
    assert len(result["sources"]) == 1
