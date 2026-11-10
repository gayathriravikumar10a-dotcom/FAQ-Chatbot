
from unittest.mock import MagicMock

from chatbot.rag_pipeline import FAQRAG


def make_rag_with_mocked_retrieval(score):
    """Build a lightweight RAG object without loading Gemini."""

    rag = FAQRAG.__new__(FAQRAG)
    rag.client = MagicMock()
    rag.model_name = "test-model"

    rag.retrieve = MagicMock(
        return_value=[
            {
                "question": "How can I reset my password?",
                "answer": (
                    'Click "Forgot Password" on the login page.'
                ),
                "score": score,
            }
        ]
    )

    return rag


def test_unrelated_question_is_rejected():
    rag = make_rag_with_mocked_retrieval(0.10)

    result = rag.answer_question("Explain the solar system")

    assert "couldn't find relevant information" in result["answer"]
    assert result["sources"] == []
    rag.client.models.generate_content.assert_not_called()


def test_relevant_question_calls_generation():
    rag = make_rag_with_mocked_retrieval(0.90)

    response = MagicMock()
    response.text = 'Click "Forgot Password" on the login page.'
    rag.client.models.generate_content.return_value = response

    result = rag.answer_question("How can I reset my password?")

    assert "Forgot Password" in result["answer"]
    assert len(result["sources"]) == 1
    rag.client.models.generate_content.assert_called_once()


def test_empty_question_never_calls_generation():
    rag = make_rag_with_mocked_retrieval(0.90)

    result = rag.answer_question("   ")

    assert result["answer"] == "Please enter a question."
    assert result["sources"] == []
    rag.client.models.generate_content.assert_not_called()
