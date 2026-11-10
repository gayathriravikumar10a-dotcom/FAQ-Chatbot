
from unittest.mock import MagicMock

from chatbot.rag_pipeline import FAQRAG


def main():
    rag = FAQRAG()

    # Replace Gemini generation with a predictable test response.
    rag.client = MagicMock()

    test_cases = [
        {
            "question": "How do I reset my password?",
            "expected_keyword": "password",
            "mock_answer": (
                'Click "Forgot Password" on the login page '
                "to reset your password."
            ),
        },
        {
            "question": "How do I get a refund?",
            "expected_keyword": "refund",
            "mock_answer": (
                "Please consult the FAQ information about refunds."
            ),
        },
    ]

    passed = 0

    print("\nEND-TO-END RAG EVALUATION")
    print("=" * 45)

    for case in test_cases:
        mock_response = MagicMock()
        mock_response.text = case["mock_answer"]

        rag.client.models.generate_content.return_value = mock_response

        result = rag.answer_question(case["question"])

        retrieved_questions = [
            source["question"].lower()
            for source in result["sources"]
        ]

        retrieval_ok = any(
            case["expected_keyword"] in question
            for question in retrieved_questions
        )

        answer_ok = bool(result["answer"].strip())

        test_passed = retrieval_ok and answer_ok
        passed += int(test_passed)

        print(f"\nQuestion: {case['question']}")
        print(f"Retrieved relevant FAQ: {retrieval_ok}")
        print(f"Answer produced: {answer_ok}")
        print(f"Result: {'PASS' if test_passed else 'FAIL'}")

    print("\n" + "=" * 45)
    print(f"Passed: {passed}/{len(test_cases)}")
    print(
        "Note: Generation is mocked; this checks pipeline integration "
        "and retrieval, not real LLM answer quality."
    )


if __name__ == "__main__":
    main()
