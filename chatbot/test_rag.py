
from chatbot.rag_pipeline import FAQRAG


def main():
    rag = FAQRAG()

    tests = [
        {
            "name": "Relevant password question",
            "question": "How can I reset my password?",
            "expected_topic": "password",
        },
        {
            "name": "Relevant refund question",
            "question": "How do I get a refund?",
            "expected_topic": "refund",
        },
        {
            "name": "Out-of-scope question",
            "question": "Explain the solar system.",
            "expected_topic": None,
        },
    ]

    passed = 0

    print("\nRAG AUTOMATED RETRIEVAL TESTS")
    print("=" * 45)

    for test in tests:
        question = test["question"]
        expected = test["expected_topic"]

        results = rag.retrieve(question, top_k=3)

        if expected is None:
            # An out-of-scope question should not match
            # any FAQ above the configured threshold.
            actual_score = results[0]["score"] if results else 0.0
            success = actual_score < 0.35
        else:
            # At least one retrieved FAQ should contain
            # the expected topic in its question.
            success = any(
                expected in item["question"].lower()
                for item in results
            )

        status = "PASS" if success else "FAIL"

        if success:
            passed += 1

        print(f"\nTest: {test['name']}")
        print(f"Question: {question}")
        print(f"Result: {status}")

        if results:
            print(
                f"Top FAQ: {results[0]['question']}"
            )
            print(
                f"Top similarity: {results[0]['score']:.3f}"
            )

    print("\n" + "=" * 45)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Pass rate: {passed / len(tests) * 100:.1f}%")


if __name__ == "__main__":
    main()
