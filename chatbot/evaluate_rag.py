
from chatbot.rag_pipeline import FAQRAG


def evaluate_recall_at_k():
    rag = FAQRAG()

    # Each question has a known correct FAQ.
    test_cases = [
        {
            "question": "I forgot my password",
            "expected_faq": "How can I reset my password?",
        },
        {
            "question": "How can I register for an account?",
            "expected_faq": "How do I create an account?",
        },
        {
            "question": "How can I contact customer support?",
            "expected_faq": "How do I contact customer support?",
        },
        {
            "question": "How do I get my money back?",
            "expected_faq": "How do I get a refund?",
        },
    ]

    k = 3
    hits = 0

    print("\nRECALL@K EVALUATION")
    print("=" * 55)

    for test in test_cases:
        question = test["question"]
        expected_faq = test["expected_faq"]

        results = rag.retrieve(question, top_k=k)

        retrieved_questions = [
            item["question"].strip().lower()
            for item in results
        ]

        expected = expected_faq.strip().lower()
        found = expected in retrieved_questions

        if found:
            hits += 1

        print(f"\nQuestion: {question}")
        print(f"Expected FAQ: {expected_faq}")
        print(f"Found in top {k}: {'YES' if found else 'NO'}")

        if results:
            print("Retrieved FAQs:")
            for item in results:
                print(
                    f"  - {item['question']} "
                    f"(score: {item['score']:.3f})"
                )

    recall = hits / len(test_cases)

    print("\n" + "=" * 55)
    print(f"Recall@{k}: {recall:.1%}")
    print(f"Correct FAQs retrieved: {hits}/{len(test_cases)}")


if __name__ == "__main__":
    evaluate_recall_at_k()
