
import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from chatbot.rag_pipeline import FAQRAG


PROJECT_DIR = Path(__file__).resolve().parent.parent
FAQ_FILE = PROJECT_DIR / "data" / "faqs.json"


def build_retriever():
    """Load the FAQ retrieval components without initializing Gemini."""

    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        faqs = json.load(file)

    rag = FAQRAG.__new__(FAQRAG)
    rag.faqs = faqs
    rag.questions = [faq["question"] for faq in faqs]

    rag.embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    rag.embeddings = rag.embedding_model.encode(
        rag.questions,
        normalize_embeddings=True,
    )

    return rag


def main():
    rag = build_retriever()

    test_cases = [
        {
            "question": "I forgot my password",
            "expected": "How can I reset my password?",
        },
        {
            "question": "How can I register for an account?",
            "expected": "How do I create an account?",
        },
        {
            "question": "How can I contact customer support?",
            "expected": "How do I contact customer support?",
        },
        {
            "question": "How do I get my money back?",
            "expected": "How do I get a refund?",
        },
    ]

    total_reciprocal_rank = 0.0
    total_recall = 0
    total_latency = 0.0
    k = 3

    print("\nRAG RETRIEVAL EVALUATION")
    print("=" * 60)

    for case in test_cases:
        start = time.perf_counter()
        results = rag.retrieve(case["question"], top_k=k)
        latency = time.perf_counter() - start

        total_latency += latency

        retrieved_questions = [
            item["question"].strip().lower() for item in results
        ]
        expected = case["expected"].strip().lower()

        hit = expected in retrieved_questions
        total_recall += int(hit)

        if hit:
            rank = retrieved_questions.index(expected) + 1
            reciprocal_rank = 1 / rank
        else:
            reciprocal_rank = 0.0

        total_reciprocal_rank += reciprocal_rank

        status = "PASS" if hit else "FAIL"

        print(f"\nQuestion: {case['question']}")
        print(f"Expected FAQ: {case['expected']}")
        print(f"Retrieved: {results[0]['question'] if results else 'None'}")
        print(f"Recall@{k} hit: {status}")
        print(f"Reciprocal rank: {reciprocal_rank:.3f}")
        print(f"Retrieval latency: {latency * 1000:.2f} ms")

    count = len(test_cases)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Recall@{k}: {total_recall / count:.2%}")
    print(f"MRR: {total_reciprocal_rank / count:.3f}")
    print(f"Pass rate: {total_recall / count:.2%}")
    print(f"Average retrieval latency: {total_latency / count * 1000:.2f} ms")
    print("\nNote: These metrics evaluate retrieval, not answer correctness.")


if __name__ == "__main__":
    main()
