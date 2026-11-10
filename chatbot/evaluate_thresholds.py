

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent.parent
FAQ_FILE = PROJECT_DIR / "data" / "faqs.json"
EVAL_FILE = PROJECT_DIR / "data" / "retrieval_eval.json"


def main():
    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        faqs = json.load(file)

    with open(EVAL_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    faq_questions = [faq["question"] for faq in faqs]
    faq_embeddings = model.encode(
        faq_questions,
        normalize_embeddings=True,
    )

    # Calculate each question's best similarity score once.
    scored_cases = []

    for case in test_cases:
        query_embedding = model.encode(
            [case["question"]],
            normalize_embeddings=True,
        )

        scores = cosine_similarity(
            query_embedding,
            faq_embeddings,
        )[0]

        scored_cases.append({
            "question": case["question"],
            "answerable": case["answerable"],
            "score": float(scores.max()),
        })

    thresholds = [0.20, 0.30, 0.35, 0.40, 0.50]

    print("\nRETRIEVAL THRESHOLD EVALUATION")
    print("=" * 65)
    print(f"Evaluation questions: {len(scored_cases)}")

    for threshold in thresholds:
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for case in scored_cases:
            predicted_answerable = case["score"] >= threshold
            actual_answerable = case["answerable"]

            if predicted_answerable and actual_answerable:
                true_positive += 1
            elif not predicted_answerable and not actual_answerable:
                true_negative += 1
            elif predicted_answerable and not actual_answerable:
                false_positive += 1
            else:
                false_negative += 1

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )

        accuracy = (
            (true_positive + true_negative) / len(scored_cases)
        )

        print(f"\nThreshold: {threshold:.2f}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"True accepts: {true_positive}")
        print(f"True rejects: {true_negative}")
        print(f"False accepts: {false_positive}")
        print(f"False rejects: {false_negative}")

    print(
        "\nReminder: Similarity scores are not probabilities. "
        "These results apply only to this small labeled dataset."
    )


if __name__ == "__main__":
    main()
