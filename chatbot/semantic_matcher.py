
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent.parent
FAQ_FILE = PROJECT_DIR / "data" / "faqs.json"


class SemanticFAQMatcher:
    def __init__(self):
        with open(FAQ_FILE, "r", encoding="utf-8") as file:
            self.faqs = json.load(file)

        if not self.faqs:
            raise ValueError("The FAQ knowledge base is empty.")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.questions = [
            faq["question"] for faq in self.faqs
        ]

        self.embeddings = self.model.encode(
            self.questions,
            normalize_embeddings=True
        )

    def get_answer(self, user_question, threshold=0.35):
        if not user_question or not user_question.strip():
            return {
                "answer": "Please enter a question.",
                "matched_question": None,
                "score": 0.0,
            }

        query_embedding = self.model.encode(
            [user_question],
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        best_index = int(scores.argmax())
        best_score = float(scores[best_index])
        best_faq = self.faqs[best_index]

        if best_score < threshold:
            return {
                "answer": (
                    "I couldn't find a sufficiently relevant FAQ. "
                    "Please try another question."
                ),
                "matched_question": None,
                "score": round(best_score, 3),
            }

        return {
            "answer": best_faq["answer"],
            "matched_question": best_faq["question"],
            "score": round(best_score, 3),
        }


if __name__ == "__main__":
    matcher = SemanticFAQMatcher()

    tests = [
        "I cannot remember my login password",
        "I want to register for the first time",
        "Where can I get assistance?",
        "What is the capital of Mars?",
    ]

    for question in tests:
        result = matcher.get_answer(question)

        print("\nUser:", question)
        print("Matched FAQ:", result["matched_question"])
        print("Similarity:", result["score"])
        print("Bot:", result["answer"])
