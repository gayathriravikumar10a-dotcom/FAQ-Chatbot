
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chatbot.nlp import preprocess_text


# Locate the FAQ file relative to the project directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
FAQ_FILE = PROJECT_DIR / "data" / "faqs.json"


class FAQMatcher:
    def __init__(self, faq_file=FAQ_FILE):
        with open(faq_file, "r", encoding="utf-8") as file:
            self.faqs = json.load(file)

        if not self.faqs:
            raise ValueError("The FAQ knowledge base is empty.")

        self.questions = [
            faq["question"] for faq in self.faqs
        ]

        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text
        )

        self.question_vectors = self.vectorizer.fit_transform(
            self.questions
        )

    def get_answer(self, user_question, threshold=0.20):
        if not user_question or not user_question.strip():
            return {
                "answer": "Please enter a question.",
                "matched_question": None,
                "score": 0.0,
            }

        user_vector = self.vectorizer.transform([user_question])

        scores = cosine_similarity(
            user_vector,
            self.question_vectors
        )[0]

        best_index = scores.argmax()
        best_score = float(scores[best_index])
        best_faq = self.faqs[best_index]

        if best_score < threshold:
            return {
                "answer": (
                    "Sorry, I couldn't find a relevant FAQ. "
                    "Please try rephrasing your question."
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
    matcher = FAQMatcher()

    test_questions = [
        "I forgot my password",
        "How can I create a new account?",
        "How do I get help from support?",
        "Tell me about dinosaurs",
    ]

    for question in test_questions:
        result = matcher.get_answer(question)

        print("\nUser:", question)
        print("Matched FAQ:", result["matched_question"])
        print("Similarity:", result["score"])
        print("Bot:", result["answer"])
