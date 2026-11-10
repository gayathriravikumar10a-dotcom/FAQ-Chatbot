
import re


def normalize_text(text):
    """Normalize text for simple word-level comparisons."""
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def evaluate_grounding(answer, context):
    """
    Estimate lexical support for an answer using its source context.

    This is a simple heuristic, not a reliable semantic
    hallucination detector.
    """
    answer_words = normalize_text(answer)
    context_words = normalize_text(context)

    if not answer_words:
        return {
            "support_ratio": 0.0,
            "supported": False,
            "message": "The answer is empty.",
        }

    supported_words = answer_words & context_words
    ratio = len(supported_words) / len(answer_words)

    return {
        "support_ratio": round(ratio, 3),
        "supported": ratio >= 0.5,
        "message": (
            "Many answer words occur in the source context."
            if ratio >= 0.5
            else "The answer may contain unsupported wording."
        ),
    }


if __name__ == "__main__":
    context = (
        'To reset your password, click "Forgot Password" '
        "on the login page."
    )

    answer = (
        'Click "Forgot Password" on the login page '
        "to reset your password."
    )

    result = evaluate_grounding(answer, context)

    print("\nGROUNDING EVALUATION")
    print("=" * 35)
    print(f"Support ratio: {result['support_ratio']}")
    print(f"Supported by heuristic: {result['supported']}")
    print(f"Assessment: {result['message']}")
    print("\nNote: Word overlap does not prove factual correctness.")
