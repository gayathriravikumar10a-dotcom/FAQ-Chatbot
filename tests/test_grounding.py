

from chatbot.grounding_evaluator import evaluate_grounding


def test_supported_answer_has_high_word_overlap():
    context = (
        'To reset your password, click "Forgot Password" '
        "on the login page."
    )

    answer = (
        'Click "Forgot Password" on the login page '
        "to reset your password."
    )

    result = evaluate_grounding(answer, context)

    assert result["support_ratio"] >= 0.5
    assert result["supported"] is True


def test_unrelated_answer_has_low_word_overlap():
    context = (
        'To reset your password, click "Forgot Password" '
        "on the login page."
    )

    answer = (
        "Elephants live in forests and eat plants."
    )

    result = evaluate_grounding(answer, context)

    assert result["support_ratio"] < 0.5
    assert result["supported"] is False


def test_empty_answer_is_not_supported():
    result = evaluate_grounding("", "Some source information.")

    assert result["support_ratio"] == 0.0
    assert result["supported"] is False
