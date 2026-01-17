# Evaluation module for agent responses
# Citation evaluators for offline evaluation against ground truth
from .citation_evaluators import (
    evaluate_citation_grounding,
    evaluate_citation_presence,
    evaluate_citation_accuracy,
    evaluate_answer_correctness,
)

# Legacy response format evaluator (keeping for backwards compatibility)
try:
    from .response_format import evaluate_response_format
    __all__ = [
        "evaluate_citation_grounding",
        "evaluate_citation_presence",
        "evaluate_citation_accuracy",
        "evaluate_answer_correctness",
        "evaluate_response_format",  # legacy
    ]
except ImportError:
    __all__ = [
        "evaluate_citation_grounding",
        "evaluate_citation_presence",
        "evaluate_citation_accuracy",
        "evaluate_answer_correctness",
    ]
