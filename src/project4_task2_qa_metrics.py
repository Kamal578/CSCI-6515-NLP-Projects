from __future__ import annotations

import collections
import string
from dataclasses import dataclass

import torch


_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT_TABLE)
    no_articles = " ".join(token for token in no_punct.split() if token not in _ARTICLES)
    return " ".join(no_articles.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    if not ground_truths:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, truth) for truth in ground_truths)


@dataclass(slots=True)
class SquadMetricRecord:
    question_id: str
    prediction_text: str
    gold_answers: list[str]
    exact_match: float
    f1: float
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.question_id,
            "prediction_text": self.prediction_text,
            "gold_answers": self.gold_answers,
            "exact_match": self.exact_match,
            "f1": self.f1,
            "score": self.score,
        }


def aggregate_squad_metrics(records: list[SquadMetricRecord]) -> dict[str, float]:
    if not records:
        return {"exact_match": 0.0, "f1": 0.0, "num_examples": 0}
    return {
        "exact_match": sum(r.exact_match for r in records) / len(records),
        "f1": sum(r.f1 for r in records) / len(records),
        "num_examples": len(records),
    }


def select_best_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    mask: torch.Tensor,
    max_answer_words: int,
) -> tuple[int, int, float]:
    valid_indices = mask.nonzero(as_tuple=False).flatten().tolist()
    if not valid_indices:
        return 0, 0, float("-inf")

    best_start = valid_indices[0]
    best_end = valid_indices[0]
    best_score = float("-inf")
    for start_idx in valid_indices:
        max_end = min(valid_indices[-1], start_idx + max_answer_words - 1)
        for end_idx in range(start_idx, max_end + 1):
            if not bool(mask[end_idx]):
                continue
            score = float(start_logits[start_idx].item() + end_logits[end_idx].item())
            if score > best_score:
                best_start = start_idx
                best_end = end_idx
                best_score = score
    return best_start, best_end, best_score
