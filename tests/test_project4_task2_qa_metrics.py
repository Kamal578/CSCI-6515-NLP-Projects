from __future__ import annotations

import torch

from src.project4_task2_qa_metrics import (
    exact_match_score,
    f1_score,
    metric_max_over_ground_truths,
    normalize_answer,
    select_best_span,
)


def test_normalize_answer_strips_case_punctuation_and_articles():
    assert normalize_answer("The, Quick Brown Fox!") == "quick brown fox"


def test_exact_match_and_f1_follow_squad_style():
    assert exact_match_score("The answer", "answer") == 1.0
    assert f1_score("alpha beta", "alpha gamma") == 0.5


def test_metric_max_over_ground_truths_uses_best_gold_answer():
    prediction = "new york city"
    gold_answers = ["york", "new york city", "city"]
    assert metric_max_over_ground_truths(exact_match_score, prediction, gold_answers) == 1.0


def test_select_best_span_respects_mask_and_answer_length():
    start_logits = torch.tensor([0.1, 2.5, 0.3, 4.0, -5.0])
    end_logits = torch.tensor([0.1, 0.2, 3.0, 0.5, -5.0])
    mask = torch.tensor([True, True, True, True, False])

    start_idx, end_idx, score = select_best_span(start_logits, end_logits, mask, max_answer_words=2)
    assert (start_idx, end_idx) == (1, 2)
    assert score > 0
