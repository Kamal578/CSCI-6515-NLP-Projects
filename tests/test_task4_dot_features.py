from __future__ import annotations

from src.project2_task4_dot_features import extract_dot_features


def test_decimal_dot_feature_detected():
    text = "Qiymət 50.5 manatdır."
    idx = text.index(".")
    feats = extract_dot_features(text, idx)
    assert feats["is_decimal_pattern"] is True
    assert feats["prev_is_digit"] is True
    assert feats["next_is_digit"] is True


def test_compact_initials_pattern_detected():
    text = "A.Məlikli gəldi."
    idx = text.index(".")
    feats = extract_dot_features(text, idx)
    assert feats["joined_no_space_around_dot"] is True
    assert feats["is_compact_initials_pattern"] is True


def test_neighbor_case_and_space_flags():
    text = "Son cümlə. Yeni cümlə"
    idx = text.index(".")
    feats = extract_dot_features(text, idx)
    assert feats["prev_is_lower"] is True
    assert feats["next_is_space"] is True
    assert feats["next_nonspace_is_upper"] is True

