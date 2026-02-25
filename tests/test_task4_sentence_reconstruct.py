from __future__ import annotations

from src.task4_sentence_utils import split_on_boundaries


def test_split_on_predicted_dot_boundaries_preserves_order():
    text = "Salam. Necəsən. Mən yaxşıyam"
    dot_indices = {5, 14}
    sents, bounds = split_on_boundaries(text, dot_indices)
    assert sents == ["Salam.", "Necəsən.", "Mən yaxşıyam"]
    assert bounds == [5, 14]


def test_split_handles_quotes_and_question_marks():
    text = 'O dedi." Salam?" Son.'
    first_dot_idx = text.index(".")
    dot_idx = text.rindex(".")
    sents, bounds = split_on_boundaries(text, {first_dot_idx, dot_idx})
    assert len(sents) == 3
    assert sents[0].endswith('."')
    assert sents[1].endswith('?"')
    assert bounds[-1] == dot_idx
