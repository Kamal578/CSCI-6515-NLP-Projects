from __future__ import annotations

from src.project4_task2_qa_data import (
    SquadExample,
    build_windowed_examples,
    load_glove_embedding_matrix,
    map_answer_chars_to_word_span,
    tokenize_qa_example,
    word_tokenize_with_offsets,
)


def test_map_answer_chars_to_word_span_handles_punctuation():
    context = "alpha, beta gamma."
    _, offsets = word_tokenize_with_offsets(context)
    answer_start = context.index("beta")
    assert map_answer_chars_to_word_span(offsets, answer_start, "beta") == (2, 2)


def test_map_answer_chars_to_word_span_uses_character_position_for_repeated_answers():
    context = "beta beta gamma"
    _, offsets = word_tokenize_with_offsets(context)
    second_answer_start = context.rindex("beta")
    assert map_answer_chars_to_word_span(offsets, second_answer_start, "beta") == (1, 1)


def test_word_tokenize_with_offsets_skips_format_only_unicode_marks():
    context = "alpha \u200e \u0651 beta"
    words, offsets = word_tokenize_with_offsets(context)

    assert words == ["alpha", "beta"]
    assert offsets == [(0, 5), (10, 14)]


def test_build_windowed_examples_keeps_only_train_windows_containing_gold_span():
    example = SquadExample(
        question_id="q1",
        title="demo",
        context="zero one two three four five",
        question="which tokens form the answer",
        answers=["three four"],
        answer_starts=[13],
    )
    tokenized = tokenize_qa_example(example, max_question_words=16)
    windows = build_windowed_examples(
        [tokenized],
        context_window_words=4,
        doc_stride_words=2,
        is_train=True,
    )

    assert len(windows) == 1
    assert windows[0].context_start_word == 2
    assert windows[0].start_position == 1
    assert windows[0].end_position == 2


def test_build_windowed_examples_emits_overlapping_eval_windows():
    example = SquadExample(
        question_id="q2",
        title="demo",
        context="zero one two three four five",
        question="which token is four",
        answers=["four"],
        answer_starts=[19],
    )
    tokenized = tokenize_qa_example(example, max_question_words=16)
    windows = build_windowed_examples(
        [tokenized],
        context_window_words=4,
        doc_stride_words=2,
        is_train=False,
    )

    starts = [window.context_start_word for window in windows]
    assert starts == [0, 2]


def test_load_glove_embedding_matrix_infers_dim_from_headered_text_vectors(tmp_path):
    glove_path = tmp_path / "vectors.txt"
    glove_path.write_text("2 4\nalpha 1 2 3 4\nbeta 5 6 7 8\n", encoding="utf-8")

    matrix, stats = load_glove_embedding_matrix(
        vocab={"<pad>": 0, "<unk>": 1, "alpha": 2, "beta": 3},
        glove_path=glove_path,
        embedding_dim=100,
        seed=42,
    )

    assert matrix.shape == (4, 4)
    assert stats["embedding_dim"] == 4
    assert stats["requested_embedding_dim"] == 100
    assert stats["requested_dim_matches_file"] is False
