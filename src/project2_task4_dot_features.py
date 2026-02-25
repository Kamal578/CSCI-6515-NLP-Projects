from __future__ import annotations

from typing import Any

import regex as re

from .sentence_segment import (
    ABBREVIATIONS,
    QUOTE_CLOSE,
    QUOTE_OPEN,
    is_abbreviation,
    is_compact_initials,
    is_decimal_dot_or_comma,
    is_initial_period,
    is_surrounded_by_non_space,
)


TOKEN_CHAR_RE = re.compile(r"[\p{L}\p{N}_'’\-]+", re.UNICODE)
WINDOW_SIZE = 40
CHAR_WINDOW_RADIUS = 4


def _char(text: str, idx: int) -> str:
    return text[idx] if 0 <= idx < len(text) else ""


def _safe_slice(text: str, start: int, end: int) -> str:
    return text[max(0, start) : min(len(text), end)]


def _find_prev_nonspace(text: str, i: int) -> int:
    j = i - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    return j


def _find_next_nonspace(text: str, i: int) -> int:
    j = i + 1
    while j < len(text) and text[j].isspace():
        j += 1
    return j


def _scan_prev_token(text: str, i: int) -> str:
    j = i - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    if j < 0:
        return ""
    end = j + 1
    while j >= 0 and TOKEN_CHAR_RE.fullmatch(text[j]):
        j -= 1
    tok = text[j + 1 : end]
    return tok


def _scan_next_token(text: str, i: int) -> str:
    j = i + 1
    while j < len(text) and text[j].isspace():
        j += 1
    if j >= len(text):
        return ""
    start = j
    while j < len(text) and TOKEN_CHAR_RE.fullmatch(text[j]):
        j += 1
    return text[start:j]


def _normalize_token(tok: str) -> str:
    tok = tok.strip()
    tok = tok.strip("".join(QUOTE_OPEN | QUOTE_CLOSE))
    tok = tok.rstrip(".,!?;:")
    return tok.lower()


def _prefix(tok: str, n: int) -> str:
    return tok[:n] if tok else ""


def _suffix(tok: str, n: int) -> str:
    return tok[-n:] if tok else ""


def _char_type(ch: str) -> str:
    if not ch:
        return "none"
    if ch.isspace():
        return "space"
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        if ch.isupper():
            return "upper"
        if ch.islower():
            return "lower"
        return "alpha"
    if ch in QUOTE_OPEN or ch in QUOTE_CLOSE:
        return "quote"
    return "other"


def _next_nonspace_char(text: str, i: int) -> str:
    j = _find_next_nonspace(text, i)
    return _char(text, j)


def _prev_nonspace_char(text: str, i: int) -> str:
    j = _find_prev_nonspace(text, i)
    return _char(text, j)


def rule_guess_eos_for_dot(text: str, i: int) -> int:
    """
    Approximate the existing rule-based segmenter's dot decision for this position.
    Returns 1 if '.' is treated as end-of-sentence, else 0.
    """
    if i < 0 or i >= len(text) or text[i] != ".":
        raise ValueError("rule_guess_eos_for_dot expects a '.' index")

    if is_decimal_dot_or_comma(text, i):
        return 0
    if is_surrounded_by_non_space(text, i):
        return 0
    if is_initial_period(text, i):
        return 0

    chunk = text[: i + 1]
    prev_token = chunk.rstrip().split()[-1] if chunk.rstrip().split() else ""

    if is_abbreviation(prev_token):
        return 0
    if is_compact_initials(prev_token):
        return 0

    j = i + 1
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j].islower():
        return 0

    if i + 1 < len(text) and text[i + 1] in QUOTE_CLOSE:
        return 0

    return 1


def extract_dot_features(text: str, dot_index: int) -> dict[str, Any]:
    if dot_index < 0 or dot_index >= len(text):
        raise ValueError("dot_index out of range")
    if text[dot_index] != ".":
        raise ValueError(f"Expected '.' at index {dot_index}, got {text[dot_index]!r}")

    prev_char = _char(text, dot_index - 1)
    next_char = _char(text, dot_index + 1)
    prev2_char = _char(text, dot_index - 2)
    next2_char = _char(text, dot_index + 2)
    prev_ns = _prev_nonspace_char(text, dot_index)
    next_ns = _next_nonspace_char(text, dot_index)

    prev_token = _scan_prev_token(text, dot_index)
    next_token = _scan_next_token(text, dot_index)
    prev_token_norm = _normalize_token(prev_token)
    next_token_norm = _normalize_token(next_token)

    joined_no_space = (
        dot_index > 0
        and dot_index + 1 < len(text)
        and (not text[dot_index - 1].isspace())
        and (not text[dot_index + 1].isspace())
    )

    char_window = _safe_slice(text, dot_index - CHAR_WINDOW_RADIUS, dot_index + CHAR_WINDOW_RADIUS + 1)
    window_text = _safe_slice(text, dot_index - WINDOW_SIZE, dot_index + WINDOW_SIZE + 1)

    feat: dict[str, Any] = {
        "prev_char": prev_char,
        "next_char": next_char,
        "prev2_char": prev2_char,
        "next2_char": next2_char,
        "prev_nonspace_char": prev_ns,
        "next_nonspace_char": next_ns,
        "char_window": char_window,
        "window_text_short": window_text,
        "prev_is_digit": prev_char.isdigit(),
        "next_is_digit": next_char.isdigit(),
        "prev_is_upper": prev_char.isupper(),
        "next_is_upper": next_char.isupper(),
        "prev_is_lower": prev_char.islower(),
        "next_is_lower": next_char.islower(),
        "prev_is_space": prev_char.isspace() if prev_char else False,
        "next_is_space": next_char.isspace() if next_char else False,
        "prev_is_quote": prev_char in QUOTE_OPEN or prev_char in QUOTE_CLOSE,
        "next_is_quote": next_char in QUOTE_OPEN or next_char in QUOTE_CLOSE,
        "next_nonspace_is_upper": next_ns.isupper() if next_ns else False,
        "next_nonspace_is_lower": next_ns.islower() if next_ns else False,
        "prev_nonspace_is_digit": prev_ns.isdigit() if prev_ns else False,
        "next_nonspace_is_digit": next_ns.isdigit() if next_ns else False,
        "prev_token": prev_token[:40],
        "next_token": next_token[:40],
        "prev_token_lower": prev_token_norm[:40],
        "next_token_lower": next_token_norm[:40],
        "prev_token_len": len(prev_token_norm),
        "next_token_len": len(next_token_norm),
        "prev_token_is_single_upper": len(prev_token_norm) == 1 and prev_token.isupper(),
        "prev_token_looks_abbrev": bool(prev_token_norm and is_abbreviation(prev_token)),
        "prev_token_looks_initials": bool(prev_token and is_compact_initials(prev_token)),
        "prev_token_contains_digit": any(c.isdigit() for c in prev_token),
        "next_token_contains_digit": any(c.isdigit() for c in next_token),
        "joined_no_space_around_dot": joined_no_space,
        "is_decimal_pattern": is_decimal_dot_or_comma(text, dot_index),
        "is_compact_initials_pattern": bool(
            (prev_token and is_compact_initials(prev_token))
            or (
                joined_no_space
                and prev_char.isupper()
                and next_char.isalpha()
                and next_char.isupper()
            )
        ),
        "is_known_abbreviation": prev_token_norm in ABBREVIATIONS,
        "quote_then_space_upper": (
            dot_index + 3 < len(text)
            and text[dot_index + 1] in QUOTE_CLOSE
            and text[dot_index + 2].isspace()
            and text[dot_index + 3].isupper()
        ),
        "end_of_text_after_dot": _find_next_nonspace(text, dot_index) >= len(text),
        "line_break_after_dot": "\n" in _safe_slice(text, dot_index + 1, dot_index + 4),
        "prev_prefix_1": _prefix(prev_token_norm, 1),
        "prev_prefix_2": _prefix(prev_token_norm, 2),
        "prev_prefix_3": _prefix(prev_token_norm, 3),
        "prev_suffix_1": _suffix(prev_token_norm, 1),
        "prev_suffix_2": _suffix(prev_token_norm, 2),
        "prev_suffix_3": _suffix(prev_token_norm, 3),
        "next_token_first_char_type": _char_type(next_token[0] if next_token else ""),
    }
    return feat


def extract_features_from_row(row: dict[str, Any], doc_text: str) -> dict[str, Any]:
    dot_index = int(row["char_index"])
    return extract_dot_features(doc_text, dot_index)


def dot_candidate_preview_fields(text: str, dot_index: int, window: int = 40) -> dict[str, Any]:
    prev_token = _scan_prev_token(text, dot_index)
    next_token = _scan_next_token(text, dot_index)
    return {
        "left_context": _safe_slice(text, max(0, dot_index - window), dot_index),
        "right_context": _safe_slice(text, dot_index + 1, min(len(text), dot_index + 1 + window)),
        "window_text": _safe_slice(text, dot_index - window, dot_index + window + 1),
        "prev_char": _char(text, dot_index - 1),
        "next_char": _char(text, dot_index + 1),
        "prev_token": prev_token,
        "next_token": next_token,
        "rule_guess": rule_guess_eos_for_dot(text, dot_index),
    }
