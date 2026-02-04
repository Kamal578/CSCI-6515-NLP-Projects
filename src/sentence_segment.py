# src/sentence_segment.py (Updated version with better logging)
from __future__ import annotations

import regex as re
from typing import List
import argparse
from pathlib import Path
import pandas as pd

# ----------------------------
# Quote characters (robust)
# ----------------------------
QUOTE_OPEN = set(['"', "'", "“", "«", "‹", "„", "‘", "‚", "「", "『", "（", "(", "[", "{"])
QUOTE_CLOSE = set(['"', "'", "”", "»", "›", "‟", "’", "‛", "」", "』", "）", ")", "]", "}"])

# ----------------------------
# Abbreviations (normalize)
# ----------------------------
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "etc", "e.g", "i.e",
    "a.m", "s.a", "b.c", "m.a", "ph.d", "u.s"
}

# Sentence-ending punctuation candidates
SENT_END = {".", "!", "?"}


def normalize_token(tok: str) -> str:
    # strip surrounding quotes and trailing punctuation
    tok = tok.strip()
    tok = tok.strip("".join(QUOTE_OPEN | QUOTE_CLOSE))
    tok = tok.rstrip(".,!?;:")
    return tok.lower()


def is_abbreviation(token: str) -> bool:
    return normalize_token(token) in ABBREVIATIONS


def is_surrounded_by_non_space(text: str, i: int) -> bool:
    """
    For '.' or ',' inside tokens like:
      154.5, a=5,1, S.Rustamov, U.S., 10km/saat (not spaces)
    """
    if i <= 0 or i >= len(text) - 1:
        return False
    return (text[i - 1] != " " and text[i + 1] != " ")


def is_decimal_dot_or_comma(text: str, i: int) -> bool:
    """
    Detect 3.14 or 2,5 (digit on both sides)
    """
    if i <= 0 or i >= len(text) - 1:
        return False
    return text[i - 1].isdigit() and text[i + 1].isdigit()


def is_initial_period(text: str, i: int) -> bool:
    """
    If '.' after a single capital letter, e.g., "A." or "S."
    """
    return i > 0 and text[i] == "." and text[i - 1].isupper()


def is_compact_initials(token: str) -> bool:
    """
    Detect forms like A.M., S.B., J.Epstein (partial), S.Rustamov
    We'll treat tokens containing a capital + '.' + (capital OR letter) as non-boundary token.
    """
    return bool(re.search(r"\b[A-Z]\.[A-ZƏÖÜİĞÇŞ]", token))


def quote_followed_by_space_upper(text: str, i: int) -> bool:
    """
    NEW RULE (generalized):
    If a closing quote is followed by space + uppercase letter, we break.
    Example: ... "citation." NewSentence
             ... » NewSentence
    """
    if text[i] not in QUOTE_CLOSE:
        return False
    if i + 2 >= len(text):
        return False
    return text[i + 1] == " " and text[i + 2].isupper()

SENT_END = {".", "!", "?"}
DECIMAL_RE = re.compile(r"\d+[.,]\d+")
INITIALS_RE = re.compile(r"(?:\b\p{L}\.){2,}$", re.UNICODE)
CATEGORY_GARBAGE_RE = re.compile(
    r"""
    (?im)
    ^\s*(kateqoriya|istinadlar|qeydlər|əlavə ədəbiyyat)\b.*$
    """,
    re.VERBOSE,
)

def is_abbreviation(token: str) -> bool:
    tok = token.strip()
    tok = tok.strip("".join(QUOTE_OPEN | QUOTE_CLOSE))
    tok = tok.rstrip(".,!?;:")
    return tok.lower() in ABBREVIATIONS

def strip_wiki_garbage(text: str) -> str:
    # Remove category/navigation-like lines
    text = CATEGORY_GARBAGE_RE.sub(" ", text)
    # Also remove standalone occurrences like "Kateqoriya:" if embedded
    text = re.sub(r"(?i)\bkateqoriya\b\s*:\s*", " ", text)
    return text

def normalize_text(s: str) -> str:
    # Clean up excessive punctuation
    s = re.sub(r"\s*[;:]+\s*", " ", s)  # Replace excessive semicolons and colons
    s = re.sub(r"\(\s*;\s*\)", " ", s)  # Clean up ( ; )
    s = re.sub(r"\s+", " ", s).strip()  # Normalize multiple spaces
    return s

def sentence_segment(text: str) -> List[str]:
    sentences: List[str] = []
    start = 0
    i = 0

    text = normalize_text(strip_wiki_garbage(text))
    n = len(text)
    if n == 0:
        return sentences

    while i < n:
        ch = text[i]

        # Closing quote + space + Uppercase => boundary
        if ch in QUOTE_CLOSE and quote_followed_by_space_upper(text, i):
            sent = text[start:i + 1].strip()
            if sent:
                sentences.append(sent)
            start = i + 1
            i += 1
            continue

        if ch in SENT_END:
            if ch == ".":
                if is_decimal_dot_or_comma(text, i):
                    i += 1
                    continue
                if is_surrounded_by_non_space(text, i):
                    i += 1
                    continue
                if is_initial_period(text, i):
                    i += 1
                    continue

            chunk = text[start:i + 1]
            prev_token = chunk.rstrip().split()[-1] if chunk.rstrip().split() else ""

            if is_abbreviation(prev_token):
                i += 1
                continue

            if ch == "." and is_compact_initials(prev_token):
                i += 1
                continue

            if i + 1 < n and text[i + 1] in QUOTE_CLOSE:
                i += 1
                continue

            sent = chunk.strip()
            if sent:
                sentences.append(sent)
            start = i + 1

        elif ch == ":":
            pass

        i += 1

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)

    return sentences


def _load_text_from_csv(corpus_path: str, limit: int | None = None) -> str:
    df = pd.read_csv(corpus_path)
    texts = df["text"].dropna().astype(str)
    if limit:
        texts = texts.head(limit)
    return " ".join(texts.tolist())


def main():
    ap = argparse.ArgumentParser(description="Sentence segmentation for Azerbaijani corpus.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text_path", type=str, help="Plain text file to segment.")
    src.add_argument("--corpus_path", type=str, help="CSV with a 'text' column (e.g., data/raw/corpus.csv).")
    ap.add_argument("--limit", type=int, default=None, help="Optional: limit number of rows from corpus.")
    ap.add_argument("--out", type=str, default="outputs/sentences.txt", help="Where to write segmented sentences.")
    args = ap.parse_args()

    if args.text_path:
        text = Path(args.text_path).read_text(encoding="utf-8")
    else:
        text = _load_text_from_csv(args.corpus_path, limit=args.limit)

    sentences = sentence_segment(text)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sentences), encoding="utf-8")
    print(f"Saved {len(sentences)} sentences to {out_path}")


if __name__ == "__main__":
    main()