import re
from typing import List

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


def sentence_segment(text: str) -> List[str]:
    sentences = []
    start = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # ----------------------------
        # Priority rule: closing quote + space + Uppercase => boundary
        # ----------------------------
        if ch in QUOTE_CLOSE and quote_followed_by_space_upper(text, i):
            sent = text[start:i + 1].strip()
            if sent:
                sentences.append(sent)
            start = i + 1
            i += 1
            continue

        # ----------------------------
        # Sentence-ending punctuation: . ! ?
        # ----------------------------
        if ch in SENT_END:

            # 1) Never treat ':' as an end (handled elsewhere) — no action needed here

            # 2) For '.' specifically, ignore decimals / inside-token punctuation
            if ch == ".":
                if is_decimal_dot_or_comma(text, i):
                    i += 1
                    continue
                if is_surrounded_by_non_space(text, i):
                    # catches S.Rustamov, U.S., 154.5$
                    i += 1
                    continue
                if is_initial_period(text, i):
                    i += 1
                    continue

            # Examine the token before punctuation
            chunk = text[start:i + 1]
            prev_token = chunk.rstrip().split()[-1] if chunk.rstrip().split() else ""

            # 3) abbreviations (prof., dr., etc.)
            if is_abbreviation(prev_token):
                i += 1
                continue

            # 4) compact initials like A.M., S.B., S.Rustamov
            if ch == "." and is_compact_initials(prev_token):
                i += 1
                continue

            # 5) If punctuation is followed by a closing quote, we *might* want to
            #    delay decision until after the quote rule triggers.
            #    Example: ..."citation." NewSentence
            #    We'll allow boundary here only if the next char isn't a quote close.
            if i + 1 < n and text[i + 1] in QUOTE_CLOSE:
                # don't split yet; let the quote rule decide
                i += 1
                continue

            # Otherwise, accept boundary
            sent = chunk.strip()
            if sent:
                sentences.append(sent)
            start = i + 1

        # ----------------------------
        # Colon is explicitly NOT a sentence ender
        # ----------------------------
        elif ch == ":":
            # do nothing; keep scanning
            pass

        i += 1

    # Add remaining tail
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)

    return sentences


if __name__ == "__main__":
    text = r'Salam. J.Epstein was in ADA! today "citation." and "another citation." New sentence? prof. Rustamov salam deyir. S. Rustamov salam deyir! S.Rustamov salam deyir!'
    for s in sentence_segment(text):
        print("--------\n" + s)
