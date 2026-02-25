from __future__ import annotations


def split_on_boundaries(text: str, dot_eos_indices: set[int]) -> tuple[list[str], list[int]]:
    """
    Returns (sentences, boundary_char_indices).
    Splits on predicted EOS dots and always on !/?. Includes trailing quotes/brackets with the sentence.
    """
    boundaries: list[int] = []
    sentences: list[str] = []
    start = 0
    i = 0
    closers = set(['"', "'", "”", "»", "›", ")", "]", "}", "’"])
    while i < len(text):
        ch = text[i]
        split_here = False
        if ch == "." and i in dot_eos_indices:
            split_here = True
        elif ch in {"!", "?"}:
            split_here = True
        if split_here:
            punct_i = i
            j = i + 1
            while j < len(text) and text[j] in closers:
                j += 1
            sent = text[start:j].strip()
            if sent:
                sentences.append(sent)
                boundaries.append(punct_i)
            start = j
            i = j
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences, boundaries

