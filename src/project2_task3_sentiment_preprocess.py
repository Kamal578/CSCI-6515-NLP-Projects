from __future__ import annotations

import re


# Adapted from the user's original notebook/script preprocessing to keep results comparable.
NON_LATIN_LETTERS = {
    "ç": "c",
    "ə": "e",
    "ı": "i",
    "ğ": "g",
    "ö": "o",
    "ş": "s",
    "ü": "u",
    "ch": "c",
    "sh": "s",
    "gh": "g",
}

COMMON_WORDS = {
    "men", "sen", "o", "biz", "siz", "onlar", "ne", "kim", "hara", "niye", "nece", "hansi", "ne", "vaxt",
    "sonra", "eger", "heqiqeten", "lakin", "cunki", "bu", "ki", "butun", "ve", "ya", "veya", "amma", "yoxsa",
    "ancag", "sadece", "qisa", "uzun", "kicik", "boyuk", "ora", "bura", "sag", "sol", "salam", "bele", "cox",
    "az", "e", "bir", "her",
}

SUFFIXES = [
    "lar", "ler", "larin", "lerin", "mis", "mak", "mek", "liq", "luq", "acaq", "eceq", "ma", "m", "am", "em",
    "ar", "er", "araq", "ereq", "arak", "erek", "ca", "ce", "ci", "cu", "da", "de", "dan", "den", "di", "diq",
    "dir", "du", "duq", "dur", "duk", "ib", "ici", "il", "inci", "uncu", "istan", "is", "in", "la", "le",
    "las", "les", "luk", "maq", "mus", "n", "nci", "ncu", "s", "ub", "ucu", "ul", "ustan", "us", "y", "cil",
    "dar", "der", "an", "en", "gec", "kar", "kes", "ken", "lik", "t", "me", "nan", "nen", "ova", "ov", "san",
    "siniz", "sul", "sunas",
]

EMOJIS = {
    "👍", "👎", "👌", "🙃", "😉", "❤️", "🖤", "💔", "💕", "💖", "💗", "💘", "💙", "💚", "💛", "💜", "💝", "💞", "💟",
    "💠", "🤗", "🤔", "🤣", "🤤", "🤥", "🤦", "🤧", "🤨", "🤩", "🤪", "🤫", "🤬", "🤭", "🤮", "🤯", "🤰", "🤱", "🤲",
    "🎉", "😡", "🥰",
}

NON_LATIN_PATTERN = re.compile("|".join(map(re.escape, sorted(NON_LATIN_LETTERS, key=len, reverse=True))))
PUNCT_TABLE = str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")


def replace_non_latins(text: str) -> str:
    return NON_LATIN_PATTERN.sub(lambda m: NON_LATIN_LETTERS[m.group(0)], text)


def legacy_preprocess_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = replace_non_latins(text)
    text = text.replace("-", " ")
    text = text.translate(PUNCT_TABLE)
    tokens = [w for w in text.split() if w and w not in COMMON_WORDS]
    return " ".join(tokens)


def legacy_tokenize(text: str) -> list[str]:
    text = legacy_preprocess_text(text)
    if not text:
        return []

    out: list[str] = []
    for raw_word in text.split():
        word = raw_word
        # Approximate a lightweight stemming pass from the original experiment.
        for suffix in SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix):
                word = word[: -len(suffix)]

        morphemes: list[str] = []
        for emoji in EMOJIS:
            if emoji in word:
                morphemes.append(emoji)
                word = word.replace(emoji, "")

        if word and word not in COMMON_WORDS:
            morphemes.append(word)

        morphemes.reverse()
        out.extend([m for m in morphemes if m])
    return out
