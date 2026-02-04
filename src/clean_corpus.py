# src/clean_corpus.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List

import langid

# ====================
# Patterns to remove
# ====================

CATEGORY_LINE = re.compile(r"(?im)^\s*(kateqoriya:|category:).*$")
WIKI_HEADER  = re.compile(r"(?m)^==\s*(.+?)\s*==\s*$")

# Known section markers to truncate at
SECTION_BLACKLIST = {
    "İstinadlar", "References", "Xarici keçidlər",
    "External links", "Qeydlər", "Notes", "See also"
}

HTML_TAGS = re.compile(r"<[^>]+>")
WIKITEXT_TEMPLATES = re.compile(r"\{\{[^}]+\}\}")

def remove_category_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines()
        if not CATEGORY_LINE.match(line)
    )

def remove_templates(text: str) -> str:
    return WIKITEXT_TEMPLATES.sub(" ", text)

def remove_html_tags(text: str) -> str:
    return HTML_TAGS.sub(" ", text)

def remove_blacklisted_sections(text: str) -> str:
    """
    Removes whole sections like İstinadlar / References / External links.
    Everything after the first of these section headings is deleted.
    """
    lines = text.splitlines()
    cleaned: List[str] = []
    drop_remaining = False

    for line in lines:
        # check header
        m = WIKI_HEADER.match(line)
        if m:
            section_title = m.group(1).strip()
            if section_title in SECTION_BLACKLIST:
                drop_remaining = True
                break
            else:
                cleaned.append(line)
                continue

        if not drop_remaining:
            cleaned.append(line)

    return "\n".join(cleaned)

def remove_english_sentences(text: str, threshold: float = 0.7) -> str:
    """
    Removes lines where dominant language is English.
    threshold: fraction of words predicted as English
    """
    lines = text.splitlines()
    result: List[str] = []

    for line in lines:
        lang, score = langid.classify(line)
        if lang == "en" and score > threshold:
            continue
        result.append(line)
    return "\n".join(result)

def clean_wiki_page(text: str) -> str:
    # 1) remove categories
    txt = remove_category_lines(text)

    # 2) remove structured sections we don't want
    txt = remove_blacklisted_sections(txt)

    # 3) remove templates and html
    txt = remove_templates(txt)
    txt = remove_html_tags(txt)

    # 4) remove stray underscores/multiple spaces
    txt = re.sub(r"_+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    # 5) optionally remove English-heavy lines
    txt = remove_english_sentences(txt)

    return txt
