"""
Downloader/collector for Azerbaijani Wikipedia pages.
Example Run:
  python -m src.pull_wikipedia --random 1000 --out data/raw/corpus.csv
"""

import argparse
import csv
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm
import mwparserfromhell


AZWIKI_API = "https://az.wikipedia.org/w/api.php"


@dataclass
class Page:
    page_id: int
    title: str
    revision_id: Optional[int]
    timestamp: Optional[str]
    url: str
    text: str


def api_get(params: Dict, session: requests.Session, retries: int = 5, backoff: float = 1.0) -> Dict:
    """GET wrapper with basic retries."""
    for attempt in range(retries):
        try:
            r = session.get(AZWIKI_API, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))


def random_titles(n: int, session: requests.Session, namespace: int = 0) -> List[str]:
    """Fetch random page titles from main namespace (namespace=0)."""
    titles: List[str] = []
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": namespace,
        "rnlimit": min(500, n),
    }
    while len(titles) < n:
        params["rnlimit"] = min(500, n - len(titles))
        data = api_get(params, session)
        items = data.get("query", {}).get("random", [])
        titles.extend([it["title"] for it in items])
    return titles[:n]


def category_titles(category: str, n: int, session: requests.Session, namespace: int = 0) -> List[str]:
    """
    Fetch page titles from a given category (non-recursive).
    Example category: "Azərbaycan"
    """
    titles: List[str] = []
    cmcontinue = None

    # MediaWiki expects "Category:NAME" in many contexts, but the API parameter is usually the raw category title.
    # We'll accept both; normalize lightly:
    category = category.replace("Kateqoriya:", "").replace("Category:", "").strip()

    while len(titles) < n:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Kateqoriya:{category}",
            "cmnamespace": namespace,
            "cmlimit": 500,
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = api_get(params, session)
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            if "title" in m:
                titles.append(m["title"])
                if len(titles) >= n:
                    break

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return titles[:n]


def fetch_pages_wikitext(titles: List[str], session: requests.Session) -> List[Tuple[int, str, int, str, str]]:
    """
    Fetch latest revision wikitext for each title.
    Returns list of tuples:
      (page_id, title, revision_id, timestamp, wikitext)
    """
    out: List[Tuple[int, str, int, str, str]] = []

    # Batch titles for API (max URL length; typical safe batch ~ 20-50)
    batch_size = 25
    for i in tqdm(range(0, len(titles), batch_size), desc="Fetching wikitext"):
        batch = titles[i:i + batch_size]
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "titles": "|".join(batch),
            "formatversion": "2",
        }
        data = api_get(params, session)
        pages = data.get("query", {}).get("pages", [])
        for p in pages:
            if p.get("missing"):
                continue
            page_id = p.get("pageid")
            title = p.get("title")
            revs = p.get("revisions", [])
            if not revs:
                continue
            rev = revs[0]
            rev_id = rev.get("revid")
            ts = rev.get("timestamp")
            content = ""
            # Newer format puts content under slots.main.content
            slots = rev.get("slots", {})
            if isinstance(slots, dict):
                main = slots.get("main", {})
                content = main.get("content", "") or ""
            else:
                # fallback older
                content = rev.get("*", "") or ""
            if content:
                out.append((page_id, title, rev_id, ts, content))

    return out


import langid
import re
import mwparserfromhell

def clean_wikitext_to_text(wikitext: str) -> str:
    """
    Convert wikitext to plain text while removing templates, references, tables, files, categories, and sections.
    Also removes English content using language detection.
    """
    code = mwparserfromhell.parse(wikitext)

    # Remove templates, tags, files, and category links
    for tpl in code.filter_templates(recursive=True):
        try:
            code.remove(tpl)
        except Exception:
            pass

    for tag in code.filter_tags(recursive=True):
        try:
            code.remove(tag)
        except Exception:
            pass

    text = code.strip_code(normalize=True, collapse=True)

    # Remove category links and file links
    text = re.sub(r"\[\[Kateqoriya:[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(Fayl|File):[^\]]+\]\]", " ", text, flags=re.IGNORECASE)

    # Remove specific sections like "External Links" and "See also"
    text = re.sub(r"(?i)^(==\s*(External links|See also|Notes|References).+?==\s*$)(.|\n)*", "", text)

    # Remove leftover markup noise and multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Filter out English-heavy sentences based on language detection
    return remove_english_sentences(text)

def remove_english_sentences(text: str, threshold: float = 0.7) -> str:
    """
    Removes lines where the dominant language is English based on language detection.
    Threshold: fraction of words predicted as English
    """
    lines = text.splitlines()
    result = []

    for line in lines:
        lang, score = langid.classify(line)
        if lang == "en" and score > threshold:
            continue
        result.append(line)
    return "\n".join(result)



def save_corpus_csv(pages: List[Page], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["doc_id", "page_id", "title", "revision_id", "timestamp", "source", "url", "text"],
        )
        writer.writeheader()
        for idx, p in enumerate(pages):
            writer.writerow({
                "doc_id": idx,
                "page_id": p.page_id,
                "title": p.title,
                "revision_id": p.revision_id,
                "timestamp": p.timestamp,
                "source": "az.wikipedia.org",
                "url": p.url,
                "text": p.text,
            })


def main():
    ap = argparse.ArgumentParser(description="Collect Azerbaijani Wikipedia corpus via MediaWiki API.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--random", type=int, help="Number of random articles from main namespace.")
    mode.add_argument("--category", type=str, help="Category name (e.g., 'Azərbaycan' or 'Tarix'). Use with --limit.")
    ap.add_argument("--limit", type=int, default=500, help="Max number of pages to fetch (used with --category).")
    ap.add_argument("--min_chars", type=int, default=400, help="Drop docs shorter than this after cleaning.")
    ap.add_argument("--sleep", type=float, default=0.1, help="Sleep between API batches (politeness).")
    ap.add_argument("--out", type=str, default="data/raw/corpus.csv", help="Output CSV path.")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "NLP-AZWIKI-Corpus-Collector/1.0 (educational project)"})

    if args.random is not None:
        titles = random_titles(args.random, session=session)
    else:
        titles = category_titles(args.category, n=args.limit, session=session)

    raw_pages = fetch_pages_wikitext(titles, session=session)
    time.sleep(args.sleep)

    pages: List[Page] = []
    for page_id, title, rev_id, ts, wikitext in tqdm(raw_pages, desc="Cleaning"):
        clean = clean_wikitext_to_text(wikitext)
        if len(clean) < args.min_chars:
            continue
        url_title = title.replace(" ", "_")
        url = f"https://az.wikipedia.org/wiki/{url_title}"
        pages.append(Page(page_id=page_id, title=title, revision_id=rev_id, timestamp=ts, url=url, text=clean))

    save_corpus_csv(pages, args.out)
    print(f"Saved {len(pages)} docs to {args.out}")


if __name__ == "__main__":
    main()
