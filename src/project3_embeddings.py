from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EmbeddingSpace:
    words: list[str]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.normed = self.vectors / np.clip(norms, 1e-12, None)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.vectors.shape[0])

    def has_word(self, word: str) -> bool:
        return word in self.word_to_idx

    def get_vector(self, word: str) -> np.ndarray:
        return self.vectors[self.word_to_idx[word]]

    def nearest_neighbors(self, word: str, top_k: int = 10) -> list[tuple[str, float]]:
        if word not in self.word_to_idx:
            return []
        q = self.normed[self.word_to_idx[word]]
        sims = self.normed @ q
        sims[self.word_to_idx[word]] = -np.inf
        k = min(top_k, len(self.words) - 1)
        idx = np.argpartition(-sims, kth=max(0, k - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.words[i], float(sims[i])) for i in idx]

    def analogy(self, a: str, b: str, c: str, top_k: int = 10) -> list[tuple[str, float]]:
        if any(w not in self.word_to_idx for w in (a, b, c)):
            return []
        q = self.get_vector(b) - self.get_vector(a) + self.get_vector(c)
        q = q / np.clip(np.linalg.norm(q), 1e-12, None)
        sims = self.normed @ q
        for w in (a, b, c):
            sims[self.word_to_idx[w]] = -np.inf
        k = min(top_k, len(self.words) - 3)
        idx = np.argpartition(-sims, kth=max(0, k - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.words[i], float(sims[i])) for i in idx]


def load_text_vectors(path: str | Path, max_words: int | None = None) -> EmbeddingSpace:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding file not found: {p}")

    words: list[str] = []
    vectors: list[np.ndarray] = []
    for word, vector in iter_text_vectors(p, max_words=max_words):
        words.append(word)
        vectors.append(vector)

    if not words:
        raise ValueError(f"No vectors could be read from: {p}")
    mat = np.vstack(vectors)
    return EmbeddingSpace(words=words, vectors=mat)


def iter_text_vectors(path: str | Path, max_words: int | None = None):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding file not found: {p}")

    dim: int | None = None
    yielded = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip().split()
        if len(first) == 2 and first[0].isdigit() and first[1].isdigit():
            dim = int(first[1])
        else:
            f.seek(0)

        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 3:
                continue
            word = parts[0]
            vals = parts[1:]
            if dim is None:
                dim = len(vals)
            if len(vals) != dim:
                continue
            try:
                vector = np.asarray([float(x) for x in vals], dtype=np.float32)
            except ValueError:
                continue
            yield word, vector
            yielded += 1
            if max_words is not None and yielded >= max_words:
                break


def evaluate_targets(space: EmbeddingSpace, targets: list[str], top_k: int = 10) -> list[dict]:
    rows: list[dict] = []
    for t in targets:
        nns = space.nearest_neighbors(t, top_k=top_k)
        if not nns:
            rows.append({"target": t, "rank": None, "neighbor": None, "cosine": None, "status": "oov"})
            continue
        for rank, (w, sim) in enumerate(nns, start=1):
            rows.append({"target": t, "rank": rank, "neighbor": w, "cosine": sim, "status": "ok"})
    return rows


def evaluate_analogies(
    space: EmbeddingSpace,
    analogies: list[tuple[str, str, str, str]],
    top_k: int = 10,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    evaluated = 0
    hit1 = 0
    hitk = 0

    for a, b, c, expected in analogies:
        missing = [w for w in (a, b, c, expected) if not space.has_word(w)]
        if missing:
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "c": c,
                    "expected": expected,
                    "predicted_top1": None,
                    "predicted_topk": None,
                    "hit_at_1": False,
                    "hit_at_k": False,
                    "status": "skipped_oov",
                    "missing_words": ",".join(missing),
                }
            )
            continue

        preds = space.analogy(a, b, c, top_k=top_k)
        top_words = [w for w, _ in preds]
        top1 = top_words[0] if top_words else None
        is_hit1 = bool(top1 == expected)
        is_hitk = bool(expected in top_words)

        evaluated += 1
        hit1 += int(is_hit1)
        hitk += int(is_hitk)

        rows.append(
            {
                "a": a,
                "b": b,
                "c": c,
                "expected": expected,
                "predicted_top1": top1,
                "predicted_topk": ",".join(top_words),
                "hit_at_1": is_hit1,
                "hit_at_k": is_hitk,
                "status": "ok",
                "missing_words": "",
            }
        )

    summary = {
        "num_total": len(analogies),
        "num_evaluated": evaluated,
        "num_skipped": len(analogies) - evaluated,
        "hit_at_1": hit1 / evaluated if evaluated else 0.0,
        "hit_at_k": hitk / evaluated if evaluated else 0.0,
    }
    return rows, summary


def average_neighbor_cosine(rows: list[dict], top_n: int = 3) -> float:
    vals = [float(r["cosine"]) for r in rows if r.get("status") == "ok" and r.get("rank") is not None and int(r["rank"]) <= top_n]
    if not vals:
        return 0.0
    return float(np.mean(vals))
