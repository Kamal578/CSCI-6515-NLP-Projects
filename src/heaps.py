# src/heaps.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .load_data import load_corpus_csv
from .tokenize import iter_tokens


@dataclass
class HeapsResult:
    k: float
    beta: float
    points: List[Tuple[int, int]]  # (N, V)


def compute_heaps_points(tokens: Iterable[str], step: int = 1000) -> List[Tuple[int, int]]:
    """
    Stream tokens and record (N, V) every `step` tokens:
      N = total tokens seen
      V = unique types seen
    """
    seen = set()
    N = 0
    points: List[Tuple[int, int]] = []
    next_mark = step

    for tok in tokens:
        N += 1
        seen.add(tok)
        if N >= next_mark:
            points.append((N, len(seen)))
            next_mark += step

    # ensure final point exists
    if not points or points[-1][0] != N:
        points.append((N, len(seen)))

    return points


def fit_heaps(points: List[Tuple[int, int]]) -> Tuple[float, float]:
    """
    Fit V(N) = k * N^beta by linear regression in log space:
      log(V) = log(k) + beta * log(N)
    """
    N = np.array([p[0] for p in points], dtype=float)
    V = np.array([p[1] for p in points], dtype=float)

    # Remove any zero values (shouldn't happen, but safe)
    mask = (N > 0) & (V > 0)
    N = N[mask]
    V = V[mask]

    x = np.log(N)
    y = np.log(V)

    beta, logk = np.polyfit(x, y, 1)  # y ≈ beta*x + logk
    k = float(np.exp(logk))
    return float(k), float(beta)


def plot_heaps(points: List[Tuple[int, int]], k: float, beta: float, out_path: Path) -> None:
    N = np.array([p[0] for p in points], dtype=float)
    V = np.array([p[1] for p in points], dtype=float)
    V_hat = k * (N ** beta)

    plt.figure()
    plt.loglog(N, V, label="Observed V(N)")
    plt.loglog(N, V_hat, label=f"Fit: k={k:.2f}, β={beta:.3f}")
    plt.xlabel("Tokens N (log)")
    plt.ylabel("Vocabulary size V (log)")
    plt.title("Heaps' Law Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_heaps(
    corpus_path: str = "data/raw/corpus.csv",
    out_stats: str = "outputs/stats/heaps_params.json",
    out_plot: str = "outputs/plots/heaps.png",
    lowercase: bool = True,
    step: int = 1000,
) -> HeapsResult:
    out_stats_path = Path(out_stats)
    out_plot_path = Path(out_plot)
    out_stats_path.parent.mkdir(parents=True, exist_ok=True)
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_corpus_csv(corpus_path)
    tokens = iter_tokens(df["text"].tolist(), lowercase=lowercase)

    points = compute_heaps_points(tokens, step=step)
    k, beta = fit_heaps(points)
    plot_heaps(points, k, beta, out_plot_path)

    payload = {
        "documents": int(len(df)),
        "lowercase": bool(lowercase),
        "step": int(step),
        "k": k,
        "beta": beta,
        "final_N": int(points[-1][0]),
        "final_V": int(points[-1][1]),
    }
    out_stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Task 2] k={k:.2f}, beta={beta:.3f}")
    print(f"Saved: {out_stats_path}")
    print(f"Saved: {out_plot_path}")

    return HeapsResult(k=k, beta=beta, points=points)


if __name__ == "__main__":
    run_heaps()
