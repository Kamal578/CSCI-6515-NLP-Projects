from __future__ import annotations

import json
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_confusion(confusion_json: str):
    data = json.loads(Path(confusion_json).read_text(encoding="utf-8"))
    sub = data.get("confusion", {}).get("sub", {})
    # sub keys are like "a->b": count
    pairs = []
    for k, v in sub.items():
        if "->" not in k:
            continue
        a, b = k.split("->", 1)
        pairs.append((a, b, v))
    df = pd.DataFrame(pairs, columns=["src", "tgt", "count"])
    return df


def plot_confusion(df: pd.DataFrame, out_path: str, top_n: int = 25):
    # keep top pairs by count
    df = df.sort_values("count", ascending=False).head(top_n)
    pivot = df.pivot_table(index="src", columns="tgt", values="count", aggfunc="sum", fill_value=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt="g", cmap="OrRd")
    plt.xlabel("Target character")
    plt.ylabel("Source character")
    plt.title(f"Top {top_n} substitution confusions")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot substitution confusion matrix heatmap.")
    ap.add_argument("--confusion", type=str, default="outputs/spellcheck/confusion.json")
    ap.add_argument("--out", type=str, default="outputs/spellcheck/confusion_heatmap.png")
    ap.add_argument("--top_n", type=int, default=25, help="Top substitution pairs to plot.")
    args = ap.parse_args()

    df = load_confusion(args.confusion)
    if df.empty:
        raise SystemExit("No substitution data found in confusion.json")
    plot_confusion(df, args.out, top_n=args.top_n)
    print(f"Saved confusion heatmap to {args.out}")


if __name__ == "__main__":
    main()
