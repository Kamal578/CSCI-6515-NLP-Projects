from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple


def align_ops(src: str, tgt: str) -> List[Tuple[str, str | None, str | None]]:
    """
    Return list of operations to transform src -> tgt.
    Ops: ("match", a, a), ("sub", a, b), ("ins", None, b), ("del", a, None)
    """
    m, n = len(src), len(tgt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if src[i - 1] == tgt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # del
                dp[i][j - 1] + 1,        # ins
                dp[i - 1][j - 1] + cost, # sub/match
            )
    ops: List[Tuple[str, str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", src[i - 1], None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", None, tgt[j - 1]))
            j -= 1
        else:
            if src[i - 1] == tgt[j - 1]:
                ops.append(("match", src[i - 1], tgt[j - 1]))
            else:
                ops.append(("sub", src[i - 1], tgt[j - 1]))
            i -= 1
            j -= 1
    ops.reverse()
    return ops


def build_confusion(pairs: List[Tuple[str, str]]) -> Dict[str, Counter]:
    """
    pairs: list of (misspelled, correct)
    Returns dict with keys 'sub', 'ins', 'del' mapping to Counter.
      sub counter keyed by (src_char, tgt_char)
      ins counter keyed by tgt_char
      del counter keyed by src_char
    """
    subs = Counter()
    ins = Counter()
    dels = Counter()
    for miss, corr in pairs:
        for op, a, b in align_ops(miss, corr):
            if op == "sub":
                subs[(a, b)] += 1
            elif op == "ins":
                ins[b] += 1
            elif op == "del":
                dels[a] += 1
    return {"sub": subs, "ins": ins, "del": dels}


def weights_from_confusion(conf: Dict[str, Counter], smoothing: float = 1.0) -> Dict[Tuple, float]:
    """
    Convert confusion counts to costs (lower cost for frequent confusions).
    cost = 1 / (count + smoothing)
    """
    weights: Dict[Tuple, float] = {}
    for (a, b), c in conf["sub"].items():
        weights[("sub", a, b)] = 1.0 / (c + smoothing)
    for b, c in conf["ins"].items():
        weights[("ins", b)] = 1.0 / (c + smoothing)
    for a, c in conf["del"].items():
        weights[("del", a)] = 1.0 / (c + smoothing)
    return weights


def load_pairs(csv_path: str) -> List[Tuple[str, str]]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    return list(zip(df["misspelled"].astype(str), df["correct"].astype(str)))


def main():
    ap = argparse.ArgumentParser(description="Build character-level confusion matrix from misspelled/correct pairs.")
    ap.add_argument("--pairs_csv", type=str, default="data/processed/spell_test.csv", help="CSV with misspelled,correct")
    ap.add_argument("--out_confusion", type=str, default="outputs/spellcheck/confusion.json")
    ap.add_argument("--smoothing", type=float, default=1.0)
    args = ap.parse_args()

    pairs = load_pairs(args.pairs_csv)
    conf = build_confusion(pairs)
    weights = weights_from_confusion(conf, smoothing=args.smoothing)

    out_path = Path(args.out_confusion)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "confusion": {
            "sub": {f"{a}->{b}": int(c) for (a, b), c in conf["sub"].items()},
            "ins": {k: int(v) for k, v in conf["ins"].items()},
            "del": {k: int(v) for k, v in conf["del"].items()},
        },
        "weights": {str(k): v for k, v in weights.items()},
        "smoothing": args.smoothing,
        "pairs_csv": args.pairs_csv,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote confusion + weights to {out_path}")


if __name__ == "__main__":
    main()
