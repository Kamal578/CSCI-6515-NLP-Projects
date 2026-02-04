from __future__ import annotations

from typing import Dict, Tuple, Optional


# weights keys:
#   ("sub", a, b) : cost to substitute a->b
#   ("ins", b)    : cost to insert b
#   ("del", a)    : cost to delete a
# Defaults to 1.0 when not found.


def get_cost(weights: Dict[Tuple, float], op: str, a: str | None, b: str | None) -> float:
    if op == "sub":
        return float(weights.get(("sub", a, b), 1.0))
    if op == "ins":
        return float(weights.get(("ins", b), 1.0))
    if op == "del":
        return float(weights.get(("del", a), 1.0))
    raise ValueError(f"Unknown op {op}")


def weighted_levenshtein(a: str, b: str, weights: Dict[Tuple, float], max_cost: Optional[float] = None) -> float:
    if a == b:
        return 0.0

    la, lb = len(a), len(b)
    if max_cost is not None and abs(la - lb) > max_cost:
        return max_cost + 1

    # Ensure b is shorter
    if lb > la:
        a, b = b, a
        la, lb = lb, la

    prev = [float(j) for j in range(lb + 1)]  # deletion of j chars

    for i, ca in enumerate(a, 1):
        curr = [float(i)]  # deletion cost baseline

        # band if max_cost provided
        if max_cost is not None:
            low = max(1, i - int(max_cost) - 1)
            high = min(lb, i + int(max_cost) + 1)
        else:
            low, high = 1, lb

        # fill band prefix with large cost when skipping
        if low > 1:
            curr.extend([max_cost + 1 if max_cost is not None else 1e9] * (low - 1))

        for j in range(low, high + 1):
            cb = b[j - 1]
            ins = curr[j - 1] + get_cost(weights, "ins", None, cb)
            delete = prev[j] + get_cost(weights, "del", ca, None)
            sub = prev[j - 1] + (0.0 if ca == cb else get_cost(weights, "sub", ca, cb))
            curr.append(min(ins, delete, sub))

        if max_cost is not None and min(curr) > max_cost:
            return max_cost + 1

        prev = curr + prev[len(curr):]

    return prev[lb]
