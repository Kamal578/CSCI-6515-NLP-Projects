from __future__ import annotations

from typing import Optional


def levenshtein(a: str, b: str, max_dist: Optional[int] = None) -> int:
    """
    Standard Levenshtein distance with optional cutoff.
    Early-aborts if distance exceeds max_dist (if provided).
    Costs: ins=1, del=1, sub=1.
    """
    if a == b:
        return 0

    la, lb = len(a), len(b)

    if max_dist is not None and abs(la - lb) > max_dist:
        return max_dist + 1

    # Ensure b is the shorter word to reduce memory
    if lb > la:
        a, b = b, a
        la, lb = lb, la

    prev = list(range(lb + 1))

    for i, ca in enumerate(a, 1):
        curr = [i]
        # Narrow band if cutoff provided
        if max_dist is not None:
            low = max(1, i - max_dist)
            high = min(lb, i + max_dist) + 1
        else:
            low, high = 1, lb + 1

        # Fill untouched prefix with sentinel large values when banding
        if max_dist is not None and low > 1:
            curr.extend([max_dist + 1] * (low - 1))

        for j in range(low, high):
            cb = b[j - 1]
            ins = curr[j - 1] + 1
            del_ = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, del_, sub))

        if max_dist is not None and min(curr) > max_dist:
            return max_dist + 1

        prev = curr + prev[len(curr):]  # keep same length when banding

    return prev[lb]
