"""Inter-rater agreement metrics.

Krippendorff's α for interval-scale data. Implemented from the canonical
formula (Krippendorff 2011) rather than depending on the `krippendorff`
package to keep external deps minimal.

For a row of N items and K judges where every item is rated by all K judges,
α = 1 - (D_o / D_e)
where:
    D_o (observed disagreement) = sum over all pairs of judges and all items
        of (rating_i - rating_j)^2 / (N * K * (K-1))
    D_e (expected disagreement) = sum over all pairs of distinct ratings c, d
        of n_c * n_d * (c - d)^2 / (T * (T-1))
        where n_c = total count of rating c across all (judge, item) pairs
        and T = total number of (judge, item) ratings
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


def _validate(rows: Mapping[str, Sequence[float]]) -> None:
    if not rows:
        raise ValueError("rows must be non-empty")
    if len(rows) < 2:
        raise ValueError("Krippendorff's α requires at least 2 judges")
    lengths = {len(v) for v in rows.values()}
    if len(lengths) != 1:
        raise ValueError(f"all judges must rate the same items; got lengths {lengths}")


def _observed_disagreement(rows: Mapping[str, Sequence[float]]) -> float:
    judges = list(rows.keys())
    k = len(judges)
    n_items = len(rows[judges[0]])
    total = 0.0
    for item_idx in range(n_items):
        for i in range(k):
            for j in range(i + 1, k):
                ratings_i = rows[judges[i]][item_idx]
                ratings_j = rows[judges[j]][item_idx]
                total += (ratings_i - ratings_j) ** 2
    # pairs per item = k*(k-1)/2, total pairs across items = n_items * k*(k-1)/2
    # for the interval-scale α formula we divide by total ratings (not pairs)
    # times (k-1):
    return 2 * total / (n_items * k * (k - 1))


def _expected_disagreement(rows: Mapping[str, Sequence[float]]) -> float:
    all_ratings: list[float] = []
    for v in rows.values():
        all_ratings.extend(v)
    counts = Counter(all_ratings)
    distinct = list(counts.keys())
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    sum_sq = 0.0
    for c in distinct:
        for d in distinct:
            sum_sq += counts[c] * counts[d] * (c - d) ** 2
    return sum_sq / (total * (total - 1))


def krippendorffs_alpha(rows: Mapping[str, Sequence[float]]) -> float:
    """Krippendorff's α (interval scale) for the given judges×items ratings.

    Args:
        rows: mapping of judge_name -> list of numeric ratings (one per item).
            All judges must have rated the same number of items.

    Returns:
        Float in roughly [-1, 1]. 1 = perfect agreement, 0 ≈ chance,
        <0 = systematic disagreement. Returns 1.0 when expected disagreement
        is 0 (degenerate case, e.g. all ratings identical).
    """
    _validate(rows)
    d_o = _observed_disagreement(rows)
    d_e = _expected_disagreement(rows)
    if d_e == 0:
        return 1.0
    return 1.0 - (d_o / d_e)
