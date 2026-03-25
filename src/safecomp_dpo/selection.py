"""
Shared allocation helper for stratified prompt selection.

Used by all select_prompts_*.py scripts to ensure consistent
two-phase floor + proportional allocation behaviour.
"""

from __future__ import annotations

from typing import Any


def allocate(
    category_counts: dict[Any, int],
    target_n: int,
    min_per_category: int,
) -> dict[Any, int]:
    """Two-phase stratified allocation.

    Phase 1: base_i = min(count_i, min_per_category)
    Phase 2: distribute (target_n - sum(base)) proportionally over remaining
             per-category capacity using largest-remainder rounding.

    Tie-break in largest-remainder step: smaller category key wins.
    Returns a quota dict; sum(quota) == target_n when total pool >= target_n
    and base_total <= target_n.
    """
    cats = sorted(category_counts.keys())

    # Phase 1 — base floor
    base = {c: min(category_counts[c], min_per_category) for c in cats}
    base_total = sum(base.values())
    remaining_budget = target_n - base_total

    quota = dict(base)

    if remaining_budget <= 0:
        return quota

    # Phase 2 — proportional distribution over remaining capacity
    remaining_cap = {c: category_counts[c] - base[c] for c in cats}
    total_cap = sum(remaining_cap.values())

    if total_cap == 0:
        return quota

    exact = {c: remaining_budget * remaining_cap[c] / total_cap for c in cats}
    floored = {c: int(exact[c]) for c in cats}
    leftover = remaining_budget - sum(floored.values())

    # Largest-remainder: sort by fractional part descending, tie-break by key ascending.
    order = sorted(cats, key=lambda c: (-(exact[c] - floored[c]), c))
    for c in order[:leftover]:
        floored[c] += 1

    for c in cats:
        quota[c] = base[c] + floored[c]

    return quota
