"""Combinatorial Purged Cross-Validation (CPCV) — pure index math.

CPCV (López de Prado, *Advances in Financial Machine Learning*) generates many
overlapping train/test splits to estimate the *distribution* of a strategy's
out-of-sample performance rather than a single point estimate. The samples are
partitioned into ``n_groups`` contiguous groups; each split picks
``n_test_groups`` of them as the test set, and the train set is everything else.

Because financial samples are serially correlated, train indices adjacent to a
test index leak information across the boundary. CPCV therefore **purges** the
train set: any train index lying within ``embargo`` positions of any test index
is dropped. This module is pure index arithmetic — no PnLs, no numpy/pandas — so
it composes with any downstream backtest harness.

Deterministic: combinations are emitted in :func:`itertools.combinations` order,
and all index tuples are sorted ascending.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True, slots=True)
class CPCVSplit:
    """A single combinatorial purged cross-validation split.

    * ``train`` — sorted, purged training indices (no overlap with ``test``).
    * ``test`` — sorted test indices (the union of the chosen groups).
    """

    train: tuple[int, ...]
    test: tuple[int, ...]


def _group_bounds(n_samples: int, n_groups: int) -> list[tuple[int, int]]:
    """Partition ``range(n_samples)`` into ``n_groups`` contiguous half-open
    ranges, as even as possible (earlier groups absorb the remainder).

    :returns: a list of ``(start, stop)`` bounds, one per group.
    """
    base, extra = divmod(n_samples, n_groups)
    bounds: list[tuple[int, int]] = []
    start = 0
    for g in range(n_groups):
        size = base + (1 if g < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def combinatorial_splits(
    n_samples: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 0,
) -> list[CPCVSplit]:
    """Generate all CPCV splits as pure index math.

    The samples ``range(n_samples)`` are partitioned into ``n_groups``
    contiguous groups (as even as possible). For **every** combination of
    ``n_test_groups`` groups chosen from the ``n_groups``, the test set is the
    union of those groups' indices and the train set is all remaining indices,
    minus a purge/embargo: any train index within ``embargo`` positions of any
    test index is dropped to avoid leakage across the boundary.

    There are ``C(n_groups, n_test_groups)`` splits, returned in
    :func:`itertools.combinations` order. Train and test indices are sorted
    ascending and never overlap.

    :param n_samples: total number of samples (indices ``0 .. n_samples-1``).
    :param n_groups: number of contiguous groups to partition into (``>= 2``).
    :param n_test_groups: groups assigned to the test set per split
        (``>= 1`` and ``< n_groups``).
    :param embargo: purge radius; train indices within this many positions of
        any test index are excluded (``>= 0``).
    :raises ValueError: if ``n_groups < 2``, ``n_test_groups < 1``,
        ``n_test_groups >= n_groups``, ``n_samples < n_groups``, or
        ``embargo < 0``.
    :returns: one :class:`CPCVSplit` per combination of test groups.
    """
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}")
    if n_test_groups < 1:
        raise ValueError(f"n_test_groups must be >= 1, got {n_test_groups}")
    if n_test_groups >= n_groups:
        raise ValueError(
            f"n_test_groups must be < n_groups, got {n_test_groups} >= {n_groups}"
        )
    if n_samples < n_groups:
        raise ValueError(
            f"n_samples must be >= n_groups, got {n_samples} < {n_groups}"
        )
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")

    bounds = _group_bounds(n_samples, n_groups)
    splits: list[CPCVSplit] = []

    for test_groups in combinations(range(n_groups), n_test_groups):
        test_index: set[int] = set()
        for g in test_groups:
            start, stop = bounds[g]
            test_index.update(range(start, stop))

        # Purge: drop any train index within `embargo` positions of a test index.
        purged: set[int] = set()
        if embargo > 0:
            for idx in test_index:
                lo = max(0, idx - embargo)
                hi = min(n_samples, idx + embargo + 1)
                purged.update(range(lo, hi))
        else:
            purged = set(test_index)

        train_index = sorted(
            i for i in range(n_samples) if i not in test_index and i not in purged
        )
        splits.append(
            CPCVSplit(train=tuple(train_index), test=tuple(sorted(test_index)))
        )

    return splits
