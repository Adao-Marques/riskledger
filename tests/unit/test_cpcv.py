"""Unit tests for Combinatorial Purged Cross-Validation split generation."""

from __future__ import annotations

import math

import pytest

from riskledger.validation.cpcv import CPCVSplit, combinatorial_splits


def test_number_of_splits_equals_n_choose_k() -> None:
    splits = combinatorial_splits(120, n_groups=6, n_test_groups=2)
    assert len(splits) == math.comb(6, 2) == 15


def test_train_and_test_never_overlap() -> None:
    splits = combinatorial_splits(120, n_groups=6, n_test_groups=2)
    for split in splits:
        assert isinstance(split, CPCVSplit)
        assert set(split.train).isdisjoint(split.test)


def test_indices_are_sorted_ascending() -> None:
    for split in combinatorial_splits(97, n_groups=5, n_test_groups=2):
        assert list(split.train) == sorted(split.train)
        assert list(split.test) == sorted(split.test)


def test_no_embargo_train_is_complement_of_test() -> None:
    n = 60
    for split in combinatorial_splits(n, n_groups=6, n_test_groups=2, embargo=0):
        assert set(split.train) | set(split.test) == set(range(n))


def test_embargo_excludes_neighbours_of_test_indices() -> None:
    n = 60
    embargo = 3
    splits = combinatorial_splits(n, n_groups=6, n_test_groups=2, embargo=embargo)
    for split in splits:
        test = set(split.test)
        # Every kept train index must be more than `embargo` away from all test
        # indices.
        for t in test:
            for delta in range(1, embargo + 1):
                assert (t - delta) not in split.train
                assert (t + delta) not in split.train
        # And the purge only removes things; train must still be disjoint.
        assert test.isdisjoint(split.train)


def test_embargo_shrinks_train_set() -> None:
    n = 120
    no_purge = combinatorial_splits(n, n_groups=6, n_test_groups=2, embargo=0)
    purged = combinatorial_splits(n, n_groups=6, n_test_groups=2, embargo=5)
    for a, b in zip(no_purge, purged, strict=True):
        assert a.test == b.test
        assert len(b.train) <= len(a.train)
    # At least one split must actually lose train indices to the embargo.
    assert any(
        len(b.train) < len(a.train) for a, b in zip(no_purge, purged, strict=True)
    )


def test_union_of_test_sets_covers_every_sample() -> None:
    n = 120
    splits = combinatorial_splits(n, n_groups=6, n_test_groups=2)
    covered: set[int] = set()
    for split in splits:
        covered.update(split.test)
    assert covered == set(range(n))


def test_uneven_partition_covers_all_samples() -> None:
    # 100 / 6 is not even; the partition must still cover 0..99 exactly once.
    n = 100
    splits = combinatorial_splits(n, n_groups=6, n_test_groups=1)
    covered: set[int] = set()
    for split in splits:
        covered.update(split.test)
    assert covered == set(range(n))
    # With n_test_groups=1 the test sets are exactly the groups: disjoint.
    test_sets = [set(s.test) for s in splits]
    for i in range(len(test_sets)):
        for j in range(i + 1, len(test_sets)):
            assert test_sets[i].isdisjoint(test_sets[j])


def test_deterministic() -> None:
    a = combinatorial_splits(120, n_groups=6, n_test_groups=2, embargo=4)
    b = combinatorial_splits(120, n_groups=6, n_test_groups=2, embargo=4)
    assert a == b


@pytest.mark.parametrize(
    ("n_samples", "n_groups", "n_test_groups", "embargo"),
    [
        (100, 1, 1, 0),  # n_groups < 2
        (100, 6, 0, 0),  # n_test_groups < 1
        (100, 6, 6, 0),  # n_test_groups >= n_groups
        (100, 6, 7, 0),  # n_test_groups > n_groups
        (4, 6, 2, 0),  # n_samples < n_groups
        (100, 6, 2, -1),  # negative embargo
    ],
)
def test_invalid_params_raise(
    n_samples: int, n_groups: int, n_test_groups: int, embargo: int
) -> None:
    with pytest.raises(ValueError):
        combinatorial_splits(
            n_samples,
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            embargo=embargo,
        )
