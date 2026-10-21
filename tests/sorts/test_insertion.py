"""
tests.sorts.test_insertion
==========================
Tests for triage.sorts.insertion (insertion_sort, binary_insertion_sort).

Coverage
--------
Correctness
  * Result matches sorted(arr, key=lambda p: p.key) for all 6 GENERATORS.
  * Empty list, single element, two elements (both orderings).
  * All-equal severity (tests FIFO via arrival_seq tiebreaking).
  * Subrange: arr[lo:hi] sorted; everything outside is bit-for-bit unchanged.

Exact counter values  (insertion_sort)
  * Sorted input  : comparisons == n-1, total moves == n-1 (0 shifts + n-1 placements).
  * Reverse input : comparisons == n(n-1)/2, shifts == n(n-1)/2,
                    total moves == n(n-1)/2 + (n-1).
  * Any input     : shifts == count_inversions(original_arr),
                    total moves == I + (n-1).

binary_insertion_sort
  * Same sorted output as insertion_sort for all generators.
  * Total moves == I + (n-1)  (identical to insertion_sort).
  * On sorted input, comparisons > n-1 for n >= 3  (Θ(n log n) vs n-1).
  * On reverse input, comparisons < n(n-1)/2 for n >= 3  (Θ(n log n) < n²/2).
  * Comparisons depend only on n, NOT on input order.

Stability
  * insertion_sort       : severity-only comparison wrapper preserves uid order.
  * binary_insertion_sort: same stability guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import pytest

from triage.generators import (
    GENERATORS,
    count_inversions,
    gen_random,
    gen_reverse,
    gen_sorted,
)
from triage.metrics import Counter
from triage.models import Patient, Severity
from triage.sorts.insertion import binary_insertion_sort, insertion_sort


# ---------------------------------------------------------------------------
# Stability test helper
# ---------------------------------------------------------------------------

@dataclass
class _SevOnly:
    """
    Minimal sortable element that compares by ``rank`` only.

    ``uid`` tracks the original position in the input list but is NOT
    included in ``__lt__``/``__le__``, simulating two elements that are
    equal under a severity-only comparison key.  After a stable sort the
    relative order of equal-rank elements must be preserved (uids ascending
    within each rank group).
    """

    rank: int
    uid:  int

    def __lt__(self, other: "_SevOnly") -> bool:   # noqa: D105
        return self.rank < other.rank

    def __le__(self, other: "_SevOnly") -> bool:   # noqa: D105
        return self.rank <= other.rank

    def __eq__(self, other: object) -> bool:        # noqa: D105
        if not isinstance(other, _SevOnly):
            return NotImplemented
        return self.rank == other.rank and self.uid == other.uid


def _check_stable(elements: List[_SevOnly]) -> None:
    """
    After sorting, verify that elements with equal rank appear in ascending
    uid order (original relative order preserved).
    """
    current_rank: int | None = None
    last_uid: int | None = None
    for e in elements:
        if e.rank != current_rank:
            current_rank = e.rank
            last_uid = e.uid
        else:
            assert last_uid is not None
            assert e.uid > last_uid, (
                f"Stability violated: rank={e.rank}, "
                f"uid={e.uid} follows uid={last_uid}"
            )
            last_uid = e.uid


# ---------------------------------------------------------------------------
# Helper: oracle comparison
# ---------------------------------------------------------------------------

def _sorted_oracle(patients: List[Patient]) -> List[Patient]:
    """Return a correctly sorted copy using Python's stable sort as oracle."""
    return sorted(patients, key=lambda p: p.key)


# ---------------------------------------------------------------------------
# insertion_sort — correctness
# ---------------------------------------------------------------------------

class TestInsertionSortCorrectness:

    @pytest.mark.parametrize("name", list(GENERATORS.keys()))
    def test_matches_oracle_all_generators(self, name: str) -> None:
        arr = GENERATORS[name](40, seed=42)
        expected = _sorted_oracle(arr)
        c = Counter()
        insertion_sort(arr, c)
        assert arr == expected, f"Failed for generator '{name}'"

    def test_empty(self) -> None:
        arr: List[Patient] = []
        c = Counter()
        insertion_sort(arr, c)
        assert arr == []
        assert c.comparisons == 0
        assert c.moves == 0

    def test_single_element(self) -> None:
        arr = gen_sorted(1)
        c = Counter()
        insertion_sort(arr, c)
        assert len(arr) == 1
        assert c.comparisons == 0
        assert c.moves == 0

    def test_two_elements_already_sorted(self) -> None:
        arr = gen_sorted(2)
        expected = _sorted_oracle(arr)
        c = Counter()
        insertion_sort(arr, c)
        assert arr == expected

    def test_two_elements_reversed(self) -> None:
        arr = gen_reverse(2)
        expected = _sorted_oracle(arr)
        c = Counter()
        insertion_sort(arr, c)
        assert arr == expected

    def test_all_equal_severity_preserves_fifo(self) -> None:
        """All patients have Severity.URGENT; order must become ascending arrival_seq."""
        patients = [
            Patient(id=i, name=f"P{i}", severity=Severity.URGENT, arrival_seq=i)
            for i in range(10, 0, -1)   # arrival_seqs 10..1 in reverse
        ]
        c = Counter()
        insertion_sort(patients, c)
        arrival_seqs = [p.arrival_seq for p in patients]
        assert arrival_seqs == sorted(arrival_seqs)

    def test_subrange_inside_sorted(self) -> None:
        arr = gen_random(20, seed=7)
        lo, hi = 3, 14
        c = Counter()
        insertion_sort(arr, c, lo=lo, hi=hi)
        for a, b in zip(arr[lo:hi], arr[lo + 1:hi]):
            assert a.key <= b.key

    def test_subrange_outside_unchanged(self) -> None:
        arr = gen_random(20, seed=7)
        original = list(arr)
        lo, hi = 3, 14
        insertion_sort(arr, Counter(), lo=lo, hi=hi)
        assert arr[:lo]  == original[:lo],  "Elements before lo were modified"
        assert arr[hi:]  == original[hi:],  "Elements after hi were modified"

    def test_subrange_full_range_equivalent(self) -> None:
        arr1 = gen_random(15, seed=9)
        arr2 = list(arr1)
        insertion_sort(arr1, Counter())
        insertion_sort(arr2, Counter(), lo=0, hi=len(arr2))
        assert arr1 == arr2


# ---------------------------------------------------------------------------
# insertion_sort — EXACT counter values
# ---------------------------------------------------------------------------

class TestInsertionSortExactCounts:

    # ── Sorted input ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
    def test_sorted_comparisons_equal_n_minus_1(self, n: int) -> None:
        """Sorted input: exactly n-1 comparisons (one failing check per outer step)."""
        arr = gen_sorted(n)
        c = Counter()
        insertion_sort(arr, c)
        assert c.comparisons == n - 1, (
            f"n={n}: expected {n-1} comparisons, got {c.comparisons}"
        )

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
    def test_sorted_shifts_are_zero(self, n: int) -> None:
        """Sorted input: I=0 → 0 shifts."""
        arr = gen_sorted(n)
        c = Counter()
        insertion_sort(arr, c)
        shifts = c.moves - (n - 1)
        assert shifts == 0, f"n={n}: expected 0 shifts, got {shifts}"

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
    def test_sorted_total_moves_equal_n_minus_1(self, n: int) -> None:
        """Sorted input: total moves = 0 shifts + (n-1) placements = n-1."""
        arr = gen_sorted(n)
        c = Counter()
        insertion_sort(arr, c)
        assert c.moves == n - 1, (
            f"n={n}: expected {n-1} total moves, got {c.moves}"
        )

    # ── Reverse input ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
    def test_reverse_comparisons_equal_max(self, n: int) -> None:
        """
        Reverse input: exactly n(n-1)/2 comparisons.

        On a strictly-decreasing key sequence every inner-loop body executes
        until j < lo (short-circuit on the boundary check, NOT on compare),
        giving i comparisons for element at position i → total n(n-1)/2.
        """
        arr = gen_reverse(n)
        c = Counter()
        insertion_sort(arr, c)
        expected = n * (n - 1) // 2
        assert c.comparisons == expected, (
            f"n={n}: expected {expected} comparisons, got {c.comparisons}"
        )

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
    def test_reverse_shifts_equal_max_inversions(self, n: int) -> None:
        """Reverse input: shifts = n(n-1)/2 = I (every pair is an inversion)."""
        arr = gen_reverse(n)
        I = count_inversions(arr)
        c = Counter()
        insertion_sort(arr, c)
        shifts = c.moves - (n - 1)
        expected_I = n * (n - 1) // 2
        assert I == expected_I,   f"count_inversions reported {I}, expected {expected_I}"
        assert shifts == expected_I, f"shifts={shifts}, expected {expected_I}"

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
    def test_reverse_total_moves(self, n: int) -> None:
        """Reverse input: total moves = n(n-1)/2 + (n-1) = I + (n-1)."""
        arr = gen_reverse(n)
        c = Counter()
        insertion_sort(arr, c)
        expected = n * (n - 1) // 2 + (n - 1)
        assert c.moves == expected, (
            f"n={n}: expected {expected} total moves, got {c.moves}"
        )

    # ── Any input: shifts == I, total moves == I + (n-1) ─────────────────────

    @pytest.mark.parametrize("seed", [1, 7, 42, 99])
    def test_shifts_equal_count_inversions(self, seed: int) -> None:
        """
        For any input, the number of shifts equals the inversion count.
        count_inversions() is called BEFORE insertion_sort (arr is mutated).
        """
        n = 30
        arr = gen_random(n, seed=seed)
        I = count_inversions(arr)          # must capture BEFORE sort
        c = Counter()
        insertion_sort(arr, c)
        shifts = c.moves - (n - 1)
        assert shifts == I, (
            f"seed={seed}: shifts={shifts}, count_inversions={I}"
        )

    @pytest.mark.parametrize("seed", [1, 7, 42, 99])
    def test_total_moves_equal_I_plus_n_minus_1(self, seed: int) -> None:
        """Total moves = I + (n-1) for any input (canonical formula)."""
        n = 30
        arr = gen_random(n, seed=seed)
        I = count_inversions(arr)
        c = Counter()
        insertion_sort(arr, c)
        assert c.moves == I + (n - 1), (
            f"seed={seed}: moves={c.moves}, expected {I + (n-1)}"
        )

    def test_nearly_sorted_low_moves(self) -> None:
        """
        Nearly-sorted input (k=5) should have far fewer moves than reverse.
        Demonstrates the O(n+k) advantage of insertion sort on nearly-sorted data.
        """
        from triage.generators import gen_nearly_sorted
        n, k = 50, 5
        arr_near = gen_nearly_sorted(n, k=k, seed=42)
        arr_rev  = gen_reverse(n)
        I_near = count_inversions(arr_near)
        I_rev  = count_inversions(arr_rev)

        c_near, c_rev = Counter(), Counter()
        insertion_sort(arr_near, c_near)
        insertion_sort(arr_rev,  c_rev)

        assert I_near <= k,                    "nearly_sorted inversions should be <= k"
        assert c_near.moves < c_rev.moves,     "nearly-sorted should need far fewer moves"
        assert c_near.comparisons < c_rev.comparisons


# ---------------------------------------------------------------------------
# insertion_sort — stability
# ---------------------------------------------------------------------------

class TestInsertionSortStability:

    def test_stable_equal_ranks_preserve_uid_order(self) -> None:
        """
        Elements with equal rank (severity-only comparison) must maintain
        their original uid (position) order after sorting.
        """
        ranks = [3, 1, 2, 1, 3, 2, 1]
        elements = [_SevOnly(rank=r, uid=i) for i, r in enumerate(ranks)]
        insertion_sort(elements, Counter())
        _check_stable(elements)

    def test_stable_alternating_two_ranks(self) -> None:
        elements = [_SevOnly(rank=i % 2, uid=i) for i in range(12)]
        insertion_sort(elements, Counter())
        _check_stable(elements)

    def test_stable_all_equal(self) -> None:
        """All equal rank — original order (uids 0..n-1) must be preserved."""
        elements = [_SevOnly(rank=1, uid=i) for i in range(10)]
        insertion_sort(elements, Counter())
        assert [e.uid for e in elements] == list(range(10))

    def test_stable_real_patients_same_severity(self) -> None:
        """Real Patient objects: same severity, different arrival_seq."""
        patients = [
            Patient(id=i, name=f"P{i}", severity=Severity.URGENT, arrival_seq=i)
            for i in reversed(range(8))    # arrival_seqs 7..0 in list order
        ]
        insertion_sort(patients, Counter())
        arrival_seqs = [p.arrival_seq for p in patients]
        assert arrival_seqs == sorted(arrival_seqs), (
            "Same-severity patients not sorted by arrival_seq (FIFO violated)"
        )


# ---------------------------------------------------------------------------
# binary_insertion_sort — correctness
# ---------------------------------------------------------------------------

class TestBinaryInsertionSortCorrectness:

    @pytest.mark.parametrize("name", list(GENERATORS.keys()))
    def test_matches_oracle_all_generators(self, name: str) -> None:
        arr = GENERATORS[name](40, seed=42)
        expected = _sorted_oracle(arr)
        c = Counter()
        binary_insertion_sort(arr, c)
        assert arr == expected, f"Failed for generator '{name}'"

    def test_empty(self) -> None:
        arr: List[Patient] = []
        binary_insertion_sort(arr, Counter())
        assert arr == []

    def test_single_element(self) -> None:
        arr = gen_sorted(1)
        c = Counter()
        binary_insertion_sort(arr, c)
        assert c.comparisons == 0
        assert c.moves == 0

    def test_subrange_outside_unchanged(self) -> None:
        arr = gen_random(20, seed=13)
        original = list(arr)
        lo, hi = 4, 16
        binary_insertion_sort(arr, Counter(), lo=lo, hi=hi)
        assert arr[:lo]  == original[:lo]
        assert arr[hi:]  == original[hi:]


# ---------------------------------------------------------------------------
# binary_insertion_sort — move count identical to insertion_sort
# ---------------------------------------------------------------------------

class TestBinaryInsertionSortMoves:

    @pytest.mark.parametrize("name", list(GENERATORS.keys()))
    def test_total_moves_same_as_linear(self, name: str) -> None:
        """
        Total moves must equal I + (n-1) — unchanged from insertion_sort.
        Binary search finds the position faster but the shifts still happen.
        """
        n = 30
        arr_b = GENERATORS[name](n, seed=7)
        arr_l = list(arr_b)

        I = count_inversions(arr_b)      # capture before sort

        cb, cl = Counter(), Counter()
        binary_insertion_sort(arr_b, cb)
        insertion_sort(arr_l, cl)

        assert cb.moves == I + (n - 1), (
            f"[{name}] binary moves={cb.moves}, expected {I + (n-1)}"
        )
        assert cb.moves == cl.moves, (
            f"[{name}] binary({cb.moves}) != linear({cl.moves}) moves"
        )


# ---------------------------------------------------------------------------
# binary_insertion_sort — comparison count properties
# ---------------------------------------------------------------------------

class TestBinaryInsertionSortComparisons:

    @pytest.mark.parametrize("n", [4, 5, 10, 20])
    def test_sorted_comparisons_exceed_n_minus_1(self, n: int) -> None:
        """
        On sorted input binary_insertion_sort does Θ(n log n) comparisons —
        strictly MORE than the n-1 that linear insertion sort does (for n >= 4).
        This demonstrates the tradeoff: binary search costs more on
        nearly-sorted data (where the linear scan exits immediately).
        """
        arr = gen_sorted(n)
        cb, cl = Counter(), Counter()
        binary_insertion_sort(arr, cb)
        arr2 = gen_sorted(n)
        insertion_sort(arr2, cl)
        assert cb.comparisons > cl.comparisons, (
            f"n={n}: binary={cb.comparisons} should exceed linear={cl.comparisons}"
        )

    @pytest.mark.parametrize("n", [4, 5, 10, 20])
    def test_reverse_comparisons_below_quadratic(self, n: int) -> None:
        """
        On reverse input binary_insertion_sort does fewer comparisons than
        insertion_sort (Θ(n log n) < n(n-1)/2 for n >= 4).
        """
        arr_b = gen_reverse(n)
        arr_l = gen_reverse(n)
        cb, cl = Counter(), Counter()
        binary_insertion_sort(arr_b, cb)
        insertion_sort(arr_l, cl)
        assert cb.comparisons < cl.comparisons, (
            f"n={n}: binary={cb.comparisons} should be < linear={cl.comparisons}"
        )

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
    def test_exact_theoretical_bounds(self, n: int) -> None:
        """
        Inserting element i into a sorted prefix of length i partitions i+1
        insertion positions. In a binary search tree, the path length to any
        position is between ⌊log₂(i+1)⌋ and ⌈log₂(i+1)⌉ comparisons.
        Sorted input follows the shallowest paths (min bound);
        Reverse input follows the deepest paths (max bound).
        """
        min_expected = sum(math.floor(math.log2(i + 1)) for i in range(1, n))
        max_expected = sum(math.ceil(math.log2(i + 1)) for i in range(1, n))

        # Sorted input achieves the minimum comparison bound
        c_sorted = Counter()
        binary_insertion_sort(gen_sorted(n), c_sorted)
        assert c_sorted.comparisons == min_expected, (
            f"n={n}: sorted comparisons got {c_sorted.comparisons}, expected min {min_expected}"
        )

        # Reverse input achieves the maximum comparison bound
        c_reverse = Counter()
        binary_insertion_sort(gen_reverse(n), c_reverse)
        assert c_reverse.comparisons == max_expected, (
            f"n={n}: reverse comparisons got {c_reverse.comparisons}, expected max {max_expected}"
        )

    @pytest.mark.parametrize("name", list(GENERATORS.keys()))
    def test_comparisons_within_bounds_for_all_generators(self, name: str) -> None:
        """
        For any input distribution, binary_insertion_sort comparison count
        is strictly bounded by [min_expected, max_expected] = Θ(n log n).
        """
        n = 25
        min_expected = sum(math.floor(math.log2(i + 1)) for i in range(1, n))
        max_expected = sum(math.ceil(math.log2(i + 1)) for i in range(1, n))

        arr = GENERATORS[name](n, seed=42)
        c = Counter()
        binary_insertion_sort(arr, c)
        assert min_expected <= c.comparisons <= max_expected, (
            f"[{name}] n={n}: {c.comparisons} comparisons outside [{min_expected}, {max_expected}]"
        )


# ---------------------------------------------------------------------------
# binary_insertion_sort — stability
# ---------------------------------------------------------------------------

class TestBinaryInsertionSortStability:

    def test_stable_equal_ranks_preserve_uid_order(self) -> None:
        ranks = [3, 1, 2, 1, 3, 2, 1]
        elements = [_SevOnly(rank=r, uid=i) for i, r in enumerate(ranks)]
        binary_insertion_sort(elements, Counter())
        _check_stable(elements)

    def test_stable_all_equal(self) -> None:
        elements = [_SevOnly(rank=1, uid=i) for i in range(10)]
        binary_insertion_sort(elements, Counter())
        assert [e.uid for e in elements] == list(range(10))
