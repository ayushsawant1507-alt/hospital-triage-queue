"""
tests.sorts.test_heap_sort
==========================
Tests for triage.sorts.heap_sort (heap_sort).
"""

import math
from dataclasses import dataclass
import pytest

from triage.generators import GENERATORS, gen_random, gen_reverse, gen_sorted
from triage.metrics import Counter
from triage.sorts.heap_sort import heap_sort


@dataclass
class _SevOnly:
    """Wrapper that compares only by severity/rank to demonstrate stability failure."""
    rank: int
    uid: int

    def __lt__(self, other: "_SevOnly") -> bool:
        return self.rank < other.rank


class TestHeapSortCorrectness:

    @pytest.mark.parametrize("name", list(GENERATORS.keys()))
    def test_matches_sorted_oracle_on_all_generators(self, name: str) -> None:
        n = 35
        arr = GENERATORS[name](n, seed=42)
        oracle = sorted(arr, key=lambda p: p.key)
        c = Counter()
        heap_sort(arr, c)
        assert arr == oracle

    def test_empty(self) -> None:
        arr: list = []
        c = Counter()
        heap_sort(arr, c)
        assert arr == []
        assert c.comparisons == 0
        assert c.moves == 0

    def test_single_element(self) -> None:
        arr = gen_sorted(1)
        original = list(arr)
        c = Counter()
        heap_sort(arr, c)
        assert arr == original
        assert c.comparisons == 0
        assert c.moves == 0

    def test_two_elements(self) -> None:
        arr = gen_reverse(2)
        oracle = sorted(arr, key=lambda p: p.key)
        c = Counter()
        heap_sort(arr, c)
        assert arr == oracle

    def test_subrange_isolation(self) -> None:
        arr = gen_random(30, seed=99)
        original = list(arr)
        lo, hi = 5, 22
        heap_sort(arr, Counter(), lo=lo, hi=hi)
        assert arr[:lo] == original[:lo]
        assert arr[hi:] == original[hi:]
        assert arr[lo:hi] == sorted(original[lo:hi], key=lambda p: p.key)


class TestHeapSortPerformanceAndBounds:

    @pytest.mark.parametrize("n", [16, 64, 256])
    def test_comparisons_stay_within_2n_log2n_bound(self, n: int) -> None:
        arr = gen_random(n, seed=123)
        c = Counter()
        heap_sort(arr, c)
        # Standard upper bound for binary heap sort comparisons is 2*n*log2(n)
        upper_bound = 2 * n * math.log2(n)
        assert c.comparisons <= upper_bound

    def test_moves_and_comparisons_are_deterministic(self) -> None:
        n = 50
        arr1 = gen_random(n, seed=77)
        arr2 = gen_random(n, seed=77)
        c1, c2 = Counter(), Counter()
        heap_sort(arr1, c1)
        heap_sort(arr2, c2)
        assert c1.comparisons == c2.comparisons
        assert c1.moves == c2.moves
        assert arr1 == arr2


class TestHeapSortStabilityDemo:

    def test_heap_sort_severity_only_is_not_stable(self) -> None:
        """
        Demonstrate that heap sort on severity ONLY is NOT stable:
        Original relative order among elements with identical rank is disrupted.
        """
        # An array with equal rank elements: in a stable sort, relative order is preserved.
        # In heap sort, root extractions place early elements at the end of the array.
        items = [_SevOnly(rank=2, uid=i) for i in range(6)]
        c = Counter()
        heap_sort(items, c)

        uids = [item.uid for item in items]
        # In a stable sort, uids would strictly remain [0, 1, 2, 3, 4, 5].
        # In heap sort, the relative ordering is demonstrably disrupted:
        assert uids != list(range(6)), "Heap sort unexpectedly preserved stability"

    def test_patient_composite_key_forces_total_deterministic_order(self) -> None:
        """
        By contrast, Patient.key incorporates (severity, arrival_seq), turning the
        preorder into a strict total order, guaranteeing deterministic output.
        """
        arr = gen_random(25, seed=12)
        c1, c2 = Counter(), Counter()
        arr1 = list(arr)
        arr2 = list(arr)
        heap_sort(arr1, c1)
        heap_sort(arr2, c2)
        assert arr1 == arr2
        # Verify arrival_seq strictly increases within any severity level
        for i in range(len(arr1) - 1):
            if arr1[i].severity == arr1[i + 1].severity:
                assert arr1[i].arrival_seq < arr1[i + 1].arrival_seq
