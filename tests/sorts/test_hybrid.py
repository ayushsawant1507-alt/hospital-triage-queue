"""
tests.sorts.test_hybrid
=======================
Tests for triage.sorts.hybrid (timsort_style, introsort).
"""

import pytest

from triage.generators import GENERATORS, gen_random, gen_reverse, gen_sorted
from triage.metrics import Counter
from triage.sorts.hybrid import introsort, timsort_style


class TestHybridSortsCorrectness:

    @pytest.mark.parametrize("size", [0, 1, 2, 3, 15, 16, 17, 100])
    @pytest.mark.parametrize("name", ["random", "reverse", "nearly_sorted"])
    def test_timsort_matches_sorted_oracle(self, size: int, name: str) -> None:
        arr = [] if (name == "nearly_sorted" and size == 0) else GENERATORS[name](size, seed=123)
        oracle = sorted(arr, key=lambda p: p.key)
        c = Counter()
        timsort_style(arr, c)
        assert arr == oracle

    @pytest.mark.parametrize("size", [0, 1, 2, 3, 15, 16, 17, 100])
    @pytest.mark.parametrize("name", ["random", "reverse", "nearly_sorted"])
    def test_introsort_matches_sorted_oracle(self, size: int, name: str) -> None:
        arr = [] if (name == "nearly_sorted" and size == 0) else GENERATORS[name](size, seed=456)
        oracle = sorted(arr, key=lambda p: p.key)
        c = Counter()
        introsort(arr, c)
        assert arr == oracle

    def test_large_size_1000(self) -> None:
        n = 1000
        arr_tim = gen_random(n, seed=99)
        arr_intro = list(arr_tim)
        oracle = sorted(arr_tim, key=lambda p: p.key)

        timsort_style(arr_tim, Counter())
        assert arr_tim == oracle

        introsort(arr_intro, Counter())
        assert arr_intro == oracle


class TestTimsortAdaptivity:

    @pytest.mark.parametrize("n", [50, 100, 200])
    def test_timsort_linear_on_sorted(self, n: int) -> None:
        arr = gen_sorted(n)
        c = Counter()
        timsort_style(arr, c)
        # One pass over n elements to detect single ascending run
        # Comparisons are exactly n - 1
        assert c.comparisons <= 2 * n

    @pytest.mark.parametrize("n", [50, 100, 200])
    def test_timsort_linear_on_reverse(self, n: int) -> None:
        arr = gen_reverse(n)
        c = Counter()
        timsort_style(arr, c)
        # One pass to detect descending run + reverse in place
        assert c.comparisons <= 2 * n


class TestIntrosortThresholdAndFallback:

    @pytest.mark.parametrize("T", [1, 4, 16, 64])
    def test_threshold_sweep_correctness(self, T: int) -> None:
        n = 120
        arr = gen_random(n, seed=789)
        oracle = sorted(arr, key=lambda p: p.key)
        c = Counter()
        introsort(arr, c, threshold=T)
        assert arr == oracle

    def test_adversarial_depth_limit_triggers_heap_sort_fallback(self) -> None:
        """
        Verify that exceeding recursion depth limit triggers the heap-sort safety net.
        """
        n = 80
        arr = gen_random(n, seed=42)
        oracle = sorted(arr, key=lambda p: p.key)
        stats = {"heap_sort_fallbacks": 0}
        c = Counter()

        # Force a very tight depth limit (e.g. max_depth_override = 2)
        introsort(arr, c, threshold=4, max_depth_override=2, stats=stats)
        assert arr == oracle
        assert stats["heap_sort_fallbacks"] > 0, "Heap sort fallback did not trigger under depth constraint"
