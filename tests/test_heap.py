"""
tests.test_heap
===============
Tests for triage.heap (BinaryHeap).
"""

import pytest

from triage.generators import gen_random, gen_reverse, gen_sorted
from triage.heap import BinaryHeap
from triage.metrics import Counter


def _check_heap_property(heap: BinaryHeap) -> None:
    data = heap.data
    n = len(data)
    is_max = heap._max_heap
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n:
            if is_max:
                assert not (data[i] < data[left]), f"Max-heap violated at parent={i}, left={left}"
            else:
                assert not (data[left] < data[i]), f"Min-heap violated at parent={i}, left={left}"
        if right < n:
            if is_max:
                assert not (data[i] < data[right]), f"Max-heap violated at parent={i}, right={right}"
            else:
                assert not (data[right] < data[i]), f"Min-heap violated at parent={i}, right={right}"


class TestBinaryHeap:

    def test_empty_heap(self) -> None:
        c = Counter()
        h = BinaryHeap(counter=c)
        assert len(h) == 0
        assert h.is_empty()
        with pytest.raises(IndexError):
            h.pop()
        with pytest.raises(IndexError):
            h.peek()

    def test_single_element(self) -> None:
        c = Counter()
        h = BinaryHeap(counter=c)
        h.push(42)
        assert len(h) == 1
        assert not h.is_empty()
        assert h.peek() == 42
        assert h.pop() == 42
        assert h.is_empty()

    def test_min_heap_invariant_push_pop(self) -> None:
        c = Counter()
        h = BinaryHeap(counter=c, max_heap=False)
        patients = gen_random(40, seed=123)
        for p in patients:
            h.push(p)
            _check_heap_property(h)

        extracted = []
        while not h.is_empty():
            extracted.append(h.pop())
            _check_heap_property(h)

        expected = sorted(patients, key=lambda p: p.key)
        assert extracted == expected

    def test_max_heap_invariant_push_pop(self) -> None:
        c = Counter()
        h = BinaryHeap(counter=c, max_heap=True)
        patients = gen_random(40, seed=456)
        for p in patients:
            h.push(p)
            _check_heap_property(h)

        extracted = []
        while not h.is_empty():
            extracted.append(h.pop())
            _check_heap_property(h)

        expected = sorted(patients, key=lambda p: p.key, reverse=True)
        assert extracted == expected

    def test_build_heap_matches_repeated_push(self) -> None:
        patients = gen_random(50, seed=789)
        c1, c2 = Counter(), Counter()
        h_built = BinaryHeap(counter=c1)
        h_built.build_heap(patients)
        _check_heap_property(h_built)

        h_pushed = BinaryHeap(counter=c2)
        for p in patients:
            h_pushed.push(p)

        extracted_built = []
        while not h_built.is_empty():
            extracted_built.append(h_built.pop())

        extracted_pushed = []
        while not h_pushed.is_empty():
            extracted_pushed.append(h_pushed.pop())

        assert extracted_built == extracted_pushed
        assert extracted_built == sorted(patients, key=lambda p: p.key)

    def test_build_heap_complexity_is_linear(self) -> None:
        n = 100
        patients = gen_reverse(n)
        c = Counter()
        h = BinaryHeap(counter=c)
        h.build_heap(patients)
        # Floyd's build_heap does at most 2*n comparisons (for heapifying)
        # and moves bounded by O(n).
        assert c.comparisons <= 4 * n

    def test_on_move_callback(self) -> None:
        moves_tracked = {}

        def track(item: int, idx: int) -> None:
            moves_tracked[item] = idx

        c = Counter()
        h = BinaryHeap(counter=c, on_move=track)
        for x in [10, 20, 5, 15]:
            h.push(x)

        # In min-heap, 5 must be at root index 0
        assert moves_tracked[5] == 0
