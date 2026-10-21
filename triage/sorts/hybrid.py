"""
triage.sorts.hybrid
===================
Hybrid sorting algorithms: Timsort-style and Introsort.

1. timsort_style(arr, counter, minrun=32)
   - Weakness fixed: Pure merge sort does not exploit existing order and uses
     O(n) auxiliary space unconditionally; pure insertion sort is O(n²) on
     random data.
   - Hybrid mechanism: Exploits natural ascending and strictly descending runs
     in O(n) time, extends short runs to `minrun` with binary insertion sort,
     and merges balanced runs using a run-stack invariant.
   - Note on omissions: Merging without galloping mode (galloping is omitted to
     keep pedagogical clarity; standard 2-way linear merging is used).

2. introsort(arr, counter, threshold=16)
   - Weakness fixed: Quicksort has catastrophic O(n²) worst-case on adversarial
     inputs; heap sort has poor cache locality despite guaranteed O(n log n);
     insertion sort is quadratic on large unsorted arrays but fastest on small ones.
   - Hybrid mechanism: Starts with quicksort (median-of-three Lomuto partition);
     monitors recursion depth and falls back to heap_sort when depth exceeds
     2 * floor(log2 n) to guarantee O(n log n) worst-case; switches small partitions
     (<= threshold) to insertion sort for minimal overhead.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from triage.metrics import Counter
from triage.sorts.heap_sort import heap_sort
from triage.sorts.insertion import binary_insertion_sort, insertion_sort


# ===========================================================================
# A) Timsort-Style
# ===========================================================================

def _reverse_slice(arr: List[Any], counter: Counter, lo: int, hi: int) -> None:
    """Reverse arr[lo:hi] in place, counting 2 moves per swap."""
    left = lo
    right = hi - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        counter.move()
        counter.move()
        left += 1
        right -= 1


def _merge_runs(
    arr: List[Any],
    counter: Counter,
    start_a: int,
    len_a: int,
    start_b: int,
    len_b: int,
) -> None:
    """
    Merge adjacent runs arr[start_a : start_a + len_a] and arr[start_b : start_b + len_b].
    Assumes start_b == start_a + len_a.
    Copies run A to a temporary buffer and merges back into arr.
    """
    # Copy run A into temporary memory
    buf_a = [arr[start_a + i] for i in range(len_a)]
    for _ in range(len_a):
        counter.move()

    idx_a = 0
    idx_b = start_b
    dest = start_a
    end_b = start_b + len_b

    while idx_a < len_a and idx_b < end_b:
        # Stable comparison: if arr[idx_b] < buf_a[idx_a] take b, else take a
        if counter.compare(arr[idx_b], buf_a[idx_a]):
            arr[dest] = arr[idx_b]
            counter.move()
            idx_b += 1
        else:
            arr[dest] = buf_a[idx_a]
            counter.move()
            idx_a += 1
        dest += 1

    # Drain remaining elements of A
    while idx_a < len_a:
        arr[dest] = buf_a[idx_a]
        counter.move()
        idx_a += 1
        dest += 1

    # Remaining elements of B are already in place


def timsort_style(
    arr: List[Any],
    counter: Counter,
    minrun: int = 32,
) -> None:
    """
    Sort arr in place using a Timsort-style adaptive hybrid sort.

    Parameters
    ----------
    arr     : List of elements to sort in place.
    counter : Shared Counter.
    minrun  : Minimum run length (default 32). Short runs are extended
              using binary_insertion_sort.

    Complexity
    ----------
    Time  : Best  O(n) on already sorted or reverse-sorted input.
            Avg   O(n log n).
            Worst O(n log n).
    Space : O(minrun) to O(n) auxiliary memory for merge buffer.
    Stable: YES.
    """
    n = len(arr)
    if n <= 1:
        return

    # Stack of (start_index, run_length)
    run_stack: List[Tuple[int, int]] = []
    curr = 0

    while curr < n:
        start = curr
        if curr == n - 1:
            run_len = 1
            curr += 1
        else:
            # Detect natural run: ascending vs strictly descending
            if counter.compare(arr[curr + 1], arr[curr]):
                # Strictly descending: arr[curr+1] < arr[curr]
                curr += 1
                while curr + 1 < n and counter.compare(arr[curr + 1], arr[curr]):
                    curr += 1
                curr += 1
                run_len = curr - start
                # Reverse strictly descending run in place to make it ascending
                _reverse_slice(arr, counter, start, curr)
            else:
                # Ascending / non-descending
                curr += 1
                while curr + 1 < n and not counter.compare(arr[curr + 1], arr[curr]):
                    curr += 1
                curr += 1
                run_len = curr - start

        # Extend short runs to minrun using binary_insertion_sort
        if run_len < minrun and curr < n:
            target_end = min(n, start + minrun)
            binary_insertion_sort(arr, counter, lo=start, hi=target_end)
            run_len = target_end - start
            curr = target_end

        run_stack.append((start, run_len))

        # Enforce Timsort stack invariants:
        # 1. len[i-2] > len[i-1] + len[i]
        # 2. len[i-1] > len[i]
        while len(run_stack) >= 2:
            if len(run_stack) >= 3 and run_stack[-3][1] <= run_stack[-2][1] + run_stack[-1][1]:
                if run_stack[-3][1] < run_stack[-1][1]:
                    # Merge -3 and -2
                    s3, l3 = run_stack[-3]
                    s2, l2 = run_stack[-2]
                    _merge_runs(arr, counter, s3, l3, s2, l2)
                    run_stack.pop(-2)
                    run_stack[-2] = (s3, l3 + l2)
                else:
                    # Merge -2 and -1
                    s2, l2 = run_stack[-2]
                    s1, l1 = run_stack[-1]
                    _merge_runs(arr, counter, s2, l2, s1, l1)
                    run_stack.pop()
                    run_stack[-1] = (s2, l2 + l1)
            elif run_stack[-2][1] <= run_stack[-1][1]:
                s2, l2 = run_stack[-2]
                s1, l1 = run_stack[-1]
                _merge_runs(arr, counter, s2, l2, s1, l1)
                run_stack.pop()
                run_stack[-1] = (s2, l2 + l1)
            else:
                break

    # Force merge all remaining runs on the stack
    while len(run_stack) > 1:
        s2, l2 = run_stack[-2]
        s1, l1 = run_stack[-1]
        _merge_runs(arr, counter, s2, l2, s1, l1)
        run_stack.pop()
        run_stack[-1] = (s2, l2 + l1)


# ===========================================================================
# B) Introsort
# ===========================================================================

def _median_of_three(arr: List[Any], counter: Counter, a: int, b: int, c: int) -> int:
    """Return index of the median value among arr[a], arr[b], arr[c]."""
    if counter.compare(arr[a], arr[b]):
        # a < b
        if counter.compare(arr[b], arr[c]):
            return b  # a < b < c
        elif counter.compare(arr[a], arr[c]):
            return c  # a < c <= b
        else:
            return a  # c <= a < b
    else:
        # b <= a
        if counter.compare(arr[a], arr[c]):
            return a  # b <= a < c
        elif counter.compare(arr[b], arr[c]):
            return c  # b < c <= a
        else:
            return b  # c <= b <= a


def _lomuto_partition(arr: List[Any], counter: Counter, lo: int, hi: int) -> int:
    """
    Lomuto partition with median-of-three pivot.
    Places median of lo, mid, hi-1 at arr[hi-1] as pivot, then partitions.
    """
    mid = lo + (hi - lo) // 2
    pivot_idx = _median_of_three(arr, counter, lo, mid, hi - 1)

    # Swap pivot into arr[hi - 1]
    if pivot_idx != hi - 1:
        arr[pivot_idx], arr[hi - 1] = arr[hi - 1], arr[pivot_idx]
        counter.move()
        counter.move()

    pivot = arr[hi - 1]
    i = lo
    for j in range(lo, hi - 1):
        if counter.compare(arr[j], pivot):
            if i != j:
                arr[i], arr[j] = arr[j], arr[i]
                counter.move()
                counter.move()
            i += 1

    if i != hi - 1:
        arr[i], arr[hi - 1] = arr[hi - 1], arr[i]
        counter.move()
        counter.move()
    return i


def _introsort_loop(
    arr: List[Any],
    counter: Counter,
    lo: int,
    hi: int,
    depth_limit: int,
    threshold: int,
    stats: Optional[Dict[str, int]] = None,
) -> None:
    """
    Recursive introsort partitioning loop.
    Falls back to heap_sort when depth_limit reaches 0.
    Stops recursion when partition size <= threshold.
    """
    while (hi - lo) > threshold:
        if depth_limit <= 0:
            if stats is not None:
                stats["heap_sort_fallbacks"] = stats.get("heap_sort_fallbacks", 0) + 1
            heap_sort(arr, counter, lo=lo, hi=hi)
            return

        depth_limit -= 1
        pivot_pos = _lomuto_partition(arr, counter, lo, hi)

        # Recurse on smaller half to bound call stack, loop on larger
        if (pivot_pos - lo) < (hi - (pivot_pos + 1)):
            _introsort_loop(arr, counter, lo, pivot_pos, depth_limit, threshold, stats)
            lo = pivot_pos + 1
        else:
            _introsort_loop(arr, counter, pivot_pos + 1, hi, depth_limit, threshold, stats)
            hi = pivot_pos


def introsort(
    arr: List[Any],
    counter: Counter,
    threshold: int = 16,
    max_depth_override: Optional[int] = None,
    stats: Optional[Dict[str, int]] = None,
) -> None:
    """
    Sort arr in place using Introsort (Introspective Sort).

    Parameters
    ----------
    arr                : List of elements to sort in place.
    counter            : Shared Counter.
    threshold          : Size threshold below which partitions are left for
                         the final insertion sort pass (default 16).
    max_depth_override : Optional override for testing depth limit fallback.
                         Default is 2 * floor(log2(len(arr))).
    stats              : Optional dict to track internal algorithm telemetry
                         (e.g. heap_sort fallback count).

    Complexity
    ----------
    Time  : Best  O(n log n).
            Avg   O(n log n).
            Worst O(n log n) (guaranteed by heap sort fallback).
    Space : O(log n) call stack.
    Stable: NO.
    """
    n = len(arr)
    if n <= 1:
        return

    if max_depth_override is not None:
        depth_limit = max_depth_override
    else:
        depth_limit = 2 * math.floor(math.log2(n)) if n > 0 else 0

    # Phase 1: QuickSort with depth tracking and heap_sort fallback
    _introsort_loop(arr, counter, 0, n, depth_limit, threshold, stats)

    # Phase 2: Final cleanup pass with insertion_sort across the entire array.
    # Musser's classic choice: each partition <= threshold is sorted in a
    # single continuous, cache-friendly insertion sort pass in O(n * threshold).
    insertion_sort(arr, counter, lo=0, hi=n)


# ===========================================================================
# Algorithmic Rationale: Why each hybrid exists
# ===========================================================================
"""
Why Hybrids Exist: Architectural Rationale
-------------------------------------------
1. Timsort solves Merge Sort's blindness to pre-existing structure and Insertion Sort's
   O(n²) scaling. Real-world ER data arrives partially ordered, batched, or reversed.
   Timsort detects existing monotonic runs in O(n) comparisons and uses binary insertion
   sort only on small tiles (minrun) where CPU cache locality dominates, achieving
   O(n) best-case performance while preserving stability.

2. Introsort solves Quicksort's disastrous O(n²) worst-case on pathological inputs and
   Heap Sort's poor spatial locality. By monitoring recursion depth, Introsort achieves
   Quicksort's rapid average-case partitioning, falls back to Heap Sort if depth exceeds
   2*floor(log2 n) to guarantee an O(n log n) safety net, and delegates leaves (<= T)
   to Insertion Sort to eliminate deep recursive overhead.
"""
