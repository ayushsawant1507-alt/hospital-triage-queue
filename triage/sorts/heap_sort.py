"""
triage.sorts.heap_sort
======================
In-place heap sort built from scratch using a max-heap and hole-style sifting.

Sorts ``arr[lo:hi]`` **in place** in ascending order.
Every comparison goes through ``counter.compare()``, and every array write
goes through ``counter.move()``.

Algorithm
---------
1. Phase 1 (Heap Construction):
   Transform ``arr[lo:hi]`` into a max-heap in O(n) time using Floyd's bottom-up
   sifting from index ``lo + (n // 2) - 1`` down to ``lo``.
2. Phase 2 (Extraction / Sort-down):
   Repeatedly move the maximum element at the root (``arr[lo]``) to the end of
   the active heap range (``arr[lo + end]``) and sift down the displaced element
   within ``arr[lo : lo + end]``.

Complexity
----------
Time  : Best  Θ(n log n)
        Avg   Θ(n log n)
        Worst Θ(n log n)
Space : O(1) auxiliary (strictly in place)
Stable: NO — sift operations skip distant tree levels, destroying original relative
        order of equal elements. (In Patient, the composite key (severity, arrival_seq)
        guarantees a total deterministic order).
In-place: YES
"""

from __future__ import annotations

from typing import Any, List, Optional

from triage.metrics import Counter


def _sift_down_max(
    arr: List[Any],
    counter: Counter,
    lo: int,
    root_idx: int,
    size: int,
) -> None:
    """
    Hole-style sift_down for max-heap within subrange arr[lo : lo + size].

    root_idx is 0-indexed relative to lo.
    Moves: d copies + 1 final placement = d + 1 moves (if d > 0), else 0.
    """
    target = arr[lo + root_idx]
    curr = root_idx
    depth = 0

    while True:
        left = 2 * curr + 1
        right = 2 * curr + 2

        best_child: Optional[int] = None
        if left < size and right < size:
            # For max-heap, child with larger value has priority:
            # arr[lo + left] < arr[lo + right] -> right is bigger
            if counter.compare(arr[lo + left], arr[lo + right]):
                best_child = right
            else:
                best_child = left
        elif left < size:
            best_child = left

        # If best_child exists and target < arr[lo + best_child], shift child up into hole
        if best_child is not None and counter.compare(target, arr[lo + best_child]):
            arr[lo + curr] = arr[lo + best_child]
            counter.move()
            curr = best_child
            depth += 1
        else:
            break

    if depth > 0:
        arr[lo + curr] = target
        counter.move()


def heap_sort(
    arr: List[Any],
    counter: Counter,
    lo: int = 0,
    hi: Optional[int] = None,
) -> None:
    """
    Sort ``arr[lo:hi]`` in place in ascending order using max-heap sort.

    Parameters
    ----------
    arr     : List of elements to sort in place.
    counter : Shared Counter for tracking comparisons and moves.
    lo      : Inclusive start index (default: 0).
    hi      : Exclusive end index (default: len(arr)).
    """
    if hi is None:
        hi = len(arr)

    n = hi - lo
    if n <= 1:
        return

    # Phase 1: Build max-heap bottom-up in O(n)
    for i in range((n // 2) - 1, -1, -1):
        _sift_down_max(arr, counter, lo, i, n)

    # Phase 2: Extract max to the end and sift down
    for end in range(n - 1, 0, -1):
        # Extract max (arr[lo]) to arr[lo + end]
        max_elem = arr[lo]
        last_elem = arr[lo + end]

        arr[lo + end] = max_elem
        counter.move()

        # Place displaced last element temporarily at root and sift down
        arr[lo] = last_elem
        counter.move()

        _sift_down_max(arr, counter, lo, 0, end)
