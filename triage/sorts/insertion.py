"""
triage.sorts.insertion
======================
Insertion sort and binary insertion sort, both instrumented with Counter.

Both functions sort ``arr[lo:hi]`` **in place**.  Every comparison uses
``counter.compare()``; every array write uses ``counter.move()``, following
the convention in ``triage.metrics``:

    Shift  arr[j+1] = arr[j]   →  1 move
    Place  arr[pos] = key       →  1 move   (once per outer iteration)

For a subrange of length n = hi - lo the totals are:

    comparisons  (insertion_sort)  = I           on sorted (I=0) → n-1 total
                                     n(n-1)/2    on reverse (I=n(n-1)/2)
    shifts                         = I           always (one shift per inversion)
    placements                     = n - 1       always (one per outer loop step)
    ─────────────────────────────────────────────────────────────────
    total moves                    = I + (n - 1) always

    comparisons  (binary_insertion_sort)
                                   = Θ(n log n) comparisons
                                     (for prefix length i, between ⌊log₂ i⌋ and ⌊log₂ i⌋ + 1)
    total moves                    = I + (n - 1)   (UNCHANGED — binary search
                                     finds the position faster but the physical
                                     shifts are still required)

insertion_sort
--------------
Time  : Best  O(n)   (sorted, I=0)
        Avg   O(n²)
        Worst O(n²)  (reverse, I=n(n-1)/2)
Space : O(1)  — single temp variable
Stable: YES  — inner guard is strict ``key < arr[j]``; equal elements halt
              the loop and are never displaced.
In-place: YES

binary_insertion_sort
---------------------
Time  : Best  O(n log n) comparisons, O(n) moves   (sorted, I=0)
        Avg   O(n log n) comparisons, O(n²) moves
        Worst O(n log n) comparisons, O(n²) moves   (reverse)
Space : O(1)
Stable: YES  — _binary_right inserts AFTER equal elements (key >= arr[mid]
              pushes the search window right).
In-place: YES

Key tradeoff
------------
Halving the comparison count does NOT reduce moves.  On small, cache-hot
arrays (as used inside Timsort runs) binary insertion eliminates branch
mispredictions while the O(n²) write cost remains acceptable.  On already-
sorted input, regular insertion sort is cheaper in comparisons (n-1 vs
Θ(n log n)); binary insertion is better only when inversions dominate.
"""

from __future__ import annotations

from typing import Any, List

from triage.metrics import Counter


# ---------------------------------------------------------------------------
# insertion_sort
# ---------------------------------------------------------------------------

def insertion_sort(
    arr: List[Any],
    counter: Counter,
    lo: int = 0,
    hi: int | None = None,
) -> None:
    """
    Sort ``arr[lo:hi]`` in place in ascending order using insertion sort.

    Loop invariant
    --------------
    At the start of outer iteration ``i``, ``arr[lo:i]`` is sorted in
    non-decreasing order.

    Correctness (3-line argument)
    -----------------------------
    1. Base    : When ``i = lo + 1``, ``arr[lo:lo+1]`` is a single element —
                 trivially sorted.
    2. Step    : ``key = arr[i]`` is saved.  All ``arr[lo:i]`` elements
                 strictly greater than ``key`` are shifted one slot right,
                 creating a gap.  ``key`` fills that gap, extending the sorted
                 prefix to ``arr[lo:i+1]``.
    3. Terminal: When ``i`` reaches ``hi``, ``arr[lo:hi]`` is fully sorted. ∎

    Parameters
    ----------
    arr     : Mutable sequence (modified in place).
    counter : Shared Counter — every comparison and move is recorded here.
    lo      : Inclusive start of the subrange to sort (default 0).
    hi      : Exclusive end of the subrange to sort (default ``len(arr)``).

    Move accounting (per outer iteration ``i``, zero-indexed from lo+1)
    -------------------------------------------------------------------
    * ``k`` shifts  →  ``k`` moves  (k = number of elements shifted right)
    * 1 placement   →  1 move       (``arr[j+1] = key``, always executed)

    Total shifts across all iterations = I (number of inversions in arr[lo:hi]).
    Total placements = n - 1.
    Total moves = I + (n - 1).

    Complexity
    ----------
    Time  : Best  O(n)   sorted input, I=0, n-1 comparisons, n-1 moves
            Avg   O(n²)
            Worst O(n²)  reverse input, n(n-1)/2 comparisons, n(n-1)/2+(n-1) moves
    Space : O(1)
    Stable : YES
    In-place : YES
    """
    if hi is None:
        hi = len(arr)

    # ── Outer loop: extend the sorted prefix arr[lo:i] by one element ────────
    for i in range(lo + 1, hi):
        key = arr[i]        # save key into temp  — FREE (no array write)
        j = i - 1

        # ── Inner loop: shift elements right while arr[j] > key ──────────────
        # Short-circuit: when j < lo the second operand is NOT evaluated,
        # so no spurious compare() call is made at the boundary.  This gives
        # exactly n(n-1)/2 comparisons on fully-reverse input.
        while j >= lo and counter.compare(key, arr[j]):  # key < arr[j]
            arr[j + 1] = arr[j]   # shift right: 1 move
            counter.move()
            j -= 1

        arr[j + 1] = key    # placement: 1 move  (executed every outer iteration)
        counter.move()


# ---------------------------------------------------------------------------
# _binary_right  (private helper)
# ---------------------------------------------------------------------------

def _binary_right(
    arr: List[Any],
    key: Any,
    lo: int,
    hi: int,
    counter: Counter,
) -> int:
    """
    Return the rightmost valid insertion index for *key* in sorted
    ``arr[lo:hi]``.

    Finds the smallest ``pos ∈ [lo, hi]`` such that every element in
    ``arr[lo:pos]`` satisfies ``element <= key`` and every element in
    ``arr[pos:hi]`` satisfies ``element > key``.

    Inserting at ``pos`` preserves relative order among equal elements
    (stability): ``key`` lands to the *right* of any existing equal element
    because the ``key >= arr[mid]`` branch advances the search window right.

    Uses ``counter.compare()`` for every comparison.

    Number of comparisons: ⌊log₂(hi - lo)⌋ + 1  for hi > lo, else 0.

    Time  : O(log n)  per call
    Space : O(1)
    """
    left, right = lo, hi
    while left < right:
        mid = (left + right) // 2
        if counter.compare(key, arr[mid]):   # key < arr[mid] → pos in left half
            right = mid
        else:                                 # key >= arr[mid] → pos in right half
            left = mid + 1
    return left


# ---------------------------------------------------------------------------
# binary_insertion_sort
# ---------------------------------------------------------------------------

def binary_insertion_sort(
    arr: List[Any],
    counter: Counter,
    lo: int = 0,
    hi: int | None = None,
) -> None:
    """
    Sort ``arr[lo:hi]`` in place using binary insertion sort.

    Identical to ``insertion_sort`` in every respect **except** that the
    insertion position is found with ``_binary_right`` (O(log i) comparisons
    per element) instead of a linear backward scan.

    This reduces the comparison count from O(n²) to Θ(n log n) but does
    **not** reduce moves: the physical shifts are still required after the
    position is found.  Total moves remain exactly I + (n - 1).

    Loop invariant, correctness argument, stability
    -----------------------------------------------
    Identical to ``insertion_sort``; only the *method* of finding the gap
    changes.  ``_binary_right`` guarantees insertion AFTER equal elements,
    so the stable-sort property is preserved.

    Parameters
    ----------
    arr     : Mutable sequence (modified in place).
    counter : Shared Counter.
    lo      : Inclusive start of the subrange (default 0).
    hi      : Exclusive end of the subrange (default ``len(arr)``).

    Complexity
    ----------
    Time  : Best  Θ(n log n) comparisons, O(n) moves    (sorted, I=0)
            Avg   Θ(n log n) comparisons, O(n²) moves
            Worst Θ(n log n) comparisons, O(n²) moves   (reverse, I=n(n-1)/2)

            Note: comparisons are Θ(n log n) (between ⌊log₂ i⌋ and
            ⌊log₂ i⌋ + 1 per insertion step i). On sorted input this
            exceeds the n-1 comparisons of regular insertion sort.
    Space : O(1)
    Stable : YES  — _binary_right inserts after equal elements.
    In-place : YES

    When to prefer this over insertion_sort
    ----------------------------------------
    * Comparisons are expensive (complex key, remote memory, function call).
    * Array is small enough that O(n²) moves fit in cache (Timsort run sizes).
    * Input is NOT nearly sorted (on nearly-sorted data linear scan wins on
      comparisons AND exits the inner loop early on moves).
    """
    if hi is None:
        hi = len(arr)

    for i in range(lo + 1, hi):
        key = arr[i]                                   # save key — FREE

        # Find rightmost valid insertion position in arr[lo:i] using binary search.
        pos = _binary_right(arr, key, lo, i, counter)

        # Shift arr[pos:i] one slot right (same moves as linear insertion sort).
        j = i - 1
        while j >= pos:
            arr[j + 1] = arr[j]   # shift: 1 move
            counter.move()
            j -= 1

        arr[pos] = key             # placement: 1 move
        counter.move()
