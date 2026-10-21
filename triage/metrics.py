"""
triage.metrics
==============
Shared performance counter for algorithm analysis.

All sorting, searching, and heap algorithms in this project accept a
``Counter`` instance and MUST use ``counter.compare()`` and
``counter.move()`` instead of raw ``<`` or assignment, so that the
benchmark harness can record accurate operation counts without modifying
algorithm logic.

Design notes
------------
* ``Counter`` is a plain mutable dataclass — no threading primitives.
  Each benchmark run passes in a fresh (or reset) counter, so there is
  no shared state between runs.
* ``compare(a, b)`` returns the boolean result of ``a < b`` *and*
  increments ``comparisons``.  Wrapping the comparison ensures algorithms
  cannot accidentally use bare ``<`` without counting it.
* ``move()`` increments ``moves`` by one logical element move.  Callers
  decide what constitutes a "move" (swap = 2 moves, shift = 1 move, etc.)
  consistent with the algorithm's documented model.
* ``snapshot()`` returns a plain ``dict`` so benchmark CSVs can be written
  with ``csv.DictWriter`` without any extra serialisation step.
"""

# =============================================================================
# MOVE-COUNTING CONVENTION  (canonical reference for every algorithm)
# =============================================================================
#
# We count ELEMENT WRITES TO THE ARRAY (or heap backing list).
# A "read into a temporary variable" is FREE — it does not call move().
#
# Operation                        | Writes | move() calls | Example
# ---------------------------------|--------|--------------|---------------------------
# Swap  arr[i] <-> arr[j]         |   2    |      2       | tmp=arr[i]; arr[i]=arr[j];
#   (3 lines of Python, 2 writes) |        |              |   arr[j]=tmp
# ---------------------------------|--------|--------------|---------------------------
# Insertion-sort SHIFT             |   1    |      1       | arr[j+1] = arr[j]
#   (slide one element right)      |        |              |
# ---------------------------------|--------|--------------|---------------------------
# Insertion-sort KEY PLACEMENT     |   1    |      1       | arr[gap] = key
#   (write held-out key into gap)  |        |              |
# ---------------------------------|--------|--------------|---------------------------
# Heap sift COPY (hole-style)      |   1    |      1       | hole = child; arr[p] = arr[c]
#   d copies down the path, then   |        |              |
#   1 final placement of the key   |        |              | arr[hole] = key
# ---------------------------------|--------|--------------|---------------------------
#
# Derived totals
# --------------
# * Full swap                    = 2 moves  (always)
#
# * Insertion sort, n elements:
#   - shifts    = I  (I = number of inversions in the input; 0 <= I <= n(n-1)/2)
#   - placements = n - 1  (one key placement per outer iteration, always)
#   - TOTAL MOVES = I + (n - 1)
#   Best case (already sorted, I=0): n-1 moves.
#   Worst case (reverse sorted, I=n(n-1)/2): n(n-1)/2 + (n-1) = O(n^2) moves.
#
# * Heap sift of depth d (hole-style):
#   d copies (one per level descended) + 1 final key placement = d + 1 moves.
#   NOTE: heap.py MUST implement hole-style sift to match this count.
#         (Hole-style: save key, copy children up, write key once at the hole —
#          avoids the 3x-write cost of swap-based sifting.)
#
# Why writes, not comparisons?
# ----------------------------
# Comparisons and moves are orthogonal costs.  On modern hardware, branch
# mispredictions (comparisons) and cache misses (writes) are distinct
# bottlenecks.  Counting them separately lets the analysis state precisely:
# on nearly sorted input (k inversions) insertion sort does O(n+k) comparisons
# and O(n+k) moves; heap sort does Theta(n log n) of both regardless of input.
#
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Counter:
    """
    Mutable performance counter passed into every algorithm.

    Attributes
    ----------
    comparisons : int
        Number of element-to-element comparisons performed.
    moves : int
        Number of logical element moves (writes/shifts/swaps) performed.

    Usage
    -----
    >>> c = Counter()
    >>> c.compare(3, 5)
    True
    >>> c.move()
    >>> c.snapshot()
    {'comparisons': 1, 'moves': 1}

    Thread safety
    -------------
    Not thread-safe.  Use one Counter per benchmark run.
    """

    comparisons: int = field(default=0)
    moves:       int = field(default=0)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset both counters to zero.

        Call between benchmark runs to reuse the same Counter object.

        Time  : O(1)
        Space : O(1)
        """
        self.comparisons = 0
        self.moves = 0

    def compare(self, a: Any, b: Any) -> bool:
        """
        Increment ``comparisons`` by 1 and return ``a < b``.

        Parameters
        ----------
        a, b : Any
            Two elements that support the ``<`` operator.

        Returns
        -------
        bool
            ``True`` if ``a < b``, ``False`` otherwise.

        Time  : O(1)
        Space : O(1)

        Notes
        -----
        Algorithms must use this method exclusively for element comparisons
        so that the count is always accurate.  Using bare ``<`` is a bug.
        """
        self.comparisons += 1
        return a < b  # type: ignore[operator]

    def move(self) -> None:
        """
        Increment ``moves`` by 1.

        Call once per logical element write.  For a swap of two elements,
        call ``move()`` twice (or call it however many writes the algorithm
        actually performs — be consistent within each module).

        Time  : O(1)
        Space : O(1)
        """
        self.moves += 1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, int]:
        """
        Return a shallow copy of the current counter state as a plain dict.

        The returned dict is independent of the Counter object — subsequent
        mutations do not affect the snapshot.

        Returns
        -------
        dict
            ``{"comparisons": <int>, "moves": <int>}``

        Time  : O(1)
        Space : O(1)

        Notes
        -----
        Designed for use with ``csv.DictWriter``; keys match the expected
        CSV column names used in ``benchmarks.py``.
        """
        return {"comparisons": self.comparisons, "moves": self.moves}

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return f"Counter(comparisons={self.comparisons}, moves={self.moves})"
