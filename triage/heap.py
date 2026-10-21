"""
triage.heap
===========
Raw array-based binary heap built from scratch with hole-style sifting.

Supports both min-heap (default) and max-heap via ``max_heap: bool``.
Every comparison goes through ``counter.compare()``, and every array write
goes through ``counter.move()``.

0-Indexed Array Layout
----------------------
For element at index ``i``:
- Left child  : ``2 * i + 1``
- Right child : ``2 * i + 2``
- Parent      : ``(i - 1) // 2``

Hole-Style Sifting Convention (see ``triage.metrics``)
------------------------------------------------------
Instead of repeatedly swapping parent and child (which would cost 3 writes
per level of the heap), hole-style sifting lifts the displaced element into a
temporary variable (0 writes), copies elements along the sift path (1 write per
step of depth), and places the target into the final vacant position (1 write).
For a sift traversing depth ``d``:
    moves = d copies + 1 placement = d + 1 moves.
If ``d = 0`` (the element already satisfies the heap property): 0 copies,
0 placements -> 0 moves.

Floyd's Bottom-Up build_heap Proof: O(n) Time Complexity
--------------------------------------------------------
A naive heap construction inserts n elements one by one, each taking O(log n)
time, for O(n log n) total work.
Floyd's algorithm builds the heap bottom-up by sifting down starting from the
deepest internal node ``(n // 2) - 1`` down to 0:

Let h = ⌊log₂ n⌋ be the height of the heap.
In an n-element binary tree, there are at most ⌈n / 2^(k+1)⌉ nodes at height k.
The maximum sift-down distance for a node at height k is k.
The total number of comparisons and moves across all nodes is bounded by:
    S = ∑_{k=0}^{h} ⌈n / 2^(k+1)⌉ * k <= n * ∑_{k=1}^{∞} k / 2^k

Recall the standard geometric series identity:
    ∑_{k=1}^{∞} k * x^k = x / (1 - x)²  for |x| < 1.
For x = 1/2:
    ∑_{k=1}^{∞} k / 2^k = (1/2) / (1 - 1/2)² = (1/2) / (1/4) = 2.

Therefore:
    S <= n * 2 = 2n = O(n).

Thus, ``build_heap`` runs in strictly O(n) comparisons and moves in the worst case,
independent of initial element distribution.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from triage.metrics import Counter


class BinaryHeap:
    """
    0-indexed binary heap with hole-style sift_up and sift_down.

    Parameters
    ----------
    counter : Counter
        Shared counter measuring comparisons and element writes.
    max_heap : bool, default=False
        If False, acts as a min-heap (smaller elements higher priority).
        If True, acts as a max-heap (larger elements higher priority).
    on_move : Optional[Callable[[Any, int], None]], default=None
        Callback invoked whenever an element is written to the array:
        ``on_move(item, index)``. Used by PriorityQueue to maintain an O(1)
        id-to-index index table.
    """

    def __init__(
        self,
        counter: Counter,
        max_heap: bool = False,
        on_move: Optional[Callable[[Any, int], None]] = None,
    ) -> None:
        self._data: List[Any] = []
        self._counter: Counter = counter
        self._max_heap: bool = max_heap
        self._on_move: Optional[Callable[[Any, int], None]] = on_move

    def __len__(self) -> int:
        return len(self._data)

    def is_empty(self) -> bool:
        return len(self._data) == 0

    @property
    def data(self) -> List[Any]:
        return self._data

    def _prioritizes(self, a: Any, b: Any) -> bool:
        """
        Return True if element ``a`` has higher priority than element ``b`` in the heap.
        For a min-heap, ``a < b``. For a max-heap, ``b < a``.
        """
        if self._max_heap:
            return self._counter.compare(b, a)
        return self._counter.compare(a, b)

    def _write(self, index: int, item: Any) -> None:
        """Write item to self._data[index], counting 1 move and firing callback."""
        self._data[index] = item
        self._counter.move()
        if self._on_move is not None:
            self._on_move(item, index)

    def sift_up(self, index: int) -> int:
        """
        Sift element at ``index`` upward using hole-style copying.

        Returns final index of the element.
        Moves: d copies + 1 placement = d + 1 moves (if d > 0), else 0.
        """
        if index <= 0:
            return index

        target = self._data[index]
        curr = index
        depth = 0

        while curr > 0:
            parent = (curr - 1) // 2
            # Does target have higher priority than parent?
            if self._prioritizes(target, self._data[parent]):
                self._write(curr, self._data[parent])
                curr = parent
                depth += 1
            else:
                break

        if depth > 0:
            self._write(curr, target)
        return curr

    def sift_down(self, index: int, size: Optional[int] = None) -> int:
        """
        Sift element at ``index`` downward within ``arr[0:size]`` using hole-style copying.

        Returns final index of the element.
        Moves: d copies + 1 placement = d + 1 moves (if d > 0), else 0.
        """
        if size is None:
            size = len(self._data)

        if index >= size:
            return index

        target = self._data[index]
        curr = index
        depth = 0

        while True:
            left = 2 * curr + 1
            right = 2 * curr + 2
            best = curr

            # Find highest priority among target (or current hole), left child, right child
            candidate = target if depth == 0 else self._data[curr]

            best_child = None
            if left < size and right < size:
                if self._prioritizes(self._data[right], self._data[left]):
                    best_child = right
                else:
                    best_child = left
            elif left < size:
                best_child = left

            if best_child is not None and self._prioritizes(self._data[best_child], target):
                self._write(curr, self._data[best_child])
                curr = best_child
                depth += 1
            else:
                break

        if depth > 0:
            self._write(curr, target)
        return curr

    def push(self, item: Any) -> None:
        """
        Push item onto heap.
        Time: O(log n)
        Space: O(1) auxiliary
        """
        idx = len(self._data)
        self._data.append(item)
        self._counter.move()  # Appending element is 1 move
        if self._on_move is not None:
            self._on_move(item, idx)
        self.sift_up(idx)

    def pop(self) -> Any:
        """
        Pop and return the highest priority element.
        Time: O(log n)
        Space: O(1) auxiliary
        """
        if not self._data:
            raise IndexError("pop from empty heap")

        root = self._data[0]
        last = self._data.pop()

        if self._data:
            self._write(0, last)
            self.sift_down(0)

        return root

    def peek(self) -> Any:
        """Return highest priority element without removing it. O(1)."""
        if not self._data:
            raise IndexError("peek from empty heap")
        return self._data[0]

    def build_heap(self, items: List[Any]) -> None:
        """
        Initialize heap from an unsorted list in O(n) time using Floyd's algorithm.

        Time: O(n)
        Space: O(1) auxiliary (copies input list)
        """
        self._data = list(items)
        n = len(self._data)
        # Notify callback and count initial populating if needed
        for i in range(n):
            self._counter.move()
            if self._on_move is not None:
                self._on_move(self._data[i], i)

        # Bottom-up sift_down from deepest parent down to root
        for i in range((n // 2) - 1, -1, -1):
            self.sift_down(i)
