"""
triage.search
=============
Searching algorithms instrumented with Counter, and the PatientIndex catalog.

Algorithms
----------
- linear_search           : O(n) scan baseline.
- binary_search_iterative : O(log n) search using deferred equality (at most
                            floor(log2 n) + 1 comparisons on powers of 2).
- binary_search_recursive : O(log n) divide-and-conquer implementation.
- lower_bound             : First index with element >= target, or -1.
- upper_bound             : First index with element > target, or -1.

Classes
-------
- PatientIndex : Catalog maintained sorted by patient ID.
                 Build in O(n log n), lookup in O(log n), insert in O(n).

Functions
---------
- break_even_lookups(n) : Computes minimum query count where sorting first
                          beats repeated linear searches.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

from triage.metrics import Counter
from triage.models import Patient
from triage.sorts.insertion import binary_insertion_sort


# ---------------------------------------------------------------------------
# Linear Search
# ---------------------------------------------------------------------------

def linear_search(arr: List[Any], target: Any, counter: Counter) -> int:
    """
    Search for target in arr via sequential scan.

    Returns index if found, else -1.
    Time  : Best O(1), Avg O(n), Worst O(n).
    Space : O(1).
    """
    for i, item in enumerate(arr):
        # Equality check: neither item < target nor target < item
        if not counter.compare(item, target) and not counter.compare(target, item):
            return i
    return -1


# ---------------------------------------------------------------------------
# Binary Search (Iterative)
# ---------------------------------------------------------------------------

def binary_search_iterative(arr: List[Any], target: Any, counter: Counter) -> int:
    """
    Search for target in sorted arr using iterative binary search.

    Uses deferred equality testing to guarantee at most 1 comparison per
    search-interval halving plus 1 final validation comparison.
    For n = 2^k, exactly floor(log2 n) + 1 comparisons.

    Returns index if found, else -1.
    Time  : O(log n).
    Space : O(1).
    """
    if not arr:
        return -1

    lo = 0
    hi = len(arr)

    while lo < hi:
        mid = (lo + hi) // 2
        # Single comparison per level
        if counter.compare(arr[mid], target):
            lo = mid + 1
        else:
            hi = mid

    # Final check: arr[lo] >= target is guaranteed; check if arr[lo] == target
    if lo < len(arr) and not counter.compare(target, arr[lo]):
        return lo
    return -1


# ---------------------------------------------------------------------------
# Binary Search (Recursive)
# ---------------------------------------------------------------------------

def _bsearch_rec_helper(
    arr: List[Any],
    target: Any,
    lo: int,
    hi: int,
    counter: Counter,
) -> int:
    if lo >= hi:
        if lo < len(arr) and not counter.compare(target, arr[lo]):
            return lo
        return -1

    mid = (lo + hi) // 2
    if counter.compare(arr[mid], target):
        return _bsearch_rec_helper(arr, target, mid + 1, hi, counter)
    return _bsearch_rec_helper(arr, target, lo, mid, counter)


def binary_search_recursive(arr: List[Any], target: Any, counter: Counter) -> int:
    """
    Search for target in sorted arr using recursive binary search.

    Returns index if found, else -1.
    Time  : O(log n).
    Space : O(log n) call stack.
    """
    if not arr:
        return -1
    return _bsearch_rec_helper(arr, target, 0, len(arr), counter)


# ---------------------------------------------------------------------------
# Bounds (lower_bound / upper_bound)
# ---------------------------------------------------------------------------

def lower_bound(arr: List[Any], target: Any, counter: Counter) -> int:
    """
    Return the index of the first element in sorted arr that is >= target.

    If all elements are < target, returns -1.
    Time  : O(log n).
    Space : O(1).
    """
    if not arr:
        return -1

    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if counter.compare(arr[mid], target):
            lo = mid + 1
        else:
            hi = mid

    return lo if lo < len(arr) else -1


def upper_bound(arr: List[Any], target: Any, counter: Counter) -> int:
    """
    Return the index of the first element in sorted arr that is strictly > target.

    If all elements are <= target, returns -1.
    Time  : O(log n).
    Space : O(1).
    """
    if not arr:
        return -1

    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        # arr[mid] <= target <=> not (target < arr[mid])
        if not counter.compare(target, arr[mid]):
            lo = mid + 1
        else:
            hi = mid

    return lo if lo < len(arr) else -1


# ---------------------------------------------------------------------------
# Break-Even Analysis
# ---------------------------------------------------------------------------

def break_even_lookups(n: int) -> int:
    """
    Compute the number of lookups m at which sorting an unsorted dataset of size n
    followed by binary searches beats repeated linear searches.

    Derivation
    ----------
    Let n be the number of records, and m be the number of lookup queries.

    Strategy A (Repeated Linear Search on Unsorted Data):
        Cost = m * (n / 2) comparisons on average.

    Strategy B (Sort Once + Repeated Binary Searches):
        Cost = C_sort(n) + m * C_bsearch(n)
             ≈ n * log2(n) + m * log2(n) comparisons.

    Break-even occurs when Strategy B becomes strictly cheaper:
        n * log2(n) + m * log2(n) < m * (n / 2)
        n * log2(n) < m * (n / 2 - log2(n))
        m > (n * log2(n)) / (n / 2 - log2(n))

    For large n, (n/2 - log2(n)) ≈ n/2, simplifying to:
        m ≈ 2 * log2(n).

    Parameters
    ----------
    n : int
        Number of items in the dataset (must be >= 4 for n/2 > log2(n)).

    Returns
    -------
    int
        Minimum integer queries m needed to justify the sorting cost.
    """
    if n <= 1:
        return 1
    return max(1, math.ceil(2 * math.log2(n)))


# ---------------------------------------------------------------------------
# PatientIndex
# ---------------------------------------------------------------------------

class _PatientIdWrapper:
    """Helper wrapper for comparing Patients strictly by integer id."""
    __slots__ = ("patient",)

    def __init__(self, patient: Patient) -> None:
        self.patient = patient

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, _PatientIdWrapper):
            return self.patient.id < other.patient.id
        if isinstance(other, int):
            return self.patient.id < other
        return NotImplemented


class PatientIndex:
    """
    Catalog of Patient records maintained in sorted order by patient id.

    Build  : O(n log n)
    Lookup : O(log n)
    Insert : O(n)
    """

    def __init__(self, patients: Optional[List[Patient]] = None, counter: Optional[Counter] = None) -> None:
        self._counter = counter if counter is not None else Counter()
        self._data: List[Patient] = []
        if patients:
            self.build(patients)

    def __len__(self) -> int:
        return len(self._data)

    def build(self, patients: List[Patient]) -> None:
        """
        Build the index in O(n log n) time by sorting by patient.id.
        """
        self._data = list(patients)
        wrappers = [_PatientIdWrapper(p) for p in self._data]
        binary_insertion_sort(wrappers, self._counter)
        self._data = [w.patient for w in wrappers]

    def lookup(self, patient_id: int) -> Optional[Patient]:
        """
        Locate patient by ID in O(log n) time using binary search.
        """
        if not self._data:
            return None

        lo, hi = 0, len(self._data)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._counter.compare(self._data[mid].id, patient_id):
                lo = mid + 1
            else:
                hi = mid

        if lo < len(self._data) and not self._counter.compare(patient_id, self._data[lo].id):
            return self._data[lo]
        return None

    def insert(self, patient: Patient) -> None:
        """
        Insert a patient while maintaining sorted order by id.
        Time: O(log n) search + O(n) shift = O(n).
        """
        if not self._data:
            self._data.append(patient)
            self._counter.move()
            return

        # Find insertion position in O(log n)
        lo, hi = 0, len(self._data)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._counter.compare(self._data[mid].id, patient.id):
                lo = mid + 1
            else:
                hi = mid

        self._data.insert(lo, patient)
        self._counter.move()
