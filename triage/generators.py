"""
triage.generators
=================
Seeded, reproducible input generators for benchmarking and testing.

Every public generator returns a *new* list of ``Patient`` objects whose
``arrival_seq`` values form a permutation of ``{0, 1, …, n-1}``.  Patient
``id`` values also form the same set but are shuffled **independently** of
``arrival_seq`` using the same seeded RNG, so the two fields carry no
positional information about each other.

Generators
----------
gen_random          : Uniformly random severities.
gen_sorted          : Sorted (best case for insertion sort; I = 0 inversions).
gen_reverse         : Reverse sorted (worst case; I = n(n-1)/2 inversions).
gen_nearly_sorted   : Sorted + k random adjacent swaps (I ≤ k inversions).
gen_few_unique      : Few distinct severity values; stresses FIFO tie-breaking.
gen_er_mix          : Realistic ER distribution (see _ER_WEIGHTS below).

Utility
-------
count_inversions    : O(n log n) merge-sort inversion counter.  Returns I,
                      the exact number of inversions in the patient list
                      (= number of shifts insertion sort will perform, per the
                      convention in metrics.py).

GENERATORS          : Dict[str, Callable] mapping short names to callables
                      with signature ``(n: int, *, seed: int = 42)``.

ER severity distribution (_ER_WEIGHTS)
---------------------------------------
+-------------+--------------------+--------+
| Severity    | Label              | Weight |
+=============+====================+========+
| 1           | CRITICAL           |   5 %  |
| 2           | EMERGENT           |  15 %  |
| 3           | URGENT             |  40 %  |
| 4           | LESS_URGENT        |  40 %  |
| 5           | NON_URGENT         |   0 %  |
+-------------+--------------------+--------+
NON_URGENT weight is 0.0 — walk-in non-emergency patients are not tracked
in this simulation's triage queue (4-level model matching ESI levels 1-4).
"""

from __future__ import annotations

import random
from functools import partial
from typing import Callable, Dict, List, Sequence, Tuple

from triage.models import Patient, Severity

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Severity: CRITICAL  EMERGENT  URGENT  LESS_URGENT  NON_URGENT
_ER_WEIGHTS: Tuple[float, ...] = (0.05, 0.15, 0.40, 0.40, 0.00)

_ALL_SEVERITIES: Tuple[Severity, ...] = tuple(Severity)

# Default k used in the GENERATORS dict entry for "nearly_sorted".
_NEARLY_SORTED_DEFAULT_K: int = 10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_patient(pid: int, sev: Severity, seq: int) -> Patient:
    """Construct a Patient with a zero-padded synthetic name derived from pid."""
    return Patient(id=pid, name=f"P{pid:06d}", severity=sev, arrival_seq=seq)


def _patients_from_severities(
    severities: Sequence[Severity],
    *,
    pids: Sequence[int] | None = None,
    arrival_seqs: Sequence[int] | None = None,
) -> List[Patient]:
    """
    Wrap severity, pid, and arrival-sequence lists into a ``Patient`` list.

    Parameters
    ----------
    severities   : Severity value for each position.
    pids         : Hospital record numbers — must be a permutation of
                   ``0 … n-1``.  Defaults to ``range(n)`` (identity).
    arrival_seqs : Admission counters — must be a permutation of
                   ``0 … n-1``.  Defaults to ``range(n)`` (identity).

    The two sequences are kept independent so callers can shuffle IDs
    without disturbing the FIFO ordering that ``arrival_seq`` encodes.

    Time  : O(n)
    Space : O(n)
    """
    n = len(severities)
    _pids: Sequence[int] = range(n) if pids is None else pids
    _seqs: Sequence[int] = range(n) if arrival_seqs is None else arrival_seqs
    return [
        _make_patient(pid=_pids[i], sev=sev, seq=_seqs[i])
        for i, sev in enumerate(severities)
    ]


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def gen_random(n: int, *, seed: int = 42) -> List[Patient]:
    """
    Generate *n* patients with uniformly random severity levels.

    Patient IDs are shuffled independently of ``arrival_seq`` using the same
    RNG (after severity sampling), so the two fields carry no shared order.

    Parameters
    ----------
    n    : Number of patients.
    seed : RNG seed for full reproducibility.

    Returns
    -------
    List[Patient]
        Unsorted list.  ``arrival_seq`` values = 0 … n-1 in list order.
        ``id`` values = a shuffled permutation of 0 … n-1.

    Complexity
    ----------
    Time  : O(n)
    Space : O(n)

    Notes
    -----
    Expected inversion count ≈ n(n-1)/4 (uniform distribution).
    """
    rng = random.Random(seed)
    severities = [rng.choice(_ALL_SEVERITIES) for _ in range(n)]
    pids = list(range(n))
    rng.shuffle(pids)                          # independent of arrival_seq
    return _patients_from_severities(severities, pids=pids)


def gen_sorted(n: int, *, seed: int = 42) -> List[Patient]:
    """
    Generate *n* patients already sorted by ``(severity, arrival_seq)``.

    This is the **best case** for insertion sort: I = 0 inversions,
    so total moves = 0 + (n-1) = n-1.

    Patient IDs are shuffled with the seeded RNG so they do not coincide
    with position (simulating a real hospital where record numbers are
    assigned at registration, not in arrival order).

    Parameters
    ----------
    n    : Number of patients.
    seed : RNG seed (used for pid shuffle).

    Returns
    -------
    List[Patient]
        Severity-ascending list.  ``arrival_seq`` values = 0 … n-1.
        ``id`` values = shuffled permutation of 0 … n-1.

    Complexity
    ----------
    Time  : O(n log n)  (list.sort on the severity sequence)
    Space : O(n)
    """
    rng = random.Random(seed)
    severities = [_ALL_SEVERITIES[i % len(_ALL_SEVERITIES)] for i in range(n)]
    severities.sort()                          # sort() is on plain Severity values — OK here
    pids = list(range(n))
    rng.shuffle(pids)
    return _patients_from_severities(severities, pids=pids)


def gen_reverse(n: int, *, seed: int = 42) -> List[Patient]:
    """
    Generate *n* patients in reverse-sorted order (worst case for insertion
    sort) with exactly **n(n-1)/2 inversions**.

    Proof of inversion count
    ------------------------
    Severities are sorted descending.  ``arrival_seq`` is assigned as
    ``n-1-i`` (largest to position 0, smallest to position n-1).
    For any pair ``(i, j)`` with ``i < j``:

    * ``sev[i] >= sev[j]`` (reverse-sorted severity).
    * ``arrival_seq[i] = n-1-i > n-1-j = arrival_seq[j]``.

    Combined, ``key[i] > key[j]`` in both cases (strict when severities
    tie, because the arrival_seq parts differ).  Therefore *all* n(n-1)/2
    pairs form inversions, regardless of how many distinct severity values
    are present.

    Parameters
    ----------
    n    : Number of patients.
    seed : RNG seed (used for pid shuffle).

    Returns
    -------
    List[Patient]
        Severity-descending list.  ``arrival_seq`` values = n-1 … 0.
        ``id`` values = shuffled permutation of 0 … n-1.

    Complexity
    ----------
    Time  : O(n log n)
    Space : O(n)
    """
    rng = random.Random(seed)
    severities = [_ALL_SEVERITIES[i % len(_ALL_SEVERITIES)] for i in range(n)]
    severities.sort(reverse=True)
    pids = list(range(n))
    rng.shuffle(pids)
    # Assign arrival_seq in reverse order so every (i, j) pair is an inversion.
    arrival_seqs = list(range(n - 1, -1, -1))
    return _patients_from_severities(severities, pids=pids, arrival_seqs=arrival_seqs)


def gen_nearly_sorted(n: int, k: int, *, seed: int = 42) -> List[Patient]:
    """
    Generate *n* patients that are sorted, then disturbed by *k* random
    adjacent swaps.

    Design rationale
    ----------------
    Starting from I = 0 (sorted list) and applying k adjacent swaps adds at
    most k inversions (each swap increases the inversion count by at most 1).
    For small k the resulting inversion count I ≤ k, which directly
    demonstrates insertion sort's O(n + k) behaviour via the formula
    total_moves = I + (n-1).

    When k = 0 this generator produces identical output to ``gen_sorted``
    for the same seed (same RNG state at the pid-shuffle step, since no
    ``randint`` calls are made during the zero-iteration swap loop).

    Parameters
    ----------
    n    : Number of patients (≥ 1).
    k    : Number of random adjacent swaps (disorder parameter; ≥ 0).
    seed : RNG seed for reproducibility.

    Returns
    -------
    List[Patient]
        Nearly-sorted list.  ``arrival_seq`` values = 0 … n-1.
        ``id`` values = shuffled permutation of 0 … n-1.

    Complexity
    ----------
    Time  : O(n log n + k)
    Space : O(n)

    Raises
    ------
    ValueError
        If n < 1 or k < 0.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return gen_sorted(1)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    rng = random.Random(seed)
    severities = [_ALL_SEVERITIES[i % len(_ALL_SEVERITIES)] for i in range(n)]
    severities.sort()

    for _ in range(k):
        i = rng.randint(0, n - 2)
        severities[i], severities[i + 1] = severities[i + 1], severities[i]

    pids = list(range(n))
    rng.shuffle(pids)                          # shuffled after swaps, independent of seq
    return _patients_from_severities(severities, pids=pids)


def gen_few_unique(n: int, num_severities: int = 2, *, seed: int = 42) -> List[Patient]:
    """
    Generate *n* patients drawn from only *num_severities* distinct severity
    levels, creating many same-severity ties broken solely by ``arrival_seq``.

    Use-cases
    ---------
    * Stress-tests FIFO tie-breaking in the priority queue and sorts.
    * Mimics an ER where most patients cluster around a small set of levels.

    Parameters
    ----------
    n              : Number of patients.
    num_severities : How many distinct Severity values to sample from (1–5).
    seed           : RNG seed for reproducibility.

    Returns
    -------
    List[Patient]
        Randomly ordered list.  ``arrival_seq`` values = 0 … n-1.
        ``id`` values = shuffled permutation of 0 … n-1.

    Complexity
    ----------
    Time  : O(n)
    Space : O(n)

    Raises
    ------
    ValueError
        If ``num_severities`` is not in [1, 5].
    """
    if not (1 <= num_severities <= len(_ALL_SEVERITIES)):
        raise ValueError(
            f"num_severities must be 1–{len(_ALL_SEVERITIES)}, got {num_severities}"
        )
    rng = random.Random(seed)
    pool = _ALL_SEVERITIES[:num_severities]
    severities = [rng.choice(pool) for _ in range(n)]
    pids = list(range(n))
    rng.shuffle(pids)
    return _patients_from_severities(severities, pids=pids)


def gen_er_mix(n: int, *, seed: int = 42) -> List[Patient]:
    """
    Generate *n* patients following a realistic 4-level ER severity
    distribution (NON_URGENT weight = 0.0).

    Weights
    -------
    CRITICAL    :  5 %
    EMERGENT    : 15 %
    URGENT      : 40 %
    LESS_URGENT : 40 %
    NON_URGENT  :  0 %

    Parameters
    ----------
    n    : Number of patients.
    seed : RNG seed for reproducibility.

    Returns
    -------
    List[Patient]
        Randomly ordered list.  ``arrival_seq`` values = 0 … n-1.
        ``id`` values = shuffled permutation of 0 … n-1.

    Complexity
    ----------
    Time  : O(n)
    Space : O(n)
    """
    rng = random.Random(seed)
    severities: List[Severity] = rng.choices(
        population=list(_ALL_SEVERITIES),
        weights=list(_ER_WEIGHTS),
        k=n,
    )
    pids = list(range(n))
    rng.shuffle(pids)
    return _patients_from_severities(severities, pids=pids)


# ---------------------------------------------------------------------------
# Inversion counter — O(n log n) via modified merge sort
# ---------------------------------------------------------------------------

def count_inversions(patients: List[Patient]) -> int:
    """
    Count the number of inversions in *patients* in O(n log n) time.

    An **inversion** is a pair (i, j) with i < j and
    ``patients[i].key > patients[j].key``, i.e. the element at position i
    should appear *after* the element at position j in the sorted order.

    By the move-counting convention in ``metrics.py``, this count equals the
    exact number of *shifts* insertion sort will perform (I in the formula
    total_moves = I + (n-1)).

    Algorithm
    ---------
    Modified merge sort: whenever an element from the right sub-array is
    chosen before remaining left-sub-array elements, each of those remaining
    left elements forms an inversion with the chosen right element.
    Comparisons use ``Patient.__le__`` (delegates to ``key``).

    Parameters
    ----------
    patients : List of Patient objects.  **Not mutated.**

    Returns
    -------
    int
        Number of inversions I, where 0 ≤ I ≤ n(n-1)/2.

    Complexity
    ----------
    Time  : O(n log n)
    Space : O(n)

    Examples
    --------
    >>> count_inversions(gen_sorted(5))
    0
    >>> count_inversions(gen_reverse(5)) == 5 * 4 // 2
    True
    """
    arr = list(patients)          # copy — caller's list is never mutated
    _, inversions = _merge_sort_count(arr)
    return inversions


def _merge_sort_count(arr: List[Patient]) -> Tuple[List[Patient], int]:
    """
    Recursively sort *arr* and return ``(sorted_arr, inversion_count)``.

    Internal helper for ``count_inversions``.  Uses standard list operations
    (not Counter) because this is a measurement utility, not an algorithm
    under analysis.

    Time  : O(n log n)
    Space : O(n)
    """
    n = len(arr)
    if n <= 1:
        return arr, 0

    mid = n // 2
    left,  inv_left  = _merge_sort_count(arr[:mid])
    right, inv_right = _merge_sort_count(arr[mid:])

    merged: List[Patient] = []
    inversions = inv_left + inv_right
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:          # Patient.__le__ compares by key
            merged.append(left[i])
            i += 1
        else:
            # Every remaining element in left is > right[j] => each is an inversion.
            inversions += len(left) - i
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inversions


# ---------------------------------------------------------------------------
# GENERATORS registry
# ---------------------------------------------------------------------------

#: Mapping from short benchmark names to callables with the uniform signature
#: ``(n: int, *, seed: int = 42) -> List[Patient]``.
#: The ``nearly_sorted`` entry freezes ``k = _NEARLY_SORTED_DEFAULT_K``.
GENERATORS: Dict[str, Callable[..., List[Patient]]] = {
    "random":        gen_random,
    "sorted":        gen_sorted,
    "reverse":       gen_reverse,
    "nearly_sorted": partial(gen_nearly_sorted, k=_NEARLY_SORTED_DEFAULT_K),
    "few_unique":    gen_few_unique,
    "er_mix":        gen_er_mix,
}
