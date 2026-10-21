"""
triage.sorts — sorting algorithm sub-package.

Modules
-------
insertion  : Insertion sort with lo..hi subrange support.
heap_sort  : Heap sort built on triage.heap.
hybrid     : Timsort-style and introsort with configurable threshold T.
"""

from triage.sorts.heap_sort import heap_sort
from triage.sorts.hybrid import introsort, timsort_style
from triage.sorts.insertion import binary_insertion_sort, insertion_sort

__all__ = [
    "insertion_sort",
    "binary_insertion_sort",
    "heap_sort",
    "timsort_style",
    "introsort",
]
