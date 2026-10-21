"""
triage.priority_queue
=====================
PriorityQueue wrapping triage.heap.BinaryHeap as a min-heap for ER triage.

Patients are ordered by ``Patient.key = (severity, arrival_seq)``:
- Lower integer severity (more critical) comes first.
- Ties are broken strictly by arrival order (FIFO).

O(log n) Severity Updates
-------------------------
Maintains an internal hash map ``_id_to_idx: Dict[int, int]`` mapping each
patient ID to their current index in the heap array. This mapping is kept
in sync with every array write via the heap's ``on_move`` callback.
Thus, locating any patient by ID takes O(1) time, and sifting after modifying
their severity takes O(log n) time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from triage.heap import BinaryHeap
from triage.metrics import Counter
from triage.models import Patient, Severity


class PriorityQueue:
    """
    Min-priority queue of Patient objects backed by a binary heap.

    Attributes
    ----------
    counter : Counter
        Shared counter tracking comparisons and element moves.
    """

    def __init__(self, counter: Optional[Counter] = None) -> None:
        self.counter = counter if counter is not None else Counter()
        self._id_to_idx: Dict[int, int] = {}
        self._heap = BinaryHeap(
            counter=self.counter,
            max_heap=False,
            on_move=self._on_heap_move,
        )

    def _on_heap_move(self, item: Any, index: int) -> None:
        if isinstance(item, Patient):
            self._id_to_idx[item.id] = index

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def insert(self, patient: Patient) -> None:
        """
        Insert a patient into the priority queue.
        Time: O(log n)
        """
        if patient.id in self._id_to_idx:
            raise ValueError(f"Patient with ID {patient.id} is already in the queue")
        self._heap.push(patient)

    def extract_most_critical(self) -> Patient:
        """
        Remove and return the patient with highest priority (lowest key).
        Time: O(log n)
        Raises IndexError if empty.
        """
        if self.is_empty():
            raise IndexError("Cannot extract from an empty PriorityQueue")
        patient: Patient = self._heap.pop()
        self._id_to_idx.pop(patient.id, None)
        return patient

    def peek(self) -> Patient:
        """
        Return the patient with highest priority without removing them.
        Time: O(1)
        Raises IndexError if empty.
        """
        if self.is_empty():
            raise IndexError("Cannot peek into an empty PriorityQueue")
        return self._heap.peek()

    def update_severity(self, patient_id: int, new_severity: Severity | int) -> None:
        """
        Update the clinical severity of a patient in the queue in O(log n) time.

        Time: O(log n)
        Raises KeyError if patient_id is not in the queue.
        """
        if patient_id not in self._id_to_idx:
            raise KeyError(f"Patient with ID {patient_id} not found in PriorityQueue")

        idx = self._id_to_idx[patient_id]
        patient: Patient = self._heap.data[idx]
        patient.severity = Severity(new_severity)
        # Restore heap invariant (may need to sift up if priority increased,
        # or sift down if priority decreased).
        new_idx = self._heap.sift_up(idx)
        self._heap.sift_down(new_idx)

    def contains(self, patient_id: int) -> bool:
        """Check if a patient ID is currently in the queue in O(1) time."""
        return patient_id in self._id_to_idx

    def to_list(self) -> List[Patient]:
        """Return a copy of the underlying heap array."""
        return list(self._heap.data)
