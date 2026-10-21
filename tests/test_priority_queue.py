"""
tests.test_priority_queue
=========================
Tests for triage.priority_queue (PriorityQueue).
"""

import random
import pytest

from triage.generators import gen_random
from triage.metrics import Counter
from triage.models import Patient, Severity
from triage.priority_queue import PriorityQueue


class TestPriorityQueue:

    def test_empty_and_error_handling(self) -> None:
        pq = PriorityQueue()
        assert len(pq) == 0
        assert pq.is_empty()
        with pytest.raises(IndexError, match="empty PriorityQueue"):
            pq.extract_most_critical()
        with pytest.raises(IndexError, match="empty PriorityQueue"):
            pq.peek()
        with pytest.raises(KeyError, match="not found"):
            pq.update_severity(999, Severity.CRITICAL)

    def test_single_element(self) -> None:
        pq = PriorityQueue()
        p = Patient(id=1, name="Alice", severity=Severity.URGENT, arrival_seq=0)
        pq.insert(p)
        assert len(pq) == 1
        assert not pq.is_empty()
        assert pq.peek() == p
        assert pq.extract_most_critical() == p
        assert pq.is_empty()

    def test_duplicate_id_raises_value_error(self) -> None:
        pq = PriorityQueue()
        p1 = Patient(id=10, name="Bob", severity=Severity.URGENT, arrival_seq=0)
        p2 = Patient(id=10, name="Bob 2", severity=Severity.CRITICAL, arrival_seq=1)
        pq.insert(p1)
        with pytest.raises(ValueError, match="already in the queue"):
            pq.insert(p2)

    def test_fifo_tie_break_by_arrival_seq(self) -> None:
        pq = PriorityQueue()
        # All patients share the exact same severity (Severity.URGENT)
        patients = [
            Patient(id=1, name="P1", severity=Severity.URGENT, arrival_seq=3),
            Patient(id=2, name="P2", severity=Severity.URGENT, arrival_seq=1),
            Patient(id=3, name="P3", severity=Severity.URGENT, arrival_seq=4),
            Patient(id=4, name="P4", severity=Severity.URGENT, arrival_seq=2),
        ]
        for p in patients:
            pq.insert(p)

        extracted = [pq.extract_most_critical() for _ in range(len(patients))]
        expected = sorted(patients, key=lambda p: p.arrival_seq)
        assert extracted == expected

    def test_update_severity_improves_priority(self) -> None:
        """Patient becomes more critical -> should move to front of queue."""
        pq = PriorityQueue()
        p1 = Patient(id=1, name="Alice", severity=Severity.NON_URGENT, arrival_seq=0)
        p2 = Patient(id=2, name="Bob", severity=Severity.URGENT, arrival_seq=1)
        pq.insert(p1)
        pq.insert(p2)

        # Before update: Bob (URGENT=3) is before Alice (NON_URGENT=5)
        assert pq.peek().id == 2

        # Alice deteriorates to CRITICAL=1
        pq.update_severity(patient_id=1, new_severity=Severity.CRITICAL)

        # Now Alice must be extracted first
        assert pq.peek().id == 1
        assert pq.extract_most_critical().id == 1
        assert pq.extract_most_critical().id == 2

    def test_update_severity_worsens_priority(self) -> None:
        """Patient becomes less critical -> should move backward."""
        pq = PriorityQueue()
        p1 = Patient(id=1, name="Alice", severity=Severity.CRITICAL, arrival_seq=0)
        p2 = Patient(id=2, name="Bob", severity=Severity.EMERGENT, arrival_seq=1)
        pq.insert(p1)
        pq.insert(p2)

        assert pq.peek().id == 1

        # Alice improves clinically to NON_URGENT=5
        pq.update_severity(patient_id=1, new_severity=Severity.NON_URGENT)

        # Bob (EMERGENT=2) is now first
        assert pq.peek().id == 2
        assert pq.extract_most_critical().id == 2
        assert pq.extract_most_critical().id == 1

    def test_random_sequence_against_sorted_list_oracle(self) -> None:
        """Fuzz testing with random inserts, extractions, and severity updates."""
        rng = random.Random(42)
        pq = PriorityQueue()
        oracle: list[Patient] = []

        patient_pool = gen_random(80, seed=100)
        pool_idx = 0

        for step in range(200):
            action = rng.choice(["insert", "insert", "extract", "update"])
            if action == "insert" and pool_idx < len(patient_pool):
                p = patient_pool[pool_idx]
                pool_idx += 1
                pq.insert(p)
                oracle.append(p)
            elif action == "extract" and not pq.is_empty():
                oracle.sort(key=lambda x: x.key)
                expected = oracle.pop(0)
                actual = pq.extract_most_critical()
                assert actual.id == expected.id
            elif action == "update" and not pq.is_empty():
                target = rng.choice(oracle)
                new_sev = rng.choice(list(Severity))
                target.severity = new_sev
                pq.update_severity(target.id, new_sev)

            # Check peek consistency
            if not pq.is_empty():
                oracle.sort(key=lambda x: x.key)
                assert pq.peek().id == oracle[0].id
                assert len(pq) == len(oracle)
