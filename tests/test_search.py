"""
tests.test_search
=================
Tests for triage.search.
"""

import math
import random
import pytest

from triage.generators import gen_random
from triage.metrics import Counter
from triage.models import Patient, Severity
from triage.search import (
    PatientIndex,
    binary_search_iterative,
    binary_search_recursive,
    break_even_lookups,
    linear_search,
    lower_bound,
    upper_bound,
)


class TestSearchAlgorithms:

    def test_empty(self) -> None:
        c1, c2, c3 = Counter(), Counter(), Counter()
        assert linear_search([], 10, c1) == -1
        assert binary_search_iterative([], 10, c2) == -1
        assert binary_search_recursive([], 10, c3) == -1
        assert lower_bound([], 10, Counter()) == -1
        assert upper_bound([], 10, Counter()) == -1

    def test_single_element(self) -> None:
        arr = [42]
        c1, c2, c3 = Counter(), Counter(), Counter()
        assert linear_search(arr, 42, c1) == 0
        assert binary_search_iterative(arr, 42, c2) == 0
        assert binary_search_recursive(arr, 42, c3) == 0

        assert linear_search(arr, 99, Counter()) == -1
        assert binary_search_iterative(arr, 99, Counter()) == -1
        assert binary_search_recursive(arr, 99, Counter()) == -1

    def test_first_and_last_elements(self) -> None:
        arr = [10, 20, 30, 40, 50]
        for target, expected_idx in [(10, 0), (50, 4), (30, 2)]:
            assert linear_search(arr, target, Counter()) == expected_idx
            assert binary_search_iterative(arr, target, Counter()) == expected_idx
            assert binary_search_recursive(arr, target, Counter()) == expected_idx

    @pytest.mark.parametrize("n", [4, 8, 16, 32, 64, 128])
    def test_comparisons_bound_on_powers_of_two(self, n: int) -> None:
        """
        On array sizes that are powers of 2 (n = 2^k), lower_bound takes at most
        floor(log2 n) + 1 comparisons, and binary_search_iterative (with deferred equality)
        takes at most floor(log2 n) + 2 comparisons.
        """
        arr = list(range(0, n * 2, 2))  # [0, 2, 4, ..., 2n - 2]
        bound_lower = math.floor(math.log2(n)) + 1
        bound_bsearch = math.floor(math.log2(n)) + 2

        for target in [arr[0], arr[n // 2], arr[-1], -1, 2 * n + 5]:
            c_lb = Counter()
            lower_bound(arr, target, c_lb)
            assert c_lb.comparisons <= bound_lower, (
                f"lower_bound target {target} in n={n} took {c_lb.comparisons} > {bound_lower}"
            )

            c_bs = Counter()
            binary_search_iterative(arr, target, c_bs)
            assert c_bs.comparisons <= bound_bsearch, (
                f"binary_search target {target} in n={n} took {c_bs.comparisons} > {bound_bsearch}"
            )

    def test_agrees_with_linear_search_on_random(self) -> None:
        rng = random.Random(1234)
        n = 50
        arr = sorted(rng.sample(range(1000), n))

        # Test both existing and non-existing values
        queries = rng.sample(arr, 20) + rng.sample(range(1001, 1200), 10)
        for target in queries:
            c_lin = Counter()
            c_iter = Counter()
            c_rec = Counter()
            idx_lin = linear_search(arr, target, c_lin)
            idx_iter = binary_search_iterative(arr, target, c_iter)
            idx_rec = binary_search_recursive(arr, target, c_rec)

            assert idx_iter == idx_lin
            assert idx_rec == idx_lin

    def test_lower_and_upper_bounds(self) -> None:
        arr = [10, 20, 20, 20, 30, 40]
        # lower_bound: first >= 20 is at index 1
        assert lower_bound(arr, 20, Counter()) == 1
        # upper_bound: first > 20 is at index 4 (value 30)
        assert upper_bound(arr, 20, Counter()) == 4

        # Targets outside array range
        assert lower_bound(arr, 5, Counter()) == 0
        assert upper_bound(arr, 5, Counter()) == 0
        assert lower_bound(arr, 50, Counter()) == -1
        assert upper_bound(arr, 50, Counter()) == -1


class TestPatientIndex:

    def test_patient_index_operations(self) -> None:
        patients = [
            Patient(id=105, name="E", severity=Severity.URGENT, arrival_seq=0),
            Patient(id=101, name="A", severity=Severity.CRITICAL, arrival_seq=1),
            Patient(id=103, name="C", severity=Severity.NON_URGENT, arrival_seq=2),
        ]
        catalog = PatientIndex()
        catalog.build(patients)

        assert len(catalog) == 3
        # Lookup existing
        p = catalog.lookup(101)
        assert p is not None
        assert p.name == "A"

        # Lookup non-existent
        assert catalog.lookup(999) is None

        # Insert new patient with id=102
        new_p = Patient(id=102, name="B", severity=Severity.EMERGENT, arrival_seq=3)
        catalog.insert(new_p)
        assert len(catalog) == 4
        assert catalog.lookup(102) == new_p

        # Check catalog remains sorted by id
        ids = [p.id for p in catalog._data]
        assert ids == sorted(ids)


class TestBreakEvenAnalysis:

    def test_break_even_lookups_monotonicity(self) -> None:
        """As n grows, m ≈ 2 * log2(n) grows smoothly."""
        be16 = break_even_lookups(16)
        be64 = break_even_lookups(64)
        be256 = break_even_lookups(256)
        be1024 = break_even_lookups(1024)

        assert be16 < be64 < be256 < be1024
        # At n = 1024: 2 * log2(1024) = 20.
        assert 15 <= be1024 <= 30
