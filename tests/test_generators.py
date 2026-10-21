"""
tests.test_generators
=====================
Tests for triage.generators.

Coverage
--------
* All six generators: correct length, valid Patient objects, arrival_seq and
  id each form a permutation of {0..n-1}, ids are shuffled independently of
  arrival_seq.
* gen_sorted   : exactly 0 inversions.
* gen_reverse  : exactly n(n-1)/2 inversions for n in [0,1,2,10,100,1000]
                 with no qualifier about distinct severities.
* gen_nearly_sorted(k=0) == gen_sorted for the same seed.
* gen_nearly_sorted(k): inversions bounded by k.
* gen_few_unique: at most num_severities distinct severity values.
* gen_er_mix: weights 5/15/40/40/0 verified within ±2 % at n=10 000.
* count_inversions: sorted=0, reverse=max, non-mutation, shift-count match.
* GENERATORS dict: all six keys present, callables return correct type.
"""

from __future__ import annotations

import pytest

from triage.generators import (
    GENERATORS,
    _NEARLY_SORTED_DEFAULT_K,
    count_inversions,
    gen_er_mix,
    gen_few_unique,
    gen_nearly_sorted,
    gen_random,
    gen_reverse,
    gen_sorted,
)
from triage.models import Patient, Severity


# ---------------------------------------------------------------------------
# Shared assertion helper
# ---------------------------------------------------------------------------

def _assert_valid_list(patients: list, expected_n: int) -> None:
    """
    Assert structural invariants shared by every generator output.

    Checks
    ------
    * Correct length.
    * Every element is a ``Patient`` with a valid ``Severity``.
    * ``arrival_seq`` values form a permutation of ``{0 … n-1}``.
    * ``id`` values form a permutation of ``{0 … n-1}``.
    """
    assert len(patients) == expected_n
    for p in patients:
        assert isinstance(p, Patient)
        assert isinstance(p.severity, Severity)

    if expected_n > 0:
        arrival_seqs = sorted(p.arrival_seq for p in patients)
        assert arrival_seqs == list(range(expected_n)), (
            "arrival_seq values are not a permutation of 0…n-1"
        )
        ids = sorted(p.id for p in patients)
        assert ids == list(range(expected_n)), (
            "id values are not a permutation of 0…n-1"
        )


# ---------------------------------------------------------------------------
# gen_random
# ---------------------------------------------------------------------------

class TestGenRandom:
    def test_length(self) -> None:
        _assert_valid_list(gen_random(10), 10)

    def test_valid_patients_large(self) -> None:
        _assert_valid_list(gen_random(200), 200)

    def test_reproducible(self) -> None:
        assert gen_random(50, seed=7) == gen_random(50, seed=7)

    def test_different_seeds_differ(self) -> None:
        assert gen_random(50, seed=1) != gen_random(50, seed=2)

    def test_single_patient(self) -> None:
        _assert_valid_list(gen_random(1), 1)

    def test_all_severities_possible(self) -> None:
        patients = gen_random(500, seed=42)
        assert {p.severity for p in patients} == set(Severity)

    def test_ids_shuffled_independently_of_arrival_seq(self) -> None:
        """IDs and arrival_seqs must be independent permutations."""
        patients = gen_random(50, seed=42)
        ids          = [p.id          for p in patients]
        arrival_seqs = [p.arrival_seq for p in patients]
        # arrival_seq is always 0..n-1 in list order
        assert arrival_seqs == list(range(50))
        # ids must NOT equal position (they are shuffled)
        assert ids != list(range(50)), "ids were not shuffled"


# ---------------------------------------------------------------------------
# gen_sorted
# ---------------------------------------------------------------------------

class TestGenSorted:
    def test_length(self) -> None:
        _assert_valid_list(gen_sorted(15), 15)

    def test_zero_inversions(self) -> None:
        assert count_inversions(gen_sorted(20)) == 0

    def test_is_non_decreasing_by_key(self) -> None:
        patients = gen_sorted(20)
        for a, b in zip(patients, patients[1:]):
            assert a.key <= b.key

    def test_reproducible(self) -> None:
        assert gen_sorted(10, seed=3) == gen_sorted(10, seed=3)

    def test_ids_shuffled(self) -> None:
        patients = gen_sorted(30, seed=42)
        ids = [p.id for p in patients]
        assert ids != list(range(30))


# ---------------------------------------------------------------------------
# gen_reverse
# ---------------------------------------------------------------------------

class TestGenReverse:
    def test_length(self) -> None:
        _assert_valid_list(gen_reverse(12), 12)

    def test_is_non_increasing_by_severity(self) -> None:
        patients = gen_reverse(20)
        severities = [p.severity for p in patients]
        assert severities == sorted(severities, reverse=True)

    def test_reproducible(self) -> None:
        assert gen_reverse(20, seed=99) == gen_reverse(20, seed=99)

    def test_ids_shuffled(self) -> None:
        patients = gen_reverse(30, seed=42)
        ids = [p.id for p in patients]
        assert ids != list(range(30))

    @pytest.mark.parametrize("n", [0, 1, 2, 10, 100, 1000])
    def test_exact_max_inversions(self, n: int) -> None:
        """
        gen_reverse(n) must have exactly n*(n-1)//2 inversions for every n,
        with no qualifier about the number of distinct severity values.

        Proof sketch (in docstring of gen_reverse):
        arrival_seq[i] = n-1-i, so for any i < j the key (sev[i], n-1-i)
        is strictly greater than (sev[j], n-1-j) regardless of whether
        sev[i] == sev[j], making every pair an inversion.
        """
        assert count_inversions(gen_reverse(n)) == n * (n - 1) // 2


# ---------------------------------------------------------------------------
# gen_nearly_sorted
# ---------------------------------------------------------------------------

class TestGenNearlySorted:
    def test_length(self) -> None:
        _assert_valid_list(gen_nearly_sorted(20, k=3), 20)

    def test_k_zero_equals_sorted_same_seed(self) -> None:
        """
        With k=0 no randint calls are made, so the RNG state at the
        pid-shuffle step is identical to gen_sorted for the same seed.
        """
        seed = 42
        assert gen_nearly_sorted(20, k=0, seed=seed) == gen_sorted(20, seed=seed)

    def test_inversions_bounded_by_k(self) -> None:
        """k adjacent swaps add at most k inversions to a sorted list."""
        n, k = 30, 5
        inv = count_inversions(gen_nearly_sorted(n, k=k, seed=42))
        assert inv <= k

    def test_reproducible(self) -> None:
        assert gen_nearly_sorted(20, k=4, seed=13) == gen_nearly_sorted(20, k=4, seed=13)

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            gen_nearly_sorted(0, k=0)

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            gen_nearly_sorted(5, k=-1)

    def test_ids_shuffled(self) -> None:
        patients = gen_nearly_sorted(30, k=5, seed=42)
        ids = [p.id for p in patients]
        assert ids != list(range(30))


# ---------------------------------------------------------------------------
# gen_few_unique
# ---------------------------------------------------------------------------

class TestGenFewUnique:
    def test_length(self) -> None:
        _assert_valid_list(gen_few_unique(25, num_severities=2), 25)

    def test_at_most_num_severities_distinct(self) -> None:
        for ns in range(1, 6):
            patients = gen_few_unique(50, num_severities=ns, seed=42)
            assert len({p.severity for p in patients}) <= ns

    def test_single_severity(self) -> None:
        patients = gen_few_unique(10, num_severities=1)
        assert len({p.severity for p in patients}) == 1

    def test_reproducible(self) -> None:
        assert gen_few_unique(30, num_severities=3, seed=5) == \
               gen_few_unique(30, num_severities=3, seed=5)

    def test_invalid_num_severities_low(self) -> None:
        with pytest.raises(ValueError):
            gen_few_unique(10, num_severities=0)

    def test_invalid_num_severities_high(self) -> None:
        with pytest.raises(ValueError):
            gen_few_unique(10, num_severities=6)

    def test_ids_shuffled(self) -> None:
        patients = gen_few_unique(30, num_severities=2, seed=42)
        ids = [p.id for p in patients]
        assert ids != list(range(30))


# ---------------------------------------------------------------------------
# gen_er_mix
# ---------------------------------------------------------------------------

class TestGenErMix:
    # Expected weights: CRITICAL=5%, EMERGENT=15%, URGENT=40%, LESS_URGENT=40%, NON_URGENT=0%
    _EXPECTED = {
        Severity.CRITICAL:    0.05,
        Severity.EMERGENT:    0.15,
        Severity.URGENT:      0.40,
        Severity.LESS_URGENT: 0.40,
        Severity.NON_URGENT:  0.00,
    }

    def test_length(self) -> None:
        _assert_valid_list(gen_er_mix(100), 100)

    def test_reproducible(self) -> None:
        assert gen_er_mix(50, seed=99) == gen_er_mix(50, seed=99)

    def test_only_four_severity_levels_present(self) -> None:
        """NON_URGENT weight is 0.0 — it must never appear."""
        patients = gen_er_mix(1000, seed=42)
        assert Severity.NON_URGENT not in {p.severity for p in patients}
        assert {p.severity for p in patients} == {
            Severity.CRITICAL, Severity.EMERGENT,
            Severity.URGENT, Severity.LESS_URGENT,
        }

    def test_urgent_and_less_urgent_tied_as_top_two(self) -> None:
        """URGENT and LESS_URGENT are both at 40%; they must be the top-2."""
        from collections import Counter as PyCtr
        patients = gen_er_mix(5000, seed=42)
        counts = PyCtr(p.severity for p in patients)
        top_two = {sev for sev, _ in counts.most_common(2)}
        assert top_two == {Severity.URGENT, Severity.LESS_URGENT}

    def test_ids_shuffled(self) -> None:
        patients = gen_er_mix(30, seed=42)
        ids = [p.id for p in patients]
        assert ids != list(range(30))

    @pytest.mark.parametrize("sev,expected_frac", [
        (Severity.CRITICAL,    0.05),
        (Severity.EMERGENT,    0.15),
        (Severity.URGENT,      0.40),
        (Severity.LESS_URGENT, 0.40),
        (Severity.NON_URGENT,  0.00),
    ])
    def test_er_mix_within_2_percent_tolerance(
        self, sev: Severity, expected_frac: float
    ) -> None:
        """
        At n = 10 000 each severity's empirical fraction must lie within
        ±2 percentage points of the declared weight.
        """
        n = 10_000
        patients = gen_er_mix(n, seed=42)
        actual_frac = sum(1 for p in patients if p.severity == sev) / n
        assert abs(actual_frac - expected_frac) <= 0.02, (
            f"{sev.name}: expected {expected_frac:.0%}, got {actual_frac:.2%}"
        )


# ---------------------------------------------------------------------------
# count_inversions
# ---------------------------------------------------------------------------

class TestCountInversions:
    def test_empty(self) -> None:
        assert count_inversions([]) == 0

    def test_single_element(self) -> None:
        assert count_inversions(gen_sorted(1)) == 0

    def test_sorted_zero_inversions(self) -> None:
        assert count_inversions(gen_sorted(10)) == 0

    def test_two_elements_sorted(self) -> None:
        assert count_inversions(gen_sorted(2)) == 0

    def test_two_elements_reversed(self) -> None:
        assert count_inversions(gen_reverse(2)) == 1

    @pytest.mark.parametrize("n", [0, 1, 2, 10, 100])
    def test_reverse_equals_max_inversions(self, n: int) -> None:
        assert count_inversions(gen_reverse(n)) == n * (n - 1) // 2

    def test_does_not_mutate_input(self) -> None:
        patients = gen_random(15, seed=7)
        original_keys = [p.key for p in patients]
        count_inversions(patients)
        assert [p.key for p in patients] == original_keys

    def test_inversions_match_insertion_sort_shifts(self) -> None:
        """
        count_inversions(list) must equal the shift count a manual
        insertion sort performs on the same list's keys.
        """
        import random as _rnd
        rng = _rnd.Random(77)
        arr = [rng.randint(0, 9) for _ in range(20)]
        patients = [
            Patient(id=v, name=f"P{v}", severity=Severity(v % 5 + 1), arrival_seq=i)
            for i, v in enumerate(arr)
        ]
        expected_inversions = count_inversions(patients)

        keys = [p.key for p in patients]
        shifts = 0
        for i in range(1, len(keys)):
            key = keys[i]
            j = i - 1
            while j >= 0 and keys[j] > key:
                keys[j + 1] = keys[j]
                shifts += 1
                j -= 1
            keys[j + 1] = key

        assert shifts == expected_inversions


# ---------------------------------------------------------------------------
# GENERATORS dict
# ---------------------------------------------------------------------------

class TestGeneratorsDict:
    _EXPECTED_KEYS = {
        "random", "sorted", "reverse", "nearly_sorted", "few_unique", "er_mix"
    }

    def test_all_keys_present(self) -> None:
        assert set(GENERATORS.keys()) == self._EXPECTED_KEYS

    def test_all_values_callable(self) -> None:
        for name, fn in GENERATORS.items():
            assert callable(fn), f"GENERATORS[{name!r}] is not callable"

    @pytest.mark.parametrize("name", [
        "random", "sorted", "reverse", "nearly_sorted", "few_unique", "er_mix"
    ])
    def test_each_generator_returns_patient_list(self, name: str) -> None:
        """Each entry must accept (n, seed=42) and return a valid patient list."""
        fn = GENERATORS[name]
        n = 10
        patients = fn(n, seed=42)
        _assert_valid_list(patients, n)

    def test_nearly_sorted_uses_default_k(self) -> None:
        """The nearly_sorted entry freezes k = _NEARLY_SORTED_DEFAULT_K."""
        from functools import partial
        fn = GENERATORS["nearly_sorted"]
        assert isinstance(fn, partial)
        assert fn.keywords.get("k") == _NEARLY_SORTED_DEFAULT_K

    def test_generators_reproducible(self) -> None:
        for name, fn in GENERATORS.items():
            assert fn(20, seed=7) == fn(20, seed=7), (
                f"GENERATORS[{name!r}] not reproducible"
            )
