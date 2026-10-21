"""
tests.test_models
=================
Tests for triage.models (Severity, Patient).

Coverage
--------
* Severity integer values
* Patient.key property
* __lt__ priority ordering: severity dominates, then FIFO via arrival_seq
* Total order axioms: irreflexivity, asymmetry, transitivity (no ties)
* __le__, __gt__, __ge__ consistency
* dataclass-generated __eq__ (all fields)
* __repr__ smoke test
"""

from __future__ import annotations

import pytest

from triage.models import Patient, Severity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_patient(
    pid: int = 1,
    name: str = "Test",
    severity: Severity = Severity.URGENT,
    arrival_seq: int = 0,
) -> Patient:
    return Patient(id=pid, name=name, severity=severity, arrival_seq=arrival_seq)


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------

class TestSeverityEnum:
    def test_critical_is_one(self) -> None:
        assert int(Severity.CRITICAL) == 1

    def test_emergent_is_two(self) -> None:
        assert int(Severity.EMERGENT) == 2

    def test_urgent_is_three(self) -> None:
        assert int(Severity.URGENT) == 3

    def test_less_urgent_is_four(self) -> None:
        assert int(Severity.LESS_URGENT) == 4

    def test_non_urgent_is_five(self) -> None:
        assert int(Severity.NON_URGENT) == 5

    def test_lower_integer_means_higher_priority(self) -> None:
        """IntEnum ordering: CRITICAL < NON_URGENT => dequeued first."""
        assert Severity.CRITICAL < Severity.NON_URGENT

    def test_all_five_members_exist(self) -> None:
        assert len(Severity) == 5


# ---------------------------------------------------------------------------
# Patient.key property
# ---------------------------------------------------------------------------

class TestPatientKey:
    def test_key_is_severity_and_arrival(self) -> None:
        p = make_patient(severity=Severity.CRITICAL, arrival_seq=7)
        assert p.key == (1, 7)

    def test_key_uses_int_value_of_severity(self) -> None:
        for sev in Severity:
            p = make_patient(severity=sev, arrival_seq=0)
            assert p.key[0] == int(sev)

    def test_key_is_tuple(self) -> None:
        p = make_patient()
        assert isinstance(p.key, tuple)
        assert len(p.key) == 2


# ---------------------------------------------------------------------------
# __lt__ — severity dominates
# ---------------------------------------------------------------------------

class TestPatientOrdering:
    def test_critical_lt_emergent(self) -> None:
        p_crit = make_patient(severity=Severity.CRITICAL, arrival_seq=100)
        p_emrg = make_patient(severity=Severity.EMERGENT, arrival_seq=1)
        # Even though CRITICAL arrived much later it still wins
        assert p_crit < p_emrg

    def test_severity_dominates_arrival_seq(self) -> None:
        p_low  = make_patient(severity=Severity.NON_URGENT, arrival_seq=0)
        p_high = make_patient(severity=Severity.CRITICAL,   arrival_seq=999)
        assert p_high < p_low

    # FIFO tie-breaking
    def test_same_severity_earlier_arrival_wins(self) -> None:
        p_early = make_patient(severity=Severity.URGENT, arrival_seq=1)
        p_late  = make_patient(severity=Severity.URGENT, arrival_seq=2)
        assert p_early < p_late

    def test_all_severities_ordered(self) -> None:
        """Full chain: CRITICAL < EMERGENT < URGENT < LESS_URGENT < NON_URGENT."""
        patients = [
            make_patient(severity=sev, arrival_seq=i)
            for i, sev in enumerate(Severity)
        ]
        for a, b in zip(patients, patients[1:]):
            assert a < b

    # Total order axioms
    def test_irreflexivity(self) -> None:
        p = make_patient(arrival_seq=5)
        assert not (p < p)

    def test_asymmetry(self) -> None:
        p1 = make_patient(severity=Severity.CRITICAL, arrival_seq=1)
        p2 = make_patient(severity=Severity.URGENT,   arrival_seq=2)
        assert p1 < p2
        assert not (p2 < p1)

    def test_transitivity(self) -> None:
        p1 = make_patient(severity=Severity.CRITICAL,   arrival_seq=1)
        p2 = make_patient(severity=Severity.URGENT,     arrival_seq=2)
        p3 = make_patient(severity=Severity.NON_URGENT, arrival_seq=3)
        assert p1 < p2
        assert p2 < p3
        assert p1 < p3

    # Companion comparisons
    def test_le(self) -> None:
        p1 = make_patient(severity=Severity.CRITICAL, arrival_seq=1)
        p2 = make_patient(severity=Severity.URGENT,   arrival_seq=2)
        assert p1 <= p2
        assert p1 <= p1  # same object

    def test_gt(self) -> None:
        p1 = make_patient(severity=Severity.CRITICAL, arrival_seq=1)
        p2 = make_patient(severity=Severity.URGENT,   arrival_seq=2)
        assert p2 > p1

    def test_ge(self) -> None:
        p1 = make_patient(severity=Severity.CRITICAL, arrival_seq=1)
        p2 = make_patient(severity=Severity.URGENT,   arrival_seq=2)
        assert p2 >= p1
        assert p1 >= p1  # same object


# ---------------------------------------------------------------------------
# __eq__ — dataclass default (all fields)
# ---------------------------------------------------------------------------

class TestPatientEquality:
    def test_equal_when_all_fields_match(self) -> None:
        p1 = Patient(id=1, name="Alice", severity=Severity.CRITICAL, arrival_seq=0)
        p2 = Patient(id=1, name="Alice", severity=Severity.CRITICAL, arrival_seq=0)
        assert p1 == p2

    def test_not_equal_if_id_differs(self) -> None:
        p1 = Patient(id=1, name="Alice", severity=Severity.CRITICAL, arrival_seq=0)
        p2 = Patient(id=2, name="Alice", severity=Severity.CRITICAL, arrival_seq=0)
        assert p1 != p2

    def test_not_equal_if_arrival_seq_differs(self) -> None:
        p1 = Patient(id=1, name="Alice", severity=Severity.URGENT, arrival_seq=0)
        p2 = Patient(id=1, name="Alice", severity=Severity.URGENT, arrival_seq=1)
        assert p1 != p2


# ---------------------------------------------------------------------------
# __repr__ smoke test
# ---------------------------------------------------------------------------

class TestPatientRepr:
    def test_repr_contains_key_info(self) -> None:
        p = Patient(id=42, name="John Doe", severity=Severity.EMERGENT, arrival_seq=3)
        r = repr(p)
        assert "42" in r
        assert "John Doe" in r
        assert "EMERGENT" in r
        assert "3" in r
