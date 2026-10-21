"""
tests.test_metrics
==================
Tests for triage.metrics (Counter).

Coverage
--------
* Default initial state (comparisons=0, moves=0)
* reset() zeroes both fields
* compare(a, b): increments comparisons, returns correct bool
* move(): increments moves
* snapshot(): returns correct dict, independent of later mutations
* __repr__ smoke test
* Counter works with Patient objects (exercises real use-case)
"""

from __future__ import annotations

import pytest

from triage.metrics import Counter
from triage.models import Patient, Severity


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestCounterInit:
    def test_comparisons_start_at_zero(self) -> None:
        c = Counter()
        assert c.comparisons == 0

    def test_moves_start_at_zero(self) -> None:
        c = Counter()
        assert c.moves == 0

    def test_explicit_construction(self) -> None:
        c = Counter(comparisons=10, moves=5)
        assert c.comparisons == 10
        assert c.moves == 5


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_comparisons(self) -> None:
        c = Counter(comparisons=99)
        c.reset()
        assert c.comparisons == 0

    def test_reset_clears_moves(self) -> None:
        c = Counter(moves=42)
        c.reset()
        assert c.moves == 0

    def test_reset_both_fields(self) -> None:
        c = Counter(comparisons=7, moves=3)
        c.reset()
        assert c.comparisons == 0
        assert c.moves == 0

    def test_reset_is_idempotent(self) -> None:
        c = Counter()
        c.reset()
        c.reset()
        assert c.comparisons == 0
        assert c.moves == 0


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_returns_true_when_a_lt_b(self) -> None:
        c = Counter()
        assert c.compare(1, 2) is True

    def test_compare_returns_false_when_a_gt_b(self) -> None:
        c = Counter()
        assert c.compare(5, 3) is False

    def test_compare_returns_false_when_equal(self) -> None:
        c = Counter()
        assert c.compare(4, 4) is False

    def test_compare_increments_by_one(self) -> None:
        c = Counter()
        c.compare(1, 2)
        assert c.comparisons == 1

    def test_compare_accumulates(self) -> None:
        c = Counter()
        for _ in range(10):
            c.compare(0, 1)
        assert c.comparisons == 10

    def test_compare_does_not_affect_moves(self) -> None:
        c = Counter()
        c.compare(1, 2)
        assert c.moves == 0

    def test_compare_with_strings(self) -> None:
        c = Counter()
        assert c.compare("apple", "banana") is True
        assert c.comparisons == 1

    def test_compare_with_patients(self) -> None:
        """Exercises the real use-case: comparing Patient objects."""
        c = Counter()
        p1 = Patient(id=1, name="A", severity=Severity.CRITICAL,   arrival_seq=0)
        p2 = Patient(id=2, name="B", severity=Severity.NON_URGENT, arrival_seq=1)
        result = c.compare(p1, p2)
        assert result is True     # CRITICAL < NON_URGENT
        assert c.comparisons == 1


# ---------------------------------------------------------------------------
# move()
# ---------------------------------------------------------------------------

class TestMove:
    def test_move_increments_by_one(self) -> None:
        c = Counter()
        c.move()
        assert c.moves == 1

    def test_move_accumulates(self) -> None:
        c = Counter()
        for _ in range(7):
            c.move()
        assert c.moves == 7

    def test_move_does_not_affect_comparisons(self) -> None:
        c = Counter()
        c.move()
        assert c.comparisons == 0

    def test_compare_and_move_are_independent(self) -> None:
        c = Counter()
        c.compare(1, 2)
        c.move()
        c.move()
        assert c.comparisons == 1
        assert c.moves == 2


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_returns_dict(self) -> None:
        c = Counter()
        assert isinstance(c.snapshot(), dict)

    def test_snapshot_correct_values(self) -> None:
        c = Counter(comparisons=7, moves=3)
        assert c.snapshot() == {"comparisons": 7, "moves": 3}

    def test_snapshot_is_copy_not_reference(self) -> None:
        """Mutating counter after snapshot must not change the snapshot."""
        c = Counter(comparisons=5, moves=2)
        snap = c.snapshot()
        c.comparisons = 999
        c.moves = 999
        assert snap["comparisons"] == 5
        assert snap["moves"] == 2

    def test_snapshot_keys(self) -> None:
        c = Counter()
        assert set(c.snapshot().keys()) == {"comparisons", "moves"}

    def test_snapshot_after_reset(self) -> None:
        c = Counter(comparisons=10, moves=10)
        c.reset()
        assert c.snapshot() == {"comparisons": 0, "moves": 0}


# ---------------------------------------------------------------------------
# __repr__ smoke test
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_counts(self) -> None:
        c = Counter(comparisons=3, moves=7)
        r = repr(c)
        assert "3" in r
        assert "7" in r
