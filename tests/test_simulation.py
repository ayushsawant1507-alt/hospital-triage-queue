"""
tests.test_simulation
=====================
Tests for triage.simulation (ER discrete-event simulation).
"""

import pytest

from triage.models import Severity
from triage.simulation import run_er_simulation


class TestSimulation:

    def test_simulation_is_deterministic_for_fixed_seed(self) -> None:
        res1 = run_er_simulation(sim_duration=120, seed=12345)
        res2 = run_er_simulation(sim_duration=120, seed=12345)

        assert res1.total_admitted == res2.total_admitted
        assert res1.total_treated == res2.total_treated
        assert res1.throughput_per_hour == res2.throughput_per_hour
        assert res1.avg_wait_by_severity == res2.avg_wait_by_severity
        assert res1.service_log == res2.service_log

    def test_both_strategies_run_successfully(self) -> None:
        res_pq = run_er_simulation(sim_duration=90, strategy="priority_queue", seed=42)
        res_batch = run_er_simulation(sim_duration=90, strategy="batch_sort", seed=42)

        assert res_pq.total_admitted > 0
        assert res_pq.total_treated > 0
        assert res_batch.total_admitted > 0
        assert res_batch.total_treated > 0

    def test_higher_priority_generally_served_faster_without_aging(self) -> None:
        """
        With high aging threshold (aging disabled), critical patients must
        have lower average wait times than less urgent patients.
        """
        res = run_er_simulation(
            sim_duration=300,
            num_doctors=3,
            arrival_rate=0.4,
            aging_threshold=99999,  # Effectively disable aging
            seed=777,
        )
        avg_wait = res.avg_wait_by_severity
        # Severity 1 (Critical) vs Severity 4 (Less urgent)
        if avg_wait[1] > 0 and avg_wait[4] > 0:
            assert avg_wait[1] < avg_wait[4], (
                f"Critical wait ({avg_wait[1]}) not less than less-urgent ({avg_wait[4]})"
            )
