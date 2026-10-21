"""
tests.test_benchmarks
=====================
Tests for triage.benchmarks and triage.plots.
"""

from pathlib import Path
import pytest

from triage.benchmarks import (
    run_crossover_benchmarks,
    run_introsort_threshold_sweep,
    run_main_sort_benchmarks,
    run_nearly_sorted_k_sweep,
    run_simulation_benchmarks,
)
from triage.plots import generate_all_plots


class TestBenchmarkPipeline:

    def test_benchmarks_run_on_tiny_sizes(self, tmp_path: Path) -> None:
        """Verify benchmarks run cleanly and produce valid CSV data on tiny input sizes."""
        # 1. Main sort benchmarks on sizes 16, 32
        csv_sort = run_main_sort_benchmarks(sizes=[16, 32], data_dir=tmp_path)
        assert csv_sort.exists()
        assert csv_sort.stat().st_size > 0

        # 2. Threshold sweep on n=100
        csv_thresh = run_introsort_threshold_sweep(n=100, data_dir=tmp_path)
        assert csv_thresh.exists()
        assert csv_thresh.stat().st_size > 0

        # 3. Crossover benchmarks
        csv_cross = run_crossover_benchmarks(data_dir=tmp_path)
        assert csv_cross.exists()
        assert csv_cross.stat().st_size > 0

        # 4. Inversion k sweep on n=100
        csv_k = run_nearly_sorted_k_sweep(n=100, data_dir=tmp_path)
        assert csv_k.exists()
        assert csv_k.stat().st_size > 0

        # 5. Simulation results
        csv_sim = run_simulation_benchmarks(data_dir=tmp_path)
        assert csv_sim.exists()
        assert csv_sim.stat().st_size > 0

        # 6. Generate plots from test CSVs
        figs_dir = tmp_path / "figures"
        figs = generate_all_plots(data_dir=tmp_path, figures_dir=figs_dir)
        assert len(figs) == 6
        for f in figs:
            assert f.exists()
            assert f.stat().st_size > 0
