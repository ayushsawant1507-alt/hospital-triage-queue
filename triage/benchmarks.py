"""
triage.benchmarks
=================
Comprehensive benchmarking suite for sorting algorithms and ER triage queues.

Generates CSV datasets stored in ``report/data/``:
1. ``sort_benchmarks.csv``: Wall time (median of 5 runs after warmup), comparisons,
   and moves across all 6 input generators for n in [16..16384].
2. ``introsort_threshold_sweep.csv``: Threshold sweep T in [4..64] at n=10000.
3. ``crossover.csv``: Empirical crossover point between Insertion Sort and Heap Sort.
4. ``nearly_sorted_k_sweep.csv``: Verification of shift count == inversions across k.
5. ``simulation_results.csv``: Real-time PriorityQueue vs Batch-Sort ER outcomes.
"""

from __future__ import annotations

import csv
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Dict, List, Tuple

from triage.generators import GENERATORS, count_inversions, gen_nearly_sorted
from triage.metrics import Counter
from triage.simulation import run_er_simulation
from triage.sorts.heap_sort import heap_sort
from triage.sorts.hybrid import introsort, timsort_style
from triage.sorts.insertion import insertion_sort


DATA_DIR = Path("report/data")


def _benchmark_algorithm(
    sort_fn: Callable[[List[Any], Counter], None],
    data_gen: Callable[[], List[Any]],
    repeats: int = 5,
) -> Tuple[float, int, int]:
    """
    Measure median wall-clock time over `repeats` runs after a warmup pass,
    plus exact deterministic comparisons and moves from Counter.
    """
    # Warmup run
    warmup_arr = data_gen()
    sort_fn(warmup_arr, Counter())

    # Deterministic metrics run
    meas_arr = data_gen()
    counter = Counter()
    sort_fn(meas_arr, counter)
    comps = counter.comparisons
    moves = counter.moves

    # Timing runs (without counter overhead for accurate wall time)
    times: List[float] = []
    dummy_c = Counter()
    for _ in range(repeats):
        arr = data_gen()
        dummy_c.reset()
        t0 = time.perf_counter()
        sort_fn(arr, dummy_c)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return statistics.median(times), comps, moves


def run_main_sort_benchmarks(sizes: List[int], data_dir: Path = DATA_DIR) -> Path:
    """Run main sorting benchmarks across all generators and sizes."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "sort_benchmarks.csv"

    fieldnames = ["n", "generator", "algorithm", "median_time_sec", "comparisons", "moves"]
    records = []

    algorithms = {
        "insertion": insertion_sort,
        "heap_sort": heap_sort,
        "timsort_style": timsort_style,
        "introsort": introsort,
    }

    for n in sizes:
        print(f"Benchmarking n = {n}...")
        for gen_name, gen_func in GENERATORS.items():
            # Python sorted() reference
            times_py = []
            for _ in range(5):
                arr = gen_func(n, seed=42)
                t0 = time.perf_counter()
                _ = sorted(arr, key=lambda p: p.key)
                t1 = time.perf_counter()
                times_py.append(t1 - t0)
            records.append({
                "n": n,
                "generator": gen_name,
                "algorithm": "python_sorted",
                "median_time_sec": statistics.median(times_py),
                "comparisons": 0,
                "moves": 0,
            })

            for algo_name, algo_fn in algorithms.items():
                # Skip insertion sort on random/reverse for n > 4096 (O(n^2) timeout)
                if algo_name == "insertion" and n > 4096 and gen_name in ("random", "reverse"):
                    continue

                med_t, comps, moves = _benchmark_algorithm(
                    algo_fn,
                    lambda: gen_func(n, seed=42),
                    repeats=5,
                )
                records.append({
                    "n": n,
                    "generator": gen_name,
                    "algorithm": algo_name,
                    "median_time_sec": med_t,
                    "comparisons": comps,
                    "moves": moves,
                })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return out_file


def run_introsort_threshold_sweep(n: int = 10000, data_dir: Path = DATA_DIR) -> Path:
    """Sweep Introsort threshold T = 4..64 at n = 10000."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "introsort_threshold_sweep.csv"
    thresholds = [4, 8, 12, 16, 24, 32, 48, 64]
    records = []

    for T in thresholds:
        med_t, comps, moves = _benchmark_algorithm(
            lambda arr, c: introsort(arr, c, threshold=T),
            lambda: GENERATORS["random"](n, seed=123),
            repeats=5,
        )
        records.append({
            "threshold": T,
            "median_time_sec": med_t,
            "comparisons": comps,
            "moves": moves,
        })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "median_time_sec", "comparisons", "moves"])
        writer.writeheader()
        writer.writerows(records)

    return out_file


def run_crossover_benchmarks(data_dir: Path = DATA_DIR) -> Path:
    """Evaluate crossover point between Insertion Sort and Heap Sort."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "crossover.csv"
    sizes = [8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 200, 256, 384, 512]
    records = []

    for n in sizes:
        for input_type in ["random", "nearly_sorted"]:
            gen = GENERATORS[input_type]
            t_ins, c_ins, m_ins = _benchmark_algorithm(
                insertion_sort, lambda: gen(n, seed=99), repeats=5
            )
            t_heap, c_heap, m_heap = _benchmark_algorithm(
                heap_sort, lambda: gen(n, seed=99), repeats=5
            )
            records.append({
                "n": n,
                "input_type": input_type,
                "insertion_time": t_ins,
                "heap_time": t_heap,
                "insertion_comps": c_ins,
                "heap_comps": c_heap,
                "insertion_moves": m_ins,
                "heap_moves": m_heap,
            })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n", "input_type", "insertion_time", "heap_time",
                "insertion_comps", "heap_comps", "insertion_moves", "heap_moves"
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    return out_file


def run_nearly_sorted_k_sweep(n: int = 1000, data_dir: Path = DATA_DIR) -> Path:
    """Verify insertion shifts == inversion count across various k."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "nearly_sorted_k_sweep.csv"
    k_values = [1, 2, 5, 10, 20, 50, 100, 200]
    records = []

    for k in k_values:
        arr = gen_nearly_sorted(n, k=k, seed=77)
        inversions = count_inversions(arr)

        c = Counter()
        insertion_sort(arr, c)
        shifts = c.moves - (n - 1)

        records.append({
            "n": n,
            "k": k,
            "inversions": inversions,
            "insertion_shifts": shifts,
            "insertion_total_moves": c.moves,
        })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n", "k", "inversions", "insertion_shifts", "insertion_total_moves"],
        )
        writer.writeheader()
        writer.writerows(records)

    return out_file


def run_simulation_benchmarks(data_dir: Path = DATA_DIR) -> Path:
    """Run ER simulation comparing PriorityQueue vs Batch-Sort."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "simulation_results.csv"

    res_pq = run_er_simulation(sim_duration=480, strategy="priority_queue", seed=42)
    res_batch = run_er_simulation(sim_duration=480, strategy="batch_sort", seed=42)

    records = []
    for res in [res_pq, res_batch]:
        for sev_val in sorted(res.avg_wait_by_severity.keys()):
            records.append({
                "strategy": res.strategy,
                "severity": sev_val,
                "avg_wait_min": round(res.avg_wait_by_severity[sev_val], 2),
                "max_wait_min": round(res.max_wait_by_severity[sev_val], 2),
                "total_treated": res.total_treated,
                "throughput_per_hour": round(res.throughput_per_hour, 2),
            })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy", "severity", "avg_wait_min", "max_wait_min",
                "total_treated", "throughput_per_hour"
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    return out_file


def run_all_benchmarks(quick: bool = False) -> List[Path]:
    """Execute all benchmarks and return list of created CSV files."""
    sizes = [16, 32, 64, 128, 256, 512, 1024] if quick else [16, 32, 64, 128, 256, 512, 1024, 4096, 16384]
    n_intro = 1000 if quick else 10000

    created = [
        run_main_sort_benchmarks(sizes),
        run_introsort_threshold_sweep(n=n_intro),
        run_crossover_benchmarks(),
        run_nearly_sorted_k_sweep(),
        run_simulation_benchmarks(),
    ]
    return created


if __name__ == "__main__":
    print("Running triage benchmarks...")
    files = run_all_benchmarks(quick=False)
    for p in files:
        print(f"Created: {p}")
