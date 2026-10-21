"""
web.app
=======
Flask server powering the Hospital Emergency Triage interactive web visualizer.
Exposes REST APIs connecting directly to the triage algorithms and models.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, jsonify, request, send_from_directory

from triage.generators import GENERATORS
from triage.heap import BinaryHeap
from triage.metrics import Counter
from triage.models import Patient, Severity
from triage.priority_queue import PriorityQueue
from triage.search import (
    binary_search_iterative,
    break_even_lookups,
    linear_search,
    lower_bound,
    upper_bound,
)
from triage.simulation import run_er_simulation
from triage.sorts.heap_sort import heap_sort
from triage.sorts.hybrid import introsort, timsort_style
from triage.sorts.insertion import binary_insertion_sort, insertion_sort

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="",
)

# Global in-memory priority queue for live interactive demonstration
shared_counter = Counter()
pq_demo = PriorityQueue(counter=shared_counter)
patient_id_counter = 100
arrival_seq_counter = 0


def _patient_to_dict(p: Patient) -> Dict[str, Any]:
    return {
        "id": p.id,
        "name": p.name,
        "severity": int(p.severity),
        "severity_name": p.severity.name,
        "arrival_seq": p.arrival_seq,
        "key": list(p.key),
    }


# Initialize with a few realistic starter patients
def _seed_demo_queue() -> None:
    global patient_id_counter, arrival_seq_counter, shared_counter, pq_demo
    shared_counter.reset()
    pq_demo = PriorityQueue(counter=shared_counter)
    starters = [
        ("Sarah Connor", Severity.CRITICAL),
        ("James Holden", Severity.URGENT),
        ("Ellen Ripley", Severity.EMERGENT),
        ("Arthur Dent", Severity.LESS_URGENT),
        ("Dana Scully", Severity.URGENT),
    ]
    for name, sev in starters:
        patient_id_counter += 1
        arrival_seq_counter += 1
        p = Patient(
            id=patient_id_counter,
            name=name,
            severity=sev,
            arrival_seq=arrival_seq_counter,
        )
        pq_demo.insert(p)


_seed_demo_queue()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/queue/state", methods=["GET"])
def get_queue_state():
    """Return array representation of the binary min-heap and counter stats."""
    heap_data = [_patient_to_dict(p) for p in pq_demo._heap.data]
    return jsonify({
        "heap": heap_data,
        "length": len(pq_demo),
        "is_empty": pq_demo.is_empty(),
        "peek": _patient_to_dict(pq_demo.peek()) if not pq_demo.is_empty() else None,
        "comparisons": shared_counter.comparisons,
        "moves": shared_counter.moves,
    })


@app.route("/api/queue/admit", methods=["POST"])
def admit_patient():
    """Admit a new patient into the live triage queue."""
    global patient_id_counter, arrival_seq_counter
    data = request.json or {}
    name = data.get("name", "").strip() or f"Patient_{patient_id_counter + 1}"
    sev_val = int(data.get("severity", 3))

    patient_id_counter += 1
    arrival_seq_counter += 1

    p = Patient(
        id=patient_id_counter,
        name=name,
        severity=Severity(sev_val),
        arrival_seq=arrival_seq_counter,
    )
    c_before = shared_counter.snapshot()
    pq_demo.insert(p)
    c_after = shared_counter.snapshot()

    return jsonify({
        "success": True,
        "admitted": _patient_to_dict(p),
        "delta_comparisons": c_after["comparisons"] - c_before["comparisons"],
        "delta_moves": c_after["moves"] - c_before["moves"],
        "state": get_queue_state().json,
    })


@app.route("/api/queue/extract", methods=["POST"])
def extract_patient():
    """Treat / dequeue the highest-priority patient."""
    if pq_demo.is_empty():
        return jsonify({"error": "Priority Queue is empty"}), 400

    c_before = shared_counter.snapshot()
    treated = pq_demo.extract_most_critical()
    c_after = shared_counter.snapshot()

    return jsonify({
        "success": True,
        "treated": _patient_to_dict(treated),
        "delta_comparisons": c_after["comparisons"] - c_before["comparisons"],
        "delta_moves": c_after["moves"] - c_before["moves"],
        "state": get_queue_state().json,
    })


@app.route("/api/queue/update_severity", methods=["POST"])
def update_patient_severity():
    """Dynamically modify clinical urgency in O(log n)."""
    data = request.json or {}
    patient_id = int(data.get("id"))
    new_sev = int(data.get("severity"))

    if not pq_demo.contains(patient_id):
        return jsonify({"error": f"Patient {patient_id} not in queue"}), 404

    c_before = shared_counter.snapshot()
    pq_demo.update_severity(patient_id, Severity(new_sev))
    c_after = shared_counter.snapshot()

    return jsonify({
        "success": True,
        "patient_id": patient_id,
        "new_severity": new_sev,
        "delta_comparisons": c_after["comparisons"] - c_before["comparisons"],
        "delta_moves": c_after["moves"] - c_before["moves"],
        "state": get_queue_state().json,
    })


@app.route("/api/queue/reset", methods=["POST"])
def reset_queue():
    """Reset the demo queue with initial batch."""
    _seed_demo_queue()
    return jsonify({"success": True, "state": get_queue_state().json})


@app.route("/api/sort/compare", methods=["POST"])
def compare_sorts():
    """Run comparison of all 5 sorting algorithms on a user-chosen generator and size."""
    data = request.json or {}
    generator_name = data.get("generator", "random")
    n = max(5, min(int(data.get("n", 50)), 500))  # Bound between 5 and 500 for snappy UI

    if generator_name not in GENERATORS:
        generator_name = "random"

    gen_fn = GENERATORS[generator_name]
    baseline_patients = gen_fn(n, seed=42)

    algos = {
        "Insertion Sort": insertion_sort,
        "Binary Insertion Sort": binary_insertion_sort,
        "Heap Sort": heap_sort,
        "Timsort-Style": timsort_style,
        "Introsort": introsort,
    }

    results = []
    for name, sort_fn in algos.items():
        arr = list(baseline_patients)
        counter = Counter()

        t0 = time.perf_counter()
        sort_fn(arr, counter)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Verify correctness
        is_sorted = all(arr[i].key <= arr[i + 1].key for i in range(len(arr) - 1))

        results.append({
            "name": name,
            "comparisons": counter.comparisons,
            "moves": counter.moves,
            "time_ms": round(elapsed_ms, 3),
            "is_sorted": is_sorted,
            "sample_sorted": [_patient_to_dict(p) for p in arr[:8]],
        })

    return jsonify({
        "n": n,
        "generator": generator_name,
        "results": results,
    })


@app.route("/api/search/compare", methods=["POST"])
def search_compare():
    """Compare linear search vs binary search on Patient catalog."""
    data = request.json or {}
    n = max(10, min(int(data.get("n", 64)), 1024))
    target_id = data.get("target_id")

    # Generate patient pool sorted by ID
    patients = [
        Patient(id=i * 2 + 100, name=f"Patient_{i}", severity=Severity(i % 5 + 1), arrival_seq=i)
        for i in range(n)
    ]
    if target_id is None:
        target_id = patients[n // 2].id
    else:
        target_id = int(target_id)

    c_lin = Counter()
    idx_lin = linear_search([p.id for p in patients], target_id, c_lin)

    c_bin = Counter()
    idx_bin = binary_search_iterative([p.id for p in patients], target_id, c_bin)

    c_lb = Counter()
    idx_lb = lower_bound([p.id for p in patients], target_id, c_lb)

    c_ub = Counter()
    idx_ub = upper_bound([p.id for p in patients], target_id, c_ub)

    break_even = break_even_lookups(n)

    return jsonify({
        "n": n,
        "target_id": target_id,
        "found": idx_bin != -1,
        "index": idx_bin,
        "linear_comps": c_lin.comparisons,
        "binary_comps": c_bin.comparisons,
        "lower_bound_index": idx_lb,
        "lower_bound_comps": c_lb.comparisons,
        "upper_bound_index": idx_ub,
        "break_even_queries": break_even,
        "sample_ids": [p.id for p in patients[:12]],
    })


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run discrete-event ER simulation with custom parameters."""
    data = request.json or {}
    duration = max(60, min(int(data.get("duration", 480)), 1440))
    doctors = max(1, min(int(data.get("doctors", 4)), 10))
    rate = max(0.1, min(float(data.get("arrival_rate", 0.5)), 2.0))
    aging = max(10, min(int(data.get("aging_threshold", 45)), 120))

    res_pq = run_er_simulation(
        sim_duration=duration,
        num_doctors=doctors,
        arrival_rate=rate,
        aging_threshold=aging,
        strategy="priority_queue",
        seed=42,
    )

    res_batch = run_er_simulation(
        sim_duration=duration,
        num_doctors=doctors,
        arrival_rate=rate,
        aging_threshold=aging,
        strategy="batch_sort",
        seed=42,
    )

    return jsonify({
        "priority_queue": {
            "admitted": res_pq.total_admitted,
            "treated": res_pq.total_treated,
            "throughput_hr": round(res_pq.throughput_per_hour, 2),
            "avg_wait": {k: round(v, 1) for k, v in res_pq.avg_wait_by_severity.items()},
            "max_wait": {k: round(v, 1) for k, v in res_pq.max_wait_by_severity.items()},
            "queue_len": res_pq.queue_length_over_time[-25:],
        },
        "batch_sort": {
            "admitted": res_batch.total_admitted,
            "treated": res_batch.total_treated,
            "throughput_hr": round(res_batch.throughput_per_hour, 2),
            "avg_wait": {k: round(v, 1) for k, v in res_batch.avg_wait_by_severity.items()},
            "max_wait": {k: round(v, 1) for k, v in res_batch.max_wait_by_severity.items()},
            "queue_len": res_batch.queue_length_over_time[-25:],
        },
    })


if __name__ == "__main__":
    print("Hospital Triage Queue UI Server starting at http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
