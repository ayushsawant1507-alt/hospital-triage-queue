"""
triage.simulation
=================
Discrete-event Emergency Room (ER) simulation with aging and two queueing strategies:
1. PriorityQueue (real-time min-heap with O(log n) dynamic severity promotions).
2. Batch-sort (waiting list periodically sorted with timsort_style / introsort).

Clinical Urgency & Aging Rule
-----------------------------
To prevent starvation of less-critical patients (e.g. LESS_URGENT waiting indefinitely
due to steady arrivals of CRITICAL cases), an aging rule is applied:
If a patient waits longer than ``aging_threshold`` minutes, their clinical severity
level is promoted by one level (e.g. URGENT -> EMERGENT) up to CRITICAL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Literal, Optional, Tuple

from triage.metrics import Counter
from triage.models import Patient, Severity
from triage.priority_queue import PriorityQueue
from triage.sorts.hybrid import timsort_style


@dataclass
class SimulationResult:
    """Telemetry and outcome data from an ER simulation run."""
    strategy: str
    sim_duration: int
    num_doctors: int
    total_admitted: int
    total_treated: int
    throughput_per_hour: float
    avg_wait_by_severity: Dict[int, float]
    max_wait_by_severity: Dict[int, float]
    queue_length_over_time: List[Tuple[int, int]]
    service_log: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # service_log records: (patient_id, original_severity, arrival_time, service_start_time)


def _sample_poisson(lam: float, rng: random.Random) -> int:
    """Knuth's algorithm for generating Poisson-distributed random integers."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def _sample_er_severity(rng: random.Random) -> Severity:
    """Sample severity according to ER mix (5% CRIT, 15% EMERG, 40% URG, 40% LESS_URG)."""
    r = rng.random()
    if r < 0.05:
        return Severity.CRITICAL
    elif r < 0.20:
        return Severity.EMERGENT
    elif r < 0.60:
        return Severity.URGENT
    else:
        return Severity.LESS_URGENT


def run_er_simulation(
    sim_duration: int = 480,  # 8 hours in minutes
    num_doctors: int = 4,
    arrival_rate: float = 0.5,  # Average 0.5 patients per minute (30/hour)
    strategy: Literal["priority_queue", "batch_sort"] = "priority_queue",
    batch_interval: int = 15,  # Used if strategy == "batch_sort"
    aging_threshold: int = 45,  # Wait time before priority promotion
    seed: int = 42,
) -> SimulationResult:
    """
    Run a deterministic discrete-event ER simulation.

    Parameters
    ----------
    sim_duration     : Total simulation duration in minutes.
    num_doctors      : Number of available doctors.
    arrival_rate     : Mean arrivals per minute (Poisson lambda).
    strategy         : 'priority_queue' or 'batch_sort'.
    batch_interval   : Minutes between batch sorts when using 'batch_sort'.
    aging_threshold  : Wait duration after which a patient is promoted.
    seed             : Random seed ensuring reproducibility.
    """
    rng = random.Random(seed)

    # Doctor availability: tracks minute each doctor becomes free
    doctor_free_at = [0] * num_doctors

    # Treatment duration by severity (minutes)
    treatment_durations = {
        Severity.CRITICAL: 40,
        Severity.EMERGENT: 25,
        Severity.URGENT: 15,
        Severity.LESS_URGENT: 10,
        Severity.NON_URGENT: 10,
    }

    # Tracking state
    patient_seq = 0
    total_admitted = 0
    waiting_pq = PriorityQueue(Counter()) if strategy == "priority_queue" else None
    waiting_list: List[Patient] = []

    # Map tracking original arrival and severity for all patients
    patient_meta: Dict[int, Tuple[Severity, int]] = {}
    wait_times_by_sev: Dict[int, List[int]] = {s.value: [] for s in Severity}
    queue_len_log: List[Tuple[int, int]] = []
    service_log: List[Tuple[int, int, int, int]] = []

    for t in range(sim_duration):
        # 1. New Arrivals at minute t
        arrivals_count = _sample_poisson(arrival_rate, rng)
        for _ in range(arrivals_count):
            patient_seq += 1
            total_admitted += 1
            sev = _sample_er_severity(rng)
            patient = Patient(
                id=patient_seq,
                name=f"Patient_{patient_seq}",
                severity=sev,
                arrival_seq=patient_seq,
            )
            patient_meta[patient.id] = (sev, t)

            if strategy == "priority_queue":
                waiting_pq.insert(patient)
            else:
                waiting_list.append(patient)

        # 2. Aging Rule: Check for starvation and promote
        if strategy == "priority_queue":
            # For patients in queue, check if wait exceeds aging_threshold
            for p in waiting_pq.to_list():
                orig_sev, arr_time = patient_meta[p.id]
                wait = t - arr_time
                if wait >= aging_threshold and p.severity > Severity.CRITICAL:
                    # Promote one level (e.g. 3 -> 2)
                    promoted = Severity(p.severity.value - 1)
                    waiting_pq.update_severity(p.id, promoted)
        else:
            for p in waiting_list:
                orig_sev, arr_time = patient_meta[p.id]
                wait = t - arr_time
                if wait >= aging_threshold and p.severity > Severity.CRITICAL:
                    p.severity = Severity(p.severity.value - 1)

            # Periodic batch sort
            if t % batch_interval == 0 and len(waiting_list) > 1:
                timsort_style(waiting_list, Counter())

        # 3. Doctor Assignments
        for doc_id in range(num_doctors):
            if doctor_free_at[doc_id] <= t:
                # Doctor is free, treat next patient
                next_patient: Optional[Patient] = None
                if strategy == "priority_queue" and not waiting_pq.is_empty():
                    next_patient = waiting_pq.extract_most_critical()
                elif strategy == "batch_sort" and waiting_list:
                    # Waiting list: pick highest priority (sort if needed or pick min)
                    if len(waiting_list) > 1:
                        timsort_style(waiting_list, Counter())
                    next_patient = waiting_list.pop(0)

                if next_patient is not None:
                    orig_sev, arr_time = patient_meta[next_patient.id]
                    wait_time = t - arr_time
                    wait_times_by_sev[orig_sev.value].append(wait_time)
                    service_log.append((next_patient.id, orig_sev.value, arr_time, t))

                    dur = treatment_durations[orig_sev]
                    doctor_free_at[doc_id] = t + dur

        # Record queue length
        q_len = len(waiting_pq) if strategy == "priority_queue" else len(waiting_list)
        if t % 5 == 0:
            queue_len_log.append((t, q_len))

    # Compute summaries
    total_treated = sum(len(w) for w in wait_times_by_sev.values())
    throughput_per_hr = (total_treated / (sim_duration / 60.0)) if sim_duration > 0 else 0.0

    avg_wait = {
        s: (sum(w) / len(w) if w else 0.0)
        for s, w in wait_times_by_sev.items()
    }
    max_wait = {
        s: (max(w) if w else 0.0)
        for s, w in wait_times_by_sev.items()
    }

    return SimulationResult(
        strategy=strategy,
        sim_duration=sim_duration,
        num_doctors=num_doctors,
        total_admitted=total_admitted,
        total_treated=total_treated,
        throughput_per_hour=throughput_per_hr,
        avg_wait_by_severity=avg_wait,
        max_wait_by_severity=max_wait,
        queue_length_over_time=queue_len_log,
        service_log=service_log,
    )
