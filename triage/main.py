"""
triage.main
===========
CLI entry point and interactive demonstration.

Run with:
    python -m triage.main
"""

from __future__ import annotations

from triage.models import Patient, Severity
from triage.priority_queue import PriorityQueue
from triage.simulation import run_er_simulation


def main() -> None:
    """Execute an interactive Emergency Room triage queue demo."""
    print("=" * 70)
    print("HOSPITAL / EMERGENCY TRIAGE QUEUE - ALGORITHM DEMONSTRATION")
    print("=" * 70)

    pq = PriorityQueue()
    print("\n1. Admitting initial patients into the PriorityQueue...")
    patients = [
        Patient(id=101, name="John Doe", severity=Severity.URGENT, arrival_seq=1),
        Patient(id=102, name="Jane Smith", severity=Severity.CRITICAL, arrival_seq=2),
        Patient(id=103, name="Bob Brown", severity=Severity.LESS_URGENT, arrival_seq=3),
        Patient(id=104, name="Alice Green", severity=Severity.URGENT, arrival_seq=4),
    ]

    for p in patients:
        pq.insert(p)
        print(f"   [+] Admitted: {p.name:<12} (Severity: {p.severity.name:<11}, Arrival Seq: {p.arrival_seq})")

    print(f"\n   Current Most Critical Patient: {pq.peek().name} ({pq.peek().severity.name})")

    print("\n2. Dynamic Deterioration (O(log n) severity update):")
    print("   Bob Brown (id=103) deteriorates from LESS_URGENT (4) -> CRITICAL (1)!")
    pq.update_severity(103, Severity.CRITICAL)

    print("\n3. Serving Patients by Clinical Priority (Ties broken FIFO):")
    served_count = 1
    while not pq.is_empty():
        p = pq.extract_most_critical()
        print(f"   [{served_count}] Treating: {p.name:<12} (Severity: {p.severity.name:<11}, Arrival Seq: {p.arrival_seq})")
        served_count += 1

    print("\n4. Running 8-Hour Discrete-Event ER Simulation (Seeded)...")
    res = run_er_simulation(sim_duration=480, num_doctors=4, seed=42)
    print(f"   Total Admitted: {res.total_admitted}")
    print(f"   Total Treated : {res.total_treated}")
    print(f"   Throughput    : {res.throughput_per_hour:.2f} patients/hour")
    print("   Average Wait Times by Clinical Severity:")
    sev_names = {1: "Critical", 2: "Emergent", 3: "Urgent", 4: "Less Urgent", 5: "Non-Urgent"}
    for sev_val, wait in sorted(res.avg_wait_by_severity.items()):
        if wait > 0:
            print(f"     - {sev_names.get(sev_val, str(sev_val)):<12}: {wait:.1f} minutes")

    print("\n" + "=" * 70)
    print("Run `python -m pytest` to execute the full unit test suite.")
    print("Run `python -m triage.benchmarks` and `python -m triage.plots` to re-generate datasets.")
    print("Read `report/analysis.md` for the comprehensive academic report and viva Q&A.")
    print("=" * 70)


if __name__ == "__main__":
    main()
