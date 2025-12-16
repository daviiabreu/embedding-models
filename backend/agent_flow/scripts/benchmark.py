#!/usr/bin/env python3
"""Benchmark current system performance.

Note: This benchmark is currently disabled due to agent initialization errors.
Once Phase 1 fixes are applied, this benchmark can be re-enabled.
"""

from pathlib import Path


def create_mock_baseline() -> dict:
    """Create a mock baseline for now since the agent system has errors."""
    results = {
        "queries": [
            "oi",
            "Quais são os cursos do Inteli?",
            "Me conte sobre as bolsas de estudo",
            "Como funciona o processo de admissão?",
        ],
        "iterations": 10,
        "latencies": [],
        "note": "Mock baseline created - agent system needs Phase 1 fixes before real benchmarking",
        "mean_latency": 0.0,
        "p50_latency": 0.0,
        "p95_latency": 0.0,
        "p99_latency": 0.0,
    }
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE PERFORMANCE BENCHMARK")
    print("=" * 60)
    print("\nNOTE: Agent system has initialization errors.")
    print("Creating mock baseline for Phase 0 completion.")
    print("Real benchmarking will be done after Phase 1 fixes.\n")

    baseline = create_mock_baseline()

    print("=" * 60)
    print("BASELINE METRICS (MOCK)")
    print("=" * 60)
    print(f"Mean Latency: {baseline['mean_latency']:.2f}s (pending)")
    print(f"P50 Latency:  {baseline['p50_latency']:.2f}s (pending)")
    print(f"P95 Latency:  {baseline['p95_latency']:.2f}s (pending)")
    print(f"P99 Latency:  {baseline['p99_latency']:.2f}s (pending)")
    print("Status: Awaiting Phase 1 fixes")
    print("=" * 60)

    # Save to file
    import json

    output_file = Path(__file__).parent.parent / "baseline_metrics.json"
    with open(output_file, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nBaseline (mock) saved to {output_file}")
    print("\nNext steps:")
    print("1. Complete Phase 1 critical fixes")
    print("2. Re-run this benchmark with fixed agent system")
    print("3. Establish real performance baselines")
