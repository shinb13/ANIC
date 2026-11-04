"""Offline CPU/response-size benchmark; excludes model, SDK and HTTP latency."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from agentcc_solver import build_solution, estimate_schedule_time_us, load_json, validate_answer


def benchmark(task, topology, repeat):
    timings = {"build_s": [], "validate_s": [], "estimate_s": [], "serialize_s": []}
    for _ in range(repeat):
        start = perf_counter()
        answer = build_solution(task, topology)
        timings["build_s"].append(perf_counter() - start)
        start = perf_counter()
        validate_answer(task, topology, answer)
        timings["validate_s"].append(perf_counter() - start)
        start = perf_counter()
        replay_us = estimate_schedule_time_us(task, topology, answer)
        timings["estimate_s"].append(perf_counter() - start)
        start = perf_counter()
        wire = json.dumps({"answer": answer}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        timings["serialize_s"].append(perf_counter() - start)
    return {
        **{key: median(values) for key, values in timings.items()},
        "split": answer["split"], "rounds": answer["round_count"],
        "response_bytes": len(wire), "local_replay_us": replay_us,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--examples", type=Path, default=Path(__file__).parent / "agentcc_zip_tmp/examples/256-rank")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    tasks = sorted(args.examples.glob("*/*/task.json"))
    if not tasks:
        parser.error("no task.json fixtures found under --examples")
    results = {}
    for path in tasks:
        key = str(path.parent.relative_to(args.examples))
        results[key] = benchmark(load_json(path), load_json(path.parent / "topology.json"), args.repeat)
    print(json.dumps(results, ensure_ascii=False, indent=2))
