"""Send the 14 public example instances to a locally running agent container.

Usage (after starting the container, see README "Local test"):
    python local_test.py            # all 14 cases
    python local_test.py mesh2d     # only families whose name contains "mesh2d"

For every case it prints the status, local validation result, estimated
communication time, HTTP time, response size, model calls and the strategy
the agent selected.  This is a local check only; the official Replay score
comes from the platform.
"""

import json
import sys
import time
from pathlib import Path

import httpx

from agentcc_solver import estimate_schedule_time_us, load_json, validate_request

BASE = Path(__file__).parent / "agentcc_zip_tmp" / "examples" / "256-rank"
FAMILIES = ["uneven_pods", "directed_ring", "sparse_fabric", "cloudmatrix", "fullmesh_clos", "mesh2d", "nvlink_like"]
URL = "http://127.0.0.1:18080"


def main() -> int:
    selected = [family for family in FAMILIES if len(sys.argv) < 2 or sys.argv[1] in family]
    try:
        print("health:", httpx.get(f"{URL}/health", timeout=5).json())
    except httpx.HTTPError:
        print(f"Agent is not reachable at {URL}. Start the container first.")
        return 1
    failures = 0
    with httpx.Client(timeout=400) as client:
        for family in selected:
            for short in ("ag", "ar"):
                task = load_json(BASE / family / short / "task.json")
                topology = load_json(BASE / family / short / "topology.json")
                body = {
                    "task_id": task["task_id"],
                    "input": {"challenge": "agentcc", "instance_id": task["instance_id"], "task": task, "topology": topology},
                    "constraints": {"timeout_ms": 360000},
                }
                started = time.perf_counter()
                response = client.post(f"{URL}/solve", json=body)
                seconds = time.perf_counter() - started
                data = response.json()
                verdict, estimate = "valid", None
                try:
                    validate_request(task, topology, data)
                    estimate = estimate_schedule_time_us(task, topology, data["output"]["answer"])
                except Exception as exc:  # report every failure kind
                    verdict = f"INVALID ({exc})"
                    failures += 1
                answer = data.get("output", {}).get("answer")
                strategy = answer.get("strategy") if isinstance(answer, dict) else data.get("output")
                meta = data.get("metadata", {})
                print(
                    f"{family:14s}{short}  {data.get('status'):8s} {verdict:6s} "
                    f"est={estimate and round(estimate)}us  time={seconds:.1f}s  "
                    f"size={len(response.content) / 1e6:.1f}MB  llm_calls={meta.get('llm_calls')}  "
                    f"tokens={meta.get('total_tokens')}\n    {strategy}",
                    flush=True,
                )
    print("all valid" if not failures else f"{failures} invalid")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
