"""Public AgentCC API; numeric construction/Replay/verification live in src."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentcc_network import AgentCCError
from src.agentcc_plan import (
    Problem,
    build_baseline,
    candidate_splits as _candidate_splits,
    estimate_plan_time_us,
    plan_from_answer,
    plan_to_answer,
    validate_plan,
)


def estimate_schedule_time_us(task: dict[str, Any], topology: dict[str, Any], answer: dict[str, Any]) -> float:
    """Local V22 Replay estimate; the official platform remains authoritative."""
    problem = Problem(task, topology, answer["split"])
    return estimate_plan_time_us(problem, plan_from_answer(problem, answer))


def build_solution(
    task: dict[str, Any],
    topology: dict[str, Any],
    preferred_split: int | None = None,
) -> dict[str, Any]:
    # V1 baseline policy stays unchanged until bounded candidate search lands.
    split = min(_candidate_splits(task, preferred_split))
    problem = Problem(task, topology, split)
    plan = build_baseline(problem)  # Verifies each round before committing it.
    return plan_to_answer(problem, plan)


def solve_input(task_input: dict[str, Any], preferred_split: int | None = None) -> dict[str, Any]:
    """Return the business response expected from BaseAgent.solve()."""
    return {"answer": build_solution(task_input["task"], task_input["topology"], preferred_split)}


def validate_answer(task: dict[str, Any], topology: dict[str, Any], answer: dict[str, Any]) -> None:
    """Validate external JSON with the same state machine used during search."""
    try:
        problem = Problem(task, topology, answer["split"])
        validate_plan(problem, plan_from_answer(problem, answer))
    except (KeyError, TypeError, AttributeError) as error:
        raise AgentCCError(f"malformed answer: {error}") from error


def validate_request(task: dict[str, Any], topology: dict[str, Any], response: dict[str, Any]) -> None:
    if response.get("status") != "success":
        raise AgentCCError("response status is not success")
    if response.get("task_id") != task["task_id"]:
        raise AgentCCError("response task_id mismatch")
    if "instance_id" in response and response["instance_id"] != task.get("instance_id"):
        raise AgentCCError("response instance_id mismatch")
    validate_answer(task, topology, response["output"]["answer"])


def solve_request(task: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any]:
    response = {
        "task_id": task["task_id"], "status": "success",
        "output": {"answer": build_solution(task, topology)},
    }
    if "instance_id" in task:
        response["instance_id"] = task["instance_id"]
    return response


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("topology")
    args = parser.parse_args()
    print(json.dumps(solve_request(load_json(args.task), load_json(args.topology)), ensure_ascii=False, indent=2))
