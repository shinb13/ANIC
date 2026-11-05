import asyncio
import hashlib
import json
import logging
import os
from time import perf_counter
from typing import Any

import openai
from agenthack import BaseAgent, state, turn

from agentcc_solver import (
    AgentCCError,
    BALANCE_MAX_PATH_SEARCHES,
    ROUTE_POLICIES,
    SCHEDULE_ALGORITHMS,
    _candidate_splits,
    _routes,
    _topology_network,
    balance_schedule_routes,
    build_solution,
    doubling_round_profiles,
    estimate_schedule_time_us,
    optimal_doubling_order,
    solve_input,
    validate_answer,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_BUDGET_SECONDS = 2.0
DEFAULT_SOLVE_BUDGET_SECONDS = 25.0
RETURN_RESERVE_SECONDS = 0.25
ROUTE_BALANCE_BUDGET_SECONDS = 0.25


def _configured_seconds(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _plan_prompt(
    task: dict[str, Any],
    topology: dict[str, Any],
    baseline: dict[str, Any],
    baseline_time_us: float,
    profiles: list[dict[str, Any]],
    dimension_order: list[int] | None,
    allowed_algorithms: list[str],
    allowed_policies: list[str],
    network: tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], set[str]],
        dict[str, float],
        dict[str, list[str]],
    ],
) -> str:
    edge_capacity, _, resource_capacity, _ = network
    rank_count = len(task["ranks"])
    data_size = int(task["data_size"]["per_rank_bytes"])
    candidates = [{
        "algorithm": "recursive_doubling",
        "round_count": baseline["round_count"],
        "estimate_us": round(baseline_time_us, 3),
    }]
    if "ordered_doubling" in allowed_algorithms:
        by_step = {profile["step"]: profile for profile in profiles}
        candidates.append({
            "algorithm": "ordered_doubling",
            "round_count": baseline["round_count"],
            "dimension_order": dimension_order,
            "estimate_us": round(sum(
                by_step[step]["times_us"][position]
                for position, step in enumerate(dimension_order)
            ), 3),
        })
    if "tree" in allowed_algorithms:
        candidates.append({
            "algorithm": "tree",
            "round_count": 2 * baseline["round_count"],
            "total_rank_transfer_bytes": 2 * (rank_count - 1) * data_size,
            "recursive_doubling_total_rank_transfer_bytes":
                rank_count * baseline["round_count"] * data_size,
        })
    prompt_input = {
        "task": {
            "operator": task["operator"],
            "ranks": task["ranks"],
            "data_size": task["data_size"],
            "split_policy": task["split_policy"],
            "selected_split": baseline["split"],
            "max_nonempty_rounds": task["max_nonempty_rounds"],
            "allow_copy": task.get("allow_copy", True),
            "allow_detour": task.get("allow_detour", True),
            "communication_constraints": task.get("communication_constraints", {}),
        },
        "topology": {
            "directed_edge_count": len(edge_capacity),
            "communication_startup_time_us": topology.get("communication_startup_time_us", 0),
            "switch_processing_overhead_us": topology.get("switch_processing_overhead_us", 0),
            "resource_capacities_bytes_per_second": resource_capacity,
        },
        "candidates_on_baseline_routes": candidates,
        "route_policy_resource_penalties": ROUTE_POLICIES,
        "allowed_route_policies": allowed_policies,
        "round_load_balancing": {
            "policy": "load_balanced",
            "seed_policy": "baseline",
            "time_budget_ms": ROUTE_BALANCE_BUDGET_SECONDS * 1000,
            "max_path_searches": BALANCE_MAX_PATH_SEARCHES,
            "acceptance": "no round estimate increases; ties require lower squared utilization",
        },
        "current_dimension_evidence": [
            {
                **profile,
                "times_us": [round(value, 3) for value in profile["times_us"]],
            }
            for profile in profiles
        ],
    }
    return (
        "Choose a scheduling algorithm and route policy for this actual AgentCC input. "
        "Minimize communication time within the task constraints. The baseline is validated; "
        "only your chosen extra candidate will be constructed and checked. "
        "ordered_doubling optimizes the XOR dimension order using the shown round costs "
        "(position k sends D*2^k bytes). tree performs reduction then broadcast, reducing "
        "total traffic but adding rounds; it is useful when shared-resource congestion dominates. "
        "Edge evidence is directed; resource member examples are actual used edges, "
        "not an exhaustive graph. Route policies multiply the shared-resource path penalty; "
        "bandwidth_first and resource_first recompute static paths. load_balanced starts "
        "from baseline paths and reroutes selected transfers using actual concurrent edge "
        "and shared-resource loads, subject to the shown time and search budgets. It checks "
        "the full Replay estimate before accepting each move and never increases a round's "
        "estimate; it can help when static paths concentrate concurrent traffic. "
        "Return only JSON with exactly algorithm and policy, "
        'for example {"algorithm":"ordered_doubling","policy":"baseline"}. '
        "Choose an algorithm from candidates_on_baseline_routes and a policy from "
        "allowed_route_policies.\n"
        + json.dumps(prompt_input, ensure_ascii=False, separators=(",", ":"))
    )


async def _request_plan(prompt: str, timeout_seconds: float) -> str | None:
    async with openai.AsyncOpenAI(
        base_url=os.environ["PLATFORM_LLM_URL"],
        api_key=os.environ["PLATFORM_API_KEY"],
        timeout=timeout_seconds,
        max_retries=0,
    ) as client:
        response = await client.chat.completions.create(
            model=os.environ["AGENT_MODEL_NAME"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
        )
    return response.choices[0].message.content


async def suggest_plan_with_deadline(
    prompt: str,
    timeout_seconds: float,
    allowed_algorithms: list[str],
    allowed_policies: list[str],
) -> tuple[dict[str, str] | None, str | None, str | None]:
    """Validate the external model decision before it controls construction."""

    try:
        content = await asyncio.wait_for(
            _request_plan(prompt, timeout_seconds),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return None, "model_timeout", None
    except openai.OpenAIError as exc:
        return None, f"model_api_error:{type(exc).__name__}", None

    if content is None or not content.strip():
        return None, "model_empty_response", content
    try:
        decision = json.loads(content)
    except json.JSONDecodeError:
        return None, "model_json_error", content
    if not isinstance(decision, dict) or set(decision) != {"algorithm", "policy"}:
        return None, "model_plan_fields_error", content
    if not isinstance(decision["algorithm"], str) or decision["algorithm"] not in allowed_algorithms:
        return None, "model_unknown_algorithm", content
    if not isinstance(decision["policy"], str) or decision["policy"] not in allowed_policies:
        return None, "model_unknown_policy", content
    return decision, None, content


def _metrics_log(task_id: str, metrics: dict[str, Any]) -> None:
    LOGGER.info(
        "agentcc.solve_metrics task_id=%s %s",
        task_id,
        json.dumps(metrics, ensure_ascii=False, separators=(",", ":")),
    )


class AgentCCAgent(BaseAgent):
    async def solve(self, task_input: dict) -> dict:
        request_started = perf_counter()
        task = task_input["task"]
        topology = task_input["topology"]
        task_id = str(task["task_id"])
        solve_deadline = request_started + _configured_seconds(
            "AGENT_SOLVE_BUDGET_SECONDS", DEFAULT_SOLVE_BUDGET_SECONDS
        )
        with turn(trigger="user"):
            with state("reasoning"):
                baseline_started = perf_counter()
                network = _topology_network(topology)
                split = min(_candidate_splits(task))
                routes = _routes(task, topology, split, network=network)
                baseline = solve_input(task_input, network=network, routes=routes)
                baseline_answer = baseline["answer"]
                baseline_time_us = estimate_schedule_time_us(
                    task, topology, baseline_answer, network
                )
                baseline_ms = (perf_counter() - baseline_started) * 1000
                best = baseline
                selected_policy = "baseline"
                selected_algorithm = "recursive_doubling"
                model_plan = None
                model_content = None
                fallback_reason = None
                candidate_time_us = None
                candidate_validation = "not_built"
                candidate_ms = 0.0
                llm_ms = 0.0
                route_balancing = None

                planning_started = perf_counter()
                profiles = doubling_round_profiles(task, topology, routes, network)
                dimension_order = (
                    optimal_doubling_order(profiles)
                    if task["operator"] == "all-gather" else None
                )
                allowed_algorithms = list(SCHEDULE_ALGORITHMS[task["operator"]])
                allowed_policies = list(ROUTE_POLICIES)
                if task.get("allow_detour", topology.get("allow_detour", True)):
                    allowed_policies.append("load_balanced")
                if "tree" in allowed_algorithms and (
                    not task.get("allow_copy", True)
                    or 2 * baseline_answer["round_count"] > int(task["max_nonempty_rounds"])
                ):
                    allowed_algorithms.remove("tree")
                prompt = _plan_prompt(
                    task, topology, baseline_answer, baseline_time_us,
                    profiles, dimension_order, allowed_algorithms, allowed_policies, network,
                )
                input_sha256 = hashlib.sha256(json.dumps(
                    {"task": task, "topology": topology},
                    ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                ).encode()).hexdigest()
                LOGGER.info("agentcc.model_input %s", json.dumps({
                    "task_id": task_id, "input_sha256": input_sha256, "prompt": prompt,
                }, ensure_ascii=False, separators=(",", ":")))
                planning_ms = (perf_counter() - planning_started) * 1000
                remaining = solve_deadline - perf_counter() - RETURN_RESERVE_SECONDS
                if remaining <= 0:
                    llm_reason = "solve_budget_exhausted"
                else:
                    llm_started = perf_counter()
                    model_budget = _configured_seconds(
                        "AGENT_LLM_BUDGET_SECONDS", DEFAULT_MODEL_BUDGET_SECONDS
                    )
                    model_plan, llm_reason, model_content = await suggest_plan_with_deadline(
                        prompt, min(model_budget, remaining), allowed_algorithms, allowed_policies,
                    )
                    llm_ms = (perf_counter() - llm_started) * 1000
                LOGGER.info("agentcc.model_output %s", json.dumps({
                    "task_id": task_id, "input_sha256": input_sha256,
                    "content": model_content, "plan": model_plan, "error": llm_reason,
                }, ensure_ascii=False, separators=(",", ":")))

                if llm_reason is not None:
                    fallback_reason = llm_reason
                elif model_plan != {"algorithm": "recursive_doubling", "policy": "baseline"}:
                    remaining = solve_deadline - perf_counter() - RETURN_RESERVE_SECONDS
                    if remaining <= 0:
                        fallback_reason = "candidate_budget_exhausted"
                    else:
                        candidate_started = perf_counter()
                        try:
                            reuse_routes = model_plan["policy"] in ("baseline", "load_balanced")
                            candidate = {"answer": build_solution(
                                task, topology,
                                route_policy="baseline" if reuse_routes else model_plan["policy"],
                                network=network,
                                algorithm=model_plan["algorithm"],
                                routes=routes if reuse_routes else None,
                                dimension_order=(
                                    dimension_order if reuse_routes else None
                                ),
                            )}
                            if model_plan["policy"] == "load_balanced":
                                route_balancing = balance_schedule_routes(
                                    task, topology, candidate["answer"], network,
                                    deadline=min(
                                        perf_counter() + ROUTE_BALANCE_BUDGET_SECONDS,
                                        solve_deadline - RETURN_RESERVE_SECONDS,
                                    ),
                                )
                            validate_answer(task, topology, candidate["answer"], network)
                            candidate_validation = "passed"
                            candidate_time_us = estimate_schedule_time_us(
                                task,
                                topology,
                                candidate["answer"],
                                network,
                            )
                            if candidate_time_us < baseline_time_us:
                                best = candidate
                                selected_policy = model_plan["policy"]
                                selected_algorithm = model_plan["algorithm"]
                            else:
                                fallback_reason = "candidate_rejected_estimate_not_lower"
                        except AgentCCError as exc:
                            candidate_validation = f"failed:{exc}"
                            fallback_reason = f"candidate_error:{type(exc).__name__}"
                        candidate_ms = (perf_counter() - candidate_started) * 1000

            with state("final_answer"):
                metrics = {
                    "baseline_ms": round(baseline_ms, 3),
                    "planning_ms": round(planning_ms, 3),
                    "llm_ms": round(llm_ms, 3),
                    "candidate_ms": round(candidate_ms, 3),
                    "solve_ms": round((perf_counter() - request_started) * 1000, 3),
                    "response_bytes": len(
                        json.dumps(best, ensure_ascii=False, separators=(",", ":")).encode()
                    ),
                    "selected_policy": selected_policy,
                    "selected_algorithm": selected_algorithm,
                    "model_plan": model_plan,
                    "baseline_estimate_us": baseline_time_us,
                    "candidate_estimate_us": candidate_time_us,
                    "candidate_validation": candidate_validation,
                    "route_balancing": route_balancing,
                    "input_sha256": input_sha256,
                    "fallback_reason": fallback_reason,
                }
                _metrics_log(task_id, metrics)
                return best
