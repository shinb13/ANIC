"""Model-in-the-loop AgentCC scheduling agent.

Loop per request (every value comes from the current task and topology):

1. Parse and validate the input; profile every XOR exchange dimension on
   single-path and optimized multipath routes (per-MiB Replay cost and the
   bottleneck edge or shared resource).
2. Round 1: the model receives the constraints and profiles and proposes up
   to four schedule configurations (algorithm, split, path width, rs_dims,
   dimension order).
3. Each proposal is run through the ``evaluate_schedule`` tool: built,
   checked against every correctness constraint, and estimated with the
   published Replay formula, including a per-round bottleneck report.
4. Round 2: the model receives the measured results and proposes refinements,
   which are evaluated the same way.
5. The fastest validated candidate is returned.  If the model is unavailable
   or every proposal is invalid, a validated agent-side reference is returned
   and the reason is recorded.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from time import perf_counter
from typing import Any

import openai
from agenthack import BaseAgent, recovery, state, tool_call, turn

from agentcc_solver import (
    ALGORITHMS,
    MAX_PATH_WIDTH,
    AgentCCError,
    Planner,
    attach_metrics,
    reference_config,
    replay_estimate,
    validate_answer,
)


LOGGER = logging.getLogger(__name__)
# Solve time is scored by 35 * ((360 s - t) / (360 s - t_good))^6, so every 10 s
# of model latency costs roughly 4 of 35 points; model time is capped in total.
DEFAULT_MODEL_BUDGET_SECONDS = 15.0
DEFAULT_MODEL_TOTAL_SECONDS = 25.0
DEFAULT_SOLVE_BUDGET_SECONDS = 60.0
DEFAULT_MODEL_ROUNDS = 2
DEFAULT_MAX_OUTPUT_TOKENS = 1024
MAX_PROPOSALS_PER_ROUND = 4
RETURN_RESERVE_SECONDS = 3.0
TOOL_NAME = "evaluate_schedule"
TOOL_VERSION = "2.0"

SYSTEM_PROMPT = (
    "You are the planning component of a collective-communication scheduling agent. "
    "You choose schedule parameters; a deterministic tool builds each choice on the "
    "current topology, validates it, and reports its Replay time. "
    "Replay cost per round = max(longest single transfer path time, max startup+switch "
    "overhead + max over directed edges and shared resources of busy time); rounds add up. "
    "Algorithms: AllGather 'doubling' (XOR recursive doubling, payload doubles each round). "
    "AllReduce 'rs_ag' (reduce-scatter over the first rs_dims dimensions of the order, "
    "full reduce exchange on the remaining dimensions, then all-gather in reverse; moves "
    "about 2x the data instead of 8x but needs 2*rs_dims+remaining rounds and a split of "
    "at least 2^rs_dims), 'doubling' (full-vector reduce exchange every round), 'tree' "
    "(reduce to one rank then broadcast, 2x dimensions rounds). "
    "path_width = how many alternative switch paths each pair may split its chunks over; "
    "wider helps when one link or shared resource is the bottleneck but costs more chunks "
    "per transfer granularity. split must divide the work into enough chunks to use the "
    "width. order = 'auto' or a permutation of the XOR steps; for doubling AllGather the "
    "first dimension carries the least data, for rs_ag the first dimensions carry the most. "
    "Reply with JSON only."
)


def _configured(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _input_digest(task: dict[str, Any], topology: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        {"task": task, "topology": topology}, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _context(planner: Planner) -> dict[str, Any]:
    task, network = planner.task, planner.network
    resources = sorted(
        zip(network.resource_names, network.resource_capacity.tolist()), key=lambda item: item[1],
    )
    return {
        "operator": planner.operator,
        "rank_count": planner.rank_count,
        "per_rank_bytes": planner.data_size,
        "valid_splits": planner.splits,
        "max_nonempty_rounds": planner.max_rounds,
        "allow_copy": planner.allow_copy,
        "allow_detour": network.allow_detour,
        "xor_steps": planner.steps,
        "algorithms": list(ALGORITHMS[planner.operator]) if planner.allow_copy or planner.operator == "all-gather"
        else ["doubling"],
        "max_path_width": MAX_PATH_WIDTH,
        "communication_startup_time_us": network.startup_us,
        "switch_processing_overhead_us": network.switch_us,
        "topology_size": {
            "ranks": planner.rank_count,
            "switches": len(network.switch_names),
            "directed_edges": network.edge_count,
            "shared_resources": len(network.resource_names),
        },
        "tightest_shared_resources_Bps": [[name, capacity] for name, capacity in resources[:6]],
    }


def _first_prompt(planner: Planner, profiles: list[dict[str, Any]]) -> str:
    payload = {
        "instance": _context(planner),
        "dimension_profiles": profiles,
        "profile_note": (
            "us_per_MiB_* = round busy time if every rank sends 1 MiB to its XOR partner on "
            "that dimension at path_width 1 versus 16; the bottleneck names the limiting "
            "directed edge, shared resource, or path."
        ),
    }
    return (
        "Current instance and measured topology profiles:\n"
        + json.dumps(_round_floats(payload), ensure_ascii=False, separators=(",", ":"))
        + f"\nPropose up to {MAX_PROPOSALS_PER_ROUND} diverse configurations likely to minimize Replay "
        'time. Reply exactly as {"analysis":"<=60 words","candidates":[{"algorithm":"...",'
        '"split":N,"path_width":N,"rs_dims":N,"order":"auto"}]}. rs_dims only for rs_ag.'
    )


def _feedback_prompt(results: list[dict[str, Any]], best: dict[str, Any] | None) -> str:
    payload = {
        "evaluated": [
            {key: row[key] for key in ("config", "status", "estimate_us", "error", "slowest_rounds") if key in row}
            for row in results
        ],
        "best_so_far": None if best is None else {"config": best["config"], "estimate_us": best["estimate_us"]},
    }
    return (
        "Tool results for your candidates (validated builds on the current topology):\n"
        + json.dumps(_round_floats(payload), ensure_ascii=False, separators=(",", ":"))
        + f"\nUsing the bottlenecks above, propose up to {MAX_PROPOSALS_PER_ROUND} new configurations "
        "that could beat the best (not already evaluated). Same JSON format. If nothing can "
        'improve, reply {"analysis":"...","candidates":[]}.'
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_proposals(content: str | None) -> tuple[list[Any], str | None, str | None]:
    """Extract the candidate list from a model reply (tolerates fences and reasoning text)."""

    if content is None or not content.strip():
        return [], None, "model_empty_response"
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    match = _JSON_BLOCK.search(text)
    if not match:
        return [], None, "model_json_error"
    try:
        decision = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [], None, "model_json_error"
    if not isinstance(decision, dict) or not isinstance(decision.get("candidates"), list):
        return [], None, "model_plan_fields_error"
    analysis = decision.get("analysis")
    return decision["candidates"][:MAX_PROPOSALS_PER_ROUND], analysis if isinstance(analysis, str) else None, None


async def _chat(messages: list[dict[str, str]], timeout_seconds: float, max_tokens: int):
    async with openai.AsyncOpenAI(
        base_url=os.environ["PLATFORM_LLM_URL"],
        api_key=os.environ["PLATFORM_API_KEY"],
        timeout=timeout_seconds,
        max_retries=0,
    ) as client:
        response = await client.chat.completions.create(
            model=os.environ["AGENT_MODEL_NAME"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
    choice = response.choices[0]
    usage = response.usage
    metadata = {
        "finish_reason": choice.finish_reason,
        "usage": None if usage is None else {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    }
    return choice.message.content, metadata, getattr(response, "_agenthack_llm_call_id", None)


async def ask_model(messages: list[dict[str, str]], timeout_seconds: float, max_tokens: int):
    """Return (content, metadata, llm_call_id, error_reason); never raises for model faults."""

    try:
        content, metadata, call_id = await asyncio.wait_for(
            _chat(messages, timeout_seconds, max_tokens), timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return None, None, None, "model_timeout"
    except openai.OpenAIError as exc:
        return None, None, None, f"model_api_error:{type(exc).__name__}"
    return content, metadata, call_id, None


def _config_key(config: dict[str, Any]) -> tuple:
    return (
        config["algorithm"], config["split"], config["path_width"],
        config.get("rs_dims"), tuple(config["order"]),
    )


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in ("algorithm", "split", "path_width", "rs_dims", "order", "order_source")
            if key in config}


def evaluate_candidate(planner: Planner, raw: Any, seen: set[tuple]) -> dict[str, Any]:
    """Tool body: normalize, build, validate and estimate one proposed configuration."""

    started = perf_counter()
    row: dict[str, Any] = {"proposal": raw}
    try:
        config = planner.normalize_config(raw)
    except (AgentCCError, AttributeError, TypeError) as exc:
        row.update(status="rejected", error=str(exc))
        return row
    row["config"] = _public_config(config)
    key = _config_key(config)
    if key in seen:
        row.update(status="duplicate")
        return row
    seen.add(key)
    try:
        answer = planner.build(config)
        validate_answer(planner.task, planner.topology, answer, planner.network.raw)
    except AgentCCError as exc:
        row.update(status="invalid", error=str(exc))
        return row
    estimate = replay_estimate(planner.task, planner.topology, answer, planner.network.raw)
    slowest = sorted(enumerate(estimate["rounds"], 1), key=lambda item: -item[1]["time_us"])[:3]
    row.update(
        status="valid",
        estimate_us=estimate["total_us"],
        round_count=answer["round_count"],
        transfers=sum(len(item["transfers"]) for item in answer["rounds"]),
        slowest_rounds=[
            {"round": number, "time_us": data["time_us"], "limiter": data["limiter"]}
            for number, data in slowest
        ],
        elapsed_ms=round((perf_counter() - started) * 1000, 3),
    )
    row["_answer"] = answer
    row["_estimate"] = estimate
    return row


class AgentCCAgent(BaseAgent):
    async def solve(self, task_input: dict) -> dict:
        started = perf_counter()
        task = task_input["task"]
        topology = task_input["topology"]
        task_id = str(task.get("task_id"))
        deadline = started + _configured("AGENT_SOLVE_BUDGET_SECONDS", DEFAULT_SOLVE_BUDGET_SECONDS)
        model_budget = _configured("AGENT_LLM_BUDGET_SECONDS", DEFAULT_MODEL_BUDGET_SECONDS)
        model_total = _configured("AGENT_LLM_TOTAL_SECONDS", DEFAULT_MODEL_TOTAL_SECONDS)
        model_spent = 0.0
        model_rounds = int(_configured("AGENT_MODEL_ROUNDS", DEFAULT_MODEL_ROUNDS))
        max_tokens = int(_configured("AGENT_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))
        digest = _input_digest(task, topology)
        audit: dict[str, Any] = {"task_id": task_id, "input_sha256": digest, "model_rounds": []}

        with turn(trigger="user") as root_turn:
            with state("planning"):
                planning_started = perf_counter()
                planner = await asyncio.to_thread(Planner, task, topology)
                profiles = await asyncio.to_thread(planner.dimension_profiles)
                audit["planning_ms"] = round((perf_counter() - planning_started) * 1000, 3)

            seen: set[tuple] = set()
            evaluated: list[dict[str, Any]] = []
            best: dict[str, Any] | None = None
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _first_prompt(planner, profiles)},
            ]
            parent_turn = getattr(root_turn, "attrs", {}).get("turn_id")
            for round_index in range(max(0, model_rounds)):
                remaining = min(
                    deadline - perf_counter() - RETURN_RESERVE_SECONDS,
                    model_total - model_spent,
                )
                if remaining <= 1.0:
                    audit["model_rounds"].append({"round": round_index + 1, "skipped": "model_or_solve_budget_exhausted"})
                    break
                record: dict[str, Any] = {"round": round_index + 1, "prompt_bytes": len(messages[-1]["content"].encode())}
                with turn(trigger="tool_result" if round_index else "user", parent_turn_id=parent_turn):
                    with state("reasoning"):
                        llm_started = perf_counter()
                        content, metadata, call_id, error = await ask_model(
                            messages, min(model_budget, remaining), max_tokens,
                        )
                        model_spent += perf_counter() - llm_started
                        record["llm_ms"] = round((perf_counter() - llm_started) * 1000, 3)
                        record["response_metadata"] = metadata
                        record["content"] = content
                    proposals, analysis, parse_error = ([], None, error) if error else parse_proposals(content)
                    record["analysis"] = analysis
                    record["error"] = parse_error
                    if parse_error:
                        if metadata and metadata.get("finish_reason") == "length":
                            record["error"] = "model_token_limit"
                        audit["model_rounds"].append(record)
                        recovery("fallback_to_agent_reference" if round_index == 0 else "keep_best_candidate", count=0)
                        break
                    messages.append({"role": "assistant", "content": content})
                    rows = []
                    with state("tool_execution"):
                        for proposal in proposals:
                            if perf_counter() > deadline - RETURN_RESERVE_SECONDS:
                                rows.append({"proposal": proposal, "status": "skipped_deadline"})
                                continue
                            with tool_call(
                                TOOL_NAME, "function", arguments={"proposal": proposal},
                                tool_version=TOOL_VERSION, parent_llm_call_id=call_id,
                            ) as call:
                                row = await asyncio.to_thread(evaluate_candidate, planner, proposal, seen)
                                call.result({k: v for k, v in row.items() if not k.startswith("_")})
                            rows.append(row)
                            if row["status"] == "valid" and (best is None or row["estimate_us"] < best["estimate_us"]):
                                best = row
                    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
                    record["tool_results"] = public_rows
                    evaluated.extend(public_rows)
                    audit["model_rounds"].append(record)
                    if not proposals:
                        break
                    messages.append({"role": "user", "content": _feedback_prompt(public_rows, best)})

            with state("reflection"):
                selection = "model_candidate"
                if best is None:
                    selection = "agent_reference"
                    recovery("fallback_to_agent_reference", count=0)
                    row = await asyncio.to_thread(evaluate_candidate, planner, reference_config(planner), set())
                    if row["status"] != "valid":
                        raise AgentCCError(f"reference configuration failed: {row.get('error')}")
                    best = row
                audit["selection"] = selection
                audit["selected_config"] = best["config"]
                audit["selected_estimate_us"] = best["estimate_us"]
                audit["evaluated_valid"] = sum(1 for row in evaluated if row["status"] == "valid")
                audit["evaluated_total"] = len(evaluated)

            with state("final_answer"):
                answer = best["_answer"]
                attach_metrics(answer, best["_estimate"])
                answer["strategy"] = (
                    f"{best['config']['algorithm']} split={best['config']['split']} "
                    f"path_width={best['config']['path_width']}"
                    + (f" rs_dims={best['config']['rs_dims']}" if "rs_dims" in best["config"] else "")
                    + f" order={best['config']['order']} selected by {selection}"
                )
                audit["solve_ms"] = round((perf_counter() - started) * 1000, 3)
                LOGGER.info("agentcc.audit %s", json.dumps(_round_floats(audit), ensure_ascii=False, separators=(",", ":")))
                return {"answer": answer}
