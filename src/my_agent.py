"""Model-in-the-loop AgentCC scheduling agent (0.2.1).

Per request, every value comes from the current task and topology:

1. Parse and validate the input, build switch-only candidate routes, and
   profile every XOR exchange dimension at path width 1 and 4 (Replay cost
   per MiB and the limiting edge, shared resource or path).
2. The model receives the constraints and profiles and proposes up to three
   compact configurations (algorithm, split, path width, rs_dims).  While the
   model call is in flight, the agent evaluates its own small portfolio in a
   worker thread so the wait is not idle time.
3. Every proposal goes through the ``evaluate_schedule`` tool: built on the
   current topology and estimated with the published Replay formula.
4. The candidate with the best estimate, adjusted by a small response-size
   term, is fully validated and returned; if validation fails the next one is
   tried.  Model failures fall back to the agent portfolio, with the reason
   recorded.

Solve time is scored against a full-score line of roughly 1.5 s, so the
design keeps one short model call and overlaps local work with it.
"""

import asyncio
import hashlib
import json
import logging
import math
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
    answer_size_estimate,
    attach_metrics,
    portfolio_configs,
    replay_estimate,
    validate_answer,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_BUDGET_SECONDS = 8.0
DEFAULT_MODEL_ROUNDS = 1
DEFAULT_MAX_OUTPUT_TOKENS = 160
MAX_PROPOSALS = 3
TOOL_NAME = "evaluate_schedule"
TOOL_VERSION = "2.1"
# Selection score = ln(estimate_us) + SIZE_PENALTY_PER_MB * response MB.
# About 18 MB of extra response costs roughly as much solve-time score as a
# 3% slower schedule costs communication score.
SIZE_PENALTY_PER_MB = 0.0017

SYSTEM_PROMPT = (
    "You pick collective-communication schedule parameters. A tool builds each pick on "
    "the current topology, checks it and measures Replay time; the fastest valid one is used. "
    "Round time = max(longest transfer path time, startup+switch overhead + max busy time "
    "over directed edges and shared resources); rounds add up. "
    "Algorithms: all-gather 'doubling' (XOR recursive doubling). all-reduce 'rs_ag' "
    "(reduce-scatter on rs_dims dimensions, then all-gather; ~2x data instead of 8x; "
    "needs split>=2^rs_dims), 'doubling', 'tree'. width = parallel switch paths per pair "
    "(helps when one link/resource is the bottleneck; needs enough chunks: all-gather "
    "chunks per pair grow 1,2,4.. times split; rs_ag uses split/2^rs_dims chunks in the "
    "middle rounds). Larger split makes the response bigger and slower to return. "
    'Reply with JSON only: {"c":[[algorithm,split,width,rs_dims],...]} (rs_dims 0 unless rs_ag).'
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


def _bottleneck_label(item: dict[str, Any]) -> str:
    if item["type"] == "path_time":
        return "path_time"
    return f"{item['type']}:{item['id']}@{item['capacity_Bps'] / 1e9:.3g}GB/s"


def build_prompt(planner: Planner, profiles: list[dict[str, Any]]) -> str:
    network = planner.network
    algorithms = list(ALGORITHMS[planner.operator])
    if planner.operator == "all-reduce" and not planner.allow_copy:
        algorithms = ["doubling"]
    tight = sorted(zip(network.resource_names, network.resource_capacity.tolist()), key=lambda item: item[1])[:4]
    payload = {
        "op": planner.operator,
        "ranks": planner.rank_count,
        "bytes_per_rank": planner.data_size,
        "splits": planner.splits,
        "max_rounds": planner.max_rounds,
        "algorithms": algorithms,
        "max_width": MAX_PATH_WIDTH,
        "startup_us": network.startup_us,
        "switch_us": network.switch_us,
        "size": [len(network.switch_names), network.edge_count, len(network.resource_names)],
        "tight_resources_GBps": [[name, round(capacity / 1e9, 3)] for name, capacity in tight],
        "dims": [
            [
                row["step"],
                round(row["us_per_MiB_single_path"], 1),
                round(row["us_per_MiB_multipath"], 1),
                _bottleneck_label(row["single_path_bottleneck"]),
                _bottleneck_label(row["multipath_bottleneck"]),
            ]
            for row in profiles
        ],
    }
    return (
        "size=[switches,directed_edges,resources]; dims rows=[xor_step, us/MiB width1, "
        "us/MiB width4, bottleneck width1, bottleneck width4] measured on this topology.\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + f"\nGive up to {MAX_PROPOSALS} picks, best first."
    )


def feedback_prompt(evaluator: "Evaluator") -> str:
    rows = sorted(
        (row for row in evaluator.results.values() if "estimate_us" in row),
        key=lambda item: item["score"],
    )[:6]
    measured = [
        [row["config"]["algorithm"], row["config"]["split"], row["config"]["path_width"],
         row["config"].get("rs_dims", 0), round(row["estimate_us"], 1), row["response_mb"],
         row["slowest_round_limiter"]]
        for row in rows
    ]
    return (
        "Measured [algorithm,split,width,rs_dims,us,response_MB,slowest_round_limiter], best first:\n"
        + json.dumps(measured, ensure_ascii=False, separators=(",", ":"))
        + f"\nGive up to {MAX_PROPOSALS} new picks that could beat the best, same JSON."
    )


_ARRAY_ROW = re.compile(r"\[\s*\"(\w+)\"\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d+)\s*)?\]")


def parse_proposals(content: str | None) -> tuple[list[dict[str, Any]], str | None]:
    """Extract proposals from a reply; tolerates fences, reasoning text and truncation."""

    if content is None or not content.strip():
        return [], "model_empty_response"
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    proposals: list[dict[str, Any]] = []
    for algorithm, split, width, rs_dims in _ARRAY_ROW.findall(text):
        proposal: dict[str, Any] = {"algorithm": algorithm, "split": int(split), "path_width": int(width)}
        if algorithm == "rs_ag" and rs_dims:
            proposal["rs_dims"] = int(rs_dims)
        proposals.append(proposal)
    if not proposals:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                decision = json.loads(match.group(0))
            except json.JSONDecodeError:
                decision = None
            rows = decision.get("candidates") if isinstance(decision, dict) else None
            if isinstance(rows, list):
                proposals = [row for row in rows if isinstance(row, dict)]
    if not proposals:
        return [], "model_json_error"
    return proposals[:MAX_PROPOSALS], None


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
            temperature=0,
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
    """Return (content, metadata, llm_call_id, error_reason); model faults never raise."""

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
    return (config["algorithm"], config["split"], config["path_width"], config.get("rs_dims"), tuple(config["order"]))


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in ("algorithm", "split", "path_width", "rs_dims", "order") if key in config}


class Evaluator:
    """Tool layer shared by model proposals and the agent portfolio within one request."""

    def __init__(self, planner: Planner):
        self.planner = planner
        self.results: dict[tuple, dict[str, Any]] = {}

    def evaluate(self, raw: Any, source: str, parent_llm_call_id: str | None = None) -> dict[str, Any]:
        with tool_call(
            TOOL_NAME, "function", arguments={"source": source, "proposal": raw},
            tool_version=TOOL_VERSION, parent_llm_call_id=parent_llm_call_id,
        ) as call:
            row = self._evaluate(raw, source)
            call.result(self.public(row))
        return row

    def _evaluate(self, raw: Any, source: str) -> dict[str, Any]:
        started = perf_counter()
        row: dict[str, Any] = {"source": source, "proposal": raw}
        try:
            config = self.planner.normalize_config(raw)
        except (AgentCCError, AttributeError, TypeError) as exc:
            row.update(status="rejected", error=str(exc))
            return row
        row["config"] = _public_config(config)
        key = _config_key(config)
        previous = self.results.get(key)
        if previous is not None:
            previous["sources"].add(source)
            row.update(status="duplicate", estimate_us=previous.get("estimate_us"))
            return row
        try:
            answer = self.planner.build(config)
        except AgentCCError as exc:
            row.update(status="invalid", error=str(exc))
            return row
        estimate = replay_estimate(self.planner.task, self.planner.topology, answer, self.planner.network.raw)
        size_mb = answer_size_estimate(answer) / 1e6
        row.update(
            status="built",
            estimate_us=estimate["total_us"],
            response_mb=round(size_mb, 2),
            score=math.log(max(estimate["total_us"], 1e-9)) + SIZE_PENALTY_PER_MB * size_mb,
            round_count=answer["round_count"],
            slowest_round_limiter=max(estimate["rounds"], key=lambda item: item["time_us"])["limiter"],
            elapsed_ms=round((perf_counter() - started) * 1000, 3),
            sources={source},
            _answer=answer,
            _estimate=estimate,
        )
        self.results[key] = row
        return row

    @staticmethod
    def public(row: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in row.items() if not key.startswith("_") and key != "score"}
        if "sources" in result:
            result["sources"] = sorted(result["sources"])
        return result

    def run_portfolio(self) -> list[dict[str, Any]]:
        return [self.evaluate(config, "agent") for config in portfolio_configs(self.planner)]

    def run_proposals(self, proposals: list[Any], call_id: str | None) -> list[dict[str, Any]]:
        return [self.evaluate(proposal, "model", call_id) for proposal in proposals]

    def select(self) -> dict[str, Any]:
        """Validate built candidates in score order; return the first that passes."""

        for row in sorted(self.results.values(), key=lambda item: item["score"]):
            try:
                validate_answer(self.planner.task, self.planner.topology, row["_answer"], self.planner.network.raw)
            except AgentCCError as exc:
                row["status"] = "invalid"
                row["error"] = str(exc)
                continue
            row["status"] = "valid"
            return row
        raise AgentCCError("no candidate passed validation")


class AgentCCAgent(BaseAgent):
    async def solve(self, task_input: dict) -> dict:
        started = perf_counter()
        task = task_input["task"]
        topology = task_input["topology"]
        model_budget = _configured("AGENT_LLM_BUDGET_SECONDS", DEFAULT_MODEL_BUDGET_SECONDS)
        model_rounds = int(_configured("AGENT_MODEL_ROUNDS", DEFAULT_MODEL_ROUNDS))
        max_tokens = int(_configured("AGENT_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))
        audit: dict[str, Any] = {"task_id": str(task.get("task_id")), "input_sha256": _input_digest(task, topology)}

        with turn(trigger="user"):
            with state("planning"):
                planner = await asyncio.to_thread(Planner, task, topology)
                profiles = await asyncio.to_thread(planner.dimension_profiles, 4)
                prompt = build_prompt(planner, profiles)
                audit["planning_ms"] = round((perf_counter() - started) * 1000, 3)
                audit["prompt_bytes"] = len(prompt.encode())

            evaluator = Evaluator(planner)
            with state("tool_execution"):
                # The agent portfolio runs while the model is thinking.
                portfolio = asyncio.ensure_future(asyncio.to_thread(evaluator.run_portfolio))
            with state("reasoning"):
                llm_started = perf_counter()
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                content, metadata, call_id, error = await ask_model(messages, model_budget, max_tokens)
                audit["llm_ms"] = round((perf_counter() - llm_started) * 1000, 3)
                proposals, parse_error = ([], error) if error else parse_proposals(content)
                if parse_error == "model_json_error" and metadata and metadata.get("finish_reason") == "length":
                    parse_error = "model_token_limit"
                audit.update(model_content=content, model_metadata=metadata, model_error=parse_error)
            with state("tool_execution"):
                portfolio_rows = await portfolio
                model_rows = await asyncio.to_thread(evaluator.run_proposals, proposals, call_id) if proposals else []
            # Optional feedback rounds (AGENT_MODEL_ROUNDS > 1): the model sees the
            # measured candidates and may propose refinements.  Off by default
            # because each round adds a full model latency to the scored solve time.
            audit["feedback_rounds"] = []
            for _ in range(max(0, model_rounds - 1)):
                if parse_error or not proposals:
                    break
                with turn(trigger="tool_result"):
                    with state("reasoning"):
                        messages += [
                            {"role": "assistant", "content": content or ""},
                            {"role": "user", "content": feedback_prompt(evaluator)},
                        ]
                        round_started = perf_counter()
                        content, metadata, call_id, error = await ask_model(messages, model_budget, max_tokens)
                        proposals, round_error = ([], error) if error else parse_proposals(content)
                    with state("tool_execution"):
                        rows = await asyncio.to_thread(evaluator.run_proposals, proposals, call_id) if proposals else []
                    model_rows += rows
                    audit["feedback_rounds"].append({
                        "llm_ms": round((perf_counter() - round_started) * 1000, 3),
                        "model_content": content, "model_error": round_error,
                        "candidates": [Evaluator.public(row) for row in rows],
                    })
                    if round_error:
                        break
            with state("reflection"):
                if parse_error:
                    recovery("use_agent_portfolio", count=0)
                best = await asyncio.to_thread(evaluator.select)
                audit["agent_candidates"] = [Evaluator.public(row) for row in portfolio_rows]
                audit["model_candidates"] = [Evaluator.public(row) for row in model_rows]
                audit["selected"] = {
                    "config": best["config"],
                    "estimate_us": best["estimate_us"],
                    "response_mb": best["response_mb"],
                    "sources": sorted(best["sources"]),
                }
                agent_best = min(
                    (row["estimate_us"] for row in portfolio_rows if row.get("status") in ("built", "valid")),
                    default=None,
                )
                audit["model_gain_vs_agent_pct"] = (
                    None if agent_best is None or "model" not in best["sources"]
                    else round(100 * (agent_best - best["estimate_us"]) / agent_best, 3)
                )

            with state("final_answer"):
                answer = best["_answer"]
                attach_metrics(answer, best["_estimate"])
                config = best["config"]
                answer["strategy"] = (
                    f"{config['algorithm']} split={config['split']} path_width={config['path_width']}"
                    + (f" rs_dims={config['rs_dims']}" if "rs_dims" in config else "")
                    + f" from {'+'.join(sorted(best['sources']))}"
                )
                audit["solve_ms"] = round((perf_counter() - started) * 1000, 3)
                LOGGER.info("agentcc.audit %s", json.dumps(_round_floats(audit), ensure_ascii=False, separators=(",", ":")))
                return {"answer": answer}
