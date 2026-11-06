"""Model-in-the-loop AgentCC scheduling agent (0.2.4).

Per request, every value comes from the current task and topology:

1. Parse and validate the input, build switch-only candidate routes, and
   profile every XOR exchange dimension at path width 1 (Replay cost per MiB
   and the limiting edge, shared resource or path).
2. The model receives the constraints, the profiles and the agent's own
   predicted trade-offs (Replay time versus response size) and proposes up to
   three compact configurations.  While the call is in flight the agent ranks
   a larger portfolio with the fast predictor and builds the best few in a
   worker thread.
3. Every proposal and portfolio entry goes through the ``evaluate_schedule``
   tool: built on the current topology and estimated with the published
   Replay formula.  A model proposal whose optimistic prediction already
   loses to a built candidate is answered from the prediction, without a build.
4. The candidate with the best ln(Replay us) + SIZE_WEIGHT * response MB is
   fully validated and returned; if validation fails the next one is tried.
   Any failure of the model, of parsing or of a single candidate falls back
   to the remaining validated candidates; an answer is always returned.

Response size matters because the platform's measured solve time grows by
about 1.46 s per MB of response (0.2.1 platform run, r = 0.97).
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
    reference_config,
    replay_estimate,
    validate_answer,
)


LOGGER = logging.getLogger(__name__)
# Each second of solve time costs about as much as an 18% slower schedule, and on
# the public examples the agent's portfolio already matches the best configuration
# the model can name, so a late reply is not worth waiting for.
DEFAULT_MODEL_BUDGET_SECONDS = 3.0
DEFAULT_MAX_OUTPUT_TOKENS = 120
MAX_PROPOSALS = 3
PORTFOLIO_BUILDS = 3
# Slack on the prediction filter, in cost units (0.02 is about 2% of Replay time).
PRUNE_MARGIN = 0.02
TOOL_NAME = "evaluate_schedule"
TOOL_VERSION = "2.4"
# Selection cost = ln(Replay us) + SIZE_WEIGHT * response MB.  Derived from the
# 0.2.1 platform run (1.46 s/MB) and the new scoring curve; a sweep over slope
# assumptions 0.7-2.0 s/MB showed gains for weights 0.08-0.15.
SIZE_WEIGHT = 0.10

SYSTEM_PROMPT = (
    "You pick collective-communication schedule parameters. A tool builds each pick on the "
    "current topology, checks it and measures Replay time; the lowest cost valid one is used. "
    "cost = ln(Replay us) + 0.1*response_MB (big responses add solve time). "
    "Algorithms: all-gather 'doubling'. all-reduce 'rs_ag' (reduce-scatter on rs_dims dims then "
    "all-gather; ~2x data instead of 8x; response MB grows with split), 'tree' (tiny response, "
    "more rounds), 'doubling'. width = parallel switch paths per pair. "
    'Reply JSON only: {"c":[[algorithm,split,width,rs_dims],...]} (rs_dims 0 unless rs_ag).'
)


def _configured(name: str, default: float) -> float:
    value = os.environ.get(name)
    try:
        return default if value is None or value == "" else float(value)
    except ValueError:
        return default


def sum_notation_enabled() -> bool:
    """Write copies of complete AllReduce slots as c{slot}_sum (default on in 0.2.3)."""

    return os.environ.get("AGENTCC_SUM_NOTATION", "1").strip().lower() not in ("0", "false", "no", "off", "")


def _input_digest(task: dict[str, Any], topology: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        {"task": task, "topology": topology}, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    ).encode()).hexdigest()


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {str(key): _round_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_round_floats(item) for item in value]
    return value


def _safe_json(value: Any) -> str:
    """Serialize audit data without ever raising."""

    try:
        return json.dumps(_round_floats(value), ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return json.dumps({"unserializable": repr(value)[:500]})


def _label(item: dict[str, Any]) -> str:
    if item["type"] == "path_time":
        return "path_time"
    return f"{item['type']}:{item['id']}@{item['capacity_Bps'] / 1e9:.3g}GB/s"


def _cfg_row(config: dict[str, Any]) -> list[Any]:
    return [config["algorithm"], config["split"], config["path_width"], config.get("rs_dims", 0)]


def portfolio_raw(planner: Planner) -> list[dict[str, Any]]:
    """Agent-side search space, ranked by the fast predictor before building."""

    raws: list[dict[str, Any]] = []
    if planner.operator == "all-gather":
        for split in planner.splits[:2]:
            for width in (2, 4, 8, 16):
                raws.append({"algorithm": "doubling", "split": split, "path_width": width})
        return raws
    algorithms = ALGORITHMS[planner.operator] if planner.allow_copy else ("doubling",)
    # Tree plans are costly to rank and rs_ag beat them on every public example, so
    # the agent ranks tree only without rs_ag.  The model can still propose tree.
    rank_tree = "tree" in algorithms and not ("rs_ag" in algorithms and planner.splits[-1] >= 2)
    for split in planner.splits:
        for width in (2, 4, 8):
            if "rs_ag" in algorithms and split >= 2:
                for rs_dims in sorted({split.bit_length() - 1, max(1, split.bit_length() - 2)}):
                    raws.append({"algorithm": "rs_ag", "split": split, "path_width": width, "rs_dims": rs_dims})
            if rank_tree and split <= 8:
                raws.append({"algorithm": "tree", "split": split, "path_width": width})
    if "doubling" in algorithms:
        raws.append({"algorithm": "doubling", "split": planner.splits[0], "path_width": 4})
    return raws


def rank_portfolio(planner: Planner) -> list[tuple[float, dict[str, Any], dict[str, float]]]:
    ranked = []
    for raw in portfolio_raw(planner):
        try:
            config = planner.normalize_config(raw)
            predicted = planner.predict(config)
        except Exception:  # noqa: BLE001 - a bad portfolio entry is skipped, never fatal
            continue
        cost = math.log(max(predicted["us"], 1e-9)) + SIZE_WEIGHT * predicted["mb"]
        ranked.append((cost, config, predicted))
    ranked.sort(key=lambda item: item[0])
    return ranked


def build_prompt(planner: Planner, profiles: list[dict[str, Any]], ranked: list) -> str:
    network = planner.network
    algorithms = list(ALGORITHMS[planner.operator])
    if planner.operator == "all-reduce" and not planner.allow_copy:
        algorithms = ["doubling"]
    tight = sorted(zip(network.resource_names, network.resource_capacity.tolist()), key=lambda item: item[1])[:3]
    payload = {
        "op": planner.operator,
        "ranks": planner.rank_count,
        "splits": planner.splits,
        "max_rounds": planner.max_rounds,
        "algorithms": algorithms,
        "max_width": MAX_PATH_WIDTH,
        "tight_resources_GBps": [[name, round(capacity / 1e9, 2)] for name, capacity in tight],
        "dims_us_per_MiB_w1": [[row["step"], round(row["us_per_MiB_single_path"], 1),
                                _label(row["single_path_bottleneck"])] for row in profiles],
        "agent_predicted": [_cfg_row(config) + [round(pred["us"], 1), round(pred["mb"], 2)]
                            for _, config, pred in ranked[:6]],
    }
    return (
        "agent_predicted rows = [algorithm,split,width,rs_dims,predicted_us,response_MB] (prediction "
        "underestimates tree and width 16). Measured on this topology:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + f"\nGive up to {MAX_PROPOSALS} picks likely to lower the cost, best first; you may add picks "
        "the agent did not list."
    )


_ARRAY_ROW = re.compile(r"\[\s*\"(\w+)\"\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d+)\s*)?\]")


def parse_proposals(content: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Extract proposals from a reply; never raises."""

    try:
        if not isinstance(content, str) or not content.strip():
            return [], "model_empty_response"
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        proposals: list[dict[str, Any]] = []
        for algorithm, split, width, rs_dims in _ARRAY_ROW.findall(text):
            proposal: dict[str, Any] = {"algorithm": algorithm, "split": int(split), "path_width": int(width)}
            if algorithm == "rs_ag" and rs_dims:
                proposal["rs_dims"] = int(rs_dims)
            proposals.append(proposal)
        if not proposals:
            return [], "model_json_error"
        return proposals[:MAX_PROPOSALS], None
    except Exception as exc:  # noqa: BLE001
        return [], f"model_parse_error:{type(exc).__name__}"


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
    """Return (content, metadata, llm_call_id, error); never raises except on cancellation."""

    try:
        content, metadata, call_id = await asyncio.wait_for(
            _chat(messages, timeout_seconds, max_tokens), timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return None, None, None, "model_timeout"
    except asyncio.CancelledError:
        raise
    except openai.OpenAIError as exc:
        return None, None, None, f"model_api_error:{type(exc).__name__}"
    except Exception as exc:  # noqa: BLE001 - missing env, bad response shape, ...
        return None, None, None, f"model_error:{type(exc).__name__}"
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
        self.portfolio_ms: float | None = None

    def evaluate(self, raw: Any, source: str, parent_llm_call_id: str | None = None) -> dict[str, Any]:
        try:
            with tool_call(
                TOOL_NAME, "function", arguments={"source": source, "proposal": _round_floats(raw)},
                tool_version=TOOL_VERSION, parent_llm_call_id=parent_llm_call_id,
            ) as call:
                row = self._evaluate(raw, source)
                call.result(self.public(row))
            return row
        except Exception as exc:  # noqa: BLE001 - one candidate can never fail the request
            return {"source": source, "proposal": repr(raw)[:200], "status": "error", "error": f"{type(exc).__name__}: {exc}"}

    def _evaluate(self, raw: Any, source: str) -> dict[str, Any]:
        started = perf_counter()
        row: dict[str, Any] = {"source": source, "proposal": raw}
        try:
            config = self.planner.normalize_config(raw)
        except (AgentCCError, AttributeError, TypeError, ValueError) as exc:
            row.update(status="rejected", error=str(exc))
            return row
        row["config"] = _public_config(config)
        key = _config_key(config)
        previous = self.results.get(key)
        if previous is not None:
            previous["sources"].add(source)
            row.update(status="duplicate", estimate_us=previous.get("estimate_us"))
            return row
        if source == "model":
            predicted = self.planner.predict(config)
            cost = math.log(max(predicted["us"], 1e-9)) + SIZE_WEIGHT * predicted["mb"]
            best = min((item["score"] for item in self.results.values()), default=math.inf)
            if cost > best + PRUNE_MARGIN:
                row.update(status="pruned", predicted_us=round(predicted["us"], 3),
                           predicted_mb=round(predicted["mb"], 3))
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
            score=math.log(max(estimate["total_us"], 1e-9)) + SIZE_WEIGHT * size_mb,
            round_count=answer["round_count"],
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

    def run_portfolio(self) -> tuple[list[dict[str, Any]], list]:
        started = perf_counter()
        ranked = rank_portfolio(self.planner)
        rows = [self.evaluate(config, "agent") for _, config, _ in ranked[:PORTFOLIO_BUILDS]]
        if not any(row.get("status") == "built" for row in rows):
            rows.append(self.evaluate(reference_config(self.planner), "agent"))
        self.portfolio_ms = round((perf_counter() - started) * 1000, 3)
        return rows, ranked

    def run_proposals(self, proposals: list[Any], call_id: str | None) -> list[dict[str, Any]]:
        return [self.evaluate(proposal, "model", call_id) for proposal in proposals]

    def select(self) -> dict[str, Any]:
        """Validate built candidates in score order; return the first that passes."""

        for row in sorted(self.results.values(), key=lambda item: item["score"]):
            try:
                validate_answer(self.planner.task, self.planner.topology, row["_answer"], self.planner.network.raw)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "invalid"
                row["error"] = str(exc)
                continue
            row["status"] = "valid"
            return row
        raise AgentCCError("no candidate passed validation")

    def emergency(self) -> dict[str, Any]:
        """Last resort: build and validate the reference configuration directly."""

        config = reference_config(self.planner)
        answer = self.planner.build(config)
        validate_answer(self.planner.task, self.planner.topology, answer, self.planner.network.raw)
        estimate = replay_estimate(self.planner.task, self.planner.topology, answer, self.planner.network.raw)
        return {"config": _public_config(config), "estimate_us": estimate["total_us"], "response_mb": None,
                "sources": {"agent_emergency"}, "_answer": answer, "_estimate": estimate}


class AgentCCAgent(BaseAgent):
    async def solve(self, task_input: dict) -> dict:
        started = perf_counter()
        task = task_input["task"]
        topology = task_input["topology"]
        model_budget = _configured("AGENT_LLM_BUDGET_SECONDS", DEFAULT_MODEL_BUDGET_SECONDS)
        max_tokens = int(_configured("AGENT_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))
        audit: dict[str, Any] = {"task_id": str(task.get("task_id")), "input_sha256": _input_digest(task, topology),
                                 "sum_notation": sum_notation_enabled()}

        with turn(trigger="user"):
            with state("planning"):
                # Input errors are real task errors and are reported as such.
                planner = await asyncio.to_thread(
                    lambda: Planner(task, topology, sum_notation=sum_notation_enabled()))
                profiles = await asyncio.to_thread(planner.dimension_profiles, 1)
                ranked = await asyncio.to_thread(rank_portfolio, planner)
                prompt = build_prompt(planner, profiles, ranked)
                audit["planning_ms"] = round((perf_counter() - started) * 1000, 3)
                audit["prompt_bytes"] = len(prompt.encode())

            evaluator = Evaluator(planner)
            with state("tool_execution"):
                # The agent portfolio is built while the model is thinking.
                portfolio = asyncio.ensure_future(asyncio.to_thread(evaluator.run_portfolio))
            with state("reasoning"):
                llm_started = perf_counter()
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                content, metadata, call_id, error = await ask_model(messages, model_budget, max_tokens)
                audit["llm_ms"] = round((perf_counter() - llm_started) * 1000, 3)
                proposals, parse_error = ([], error) if error else parse_proposals(content)
                if parse_error == "model_json_error" and metadata and metadata.get("finish_reason") == "length":
                    parse_error = "model_token_limit"
                audit.update(model_content=content if isinstance(content, str) else None,
                             model_metadata=metadata, model_error=parse_error)
            with state("tool_execution"):
                try:
                    portfolio_rows, _ = await portfolio
                except Exception as exc:  # noqa: BLE001
                    portfolio_rows = []
                    audit["portfolio_error"] = f"{type(exc).__name__}: {exc}"
                audit["portfolio_ms"] = evaluator.portfolio_ms
                model_rows = []
                if proposals:
                    proposals_started = perf_counter()
                    try:
                        model_rows = await asyncio.to_thread(evaluator.run_proposals, proposals, call_id)
                    except Exception as exc:  # noqa: BLE001
                        audit["model_eval_error"] = f"{type(exc).__name__}: {exc}"
                    audit["model_eval_ms"] = round((perf_counter() - proposals_started) * 1000, 3)
            with state("reflection"):
                if parse_error:
                    recovery("use_agent_portfolio", count=0)
                select_started = perf_counter()
                try:
                    best = await asyncio.to_thread(evaluator.select)
                except Exception as exc:  # noqa: BLE001
                    audit["select_error"] = f"{type(exc).__name__}: {exc}"
                    recovery("agent_emergency_reference", count=0)
                    best = await asyncio.to_thread(evaluator.emergency)
                audit["select_ms"] = round((perf_counter() - select_started) * 1000, 3)
                audit["agent_candidates"] = [Evaluator.public(row) for row in portfolio_rows]
                audit["model_candidates"] = [Evaluator.public(row) for row in model_rows]
                audit["selected"] = {
                    "config": best["config"],
                    "estimate_us": best["estimate_us"],
                    "response_mb": best["response_mb"],
                    "sources": sorted(best["sources"]),
                }
                agent_scores = [row["score"] for row in portfolio_rows if "score" in row]
                audit["model_improved_selection"] = (
                    "model" in best["sources"] and "agent" not in best["sources"]
                    and bool(agent_scores) and best.get("score", math.inf) < min(agent_scores)
                )

            with state("final_answer"):
                answer = best["_answer"]
                attach_metrics(answer, best["_estimate"])
                config = best["config"]
                answer["strategy"] = (
                    f"{config['algorithm']} split={config['split']} path_width={config['path_width']}"
                    + (f" rs_dims={config['rs_dims']}" if "rs_dims" in config else "")
                    + f" from {'+'.join(sorted(best['sources']))}"
                    # The platform returns no logs, so the answer itself carries the evidence.
                    + f" llm_ms={round(audit['llm_ms'])} model={parse_error or 'ok'}"
                )
                audit["solve_ms"] = round((perf_counter() - started) * 1000, 3)
                LOGGER.info("agentcc.audit %s", _safe_json(audit))
                return {"answer": answer}
