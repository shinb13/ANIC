"""Model-in-the-loop AgentCC scheduling agent (0.2.10y).

Per request, every value comes from the current task and topology:

1. Parse and validate the input, build switch-only candidate routes, and
   profile every XOR exchange dimension at path width 1 (Replay cost per MiB
   and the limiting edge, shared resource or path).
2. The model receives the constraints, the profiles and a few initial
   single-path predictions (Replay time versus response size) and proposes up to
   three compact configurations.  While the call is in flight the agent ranks
   a larger portfolio with the fast predictor and builds the best few in a
   worker thread.
3. Every proposal and portfolio entry goes through the ``evaluate_schedule``
   tool: built on the current topology and estimated with the published
   Replay formula.
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
    build_dual_tree,
    prepare_dual_tree,
    reference_config,
    refine_schedule_routes,
    replay_estimate,
    split_path_bound_transfers,
    validate_answer,
)


LOGGER = logging.getLogger(__name__)
# 0.2.9y: 3 s instead of 6 s.  On the platform 0.2.4y (3 s cap) scored 24.3/25 on
# solve time against 24.1 for 0.2.8s (6 s cap); a late reply is dropped and the
# agent's own candidates are used, recorded as a recovery.
DEFAULT_MODEL_BUDGET_SECONDS = 3.0
DEFAULT_MAX_OUTPUT_TOKENS = 120
MAX_PROPOSALS = 3
PORTFOLIO_BUILDS = 4
TOOL_NAME = "evaluate_schedule"
TOOL_VERSION = "2.10y"
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
    'Reply JSON only: {"c":[[algorithm,split,width,rs_dims],...],"m":2,"t":1} (rs_dims 0 unless rs_ag). '
    "Optional m controls remapping the best rs_ag candidate: 0 original, 1 locality, 2 resource-aware; "
    "omit for agent choice. Mapping path-work gains are rough screening, not Replay times. "
    "Optional t controls a separate two-tree AllReduce trial: 0 disable, 1 split slots equally, "
    "2 first tree gets 1/4, 3 first gets 3/4; omit for agent choice. Both are four-way trees "
    "generated from current resource prices; integer Replay and full validation gate adoption."
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
    for split in planner.splits:
        for width in (2, 4, 8):
            if "rs_ag" in algorithms and split >= 2:
                for rs_dims in sorted({split.bit_length() - 1, max(1, split.bit_length() - 2)}):
                    raws.append({"algorithm": "rs_ag", "split": split, "path_width": width, "rs_dims": rs_dims})
            if "tree" in algorithms and split <= 8:
                raws.append({"algorithm": "tree", "split": split, "path_width": width})
    if "doubling" in algorithms:
        raws.append({"algorithm": "doubling", "split": planner.splits[0], "path_width": 4})
    return raws


def rank_portfolio(
    planner: Planner, candidates: list[dict[str, Any]] | None = None,
) -> list[tuple[float, dict[str, Any], dict[str, float]]]:
    ranked = []
    raws = portfolio_raw(planner) if candidates is None else candidates
    for raw in raws:
        try:
            config = planner.normalize_config(raw)
            predicted = planner.predict(config)
        except AgentCCError:
            continue
        cost = math.log(max(predicted["us"], 1e-9)) + SIZE_WEIGHT * predicted["mb"]
        ranked.append((cost, config, predicted))
    ranked.sort(key=lambda item: item[0])
    return ranked


def build_prompt(planner: Planner, profiles: list[dict[str, Any]], seeds: list, mappings: list) -> str:
    network = planner.network
    capacity = network.raw[0]
    algorithms = list(ALGORITHMS[planner.operator])
    if planner.operator == "all-reduce" and not planner.allow_copy:
        algorithms = ["doubling"]
    tight = sorted(zip(network.resource_names, network.resource_capacity.tolist()), key=lambda item: item[1])[:3]
    payload = {
        "op": planner.operator,
        "ranks": planner.rank_count,
        "bytes_per_rank": planner.data_size,
        "alignment_bytes": planner.task["split_policy"].get("alignment_bytes", 1),
        "splits": planner.splits,
        "max_rounds": planner.max_rounds,
        "algorithms": algorithms,
        "max_width": MAX_PATH_WIDTH,
        "prediction_scope": "initial_single_path_seeds",
        "links": {
            "directed": network.edge_count,
            "one_way": sum((dst, src) not in capacity for src, dst in capacity),
            "unequal_reverse": sum(capacity[(dst, src)] < bw for (src, dst), bw in capacity.items()
                                   if (dst, src) in capacity),
            "GBps_range": [float(network.edge_capacity.min()) / 1e9, float(network.edge_capacity.max()) / 1e9],
        },
        "tight_resources_GBps": [[name, round(capacity / 1e9, 2)] for name, capacity in tight],
        "dims_us_per_MiB_w1": [[row["step"], round(row["us_per_MiB_single_path"], 1),
                                _label(row["single_path_bottleneck"])] for row in profiles],
        "agent_predicted": [_cfg_row(config) + [round(pred["us"], 1), round(pred["mb"], 2)]
                            for _, config, pred in seeds],
    }
    if mappings:
        payload["rank_mapping_path_work_gain_pct"] = [
            [index, option["mapping"], round(option["proxy_gain"] * 100, 2) if option["reachable"] else "unreachable"]
            for index, option in enumerate(mappings, 1)
        ]
    tree_rounds = 2 * ((len(planner.steps) + 1) // 2)
    if (planner.operator == "all-reduce" and planner.allow_copy and planner.splits[-1] >= 4
            and tree_rounds <= planner.max_rounds):
        split = planner.splits[-1]
        payload["dual_tree_trial"] = {
            "split": split, "width": 4, "rounds": tree_rounds,
            "slot_partitions": [[split // 2, split // 2], [split // 4, 3 * split // 4],
                                [3 * split // 4, split // 4]],
            "prediction_scope": "not_yet_measured",
        }
    return (
        "agent_predicted rows = [algorithm,split,width,rs_dims,integer_Replay_us,approx_response_MB]. "
        "Initial single-path seeds on this topology, not a full search. "
        "The agent searches other widths while you decide:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + f"\nGive up to {MAX_PROPOSALS} picks likely to lower the cost, best first; you may add picks "
        "the agent did not list."
    )


_ARRAY_ROW = re.compile(r"\[\s*\"(\w+)\"\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d+)\s*)?\]")


def parse_proposals(content: Any) -> tuple[list[dict[str, Any]], int | None, int | None, str | None]:
    """Extract proposals from a reply; never raises."""

    try:
        if not isinstance(content, str) or not content.strip():
            return [], None, None, "model_empty_response"
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        mapping = None
        mapping_match = re.search(r'"m"\s*:\s*([^,}\s]+)', text)
        if mapping_match:
            mapping = json.loads(mapping_match.group(1))
            if type(mapping) is not int or mapping not in (0, 1, 2):
                return [], None, None, "model_mapping_error"
        tree = None
        tree_match = re.search(r'"t"\s*:\s*([^,}\s]+)', text)
        if tree_match:
            tree = json.loads(tree_match.group(1))
            if type(tree) is not int or tree not in (0, 1, 2, 3):
                return [], None, None, "model_tree_error"
        proposals: list[dict[str, Any]] = []
        for algorithm, split, width, rs_dims in _ARRAY_ROW.findall(text):
            proposal: dict[str, Any] = {"algorithm": algorithm, "split": int(split), "path_width": int(width)}
            if algorithm == "rs_ag" and rs_dims:
                proposal["rs_dims"] = int(rs_dims)
            proposals.append(proposal)
        if not proposals and mapping is None and tree is None:
            return [], None, None, "model_json_error"
        return proposals[:MAX_PROPOSALS], mapping, tree, None
    except Exception as exc:  # noqa: BLE001
        return [], None, None, f"model_parse_error:{type(exc).__name__}"


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


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in ("algorithm", "split", "path_width", "rs_dims", "order", "widths") if key in config}


class Evaluator:
    """Tool layer shared by model proposals and the agent portfolio within one request."""

    def __init__(self, planner: Planner):
        self.planner = planner
        self.results: dict[tuple, dict[str, Any]] = {}
        self.timings_ms: dict[str, float] = {}
        self.mapping_planners: dict[tuple[int, ...], Planner] = {}
        self.prepared_mapping: dict[str, Any] | None = None
        self.prepared_tree: dict[str, Any] = {"reason": "ineligible"}

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
        key = self.planner.schedule_key(config)
        previous = self.results.get(key)
        if previous is not None:
            previous["sources"].add(source)
            row.update(status="duplicate", estimate_us=previous["estimate_us"],
                       equivalent_config=previous["config"], elapsed_ms=round((perf_counter() - started) * 1000, 3))
            return row
        predicted = self.planner.predict(config)
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
            predicted_us=predicted["us"],
            predicted_response_mb=predicted["mb"],
            response_mb=round(size_mb, 2),
            score=math.log(max(estimate["total_us"], 1e-9)) + SIZE_WEIGHT * size_mb,
            round_count=answer["round_count"],
            elapsed_ms=round((perf_counter() - started) * 1000, 3),
            sources={source},
            _answer=answer,
            _estimate=estimate,
            _planner=self.planner,
        )
        self.results[key] = row
        return row

    @staticmethod
    def public(row: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in row.items() if not key.startswith("_") and key != "score"}
        if "sources" in result:
            result["sources"] = sorted(result["sources"])
        return result

    def run_portfolio(self, ranked: list | None = None) -> tuple[list[dict[str, Any]], list]:
        started = perf_counter()
        if ranked is None:
            ranked = rank_portfolio(self.planner)
        ranked_at = perf_counter()
        self.timings_ms["portfolio_rank"] = round((ranked_at - started) * 1000, 3)
        rows = []
        built = 0
        for _, config, _ in ranked:
            row = self.evaluate(config, "agent")
            rows.append(row)
            built += row["status"] == "built"
            if built == PORTFOLIO_BUILDS:
                break
        if not any(row.get("status") == "built" for row in rows):
            rows.append(self.evaluate(reference_config(self.planner), "agent"))
        self.timings_ms["portfolio_build"] = round((perf_counter() - ranked_at) * 1000, 3)
        prepare_started = perf_counter()
        if built:
            best = min(self.results.values(), key=lambda row: row["score"])
            if best["config"]["algorithm"] == "rs_ag":
                with tool_call("prepare_rank_mapping", "function", arguments={"config": best["config"]},
                               tool_version=TOOL_VERSION) as call:
                    self.prepared_mapping = self.public(self.mapping_trial(best["config"], None))
                    call.result(self.prepared_mapping)
        self.timings_ms["mapping_prepare"] = round((perf_counter() - prepare_started) * 1000, 3)
        prepare_started = perf_counter()
        if self.planner.operator == "all-reduce" and self.planner.allow_copy:
            with tool_call("prepare_dual_tree", "function", arguments={"split": self.planner.splits[-1]},
                           tool_version=TOOL_VERSION) as call:
                try:
                    self.prepared_tree = prepare_dual_tree(self.planner)
                except AgentCCError as exc:
                    self.prepared_tree = {"reason": "unroutable_tree", "error": str(exc)}
                call.result(self.public(self.prepared_tree))
        self.timings_ms["tree_prepare"] = round((perf_counter() - prepare_started) * 1000, 3)
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
        size_mb = answer_size_estimate(answer) / 1e6
        return {"config": _public_config(config), "estimate_us": estimate["total_us"], "response_mb": round(size_mb, 2),
                "score": math.log(estimate["total_us"]) + SIZE_WEIGHT * size_mb,
                "sources": {"agent_emergency"}, "_answer": answer, "_estimate": estimate, "_planner": self.planner}

    def mapping_trial(self, base_config: dict[str, Any], choice: int | None) -> dict[str, Any]:
        options = self.planner.mapping_candidates(base_config)
        preferred = {1: "locality", 2: "resource"}.get(choice)
        eligible = [option for option in options if option["reachable"] and option["proxy_gain"] >= 0.0025
                    and tuple(option["rank_order"]) != self.planner.rank_order
                    and (preferred is None or option["mapping"] == preferred)]
        info: dict[str, Any] = {"options": options}
        if not eligible:
            return {**info, "reason": "proxy_screen"}
        option = max(eligible, key=lambda item: item["proxy_gain"])
        key = tuple(option["rank_order"])
        mapped = self.mapping_planners.get(key)
        if mapped is None:
            mapped = self.planner.with_rank_order(option["rank_order"])
            self.mapping_planners[key] = mapped
        raw = {key: value for key, value in base_config.items() if key not in ("order", "mapping")}
        try:
            config = mapped.normalize_config(raw)
            prediction = mapped.predict(config)
        except AgentCCError as exc:
            return {**info, "reason": "unroutable_mapping", "error": str(exc)}
        return {**info, "mapping": option["mapping"], "rank_order": option["rank_order"],
                "predicted": prediction, "config": config, "_planner": mapped}

    def remap(self, best: dict[str, Any], choice: int | None, call_id: str | None
              ) -> tuple[dict[str, Any], dict[str, Any]]:
        info: dict[str, Any] = {"model_choice": choice, "adopted": False, "before_us": best["estimate_us"]}
        if best["config"]["algorithm"] != "rs_ag" or choice == 0:
            info["reason"] = "not_rs_ag" if best["config"]["algorithm"] != "rs_ag" else "model_kept_original"
            return best, info
        started = perf_counter()
        with tool_call(
            "search_rank_mapping", "function", arguments={"config": best["config"], "model_choice": choice},
            tool_version=TOOL_VERSION, parent_llm_call_id=call_id,
        ) as call:
            trial = self.mapping_trial(best["config"], choice)
            info.update(self.public(trial))
            if "predicted" not in trial:
                info["reason"] = trial["reason"]
            else:
                mapped, config, prediction = trial["_planner"], trial["config"], trial["predicted"]
                if prediction["us"] > best["estimate_us"] * 0.97:
                    info["reason"] = "integer_replay_screen"
                else:
                    source = "agent_mapping" if choice is None else "model_mapping"
                    candidate = Evaluator(mapped).evaluate({**config, "mapping": trial["mapping"]}, source, call_id)
                    info["candidate"] = self.public(candidate)
                    if candidate["status"] != "built":
                        info["reason"] = "candidate_rejected"
                    else:
                        validate_answer(mapped.task, mapped.topology, candidate["_answer"], mapped.network.raw)
                        if candidate["estimate_us"] < best["estimate_us"] and candidate["score"] <= best["score"]:
                            candidate["config"]["mapping"] = trial["mapping"]
                            candidate["sources"].update(best["sources"])
                            candidate["status"] = "valid"
                            best = candidate
                            info["adopted"] = True
                        else:
                            info["reason"] = "complete_schedule_cost"
            info.update(after_us=best["estimate_us"], elapsed_ms=(perf_counter() - started) * 1000)
            call.result(info)
        return best, info

    def refine(self, best: dict[str, Any], call_id: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
        with tool_call(
            "refine_schedule_routes", "function",
            arguments={"config": best["config"], "sources": sorted(best["sources"]), "budget_ms": 100},
            tool_version=TOOL_VERSION, parent_llm_call_id=call_id,
        ) as call:
            answer, estimate, detail = refine_schedule_routes(
                best["_planner"], best["config"], best["_answer"], best["_estimate"], size_weight=SIZE_WEIGHT,
            )
            call.result(detail)
        if detail["adopted"]:
            size_mb = answer_size_estimate(answer) / 1e6
            best = {**best, "_answer": answer, "_estimate": estimate, "estimate_us": estimate["total_us"],
                    "response_mb": round(size_mb, 2), "score": math.log(estimate["total_us"]) + SIZE_WEIGHT * size_mb}
        return best, detail

    def split_transfers(self, best: dict[str, Any], call_id: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Final pass: split path-bound transfers; adopt only a validated, cheaper schedule."""

        planner = best["_planner"]
        with tool_call(
            "split_path_bound_transfers", "function",
            arguments={"config": best["config"], "size_weight": SIZE_WEIGHT},
            tool_version=TOOL_VERSION, parent_llm_call_id=call_id,
        ) as call:
            chunk_bytes = planner.data_size // best["_answer"]["split"]
            answer, detail = split_path_bound_transfers(planner.network, best["_answer"], chunk_bytes,
                                                        size_weight=SIZE_WEIGHT)
            detail["adopted"] = False
            if answer is not best["_answer"]:
                validate_answer(planner.task, planner.topology, answer, planner.network.raw)
                estimate = replay_estimate(planner.task, planner.topology, answer, planner.network.raw)
                size_mb = answer_size_estimate(answer) / 1e6
                score = math.log(estimate["total_us"]) + SIZE_WEIGHT * size_mb
                detail.update(measured_us=estimate["total_us"], response_mb=size_mb)
                if estimate["total_us"] < best["estimate_us"] and score < best["score"]:
                    best = {**best, "_answer": answer, "_estimate": estimate, "estimate_us": estimate["total_us"],
                            "response_mb": round(size_mb, 2), "score": score}
                    detail["adopted"] = True
                else:
                    detail["rejection"] = "complete_schedule_cost"
            call.result({key: value for key, value in detail.items() if key != "rounds"})
        return best, detail

    def try_dual_tree(self, best: dict[str, Any], choice: int | None, call_id: str | None
                      ) -> tuple[dict[str, Any], dict[str, Any]]:
        info: dict[str, Any] = {"model_choice": choice, "before_us": best["estimate_us"], "adopted": False}
        if choice == 0 or "options" not in self.prepared_tree:
            info["reason"] = "model_disabled" if choice == 0 else self.prepared_tree["reason"]
            return best, info
        prepared, planner = self.prepared_tree, self.planner
        options = [option for option in prepared["options"] if choice is None or option["choice"] == choice]
        option = min(options, key=lambda item: math.log(item["us"]) + SIZE_WEIGHT * item["mb"])
        with tool_call("evaluate_dual_tree", "function", arguments={"model_choice": choice, "partition": option},
                       tool_version=TOOL_VERSION, parent_llm_call_id=call_id) as call:
            info["predicted"] = option
            if option["us"] > best["estimate_us"] * .97:
                info["reason"] = "integer_replay_screen"
            else:
                answer = build_dual_tree(planner, prepared, option["choice"])
                validate_answer(planner.task, planner.topology, answer, planner.network.raw)
                estimate = replay_estimate(planner.task, planner.topology, answer, planner.network.raw)
                size_mb = answer_size_estimate(answer) / 1e6
                score = math.log(estimate["total_us"]) + SIZE_WEIGHT * size_mb
                source = "agent_dual_tree" if choice is None else "model_dual_tree"
                info.update(candidate_us=estimate["total_us"], candidate_response_mb=size_mb,
                            before_cost=best["score"], candidate_cost=score)
                if estimate["total_us"] < best["estimate_us"] and score <= best["score"]:
                    best = {"config": {"algorithm": "dual_tree", "split": prepared["split"], "path_width": 4,
                                       "first_tree_slots": option["first_slots"]},
                            "estimate_us": estimate["total_us"], "response_mb": round(size_mb, 2), "score": score,
                            "sources": best["sources"] | {source}, "_answer": answer, "_estimate": estimate,
                            "_planner": planner, "status": "valid"}
                    info["adopted"] = True
                else:
                    info["reason"] = "complete_schedule_cost"
            info["after_us"] = best["estimate_us"]
            call.result(info)
        return best, info


class AgentCCAgent(BaseAgent):
    async def solve(self, task_input: dict) -> dict:
        started = perf_counter()
        task = task_input["task"]
        topology = task_input["topology"]
        model_budget = _configured("AGENT_LLM_BUDGET_SECONDS", DEFAULT_MODEL_BUDGET_SECONDS)
        max_tokens = int(_configured("AGENT_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))
        audit: dict[str, Any] = {"task_id": str(task.get("task_id")), "input_sha256": _input_digest(task, topology),
                                 "sum_notation": sum_notation_enabled(), "timings_ms": {}}
        timings = audit["timings_ms"]

        with turn(trigger="user"):
            with state("planning"):
                # Input errors are real task errors and are reported as such.
                phase_started = perf_counter()
                planner = await asyncio.to_thread(
                    lambda: Planner(task, topology, sum_notation=sum_notation_enabled()))
                timings["parse"] = round((perf_counter() - phase_started) * 1000, 3)
                phase_started = perf_counter()
                profiles = await asyncio.to_thread(planner.dimension_profiles, 1)
                timings["dimension_profiles"] = round((perf_counter() - phase_started) * 1000, 3)
                phase_started = perf_counter()
                seed_raws = [{"algorithm": "doubling", "split": planner.splits[0], "path_width": 1}]
                if planner.operator == "all-reduce" and planner.allow_copy:
                    if planner.splits[-1] >= 2:
                        seed_raws.append({"algorithm": "rs_ag", "split": planner.splits[-1], "path_width": 1})
                    seed_raws.append({"algorithm": "tree", "split": planner.splits[0], "path_width": 1})
                seeds = await asyncio.to_thread(rank_portfolio, planner, seed_raws)
                timings["seed_predictions"] = round((perf_counter() - phase_started) * 1000, 3)
                phase_started = perf_counter()
                rs_seed = next((config for _, config, _ in seeds if config["algorithm"] == "rs_ag"), None)
                mappings = await asyncio.to_thread(planner.mapping_candidates, rs_seed) if rs_seed is not None else []
                timings["mapping_generation"] = round((perf_counter() - phase_started) * 1000, 3)
                prompt = build_prompt(planner, profiles, seeds, mappings)
                audit["planning_ms"] = round((perf_counter() - started) * 1000, 3)
                audit["initial_candidate_count"] = len(seeds)
                audit["prompt_bytes"] = len(prompt.encode())

            evaluator = Evaluator(planner)
            with state("tool_execution"):
                # One worker owns planning caches until the portfolio has completed.
                portfolio = asyncio.ensure_future(asyncio.to_thread(evaluator.run_portfolio))
            with state("reasoning"):
                llm_started = perf_counter()
                audit["llm_started_ms"] = round((llm_started - started) * 1000, 3)
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                content, metadata, call_id, error = await ask_model(messages, model_budget, max_tokens)
                audit["llm_ms"] = round((perf_counter() - llm_started) * 1000, 3)
                timings["model"] = audit["llm_ms"]
                proposals, mapping_choice, tree_choice, parse_error = (([], None, None, error) if error
                                                                      else parse_proposals(content))
                if parse_error == "model_json_error" and metadata and metadata.get("finish_reason") == "length":
                    parse_error = "model_token_limit"
                audit.update(model_content=content if isinstance(content, str) else None,
                             model_metadata=metadata, model_error=parse_error, model_mapping_choice=mapping_choice,
                             model_tree_choice=tree_choice)
            with state("tool_execution"):
                phase_started = perf_counter()
                try:
                    portfolio_rows, _ = await portfolio
                except Exception as exc:  # noqa: BLE001
                    portfolio_rows = []
                    audit["portfolio_error"] = f"{type(exc).__name__}: {exc}"
                timings.update(evaluator.timings_ms)
                audit["mapping_preparation"] = evaluator.prepared_mapping
                audit["tree_preparation"] = Evaluator.public(evaluator.prepared_tree)
                timings["portfolio_join"] = round((perf_counter() - phase_started) * 1000, 3)
                phase_started = perf_counter()
                model_rows = []
                if proposals:
                    try:
                        model_rows = await asyncio.to_thread(evaluator.run_proposals, proposals, call_id)
                    except Exception as exc:  # noqa: BLE001
                        audit["model_eval_error"] = f"{type(exc).__name__}: {exc}"
                timings["model_candidates"] = round((perf_counter() - phase_started) * 1000, 3)
            with state("reflection"):
                if parse_error:
                    recovery("use_agent_portfolio", count=0)
                phase_started = perf_counter()
                try:
                    best = await asyncio.to_thread(evaluator.select)
                except Exception as exc:  # noqa: BLE001
                    audit["select_error"] = f"{type(exc).__name__}: {exc}"
                    recovery("agent_emergency_reference", count=0)
                    best = await asyncio.to_thread(evaluator.emergency)
                timings["select_validate"] = round((perf_counter() - phase_started) * 1000, 3)
                phase_started = perf_counter()
                best, audit["rank_mapping"] = await asyncio.to_thread(evaluator.remap, best, mapping_choice, call_id)
                timings["rank_mapping"] = round((perf_counter() - phase_started) * 1000, 3)
                phase_started = perf_counter()
                best, audit["route_refinement"] = await asyncio.to_thread(evaluator.refine, best, call_id)
                timings["refine_validate"] = round((perf_counter() - phase_started) * 1000, 3)
                phase_started = perf_counter()
                best, audit["dual_tree"] = await asyncio.to_thread(evaluator.try_dual_tree, best, tree_choice, call_id)
                timings["dual_tree_validate"] = round((perf_counter() - phase_started) * 1000, 3)
                phase_started = perf_counter()
                try:
                    best, audit["transfer_split"] = await asyncio.to_thread(evaluator.split_transfers, best, call_id)
                except Exception as exc:  # noqa: BLE001 - the unsplit schedule is already valid
                    audit["transfer_split"] = {"error": f"{type(exc).__name__}: {exc}"}
                timings["transfer_split"] = round((perf_counter() - phase_started) * 1000, 3)
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
                    + (f" mapping={config['mapping']}" if "mapping" in config else "")
                    + (f" first_tree_slots={config['first_tree_slots']}" if "first_tree_slots" in config else "")
                    + f" from {'+'.join(sorted(best['sources']))}"
                    # The platform returns no logs, so the answer itself carries the evidence.
                    + f" llm_ms={round(audit['llm_ms'])} model={parse_error or 'ok'}"
                )
                audit["solve_ms"] = round((perf_counter() - started) * 1000, 3)
                LOGGER.info("agentcc.audit %s", _safe_json(audit))
                return {"answer": answer}
