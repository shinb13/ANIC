"""LLM-driven AgentCC solver (0.3.0).

The model writes and revises the search strategy that drives the solver:

1. Analyse: the agent parses and validates the current input and measures
   each XOR dimension (Replay cost per MiB at path width 1 and 4, limiting
   edge or shared resource).
2. Generate: the model receives those measurements and the tool API and
   writes a Python function ``strategy(api)``.  The strategy decides which
   algorithms, splits, rs_dims, per-round path widths and dimension orders to
   explore, ranks them with ``api.predict`` and builds the most promising with
   ``api.measure``.
3. Execute: the code is statically checked and run in a restricted sandbox
   (strategy_sandbox.py).  Each ``measure`` builds a full schedule on the
   current topology and applies the published Replay formula.
4. Iterate: the measured results, slowest-round limiters and any error are
   sent back; the model revises its code once and the revision is run the same
   way (AGENT_MODEL_ROUNDS controls the number of rounds).
5. Select: the best measured candidate by ln(Replay us) + size_weight *
   response MB is fully validated and returned.  If the model or its code
   fails in every round, a fixed agent strategy runs instead and the failure
   is recorded.

Every prompt, every version of the code, every tool call and the selection
reason are written to an audit log line; the answer carries a short summary.
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
    StrategyAPI,
    attach_metrics,
    validate_answer,
)
from strategy_sandbox import StrategyError, check_source, run_strategy


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_BUDGET_SECONDS = 25.0
DEFAULT_MODEL_ROUNDS = 2
DEFAULT_MAX_OUTPUT_TOKENS = 900
STRATEGY_SECONDS = 4.0
MAX_MEASURES_PER_RUN = 5
TOOL_VERSION = "3.0"
# Platform 0.2.1: solve time grows ~1.46 s per response MB and costs ~0.45
# points per second; communication moves ~2 points per factor e of Replay time.
SIZE_WEIGHT = 0.15

API_DOC = """api (the only object your code can use):
  api.operator 'all-gather'|'all-reduce'; api.ranks int; api.splits list[int]; api.max_rounds int
  api.max_width int; api.algorithms list[str]; api.steps list[int] (XOR steps); api.bytes_per_rank int
  api.size_weight float; api.dims list of {step, us_per_MiB_w1, us_per_MiB_w4}
  api.predict(cfg) -> {us, mb, cost} | None   fast estimate, no build, unlimited
  api.measure(cfg) -> {us, mb, cost, slowest} | None   full build + Replay formula, at most 5 per run
  api.sweep(algorithms=None, splits=None, widths=(2,4,8,16), rs_dims=None, limit=8)
      -> [{cfg, us, mb, cost}] cheapest predicted first (None = all legal values)
  api.best() -> best measured {us, mb, cost} | None;  api.time_left() -> seconds;  api.log(text)
predict is exact for all-gather and rs_ag at width<=8 within ~5%; it underestimates tree and
wide (16) rounds by up to 2x, so measure() before trusting those.
cfg = {"algorithm": a, "split": s, "path_width": w,
       optional "rs_dims": h (rs_ag only, 1 <= h <= log2(split)),
       optional "widths": [w per round] (len = round count), optional "order": permutation of api.steps}
Round counts: doubling = log2(ranks); rs_ag = log2(ranks) + rs_dims; tree = 2*log2(ranks).
cost = ln(us) + size_weight*mb  (lower is better; mb is the response size, which adds solve time).
Algorithms: all-gather doubling. all-reduce rs_ag (reduce-scatter then all-gather; moves ~2x data
instead of 8x, but large split makes big responses), doubling (full reduce every round), tree
(reduce to rank 0 then broadcast; tiny response). width = parallel switch paths per pair.
Allowed Python: def, for, if, comprehensions, lambda, arithmetic, list/dict/set/tuple, builtins
abs all any bool dict enumerate float int len list max min range reversed round set sorted sum tuple
zip isinstance str math_log math_sqrt. No import, while, class, try, attribute other than list/dict
methods and int.bit_length."""

SYSTEM_PROMPT = (
    "You design the search strategy of a collective-communication scheduler. You write Python "
    "code that runs against a tool API on the current topology; the fastest measured valid "
    "candidate is returned. Reason about the measured bottlenecks, then write compact code. "
    "Reply with exactly one ```python code block defining strategy(api) that returns a list of "
    "cfg dicts (best first), preceded by at most 3 short lines of reasoning."
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


def _label(item: dict[str, Any]) -> str:
    if item["type"] == "path_time":
        return "path_time"
    return f"{item['type']}:{item['id']}@{item['capacity_Bps'] / 1e9:.3g}GB/s"


BASELINE_STRATEGY = '''
def strategy(api):
    picks = []
    for s in api.splits:
        if api.operator == "all-gather":
            for w in (2, 4, 8, 16):
                c = {"algorithm": "doubling", "split": s, "path_width": w}
                p = api.predict(c)
                if p: picks.append((p["cost"], c))
            continue
        for w in (2, 4, 8):
            if "rs_ag" in api.algorithms and s >= 2:
                for h in range(1, s.bit_length()):
                    c = {"algorithm": "rs_ag", "split": s, "path_width": w, "rs_dims": h}
                    p = api.predict(c)
                    if p: picks.append((p["cost"], c))
            for a in ("tree", "doubling"):
                if a in api.algorithms:
                    c = {"algorithm": a, "split": s, "path_width": w}
                    p = api.predict(c)
                    if p: picks.append((p["cost"], c))
    picks = sorted(picks, key=lambda item: item[0])
    for cost, c in picks[:4]:
        api.measure(c)
    return [c for cost, c in picks[:4]]
'''


def instance_prompt(planner: Planner, profiles: list[dict[str, Any]]) -> str:
    network = planner.network
    tight = sorted(zip(network.resource_names, network.resource_capacity.tolist()), key=lambda item: item[1])[:4]
    facts = {
        "operator": planner.operator,
        "ranks": planner.rank_count,
        "bytes_per_rank": planner.data_size,
        "splits": planner.splits,
        "max_rounds": planner.max_rounds,
        "algorithms": list(ALGORITHMS[planner.operator]) if planner.allow_copy or planner.operator == "all-gather"
        else ["doubling"],
        "startup_us": network.startup_us,
        "switch_us": network.switch_us,
        "switches_edges_resources": [len(network.switch_names), network.edge_count, len(network.resource_names)],
        "tightest_resources_GBps": [[name, round(capacity / 1e9, 3)] for name, capacity in tight],
        "dims": [
            [row["step"], round(row["us_per_MiB_single_path"], 1), round(row["us_per_MiB_multipath"], 1),
             _label(row["single_path_bottleneck"]), _label(row["multipath_bottleneck"])]
            for row in profiles
        ],
    }
    return (
        API_DOC
        + "\n\nCurrent instance (dims rows = [xor_step, us/MiB width1, us/MiB width4, bottleneck w1, bottleneck w4]):\n"
        + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
        + "\n\nBaseline strategy (valid, but blind to this instance's bottlenecks):\n```python"
        + BASELINE_STRATEGY
        + "```\nRevise it for this instance: use the bottleneck data to decide what to explore "
        "(algorithms, splits, rs_dims, per-round widths, order) and what to measure. Keep it short."
    )


def feedback_prompt(record: dict[str, Any], best: dict[str, Any] | None) -> str:
    payload = {
        "error": record.get("error"),
        "measured": [
            {key: call[key] for key in ("config", "us", "mb", "cost", "slowest", "error") if key in call}
            for call in record.get("calls", []) if call["tool"] == "measure"
        ],
        "predict_calls": sum(1 for call in record.get("calls", []) if call["tool"] == "predict"),
        "log": record.get("notes", []),
        "best_so_far": best,
        "run_seconds": record.get("run_seconds"),
    }
    return (
        "Execution result of your strategy on the current topology:\n"
        + json.dumps(_round_floats(payload), ensure_ascii=False, separators=(",", ":"))
        + "\nFix any error and revise strategy(api) to find a lower cost (explore around the best, "
        "per-round widths, other splits/rs_dims). Reply with one python block."
    )


_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(content: str | None) -> tuple[str | None, str | None]:
    if content is None or not content.strip():
        return None, "model_empty_response"
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    blocks = _CODE_BLOCK.findall(text)
    if not blocks:
        if "def strategy" in text:
            # Unterminated block (token limit): keep from the definition onwards.
            return text[text.index("def strategy"):], None
        return None, "model_no_code"
    for block in blocks:
        if "def strategy" in block:
            return block.strip(), None
    return None, "model_no_strategy_function"


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
    """Return (content, metadata, llm_call_id, error); model faults never raise."""

    try:
        content, metadata, call_id = await asyncio.wait_for(
            _chat(messages, timeout_seconds, max_tokens), timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return None, None, None, "model_timeout"
    except openai.OpenAIError as exc:
        return None, None, None, f"model_api_error:{type(exc).__name__}"
    return content, metadata, call_id, None





def execute_strategy(planner: Planner, profiles: list[dict[str, Any]], source: str,
                     origin: str) -> tuple[StrategyAPI, dict[str, Any]]:
    """Check and run one strategy version; record every tool call it makes."""

    started = perf_counter()
    api = StrategyAPI(planner, profiles, SIZE_WEIGHT, deadline=started + STRATEGY_SECONDS,
                      max_measures=MAX_MEASURES_PER_RUN)
    record: dict[str, Any] = {"origin": origin, "code": source}
    try:
        check_source(source)
        returned = run_strategy(source, api, max_seconds=STRATEGY_SECONDS)
        # Returned configs that were not measured yet are measured while budget remains.
        if isinstance(returned, list):
            for raw in returned[:3]:
                if not any(row["config"] == planner.normalize_config_safe(raw) for row in api.measured):
                    api.measure(raw)
        record["returned"] = returned if isinstance(returned, list) else None
    except StrategyError as exc:
        record["error"] = str(exc)
    except AgentCCError as exc:
        record["error"] = f"AgentCCError: {exc}"
    record["run_seconds"] = round(perf_counter() - started, 3)
    record["calls"] = api.calls
    record["notes"] = api.notes
    return api, record


class AgentCCAgent(BaseAgent):
    async def solve(self, task_input: dict) -> dict:
        started = perf_counter()
        task = task_input["task"]
        topology = task_input["topology"]
        model_budget = _configured("AGENT_LLM_BUDGET_SECONDS", DEFAULT_MODEL_BUDGET_SECONDS)
        model_rounds = int(_configured("AGENT_MODEL_ROUNDS", DEFAULT_MODEL_ROUNDS))
        max_tokens = int(_configured("AGENT_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))
        audit: dict[str, Any] = {"task_id": str(task.get("task_id")), "input_sha256": _input_digest(task, topology),
                                 "rounds": []}
        candidates: list[dict[str, Any]] = []

        with turn(trigger="user") as root:
            with state("planning"):
                planner = await asyncio.to_thread(Planner, task, topology)
                profiles = await asyncio.to_thread(planner.dimension_profiles, 4)
                audit["planning_ms"] = round((perf_counter() - started) * 1000, 3)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instance_prompt(planner, profiles)},
            ]
            parent = getattr(root, "attrs", {}).get("turn_id")
            for index in range(max(0, model_rounds)):
                with turn(trigger="user" if index == 0 else "tool_result", parent_turn_id=parent):
                    round_record: dict[str, Any] = {"round": index + 1}
                    with state("reasoning"):
                        llm_started = perf_counter()
                        content, metadata, call_id, error = await ask_model(messages, model_budget, max_tokens)
                        round_record.update(llm_ms=round((perf_counter() - llm_started) * 1000, 3),
                                            prompt=messages[-1]["content"], model_content=content,
                                            model_metadata=metadata)
                        source, code_error = (None, error) if error else extract_code(content)
                    if source is None:
                        round_record["error"] = code_error
                        audit["rounds"].append(round_record)
                        recovery("agent_strategy_fallback", count=0)
                        break
                    with state("tool_execution"):
                        with tool_call(
                            "run_strategy", "function", arguments={"code": source},
                            tool_version=TOOL_VERSION, parent_llm_call_id=call_id,
                        ) as call:
                            api, record = await asyncio.to_thread(
                                execute_strategy, planner, profiles, source, f"model_round_{index + 1}",
                            )
                            call.result({k: record.get(k) for k in ("error", "run_seconds")})
                    for row in api.measured:
                        candidates.append({**row, "origin": record["origin"]})
                    round_record.update({key: record.get(key) for key in ("code", "error", "run_seconds", "calls", "notes")})
                    audit["rounds"].append(round_record)
                    if record.get("error"):
                        recovery("retry_with_error_feedback", count=1 if index + 1 < model_rounds else 0)
                    best = min(candidates, key=lambda item: item["cost"]) if candidates else None
                    messages += [
                        {"role": "assistant", "content": content or ""},
                        {"role": "user", "content": feedback_prompt(
                            record, None if best is None else {"config": best["config"], "us": best["us"], "mb": best["mb"],
                                                               "cost": best["cost"]})},
                    ]

            with state("reflection"):
                if not candidates:
                    with tool_call("run_strategy", "function", arguments={"code": "agent_fallback"},
                                   tool_version=TOOL_VERSION) as call:
                        api, record = await asyncio.to_thread(
                            execute_strategy, planner, profiles, BASELINE_STRATEGY, "agent_fallback")
                        call.result({k: record.get(k) for k in ("error", "run_seconds")})
                    audit["fallback"] = {key: record.get(key) for key in ("error", "run_seconds", "calls")}
                    candidates = [{**row, "origin": "agent_fallback"} for row in api.measured]
                chosen = None
                for row in sorted(candidates, key=lambda item: item["cost"]):
                    try:
                        validate_answer(task, topology, row["answer"], planner.network.raw)
                    except AgentCCError as exc:
                        row["validation_error"] = str(exc)
                        continue
                    chosen = row
                    break
                if chosen is None:
                    raise AgentCCError("no strategy produced a valid schedule")
                audit["selected"] = {
                    "origin": chosen["origin"], "config": {k: v for k, v in chosen["config"].items() if k != "order_source"},
                    "us": chosen["us"], "mb": chosen["mb"], "cost": chosen["cost"],
                    "measured_total": len(candidates),
                }

            with state("final_answer"):
                answer = chosen["answer"]
                attach_metrics(answer, chosen["estimate"])
                config = chosen["config"]
                answer["strategy"] = (
                    f"{config['algorithm']} split={config['split']} path_width={config['path_width']}"
                    + (f" rs_dims={config['rs_dims']}" if "rs_dims" in config else "")
                    + (f" widths={config['widths']}" if config.get("widths") else "")
                    + f"; chosen from {len(candidates)} measured candidates; origin={chosen['origin']}"
                )
                audit["solve_ms"] = round((perf_counter() - started) * 1000, 3)
                LOGGER.info("agentcc.audit %s", json.dumps(_round_floats(audit), ensure_ascii=False,
                                                           separators=(",", ":"), default=str))
                return {"answer": answer}
