"""Local comparison of two source trees on the 14 public examples.

Runs ``AgentCCAgent.solve`` in a fresh process per case, pinned to one CPU, with
a stub SDK and a mock model that answers after a fixed delay.  It reports stage
timings, the local Replay estimate and the response size.

These are local measurements on public examples: not official Replay results,
not platform solve times and not real model latency.

    python bench_local.py --src . --delay 1 --repeat 3 --out after.json
    python bench_local.py --compare before.json after.json
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import types
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

FAMILIES = ["uneven_pods", "directed_ring", "sparse_fabric", "cloudmatrix", "fullmesh_clos", "mesh2d", "nvlink_like"]
EXAMPLES = Path(__file__).parent / "agentcc_zip_tmp" / "examples" / "256-rank"


def pin_one_cpu() -> None:
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 1 << 2)
    else:
        os.sched_setaffinity(0, {2})


def install_stub_sdk(spans: list, tools: list) -> None:
    module = types.ModuleType("agenthack")

    @contextlib.contextmanager
    def span(name=None, **_kwargs):
        started = perf_counter()
        try:
            yield SimpleNamespace(attrs={"turn_id": "t"})
        finally:
            spans.append((str(name), perf_counter() - started))

    @contextlib.contextmanager
    def tool_call(name, tool_type, **kwargs):
        started = perf_counter()
        record = {"name": name, "type": tool_type, **kwargs}
        try:
            yield SimpleNamespace(result=lambda value: record.update(result=value))
        finally:
            record["seconds"] = perf_counter() - started
            tools.append(record)

    module.BaseAgent = object
    module.state = span
    module.turn = lambda **_kwargs: span("turn")
    module.tool_call = tool_call
    module.recovery = lambda strategy, count=1: None
    sys.modules["agenthack"] = module


def mock_reply(prompt: str) -> str:
    """Echo the agent's top three predicted rows, a neutral stand-in for a model."""

    match = re.search(r'"agent_predicted":(\[\[.*?\]\])', prompt)
    rows = json.loads(match.group(1))[:3] if match else []
    return json.dumps({"c": [row[:4] for row in rows]}, separators=(",", ":"))


def install_mock_model(openai_module, delay: float, reply: str = "") -> None:
    class Completions:
        async def create(self, **kwargs):
            await asyncio.sleep(delay)
            content = reply or mock_reply(kwargs["messages"][-1]["content"])
            message = SimpleNamespace(content=content)
            delta = SimpleNamespace(content=content)
            if kwargs.get("stream"):
                async def stream():
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason="stop")], usage=None)

                return stream()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=500, completion_tokens=30, total_tokens=530),
            )

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    openai_module.AsyncOpenAI = Client


def run_one(src: str, family: str, short: str, delay: float, reply: str = "") -> dict:
    pin_one_cpu()
    sys.path.insert(0, str(Path(src).resolve()))
    sys.dont_write_bytecode = True
    os.environ.update(PLATFORM_LLM_URL="http://mock/v1", PLATFORM_API_KEY="mock", AGENT_MODEL_NAME="mock")
    spans: list = []
    tools: list = []
    install_stub_sdk(spans, tools)
    import openai

    install_mock_model(openai, delay, reply)
    audit_lines: list = []

    class Capture(logging.Handler):
        def emit(self, record):
            text = record.getMessage()
            if text.startswith("agentcc.audit "):
                audit_lines.append(text[len("agentcc.audit "):])

    logging.getLogger("src.my_agent").addHandler(Capture())
    logging.getLogger("src.my_agent").setLevel(logging.INFO)
    from agentcc_solver import estimate_schedule_time_us, validate_answer
    from src.my_agent import AgentCCAgent

    base = EXAMPLES / family / short
    task = json.loads((base / "task.json").read_text(encoding="utf-8"))
    topology = json.loads((base / "topology.json").read_text(encoding="utf-8"))
    started = perf_counter()
    result = asyncio.run(AgentCCAgent().solve({"challenge": "agentcc", "task": task, "topology": topology}))
    solve_seconds = perf_counter() - started
    started = perf_counter()
    body = json.dumps({"task_id": task["task_id"], "status": "success", "output": result}, separators=(",", ":"))
    dump_seconds = perf_counter() - started
    answer = result["answer"]
    verdict = "valid"
    estimate = None
    try:
        validate_answer(task, topology, answer)
        estimate = estimate_schedule_time_us(task, topology, answer)
    except Exception as exc:  # noqa: BLE001 - report every failure kind
        verdict = f"INVALID: {exc}"
    audit = json.loads(audit_lines[-1]) if audit_lines else {}
    peak_rss_mb = None
    with contextlib.suppress(ImportError):
        import resource

        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    stage_seconds: dict[str, float] = {}
    for name, seconds in spans:
        if name != "turn":
            stage_seconds[name] = stage_seconds.get(name, 0.0) + seconds
    return {
        "case": f"{family}/{short}",
        "verdict": verdict,
        "replay_us": estimate,
        "solve_s": solve_seconds,
        "dump_s": dump_seconds,
        "peak_rss_mb": peak_rss_mb,
        "response_mb": len(body) / 1e6,
        "rounds": answer["round_count"],
        "strategy": answer.get("strategy"),
        "stages": stage_seconds,
        "tool_calls": len(tools),
        "tool_build_s": sum(tool["seconds"] for tool in tools),
        "model_error": audit.get("model_error"),
        "llm_s": (audit.get("llm_ms") or 0) / 1000,
        "planning_s": (audit.get("planning_ms") or 0) / 1000,
        "selected_sources": (audit.get("selected") or {}).get("sources"),
        "model_late": audit.get("model_late"),
    }


def run_all(args) -> int:
    results = []
    for family in FAMILIES:
        if args.only and args.only not in family:
            continue
        for short in ("ag", "ar"):
            samples = []
            for _ in range(args.repeat):
                completed = subprocess.run(
                    [sys.executable, __file__, "--one", family, short, "--src", args.src, "--delay", str(args.delay),
                     "--reply", args.reply],
                    capture_output=True, text=True, encoding="utf-8",
                )
                if completed.returncode != 0:
                    print(completed.stderr[-2000:], file=sys.stderr)
                    return 1
                samples.append(json.loads(completed.stdout.strip().splitlines()[-1]))
            median = sorted(samples, key=lambda row: row["solve_s"])[len(samples) // 2]
            median["solve_s_samples"] = [round(row["solve_s"], 3) for row in samples]
            results.append(median)
            print(f"{median['case']:22s} {median['verdict'][:20]:7s} T={median['replay_us'] or 0:10.1f}us "
                  f"solve={median['solve_s']:6.2f}s mb={median['response_mb']:5.2f} llm={median['llm_s']:5.2f}s "
                  f"plan={median['planning_s']:5.2f}s err={median['model_error']} {median['strategy']}", flush=True)
    Path(args.out).write_text(json.dumps({"delay": args.delay, "src": args.src, "cases": results}, indent=1), encoding="utf-8")
    return 0


def compare(before_path: str, after_path: str) -> int:
    before = {row["case"]: row for row in json.loads(Path(before_path).read_text(encoding="utf-8"))["cases"]}
    after = {row["case"]: row for row in json.loads(Path(after_path).read_text(encoding="utf-8"))["cases"]}
    print("case                   valid  T before     T after    change | solve before after | MB before after")
    changes, times_before, times_after = [], [], []
    for case, old in before.items():
        new = after[case]
        change = new["replay_us"] / old["replay_us"] - 1
        changes.append(change)
        times_before.append(old["solve_s"])
        times_after.append(new["solve_s"])
        print(f"{case:22s} {new['verdict'][:5]:5s} {old['replay_us']:10.1f} {new['replay_us']:10.1f} {change:+7.2%} | "
              f"{old['solve_s']:6.2f} {new['solve_s']:6.2f} | {old['response_mb']:5.2f} {new['response_mb']:5.2f}")
    print(f"mean replay change {statistics.mean(changes):+.2%}; worst {max(changes):+.2%}; "
          f"mean solve {statistics.mean(times_before):.2f}s -> {statistics.mean(times_after):.2f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=".")
    parser.add_argument("--delay", type=float, default=1.0, help="mock model delay in seconds")
    parser.add_argument("--reply", default="", help="fixed mock model reply; default echoes the agent's top rows")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only", default="")
    parser.add_argument("--out", default="bench.json")
    parser.add_argument("--one", nargs=2, metavar=("FAMILY", "OP"))
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    if args.one:
        print(json.dumps(run_one(args.src, args.one[0], args.one[1], args.delay, args.reply)))
        return 0
    return run_all(args)


if __name__ == "__main__":
    sys.exit(main())
