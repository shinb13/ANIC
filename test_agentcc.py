import asyncio
import contextlib
import importlib
import json
import os
import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

from agentcc_solver import (
    AgentCCError,
    Planner,
    _topology_network,
    estimate_schedule_time_us,
    reference_config,
    replay_estimate,
    solve_request,
    validate_answer,
)


ROOT = Path(__file__).parent
PUBLIC = ROOT / "agentcc_zip_tmp" / "examples"
FAMILIES = ("uneven_pods", "directed_ring", "sparse_fabric", "cloudmatrix", "fullmesh_clos", "mesh2d", "nvlink_like")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def public_case(family: str, short: str):
    base = PUBLIC / "256-rank" / family / short
    return read_json(base / "task.json"), read_json(base / "topology.json")


def two_plane_input(operator="all-gather", ranks=8, candidates=(1, 2, 4, 8)):
    """Ranks attach to two switches; plane B is slower but has spare capacity."""

    names = [f"r{index}" for index in range(ranks)]
    edges = []
    for rank in names:
        edges.append({"src": rank, "dst": "A", "bw": "10GB/s", "bidirectional": True})
        edges.append({"src": rank, "dst": "B", "bw": "5GB/s", "bidirectional": True})
    return {
        "task": {
            "task_id": "synthetic",
            "instance_id": "synthetic",
            "operator": operator,
            "ranks": names,
            "data_size": {"per_rank_bytes": 65536},
            "split_policy": {"mode": "search_allowed", "candidates": list(candidates), "alignment_bytes": 4096},
            "split": 1,
            "max_nonempty_rounds": 64,
            "allow_copy": True,
            "allow_detour": True,
        },
        "topology": {
            "ranks": names,
            "switches": ["A", "B"],
            "edges": edges,
            "resources": [{
                "id": "plane-A-ingress",
                "capacity": "20GB/s",
                "edges": [{"src": rank, "dst": "A"} for rank in names],
            }],
            "communication_startup_time_us": 1,
            "switch_processing_overhead_us": 0.3,
        },
    }


class SolverTests(unittest.TestCase):
    def test_shared_resource_examples_match_published_replay(self):
        base = PUBLIC / "shared-resource"
        topology = read_json(base / "topology.json")
        for operator in ("all-gather", "all-reduce"):
            task = read_json(base / f"{operator}-task.json")
            expected = read_json(base / f"{operator}-solution.json")["output"]["answer"]
            expected["split"] = 1
            validate_answer(task, topology, expected)
            response = solve_request(task, topology)
            answer = response["output"]["answer"]
            validate_answer(task, topology, answer)
            self.assertEqual(response["instance_id"], task["instance_id"])
        ag_task = read_json(base / "all-gather-task.json")
        ag = read_json(base / "all-gather-solution.json")["output"]["answer"]
        ag["split"] = 1
        self.assertAlmostEqual(estimate_schedule_time_us(ag_task, topology, ag), 66.836, places=3)

    def test_invalid_topologies_are_rejected_with_reasons(self):
        base = two_plane_input()["topology"]
        cases = {
            "undeclared edge": lambda t: t["resources"][0]["edges"].append({"src": "A", "dst": "B"}),
            "duplicate resource": lambda t: t["resources"].append(deepcopy(t["resources"][0])),
            "undefined resource": lambda t: t["edges"][0].update(resource_id="missing"),
            "not a declared": lambda t: t["edges"].append({"src": "r0", "dst": "Z", "bw": 1}),
            "overlap": lambda t: t["switches"].append("r0"),
        }
        for message, mutate in cases.items():
            topology = deepcopy(base)
            mutate(topology)
            with self.subTest(message=message), self.assertRaisesRegex(AgentCCError, message):
                _topology_network(topology)

    def test_task_validation_rejects_bad_inputs(self):
        data = two_plane_input()
        task, topology = data["task"], data["topology"]
        bad_split = deepcopy(task)
        bad_split["split_policy"]["alignment_bytes"] = 65536 * 2
        with self.assertRaisesRegex(AgentCCError, "no split candidate"):
            Planner(bad_split, topology)
        mismatch = deepcopy(task)
        mismatch["ranks"] = list(reversed(task["ranks"]))
        with self.assertRaisesRegex(AgentCCError, "must match"):
            Planner(mismatch, topology)

    def test_all_configurations_validate_on_small_topology(self):
        for operator in ("all-gather", "all-reduce"):
            data = two_plane_input(operator)
            planner = Planner(data["task"], data["topology"])
            algorithms = ["doubling"] if operator == "all-gather" else ["rs_ag", "doubling", "tree"]
            for algorithm in algorithms:
                for split in planner.splits:
                    if algorithm == "rs_ag" and split < 2:
                        continue
                    for width in (1, 2, 4):
                        config = planner.normalize_config(
                            {"algorithm": algorithm, "split": split, "path_width": width, "order": "auto"},
                        )
                        answer = planner.build(config)
                        with self.subTest(op=operator, algorithm=algorithm, split=split, width=width):
                            validate_answer(data["task"], data["topology"], answer)
                            min_chunks = min(chunks for _, chunks in planner.round_payloads(config)) \
                                if config["algorithm"] != "tree" else 0
                            # The continuous model only applies when chunks can follow the split.
                            if min_chunks >= 4 * width:
                                self.assertAlmostEqual(
                                    estimate_schedule_time_us(data["task"], data["topology"], answer),
                                    planner.predict_us(config),
                                    delta=0.35 * planner.predict_us(config),
                                )

    def test_multipath_uses_the_second_plane_and_is_faster(self):
        data = two_plane_input("all-gather")
        planner = Planner(data["task"], data["topology"])
        single = planner.build(planner.normalize_config({"algorithm": "doubling", "split": 8, "path_width": 1}))
        multi = planner.build(planner.normalize_config({"algorithm": "doubling", "split": 8, "path_width": 2}))
        validate_answer(data["task"], data["topology"], multi)
        switches = {t["path"][1] for r in multi["rounds"] for t in r["transfers"]}
        self.assertEqual(switches, {"A", "B"})
        self.assertLess(
            estimate_schedule_time_us(data["task"], data["topology"], multi),
            estimate_schedule_time_us(data["task"], data["topology"], single),
        )

    def test_capacity_change_changes_route_split(self):
        data = two_plane_input("all-gather")
        changed = deepcopy(data)
        for edge in changed["topology"]["edges"]:
            if edge["dst"] == "B":
                edge["bw"] = "40GB/s"
        shares = []
        for case in (data, changed):
            planner = Planner(case["task"], case["topology"])
            answer = planner.build(planner.normalize_config({"algorithm": "doubling", "split": 8, "path_width": 2}))
            transfers = [t for r in answer["rounds"] for t in r["transfers"]]
            on_b = sum(len(t["chunk_ranges"]) for t in transfers if t["path"][1] == "B")
            shares.append(on_b / sum(len(t["chunk_ranges"]) for t in transfers))
        self.assertGreater(shares[1], shares[0] + 0.2)

    def test_rs_ag_moves_fewer_bytes_than_doubling(self):
        data = two_plane_input("all-reduce")
        planner = Planner(data["task"], data["topology"])
        rs = planner.build(planner.normalize_config({"algorithm": "rs_ag", "split": 8, "path_width": 2}))
        rd = planner.build(planner.normalize_config({"algorithm": "doubling", "split": 8, "path_width": 2}))
        moved = lambda a: sum(len(t["chunk_ranges"]) for r in a["rounds"] for t in r["transfers"])
        self.assertEqual(rs["round_count"], 3 + 3)
        # 8 ranks: 2*(n-1) = 14 chunk transfers per rank versus log2(n)*split = 24.
        self.assertEqual(moved(rs), 8 * 14)
        self.assertEqual(moved(rd), 8 * 24)
        self.assertLess(
            estimate_schedule_time_us(data["task"], data["topology"], rs),
            estimate_schedule_time_us(data["task"], data["topology"], rd),
        )

    def test_validator_catches_semantic_violations(self):
        data = two_plane_input("all-reduce")
        task, topology = data["task"], data["topology"]
        planner = Planner(task, topology)
        answer = planner.build(planner.normalize_config({"algorithm": "rs_ag", "split": 8, "path_width": 1}))
        duplicate = deepcopy(answer)
        first = duplicate["rounds"][0]["transfers"][0]
        duplicate["rounds"][0]["transfers"].append(deepcopy(first))
        with self.assertRaises(AgentCCError):
            validate_answer(task, topology, duplicate)
        early_copy = deepcopy(answer)
        early_copy["rounds"][0]["transfers"][0]["action"] = "copy"
        with self.assertRaisesRegex(AgentCCError, "complete reduction"):
            validate_answer(task, topology, early_copy)
        bad_path = deepcopy(answer)
        bad_path["rounds"][0]["transfers"][0]["path"] = [first["src"], "Z", first["dst"]]
        with self.assertRaises(AgentCCError):
            validate_answer(task, topology, bad_path)
        missing = deepcopy(answer)
        missing["rounds"][-1]["transfers"].pop()
        with self.assertRaisesRegex(AgentCCError, "did not finish"):
            validate_answer(task, topology, missing)

    def test_round_limit_and_bad_proposals_are_rejected(self):
        data = two_plane_input("all-reduce")
        data["task"]["max_nonempty_rounds"] = 5
        planner = Planner(data["task"], data["topology"])
        for proposal, message in [
            ({"algorithm": "tree", "split": 1, "path_width": 1}, "max_nonempty_rounds"),
            ({"algorithm": "rs_ag", "split": 1, "path_width": 1}, "requires split"),
            ({"algorithm": "rs_ag", "split": 3, "path_width": 1}, "split must be"),
            ({"algorithm": "doubling", "split": 1, "path_width": 99}, "path_width"),
            ({"algorithm": "doubling", "split": 1, "path_width": 1, "order": [1, 1, 2]}, "permutation"),
            ({"algorithm": "ring", "split": 1, "path_width": 1}, "algorithm"),
        ]:
            with self.subTest(proposal=proposal), self.assertRaisesRegex(AgentCCError, message):
                planner.normalize_config(proposal)

    def test_public_256_rank_examples(self):
        for family in FAMILIES:
            for short in ("ag", "ar"):
                task, topology = public_case(family, short)
                planner = Planner(task, topology)
                profiles = planner.dimension_profiles()
                self.assertEqual(len(profiles), 8)
                old_style = planner.build(planner.normalize_config(
                    {"algorithm": "doubling", "split": planner.splits[0], "path_width": 1},
                ))
                config = reference_config(planner)
                answer = planner.build(config)
                with self.subTest(family=family, operator=task["operator"]):
                    validate_answer(task, topology, answer, planner.network.raw)
                    self.assertLessEqual(answer["round_count"], task["max_nonempty_rounds"])
                    # Equal only where every rank has a single uplink (directed_ring AllGather).
                    self.assertLessEqual(
                        replay_estimate(task, topology, answer, planner.network.raw)["total_us"],
                        replay_estimate(task, topology, old_style, planner.network.raw)["total_us"] + 1e-6,
                    )


class AgentLoopTests(unittest.TestCase):
    @staticmethod
    def _load_agent_module(events):
        agenthack = types.ModuleType("agenthack")

        class BaseAgent:
            pass

        @contextlib.contextmanager
        def span(*_args, **_kwargs):
            yield SimpleNamespace(attrs={"turn_id": "t"})

        @contextlib.contextmanager
        def tool_call(name, tool_type, **kwargs):
            record = {"name": name, "type": tool_type, **kwargs}
            events["tools"].append(record)
            yield SimpleNamespace(result=lambda value: record.update(result=value))

        agenthack.BaseAgent = BaseAgent
        agenthack.state = span
        agenthack.turn = span
        agenthack.tool_call = tool_call
        agenthack.recovery = lambda strategy, count=1: events["recovery"].append(strategy)
        with patch.dict(sys.modules, {"agenthack": agenthack}):
            sys.modules.pop("src.my_agent", None)
            return importlib.import_module("src.my_agent")

    def _run(self, replies, task_input=None, delay=0.0, error=None, environment=None):
        events = {"requests": [], "tools": [], "recovery": []}
        module = self._load_agent_module(events)
        pending = list(replies)

        class Completions:
            async def create(_self, **kwargs):
                events["requests"].append(deepcopy(kwargs))
                if delay:
                    await asyncio.sleep(delay)
                if error is not None:
                    raise error
                content = pending.pop(0) if pending else '{"analysis":"done","candidates":[]}'
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                )

        class Client:
            def __init__(_self, **_kwargs):
                _self.chat = SimpleNamespace(completions=Completions())

            async def __aenter__(_self):
                return _self

            async def __aexit__(_self, *_args):
                return None

        env = {"PLATFORM_LLM_URL": "http://model.test/v1", "PLATFORM_API_KEY": "k", "AGENT_MODEL_NAME": "m"}
        env.update(environment or {})
        with patch.dict(os.environ, env, clear=False), patch.object(module.openai, "AsyncOpenAI", Client):
            with self.assertLogs("src.my_agent", level="INFO") as logs:
                result = asyncio.run(module.AgentCCAgent().solve(task_input or two_plane_input("all-reduce")))
        line = next(item for item in logs.output if "agentcc.audit" in item)
        audit = json.loads(line[line.index("{"):])
        return result, audit, events

    def test_model_proposals_drive_the_returned_schedule(self):
        first = json.dumps({"analysis": "try both", "candidates": [
            {"algorithm": "doubling", "split": 1, "path_width": 1, "order": "auto"},
            {"algorithm": "rs_ag", "split": 8, "path_width": 1, "rs_dims": 3, "order": "auto"},
        ]})
        second = json.dumps({"analysis": "widen", "candidates": [
            {"algorithm": "rs_ag", "split": 8, "path_width": 2, "rs_dims": 3, "order": "auto"},
        ]})
        result, audit, events = self._run([first, second])
        answer = result["answer"]
        self.assertEqual(audit["selection"], "model_candidate")
        self.assertEqual(audit["selected_config"]["algorithm"], "rs_ag")
        self.assertEqual(audit["selected_config"]["path_width"], 2)
        self.assertEqual(answer["split"], 8)
        self.assertEqual(len(events["requests"]), 2)
        self.assertEqual(len(events["tools"]), 3)
        self.assertTrue(all(tool["name"] == "evaluate_schedule" for tool in events["tools"]))
        feedback = events["requests"][1]["messages"][-1]["content"]
        self.assertIn("estimate_us", feedback)
        self.assertIn("limiter", feedback)
        self.assertIn("metrics", answer)
        data = two_plane_input("all-reduce")
        validate_answer(data["task"], data["topology"], answer)

    def test_prompt_contains_current_measurements(self):
        data = two_plane_input("all-gather")
        _, _, events = self._run(['{"candidates":[]}'], task_input=data)
        prompt = events["requests"][0]["messages"][-1]["content"]
        self.assertIn("plane-A-ingress", prompt)
        self.assertIn("us_per_MiB_multipath", prompt)
        changed = deepcopy(data)
        changed["topology"]["resources"][0]["capacity"] = "1GB/s"
        _, _, events_changed = self._run(['{"candidates":[]}'], task_input=changed)
        self.assertNotEqual(prompt, events_changed["requests"][0]["messages"][-1]["content"])

    def test_invalid_model_proposals_are_reported_and_fall_back(self):
        reply = json.dumps({"candidates": [
            {"algorithm": "tree", "split": 3, "path_width": 1},
            {"algorithm": "nonsense", "split": 1, "path_width": 1},
        ]})
        result, audit, events = self._run([reply, '{"candidates":[]}'])
        statuses = [row["status"] for row in audit["model_rounds"][0]["tool_results"]]
        self.assertEqual(statuses, ["rejected", "rejected"])
        self.assertEqual(audit["selection"], "agent_reference")
        self.assertIn("fallback_to_agent_reference", events["recovery"])
        data = two_plane_input("all-reduce")
        validate_answer(data["task"], data["topology"], result["answer"])

    def test_model_failure_and_timeout_return_valid_reference(self):
        error = openai.APIConnectionError(request=httpx.Request("POST", "http://model.test/v1/chat/completions"))
        for kwargs, reason in [
            ({"error": error}, "model_api_error:APIConnectionError"),
            ({"delay": 0.5, "environment": {"AGENT_LLM_BUDGET_SECONDS": "0.2"}}, "model_timeout"),
            ({"replies": ["not json at all"]}, "model_json_error"),
        ]:
            replies = kwargs.pop("replies", [])
            result, audit, _ = self._run(replies, **kwargs)
            with self.subTest(reason=reason):
                self.assertEqual(audit["model_rounds"][0]["error"], reason)
                self.assertEqual(audit["selection"], "agent_reference")
                data = two_plane_input("all-reduce")
                validate_answer(data["task"], data["topology"], result["answer"])

    def test_reply_with_reasoning_and_fences_is_parsed(self):
        module = self._load_agent_module({"tools": [], "recovery": []})
        content = '<think>long reasoning {not json}</think>\n```json\n{"analysis":"a","candidates":[{"algorithm":"doubling"}]}\n```'
        proposals, analysis, error = module.parse_proposals(content)
        self.assertIsNone(error)
        self.assertEqual(analysis, "a")
        self.assertEqual(proposals, [{"algorithm": "doubling"}])


if __name__ == "__main__":
    unittest.main()
