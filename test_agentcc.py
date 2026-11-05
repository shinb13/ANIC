import json
import asyncio
import contextlib
from copy import deepcopy
import importlib
from itertools import permutations
import os
from pathlib import Path
import sys
import types
import unittest
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

from agentcc_solver import (
    AgentCCError,
    _candidate_splits,
    _routes,
    _topology_network,
    balance_schedule_routes,
    doubling_round_profiles,
    estimate_schedule_time_us,
    optimal_doubling_order,
    solve_input,
    solve_request,
    validate_answer,
)


ROOT = Path(__file__).parent
PUBLIC = ROOT / "agentcc_zip_tmp" / "examples"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic_input(operator="all-gather", heterogeneous=True):
    ranks = [f"r{index}" for index in range(8)]
    edges = []
    cross_edges = []
    bandwidths = {1: 100e9, 2: 10e9, 4: 100e9}
    for source in range(8):
        for destination in range(8):
            if source == destination:
                continue
            bandwidth = (
                bandwidths.get(source ^ destination, 5e9)
                * (0.5 if source > destination else 1)
                if heterogeneous else 1e9
            )
            edge = {"src": ranks[source], "dst": ranks[destination], "bw": bandwidth}
            edges.append(edge)
            if source ^ destination == 4:
                cross_edges.append({"src": edge["src"], "dst": edge["dst"]})
    return {
        "task": {
            "task_id": "synthetic-current-input",
            "operator": operator,
            "ranks": ranks,
            "data_size": {"per_rank_bytes": 65536},
            "split_policy": {"mode": "search_allowed", "candidates": [1], "alignment_bytes": 4096},
            "split": 1,
            "max_nonempty_rounds": 64,
            "allow_copy": True,
            "allow_detour": True,
            "communication_constraints": {"send_only_possessed_chunks": True},
        },
        "topology": {
            "ranks": ranks,
            "switches": [],
            "edges": edges,
            "resources": [
                {"id": "cross", "capacity": 1e9, "edges": cross_edges},
                {"id": "overlap", "capacity": 2e9, "edges": cross_edges},
            ] if heterogeneous else [],
            "communication_startup_time_us": 1,
            "switch_processing_overhead_us": 0.3,
        },
    }


def multipath_input():
    task_input = synthetic_input()
    ranks = ["r0", "r1"]
    task_input["task"]["ranks"] = ranks
    topology = task_input["topology"]
    topology["ranks"] = ranks
    topology["switches"] = ["S0", "S1"]
    topology["edges"] = [
        {"src": rank, "dst": switch, "bw": bandwidth, "bidirectional": True}
        for rank in ranks
        for switch, bandwidth in [("S0", 100000000), ("S1", 4000000 / 3)]
    ]
    topology["resources"] = [{
        "id": "shared_ingress",
        "capacity": 1000000,
        "edges": [{"src": rank, "dst": "S0"} for rank in ranks],
    }]
    return task_input


class AgentCCBaselineTests(unittest.TestCase):
    def test_shared_resource_examples(self):
        base = PUBLIC / "shared-resource"
        topology = read_json(base / "topology.json")
        for operator in ("all-gather", "all-reduce"):
            task = read_json(base / f"{operator}-task.json")
            response = solve_request(task, topology)
            self.assertEqual(response["status"], "success")
            answer = response["output"]["answer"]
            self.assertEqual(answer["operator"], operator)
            self.assertEqual(answer["split"], 1)
            self.assertAlmostEqual(estimate_schedule_time_us(task, topology, answer), 66.836, places=3)

    def test_all_public_256_rank_examples(self):
        families = (
            "uneven_pods",
            "directed_ring",
            "sparse_fabric",
            "cloudmatrix",
            "fullmesh_clos",
            "mesh2d",
            "nvlink_like",
        )
        for family in families:
            for short_name, operator in (("ag", "all-gather"), ("ar", "all-reduce")):
                base = PUBLIC / "256-rank" / family / short_name
                task = read_json(base / "task.json")
                topology = read_json(base / "topology.json")
                with self.subTest(family=family, operator=operator):
                    network = _topology_network(topology)
                    routes = _routes(task, topology, min(_candidate_splits(task)), network=network)
                    self.assertEqual(len(routes), 256 * 8)
                    task_input = {"task": task, "topology": topology}
                    baseline = solve_input(task_input, network=network, routes=routes)["answer"]
                    algorithm = "ordered_doubling" if operator == "all-gather" else "tree"
                    answer = solve_input(
                        task_input, network=network, routes=routes, algorithm=algorithm,
                    )["answer"]
                    self.assertEqual(answer["operator"], operator)
                    self.assertEqual(answer["split"], 1)
                    self.assertEqual(baseline["round_count"], 8)
                    self.assertEqual(answer["round_count"], 8 if operator == "all-gather" else 16)
                    self.assertLess(
                        estimate_schedule_time_us(task, topology, answer, network),
                        estimate_schedule_time_us(task, topology, baseline, network),
                    )
                    before_rounds = [
                        estimate_schedule_time_us(task, topology, {"split": answer["split"], "rounds": [row]}, network)
                        for row in answer["rounds"]
                    ]
                    balance_schedule_routes(task, topology, answer, network, deadline=perf_counter() + 0.25)
                    validate_answer(task, topology, answer, network)
                    after_rounds = [
                        estimate_schedule_time_us(task, topology, {"split": answer["split"], "rounds": [row]}, network)
                        for row in answer["rounds"]
                    ]
                    for before, after in zip(before_rounds, after_rounds):
                        self.assertLessEqual(after, before)


class AgentCCOptimizationTests(unittest.TestCase):
    def test_only_needed_xor_routes_are_required_for_a_complete_collective(self):
        task_input = synthetic_input()
        task_input["task"]["ranks"] = [f"r{index}" for index in range(4)]
        topology = task_input["topology"]
        topology["ranks"] = task_input["task"]["ranks"]
        topology["resources"] = []
        topology["edges"] = [
            edge for edge in topology["edges"]
            if edge["src"] in topology["ranks"] and edge["dst"] in topology["ranks"]
            and int(edge["src"][1:]) ^ int(edge["dst"][1:]) in (1, 2)
        ]
        routes = _routes(task_input["task"], topology, 1)
        self.assertEqual(len(routes), 8)
        self.assertNotIn(("r0", "r3"), routes)
        for operator, algorithm in [("all-gather", "ordered_doubling"), ("all-reduce", "tree")]:
            task_input["task"]["operator"] = operator
            solve_input(task_input, algorithm=algorithm, routes=routes)

    def test_load_balancing_accounts_for_repeated_and_overlapping_resources(self):
        task_input = multipath_input()
        task, topology = task_input["task"], task_input["topology"]
        member_edges = [
            {"src": rank, "dst": "S0", "bidirectional": True} for rank in task["ranks"]
        ]
        topology["resources"][0]["edges"] = member_edges
        topology["resources"].append({"id": "overlap", "capacity": 2000000, "edges": member_edges})
        network = _topology_network(topology)
        answer = solve_input(task_input, network=network)["answer"]
        for transfer in answer["rounds"][0]["transfers"]:
            transfer["path"] = [transfer["src"], "S0", transfer["dst"]]
        before = estimate_schedule_time_us(task, topology, answer, network)
        self.assertAlmostEqual(before, 4 * 65536 + 1.3)
        stats = balance_schedule_routes(task, topology, answer, network, deadline=perf_counter() + 1)
        validate_answer(task, topology, answer, network)
        after = estimate_schedule_time_us(task, topology, answer, network)
        self.assertLess(after, before)
        self.assertGreater(stats["changed_paths"], 0)
        self.assertEqual(stats["initial_estimate_us"], before)
        self.assertEqual(stats["final_estimate_us"], after)

    def test_balancing_rejects_lower_utilization_when_path_latency_increases(self):
        task_input = multipath_input()
        task, topology = task_input["task"], task_input["topology"]
        topology["resources"][0]["edges"] = [{"src": "r0", "dst": "S0"}]
        for edge in topology["edges"]:
            if edge["dst"] == "S1":
                edge["bw"] = 1500000
        network = _topology_network(topology)
        answer = solve_input(task_input, network=network)["answer"]
        original = deepcopy(answer)
        stats = balance_schedule_routes(task, topology, answer, network, deadline=perf_counter() + 1)
        self.assertEqual(answer, original)
        self.assertEqual(stats["changed_paths"], 0)
        self.assertEqual(stats["initial_estimate_us"], stats["final_estimate_us"])

    def test_balancing_expired_deadline_preserves_valid_answer(self):
        task_input = multipath_input()
        task, topology = task_input["task"], task_input["topology"]
        network = _topology_network(topology)
        answer = solve_input(task_input, network=network)["answer"]
        original = deepcopy(answer)
        stats = balance_schedule_routes(task, topology, answer, network, deadline=perf_counter() - 1)
        self.assertEqual(answer, original)
        self.assertEqual(stats["path_searches"], 0)
        self.assertEqual(stats["stop_reason"], "deadline")

    def test_dimension_order_matches_exhaustive_search_and_estimator(self):
        task_input = synthetic_input()
        task, topology = task_input["task"], task_input["topology"]
        network = _topology_network(topology)
        routes = _routes(task, topology, 1, network=network)
        profiles = doubling_round_profiles(task, topology, routes, network)
        order = optimal_doubling_order(profiles)
        by_step = {profile["step"]: profile for profile in profiles}
        expected = min(
            sum(by_step[step]["times_us"][position] for position, step in enumerate(candidate))
            for candidate in permutations(by_step)
        )
        answer = solve_input(
            task_input, algorithm="ordered_doubling", network=network, routes=routes,
        )["answer"]
        self.assertEqual(order[0], 4)
        self.assertAlmostEqual(
            estimate_schedule_time_us(task, topology, answer, network), expected,
        )

    def test_resource_capacity_change_recomputes_order_without_changing_instance_name(self):
        orders = []
        original = synthetic_input()
        for task_input in [original, deepcopy(original)]:
            if orders:
                for resource in task_input["topology"]["resources"]:
                    resource["capacity"] *= 1000
            task, topology = task_input["task"], task_input["topology"]
            network = _topology_network(topology)
            routes = _routes(task, topology, 1, network=network)
            orders.append(optimal_doubling_order(
                doubling_round_profiles(task, topology, routes, network),
            ))
        self.assertEqual(orders[0][0], 4)
        self.assertEqual(orders[1][0], 2)
        self.assertNotEqual(orders[0], orders[1])
        self.assertEqual(original["topology"]["resources"][0]["capacity"], 1e9)

    def test_alignment_and_tree_with_multiple_slots(self):
        for operator, split, size, algorithm in [
            ("all-gather", 32, 1048576, "ordered_doubling"),
            ("all-reduce", 2, 65536, "tree"),
        ]:
            task_input = synthetic_input(operator)
            task = task_input["task"]
            task["data_size"]["per_rank_bytes"] = size
            task["split_policy"]["candidates"] = [split]
            answer = solve_input(task_input, algorithm=algorithm)["answer"]
            self.assertEqual(answer["split"], split)
            self.assertEqual((size // split) % 4096, 0)
            for invalid_size, alignment in [(size + 1, 4096), (size, size)]:
                invalid = deepcopy(task_input)
                invalid["task"]["data_size"]["per_rank_bytes"] = invalid_size
                invalid["task"]["split_policy"]["alignment_bytes"] = alignment
                with self.assertRaisesRegex(AgentCCError, "no split candidate"):
                    solve_input(invalid, algorithm=algorithm)

    def test_tree_respects_round_limit_and_rejects_repeated_contributors(self):
        task_input = synthetic_input("all-reduce")
        task, topology = task_input["task"], task_input["topology"]
        answer = solve_input(task_input, algorithm="tree")["answer"]
        invalid = deepcopy(answer)
        expression = invalid["rounds"][0]["transfers"][0]["chunk_ranges"][0]
        invalid["rounds"][0]["transfers"][0]["chunk_ranges"][0] = expression + "+" + expression
        with self.assertRaisesRegex(AgentCCError, "unique contributors"):
            validate_answer(task, topology, invalid)
        task["max_nonempty_rounds"] = 3
        with self.assertRaisesRegex(AgentCCError, "exceeds max_nonempty_rounds"):
            solve_input(task_input, algorithm="tree")


class AgentCCModelTests(unittest.TestCase):
    @staticmethod
    def _load_agent_module():
        agenthack = types.ModuleType("agenthack")

        class BaseAgent:
            pass

        @contextlib.contextmanager
        def state(_name):
            yield

        @contextlib.contextmanager
        def turn(**_kwargs):
            yield

        agenthack.BaseAgent = BaseAgent
        agenthack.state = state
        agenthack.turn = turn
        with patch.dict(sys.modules, {"agenthack": agenthack}):
            sys.modules.pop("src.my_agent", None)
            return importlib.import_module("src.my_agent")

    @staticmethod
    def _metrics(logs):
        line = next(line for line in logs.output if "agentcc.solve_metrics" in line)
        return json.loads(line[line.index("{") :])

    def _run_with_model(self, content=None, error=None, task_input=None, environment=None, delay=0):
        module = self._load_agent_module()
        events = {"requests": [], "closed": False, "request_finished": False}

        class Completions:
            async def create(_self, **kwargs):
                events["requests"].append(kwargs)
                try:
                    if delay:
                        await asyncio.sleep(delay)
                    if error is not None:
                        raise error
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                    )
                finally:
                    events["request_finished"] = True

        class Client:
            def __init__(_self, **_kwargs):
                _self.chat = SimpleNamespace(completions=Completions())

            async def __aenter__(_self):
                return _self

            async def __aexit__(_self, *_args):
                events["closed"] = True

        configured_environment = {
            "PLATFORM_LLM_URL": "http://model.test/v1",
            "PLATFORM_API_KEY": "test-key",
            "AGENT_MODEL_NAME": "test-model",
        }
        configured_environment.update(environment or {})
        with patch.dict(os.environ, configured_environment, clear=False):
            with patch.object(module.openai, "AsyncOpenAI", Client):
                with patch("agentcc_solver._routes", wraps=_routes) as routing:
                    with patch.object(module, "_routes", routing):
                        with self.assertLogs("src.my_agent", level="INFO") as logs:
                            result = asyncio.run(
                                module.AgentCCAgent().solve(
                                    synthetic_input() if task_input is None else task_input
                                )
                            )
                    events["routing_calls"] = routing.call_count
        events["logs"] = logs.output
        return result, self._metrics(logs), events

    def test_model_failure_returns_validated_baseline(self):
        result, metrics, events = self._run_with_model(error=openai.APIConnectionError(
            request=httpx.Request("POST", "http://model.test/v1/chat/completions"),
        ))
        self.assertEqual(result["answer"]["split"], 1)
        self.assertEqual(metrics["selected_policy"], "baseline")
        self.assertEqual(metrics["selected_algorithm"], "recursive_doubling")
        self.assertEqual(metrics["fallback_reason"], "model_api_error:APIConnectionError")
        self.assertTrue(events["closed"])

    def test_model_choice_changes_actual_schedule_and_reuses_baseline_routes(self):
        baseline, baseline_metrics, _ = self._run_with_model(
            content='{"algorithm":"recursive_doubling","policy":"baseline"}',
        )
        result, metrics, events = self._run_with_model(
            content='{"algorithm":"ordered_doubling","policy":"baseline"}',
        )
        self.assertEqual(result["answer"]["split"], 1)
        self.assertEqual(metrics["selected_algorithm"], "ordered_doubling")
        self.assertEqual(baseline_metrics["selected_algorithm"], "recursive_doubling")
        self.assertNotEqual(result["answer"]["rounds"], baseline["answer"]["rounds"])
        self.assertEqual(metrics["candidate_validation"], "passed")
        self.assertEqual(events["routing_calls"], 1)
        self.assertLess(metrics["candidate_estimate_us"], metrics["baseline_estimate_us"])
        self.assertIsNone(metrics["fallback_reason"])

    def test_worse_tree_candidate_is_rejected_using_real_estimates(self):
        result, metrics, _ = self._run_with_model(
            content='{"algorithm":"tree","policy":"baseline"}',
            task_input=synthetic_input("all-reduce", heterogeneous=False),
        )
        self.assertEqual(result["answer"]["split"], 1)
        self.assertEqual(result["answer"]["round_count"], 3)
        self.assertEqual(metrics["selected_algorithm"], "recursive_doubling")
        self.assertGreater(metrics["candidate_estimate_us"], metrics["baseline_estimate_us"])
        self.assertEqual(metrics["fallback_reason"], "candidate_rejected_estimate_not_lower")

    def test_model_route_policy_rebuilds_paths_and_improves_shared_resource_load(self):
        task_input = multipath_input()
        result, metrics, events = self._run_with_model(
            content='{"algorithm":"ordered_doubling","policy":"resource_first"}',
            task_input=task_input,
        )
        self.assertEqual(result["answer"]["rounds"][0]["transfers"][0]["path"], ["r0", "S1", "r1"])
        self.assertEqual(metrics["selected_policy"], "resource_first")
        self.assertEqual(events["routing_calls"], 2)
        self.assertLess(metrics["candidate_estimate_us"], metrics["baseline_estimate_us"])
        self.assertIsNone(metrics["fallback_reason"])

    def test_model_balancing_choice_changes_paths_and_reacts_to_current_capacity(self):
        original = multipath_input()
        baseline, _, _ = self._run_with_model(
            content='{"algorithm":"recursive_doubling","policy":"baseline"}', task_input=original,
        )
        result, metrics, events = self._run_with_model(
            content='{"algorithm":"recursive_doubling","policy":"load_balanced"}', task_input=original,
        )
        self.assertEqual(metrics["selected_policy"], "load_balanced")
        self.assertGreater(metrics["route_balancing"]["changed_paths"], 0)
        self.assertLess(metrics["candidate_estimate_us"], metrics["baseline_estimate_us"])
        self.assertEqual(events["routing_calls"], 1)
        self.assertNotEqual(baseline["answer"]["rounds"], result["answer"]["rounds"])
        changed = deepcopy(original)
        changed["topology"]["resources"][0]["capacity"] = 100000000
        changed_result, changed_metrics, _ = self._run_with_model(
            content='{"algorithm":"recursive_doubling","policy":"load_balanced"}', task_input=changed,
        )
        self.assertEqual(changed["task"]["task_id"], original["task"]["task_id"])
        self.assertNotEqual(changed_metrics["input_sha256"], metrics["input_sha256"])
        self.assertEqual(changed_metrics["route_balancing"]["changed_paths"], 0)
        self.assertNotEqual(changed_result["answer"]["rounds"], result["answer"]["rounds"])

    def test_balancing_is_not_offered_when_detours_are_disallowed(self):
        task_input = multipath_input()
        task_input["task"]["allow_detour"] = False
        _, metrics, events = self._run_with_model(
            content='{"algorithm":"recursive_doubling","policy":"load_balanced"}', task_input=task_input,
        )
        data = json.loads(events["requests"][0]["messages"][0]["content"].split("\n", 1)[1])
        self.assertNotIn("load_balanced", data["allowed_route_policies"])
        self.assertEqual(metrics["fallback_reason"], "model_unknown_policy")
        self.assertIsNone(metrics["route_balancing"])

    def test_invalid_external_model_plans_are_rejected(self):
        for content, reason in [
            ("not JSON", "model_json_error"),
            ('["tree"]', "model_plan_fields_error"),
            ('{"policy":"baseline"}', "model_plan_fields_error"),
            ('{"algorithm":"tree","policy":"baseline"}', "model_unknown_algorithm"),
            ('{"algorithm":"ordered_doubling","policy":[]}', "model_unknown_policy"),
        ]:
            with self.subTest(content=content):
                _, metrics, _ = self._run_with_model(content=content)
                self.assertEqual(metrics["selected_algorithm"], "recursive_doubling")
                self.assertEqual(metrics["fallback_reason"], reason)

    def test_model_timeout_cancels_request_and_closes_client(self):
        started = perf_counter()
        _, metrics, events = self._run_with_model(
            content='{"algorithm":"ordered_doubling","policy":"baseline"}',
            delay=1,
            environment={"AGENT_LLM_BUDGET_SECONDS": "0.01"},
        )
        self.assertLess(perf_counter() - started, 0.5)
        self.assertEqual(metrics["fallback_reason"], "model_timeout")
        self.assertTrue(events["request_finished"])
        self.assertTrue(events["closed"])

    def test_prompt_contains_actual_constraints_and_resource_memberships(self):
        original = synthetic_input()
        changed = deepcopy(original)
        for resource in changed["topology"]["resources"]:
            resource["capacity"] *= 1000
        inputs = []
        for task_input in [original, changed]:
            _, metrics, events = self._run_with_model(
                content='{"algorithm":"recursive_doubling","policy":"baseline"}',
                task_input=task_input,
            )
            line = next(line for line in events["logs"] if "agentcc.model_input " in line)
            record = json.loads(line[line.index("{"):])
            actual_prompt = events["requests"][0]["messages"][0]["content"]
            self.assertEqual(record["prompt"], actual_prompt)
            self.assertEqual(record["input_sha256"], metrics["input_sha256"])
            data = json.loads(actual_prompt.split("\n", 1)[1])
            self.assertEqual(data["task"]["split_policy"], task_input["task"]["split_policy"])
            self.assertEqual(data["task"]["max_nonempty_rounds"], 64)
            resources = data["topology"]["resource_capacities_bytes_per_second"]
            self.assertEqual(resources["cross"], task_input["topology"]["resources"][0]["capacity"])
            for profile in data["current_dimension_evidence"]:
                edge = profile["bottleneck_edge"]
                declared = next(
                    item for item in task_input["topology"]["edges"]
                    if (item["src"], item["dst"]) == (edge["src"], edge["dst"])
                )
                self.assertEqual(edge["bandwidth_bytes_per_second"], declared["bw"])
                shared = profile["bottleneck_resource"]
                if shared is not None:
                    self.assertTrue(shared["used_member_examples"])
            inputs.append(record)
        self.assertNotEqual(inputs[0]["input_sha256"], inputs[1]["input_sha256"])
        self.assertNotEqual(inputs[0]["prompt"], inputs[1]["prompt"])

    def test_exhausted_solve_budget_returns_baseline_without_model_call(self):
        _, metrics, events = self._run_with_model(
            environment={"AGENT_SOLVE_BUDGET_SECONDS": "0"},
        )
        self.assertEqual(metrics["fallback_reason"], "solve_budget_exhausted")
        self.assertEqual(events["requests"], [])

    def test_tree_is_not_offered_when_round_budget_is_too_small(self):
        task_input = synthetic_input("all-reduce")
        task_input["task"]["max_nonempty_rounds"] = 3
        _, metrics, events = self._run_with_model(
            content='{"algorithm":"tree","policy":"baseline"}',
            task_input=task_input,
        )
        data = json.loads(events["requests"][0]["messages"][0]["content"].split("\n", 1)[1])
        self.assertEqual(
            [item["algorithm"] for item in data["candidates_on_baseline_routes"]],
            ["recursive_doubling"],
        )
        self.assertEqual(metrics["fallback_reason"], "model_unknown_algorithm")

if __name__ == "__main__":
    unittest.main()
