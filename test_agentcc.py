import asyncio
import contextlib
import importlib
import json
import os
import sys
import threading
import time
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
    build_dual_tree,
    estimate_schedule_time_us,
    reference_config,
    prepare_dual_tree,
    replay_estimate,
    refine_schedule_routes,
    solve_request,
    split_path_bound_transfers,
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

    def test_integer_prediction_matches_replay_with_per_round_widths(self):
        for operator, algorithm in (("all-gather", "doubling"), ("all-reduce", "doubling"),
                                    ("all-reduce", "rs_ag"), ("all-reduce", "tree")):
            data = two_plane_input(operator)
            for notation in (False, True):
                planner = Planner(data["task"], data["topology"], sum_notation=notation)
                for split in planner.splits:
                    if algorithm == "rs_ag" and split == 1:
                        continue
                    config = planner.normalize_config({"algorithm": algorithm, "split": split, "path_width": 2})
                    config["widths"] = [1 + index % 2 for index in range(planner.round_count(config))]
                    predicted = planner.predict(config)
                    # Prediction must not need the expensive contributor expressions.
                    with patch.object(planner, "_expression_cache", side_effect=AssertionError("built expressions")):
                        self.assertEqual(planner.predict(config), predicted)
                    with patch("agentcc_solver._discretize", side_effect=AssertionError("repeated rounding")):
                        answer = planner.build(config)
                    validate_answer(data["task"], data["topology"], answer)
                    actual = replay_estimate(data["task"], data["topology"], answer)["total_us"]
                    with self.subTest(op=operator, algorithm=algorithm, split=split, notation=notation):
                        self.assertAlmostEqual(predicted["us"], actual, delta=max(1e-6, actual * 1e-9))

    def test_integer_cost_counts_repeated_and_overlapping_resources(self):
        data = two_plane_input(ranks=2, candidates=(1, 2))
        topology = data["topology"]
        topology["switches"] = ["A"]
        topology["edges"] = [edge for edge in topology["edges"] if edge["dst"] == "A"]
        topology["resources"].extend([
            {"id": "both-hops", "capacity": "1GB/s", "edges": [
                {"src": rank, "dst": "A"} for rank in data["task"]["ranks"]
            ] + [{"src": "A", "dst": rank} for rank in data["task"]["ranks"]]},
            {"id": "overlap", "capacity": "2GB/s", "edges": [
                {"src": rank, "dst": "A"} for rank in data["task"]["ranks"]]},
        ])
        planner = Planner(data["task"], topology)
        plan = planner.exchange_plan(1, 1)
        for split in planner.splits:
            config = planner.normalize_config({"algorithm": "doubling", "split": split, "path_width": 1})
            answer = planner.build(config)
            actual = replay_estimate(data["task"], topology, answer)["total_us"]
            # Two messages, two member edges per message, plus 1.3 us overhead.
            expected = 1.3 + 4 * 65536 / 1e9 * 1e6
            self.assertAlmostEqual(planner.predict(config)["us"], expected)
            self.assertAlmostEqual(actual, expected)
        # The same number of units can represent different byte sizes.
        self.assertNotEqual(planner.integer_round_time_us(plan, 1, 65536),
                            planner.integer_round_time_us(plan, 1, 32768))
        self.assertIs(planner.integer_allocation(plan, 1), planner.integer_allocation(plan, 1))

    def test_unused_long_path_does_not_add_startup_overhead(self):
        data = two_plane_input(ranks=2, candidates=(1,))
        topology = data["topology"]
        topology["switches"] = ["A", "B", "C"]
        topology["edges"] = [
            {"src": "r0", "dst": "A", "bw": "10GB/s", "bidirectional": True},
            {"src": "r1", "dst": "B", "bw": "10GB/s", "bidirectional": True},
            {"src": "A", "dst": "B", "bw": "10GB/s", "bidirectional": True},
            {"src": "A", "dst": "C", "bw": "1GB/s", "bidirectional": True},
            {"src": "C", "dst": "B", "bw": "1GB/s", "bidirectional": True},
        ]
        topology["resources"] = []
        topology["switch_processing_overhead_us"] = 100
        planner = Planner(data["task"], topology)
        config = planner.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 2})
        plan = planner.exchange_plan(1, 2)
        self.assertEqual([len(routes) for routes in plan.routes], [2, 2])
        self.assertTrue((planner.integer_allocation(plan, 1)[:, 1] == 0).all())
        answer = planner.build(config)
        actual = replay_estimate(data["task"], topology, answer)["total_us"]
        self.assertAlmostEqual(actual, 201 + 3 * 65536 / 1e10 * 1e6)
        self.assertAlmostEqual(planner.predict(config)["us"], actual)

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

    def test_sum_notation_copies_are_valid_and_smaller(self):
        data = two_plane_input("all-reduce")
        task, topology = data["task"], data["topology"]
        sizes = {}
        for notation in (False, True):
            planner = Planner(task, topology, sum_notation=notation)
            for raw in ({"algorithm": "rs_ag", "split": 8, "path_width": 2}, {"algorithm": "tree", "split": 2, "path_width": 2}):
                answer = planner.build(planner.normalize_config(raw))
                validate_answer(task, topology, answer)
                copies = [item for r in answer["rounds"] for t in r["transfers"] if t["action"] == "copy"
                          for item in t["chunk_ranges"]]
                reduces = [item for r in answer["rounds"] for t in r["transfers"] if t["action"] == "reduce"
                           for item in t["chunk_ranges"]]
                self.assertEqual(all(item.endswith("_sum") for item in copies), notation)
                self.assertFalse(any(item.endswith("_sum") for item in reduces))
                sizes[(notation, raw["algorithm"])] = len(json.dumps(answer))
                same = Planner(task, topology, sum_notation=not notation).build(planner.normalize_config(raw))
                self.assertAlmostEqual(estimate_schedule_time_us(task, topology, answer),
                                       estimate_schedule_time_us(task, topology, same))
        self.assertLess(sizes[(True, "rs_ag")], sizes[(False, "rs_ag")])
        # A copy of an incomplete slot written as c{slot}_sum is still rejected.
        planner = Planner(task, topology, sum_notation=True)
        answer = planner.build(planner.normalize_config({"algorithm": "rs_ag", "split": 8, "path_width": 1}))
        answer["rounds"][0]["transfers"][0]["action"] = "copy"
        answer["rounds"][0]["transfers"][0]["chunk_ranges"] = [
            f"c{item.split('_c')[-1]}_sum" if "+" not in item else item
            for item in answer["rounds"][0]["transfers"][0]["chunk_ranges"]
        ]
        with self.assertRaises(AgentCCError):
            validate_answer(task, topology, answer)

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


STRATEGY_REPLY = """Tree is cheap in bytes; compare with rs_ag.
```python
def strategy(api):
    rows = api.sweep(limit=3)
    for row in rows:
        api.measure(row["cfg"])
    api.log(len(rows))
    return [row["cfg"] for row in rows]
```"""

REVISION_REPLY = """Widen the last rounds.
```python
def strategy(api):
    n = 2 * len(api.steps) if api.operator == "all-reduce" else len(api.steps)
    c = {"algorithm": "tree" if api.operator == "all-reduce" else "doubling", "split": api.splits[0],
         "path_width": 2, "widths": [1] * (n // 2) + [2] * (n - n // 2)}
    api.measure(c)
    return [c]
```"""


def strategy_code(reply: str) -> str:
    return reply.split("```python\n")[1].split("```")[0]


class RankMappingTests(unittest.TestCase):
    def test_allgather_mapping_candidates_and_mapped_build(self):
        task, topology = public_case("uneven_pods", "ag")
        planner = Planner(task, topology, sum_notation=True)
        config = planner.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 2})
        candidates = planner.mapping_candidates(config)
        self.assertEqual({item["mapping"] for item in candidates}, {"locality", "resource"})
        best = max((item for item in candidates if item["reachable"]), key=lambda item: item["proxy_gain"])
        self.assertGreater(best["proxy_gain"], 0)
        mapped = planner.with_rank_order(best["rank_order"])
        answer = mapped.build(mapped.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 2}))
        validate_answer(task, topology, answer, planner.network.raw)
        baseline = replay_estimate(task, topology, planner.build(config), planner.network.raw)["total_us"]
        self.assertLess(replay_estimate(task, topology, answer, planner.network.raw)["total_us"], baseline)

    def test_logical_order_preserves_physical_data_and_round_semantics(self):
        order = [0, 2, 4, 7, 1, 6, 3, 5]
        configurations = [
            ("all-gather", {"algorithm": "doubling", "split": 8, "path_width": 2}),
            ("all-reduce", {"algorithm": "doubling", "split": 4, "path_width": 2}),
            ("all-reduce", {"algorithm": "tree", "split": 4, "path_width": 2}),
            *[("all-reduce", {"algorithm": "rs_ag", "split": 8, "path_width": 2, "rs_dims": h})
              for h in (1, 2, 3)],
        ]
        for operator, raw in configurations:
            for short in (False, True):
                with self.subTest(operator=operator, config=raw, sum_notation=short):
                    data = two_plane_input(operator)
                    original = deepcopy(data)
                    planner = Planner(data["task"], data["topology"], sum_notation=short)
                    identity = planner.normalize_config(raw)
                    baseline = planner.build(identity)
                    signature = planner.schedule_key(identity)
                    predicted = planner.predict(identity)
                    mapped = planner.with_rank_order(order)
                    config = mapped.normalize_config(raw)
                    answer = mapped.build(config)
                    validate_answer(data["task"], data["topology"], answer)
                    actual = replay_estimate(data["task"], data["topology"], answer)
                    self.assertAlmostEqual(mapped.predict(config)["us"], actual["total_us"], places=9)
                    self.assertEqual(answer["chunks"], baseline["chunks"])
                    self.assertEqual(list(answer["initial_possession"]), data["task"]["ranks"])
                    self.assertEqual(answer["initial_possession"], baseline["initial_possession"])
                    self.assertEqual(answer["round_count"], baseline["round_count"])
                    for rd in answer["rounds"]:
                        for transfer in rd["transfers"]:
                            self.assertEqual(transfer["path"][0], transfer["src"])
                            self.assertEqual(transfer["path"][-1], transfer["dst"])
                    self.assertEqual(planner.build(identity), baseline)
                    self.assertEqual(planner.predict(identity), predicted)
                    self.assertEqual(planner.schedule_key(identity), signature)
                    self.assertNotEqual(mapped.schedule_key(config), signature)
                    self.assertIs(mapped.network, planner.network)
                    self.assertIsNot(mapped._plans, planner._plans)
                    self.assertIsNot(mapped._allocations, planner._allocations)
                    self.assertEqual(data, original)

    def test_orders_react_to_actual_bandwidth_direction_and_resources(self):
        data = two_plane_input("all-reduce")
        data["topology"]["edges"].append({"src": "r0", "dst": "r4", "bw": "100GB/s", "bidirectional": True})
        data["topology"]["resources"].append({"id": "direct", "capacity": "100GB/s",
            "edges": [{"src": "r0", "dst": "r4", "bidirectional": True}]})
        original = Planner(data["task"], data["topology"]).network.rank_orders()
        for name, info in original.items():
            self.assertEqual(sorted(info["rank_order"]), list(range(8)))
            self.assertEqual(info["rank_order"][:2], [0, 4])
        bandwidth = deepcopy(data)
        bandwidth["topology"]["edges"][-1]["bw"] = "1MB/s"
        modified = Planner(bandwidth["task"], bandwidth["topology"]).network.rank_orders()
        self.assertEqual(modified["locality"]["rank_order"][:2], [0, 1])
        resource = deepcopy(data)
        resource["topology"]["resources"][-1]["capacity"] = "1MB/s"
        modified = Planner(resource["task"], resource["topology"]).network.rank_orders()
        self.assertEqual(modified["locality"]["rank_order"][:2], [0, 4])
        self.assertEqual(modified["resource"]["rank_order"][:2], [0, 1])
        direction = deepcopy(data)
        direction["topology"]["edges"][-1]["bidirectional"] = False
        direction["topology"]["resources"][-1]["edges"][0]["bidirectional"] = False
        modified = Planner(direction["task"], direction["topology"]).network.rank_orders()
        self.assertEqual(modified["locality"]["rank_order"][:2], [0, 1])
        self.assertEqual(direction["task"]["instance_id"], data["task"]["instance_id"])

    def test_mapped_refinement_uses_the_same_physical_routes(self):
        data = two_plane_input("all-reduce")
        planner = Planner(data["task"], data["topology"], sum_notation=True).with_rank_order([0, 2, 4, 7, 1, 6, 3, 5])
        config = planner.normalize_config({"algorithm": "rs_ag", "split": 8, "path_width": 2})
        answer = planner.build(config)
        before = deepcopy(answer)
        estimate = replay_estimate(data["task"], data["topology"], answer)
        refined, measured, _ = refine_schedule_routes(planner, config, answer, estimate)
        validate_answer(data["task"], data["topology"], refined)
        self.assertLessEqual(measured["total_us"], estimate["total_us"])
        self.assertEqual(answer, before)
        self.assertEqual(planner.build(config), before)


class DualTreeTests(unittest.TestCase):
    def test_public_integer_predictions_match_valid_complete_schedules(self):
        for family in FAMILIES:
            task, topology = public_case(family, "ar")
            planner = Planner(task, topology, sum_notation=True)
            prepared = prepare_dual_tree(planner)
            for option in prepared["options"]:
                answer = build_dual_tree(planner, prepared, option["choice"])
                with self.subTest(family=family, choice=option["choice"]):
                    validate_answer(task, topology, answer, planner.network.raw)
                    actual = replay_estimate(task, topology, answer, planner.network.raw)
                    self.assertAlmostEqual(option["us"], actual["total_us"], delta=1e-8)
                    self.assertLessEqual(answer["round_count"], task["max_nonempty_rounds"])

    def test_merged_trees_charge_overlapping_resources_and_keep_slot_states_independent(self):
        data = two_plane_input("all-reduce")
        data["topology"]["resources"].extend([
            {"id": name, "capacity": "1GB/s", "edges": [
                {"src": edge["src"], "dst": edge["dst"]} for edge in data["topology"]["edges"]
            ]} for name in ("shared-uplink", "overlapping-uplink")
        ])
        for compact in (False, True):
            planner = Planner(data["task"], data["topology"], sum_notation=compact)
            planner = planner.with_rank_order([2, 7, 0, 4, 1, 6, 3, 5])
            prepared = prepare_dual_tree(planner)
            for option in prepared["options"]:
                answer = build_dual_tree(planner, prepared, option["choice"])
                validate_answer(data["task"], data["topology"], answer)
                actual = replay_estimate(data["task"], data["topology"], answer)
                self.assertAlmostEqual(option["us"], actual["total_us"], delta=1e-8)
            answer = build_dual_tree(planner, prepared, 1)
            first_round = answer["rounds"][0]
            both = replay_estimate(data["task"], data["topology"], {**answer, "rounds": [first_round]})["total_us"]
            individual = []
            for tree in (0, 1):
                transfers = [transfer for transfer in first_round["transfers"]
                             if (int(transfer["chunk_ranges"][0].rsplit("_c", 1)[1]) >= 4) == bool(tree)]
                part = {**answer, "rounds": [{"round": 1, "transfers": transfers}]}
                individual.append(replay_estimate(data["task"], data["topology"], part)["total_us"])
            self.assertGreater(both, max(individual))
            self.assertEqual(answer["chunks"], [f"{rank}_c{slot}" for rank in data["task"]["ranks"] for slot in range(8)])

    def test_constraints_and_directed_switch_only_prices(self):
        cases = [two_plane_input("all-gather"), two_plane_input("all-reduce", candidates=(1, 2)),
                 two_plane_input("all-reduce"), two_plane_input("all-reduce")]
        cases[2]["task"]["allow_copy"] = False
        cases[3]["task"]["max_nonempty_rounds"] = 3
        for data in cases:
            planner = Planner(data["task"], data["topology"])
            self.assertEqual(prepare_dual_tree(planner), {"reason": "ineligible"})
        data = two_plane_input("all-reduce", ranks=4)
        data["topology"]["switches"] = []
        data["topology"]["resources"] = []
        data["topology"]["edges"] = [
            {"src": "r0", "dst": "r1", "bw": "1GB/s"},
            {"src": "r1", "dst": "r2", "bw": "2GB/s"},
        ]
        planner = Planner(data["task"], data["topology"])
        costs = planner.network.rank_costs(planner.network.static_cost)
        self.assertLess(costs[0, 1], float("inf"))
        self.assertEqual(costs[0, 2], float("inf"))
        self.assertEqual(costs[1, 0], float("inf"))
        with self.assertRaisesRegex(AgentCCError, "reachable parent"):
            prepare_dual_tree(planner)

    def test_current_bandwidth_direction_and_resources_change_predictions_without_mutating_inputs(self):
        original = two_plane_input("all-reduce")
        original_snapshot = deepcopy(original)
        results = []
        for mutation in ("original", "bandwidth", "direction", "resource"):
            data = deepcopy(original)
            if mutation == "bandwidth":
                data["topology"]["edges"][1]["bw"] = "40GB/s"
            elif mutation == "direction":
                data["topology"]["edges"][1]["bidirectional"] = False
                data["topology"]["edges"].append({"src": "B", "dst": "r0", "bw": "1GB/s"})
            elif mutation == "resource":
                data["topology"]["resources"][0]["capacity"] = "1GB/s"
            snapshot = deepcopy(data)
            planner = Planner(data["task"], data["topology"], sum_notation=True)
            prepared = prepare_dual_tree(planner)
            results.append([option["us"] for option in prepared["options"]])
            validate_answer(data["task"], data["topology"], build_dual_tree(planner, prepared, 1))
            self.assertEqual(data, snapshot)
        for changed in results[1:]:
            self.assertNotEqual(changed, results[0])
        self.assertEqual(original, original_snapshot)


def single_switch_input(operator="all-gather", ranks=4, candidates=(1, 2, 4, 8)):
    """Every rank has one link to one switch: every transfer is path-time bound."""

    names = [f"r{index}" for index in range(ranks)]
    data = two_plane_input(operator, ranks, candidates)
    data["topology"]["switches"] = ["A"]
    data["topology"]["edges"] = [{"src": rank, "dst": "A", "bw": "10GB/s", "bidirectional": True} for rank in names]
    data["topology"]["resources"] = []
    return data


class AllGatherBlockTests(unittest.TestCase):
    def test_blocks_validate_and_match_prediction(self):
        data = two_plane_input("all-gather", ranks=8)
        planner = Planner(data["task"], data["topology"], sum_notation=True)
        for blocks in ([[1, 2, 4]], [[4], [1, 2]], [[2, 4], [1]], [[1], [2], [4]]):
            with self.subTest(blocks=blocks):
                config = planner.normalize_config({"algorithm": "doubling", "split": 2, "path_width": 2, "blocks": blocks})
                self.assertEqual(planner.round_count(config), len(blocks))
                answer = planner.build(config)
                validate_answer(data["task"], data["topology"], answer)
                self.assertEqual(answer["round_count"], len(blocks))
                actual = replay_estimate(data["task"], data["topology"], answer)["total_us"]
                self.assertAlmostEqual(planner.predict(config)["us"], actual, places=6)

    def test_bad_blocks_are_rejected(self):
        data = two_plane_input("all-gather", ranks=8)
        planner = Planner(data["task"], data["topology"])
        for blocks in ([[1, 2]], [[1, 2, 4, 4]], [[1], [1, 2, 4]], [[]], "x", [[1, 2, True]]):
            with self.subTest(blocks=blocks), self.assertRaises(AgentCCError):
                planner.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 1, "blocks": blocks})
        reduce = two_plane_input("all-reduce", ranks=8)
        planner = Planner(reduce["task"], reduce["topology"])
        with self.assertRaises(AgentCCError):
            planner.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 1, "blocks": [[1, 2, 4]]})

    def test_block_search_returns_a_partition_and_singletons_match_plain_schedule(self):
        data = two_plane_input("all-gather", ranks=8)
        planner = Planner(data["task"], data["topology"], sum_notation=True)
        config = planner.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 2})
        blocks, predicted, singles = planner.allgather_blocks(config)
        self.assertEqual([step for block in blocks for step in block], config["order"])
        self.assertLessEqual(predicted, singles + 1e-9)
        split, _ = split_path_bound_transfers(planner.network, planner.build(config), planner.data_size)
        self.assertAlmostEqual(replay_estimate(data["task"], data["topology"], split)["total_us"], singles, places=6)

    def test_public_mesh2d_allgather_blocks_are_faster(self):
        task, topology = public_case("mesh2d", "ag")
        planner = Planner(task, topology, sum_notation=True)
        config = planner.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 16})
        blocks, predicted, singles = planner.allgather_blocks(config)
        self.assertLess(predicted, 0.6 * singles)
        answer = planner.build(planner.normalize_config({**{k: config[k] for k in ("algorithm", "split", "path_width")},
                                                         "blocks": blocks}))
        split, _ = split_path_bound_transfers(planner.network, answer, planner.data_size)
        validate_answer(task, topology, split, planner.network.raw)
        self.assertAlmostEqual(replay_estimate(task, topology, split, planner.network.raw)["total_us"], predicted, places=3)


class WideBlockTests(unittest.TestCase):
    def test_block_order_inside_a_block_does_not_change_pairs(self):
        data = two_plane_input("all-gather", ranks=8)
        planner = Planner(data["task"], data["topology"])
        for blocks in ([[1, 2], [4]], [[2, 1], [4]], [[4, 1, 2]], [[2, 4, 1]]):
            with self.subTest(blocks=blocks):
                config = planner.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 2,
                                                   "blocks": blocks})
                validate_answer(data["task"], data["topology"], planner.build(config))

    def test_wide_block_plans_only_lower_the_prediction_and_match_replay(self):
        task, topology = public_case("mesh2d", "ag")
        results = {}
        for wide in (False, True):
            planner = Planner(task, topology, sum_notation=True)
            config = planner.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 16})
            blocks, predicted, singles = planner.allgather_blocks(
                config, wide_deadline=time.perf_counter() + 60 if wide else None)
            results[wide] = predicted
            blocked = planner.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 16,
                                                "blocks": blocks})
            answer, _ = split_path_bound_transfers(planner.network, planner.build(blocked), planner.data_size)
            validate_answer(task, topology, answer, planner.network.raw)
            self.assertAlmostEqual(replay_estimate(task, topology, answer, planner.network.raw)["total_us"],
                                   predicted, delta=predicted * 1e-6)
        self.assertLess(results[True], 0.8 * results[False])


class AllReduceBlockTests(unittest.TestCase):
    def test_rs_ag_blocks_validate_and_match_prediction(self):
        data = two_plane_input("all-reduce", ranks=8)
        for short in (False, True):
            planner = Planner(data["task"], data["topology"], sum_notation=short)
            for rs_blocks, mid_blocks in (([[1, 2, 4]], []), ([[1, 2]], [[4]]), ([[1]], [[2, 4]]), ([[4], [2]], [[1]])):
                with self.subTest(sum_notation=short, rs=rs_blocks, mid=mid_blocks):
                    config = planner.normalize_config({"algorithm": "rs_ag", "split": 8, "path_width": 2,
                                                       "rs_blocks": rs_blocks, "mid_blocks": mid_blocks})
                    answer = planner.build(config)
                    validate_answer(data["task"], data["topology"], answer)
                    self.assertEqual(answer["round_count"], 2 * len(rs_blocks) + len(mid_blocks))
                    actual = replay_estimate(data["task"], data["topology"], answer)["total_us"]
                    self.assertAlmostEqual(planner.predict(config)["us"], actual, places=6)

    def test_bad_rs_ag_blocks_are_rejected(self):
        data = two_plane_input("all-reduce", ranks=8)
        planner = Planner(data["task"], data["topology"])
        for rs_blocks, mid_blocks, split in (([[1, 2, 4]], [], 4), ([[1, 2]], [[2, 4]], 8), ([], [[1, 2, 4]], 8),
                                             ([[1, 2]], [[4, 4]], 8), ("x", [], 8)):
            with self.subTest(rs=rs_blocks, mid=mid_blocks, split=split), self.assertRaises(AgentCCError):
                planner.normalize_config({"algorithm": "rs_ag", "split": split, "path_width": 1,
                                          "rs_blocks": rs_blocks, "mid_blocks": mid_blocks})
        with self.assertRaises(AgentCCError):
            planner.normalize_config({"algorithm": "tree", "split": 8, "path_width": 1,
                                      "rs_blocks": [[1]], "mid_blocks": [[2, 4]]})

    def test_block_search_and_public_mesh2d_gain(self):
        task, topology = public_case("mesh2d", "ar")
        planner = Planner(task, topology, sum_notation=True)
        config = planner.normalize_config({"algorithm": "rs_ag", "split": 16, "path_width": 16, "rs_dims": 4})
        rs_blocks, mid_blocks, predicted, singles = planner.allreduce_blocks(config)
        self.assertEqual([s for block in rs_blocks + mid_blocks for s in block], config["order"])
        self.assertLess(predicted, 0.8 * singles)
        blocked = planner.normalize_config({"algorithm": "rs_ag", "split": 16, "path_width": 16,
                                            "rs_blocks": rs_blocks, "mid_blocks": mid_blocks})
        split, _ = split_path_bound_transfers(planner.network, planner.build(blocked), planner.data_size // 16)
        validate_answer(task, topology, split, planner.network.raw)
        self.assertAlmostEqual(replay_estimate(task, topology, split, planner.network.raw)["total_us"], predicted,
                               places=3)
        plain, _ = split_path_bound_transfers(planner.network, planner.build(config), planner.data_size // 16)
        self.assertAlmostEqual(replay_estimate(task, topology, plain, planner.network.raw)["total_us"], singles,
                               places=3)


class HierarchicalAllReduceTests(unittest.TestCase):
    def test_hierarchical_rs_ag_validates_and_matches_prediction(self):
        data = two_plane_input("all-reduce", ranks=8)
        for short in (False, True):
            planner = Planner(data["task"], data["topology"], sum_notation=short)
            for rs_dims, hier_k in ((1, 1), (1, 2), (2, 1)):
                with self.subTest(sum_notation=short, rs_dims=rs_dims, hier_k=hier_k):
                    config = planner.normalize_config({"algorithm": "rs_ag", "split": 8, "path_width": 2,
                                                       "rs_dims": rs_dims, "hier_k": hier_k})
                    answer = planner.build(config)
                    validate_answer(data["task"], data["topology"], answer)
                    self.assertEqual(answer["round_count"], 2 * rs_dims + 3 - rs_dims + hier_k)
                    actual = replay_estimate(data["task"], data["topology"], answer)["total_us"]
                    self.assertAlmostEqual(planner.predict(config)["us"], actual, places=6)

    def test_bad_hier_k_is_rejected(self):
        data = two_plane_input("all-reduce", ranks=8)
        planner = Planner(data["task"], data["topology"])
        for raw in ({"algorithm": "rs_ag", "split": 8, "path_width": 1, "rs_dims": 1, "hier_k": 0},
                    {"algorithm": "rs_ag", "split": 8, "path_width": 1, "rs_dims": 1, "hier_k": 3},
                    {"algorithm": "rs_ag", "split": 8, "path_width": 1, "rs_dims": 1, "hier_k": True},
                    {"algorithm": "rs_ag", "split": 8, "path_width": 1, "rs_dims": 1, "hier_k": "1"},
                    {"algorithm": "rs_ag", "split": 8, "path_width": 1, "rs_blocks": [[1]], "mid_blocks": [[2, 4]],
                     "hier_k": 1},
                    {"algorithm": "tree", "split": 8, "path_width": 1, "hier_k": 1}):
            with self.subTest(raw=raw), self.assertRaises(AgentCCError):
                planner.normalize_config(raw)

    def test_public_directed_ring_split_prediction_and_gain(self):
        task, topology = public_case("directed_ring", "ar")
        planner = Planner(task, topology, sum_notation=True)
        split = planner.splits[-1]
        h = split.bit_length() - 1
        plain = planner.normalize_config({"algorithm": "rs_ag", "split": split, "path_width": 2, "rs_dims": h})
        hier = planner.normalize_config({"algorithm": "rs_ag", "split": split, "path_width": 2, "rs_dims": h,
                                         "hier_k": 2})
        results = {}
        for name, config in (("plain", plain), ("hier", hier)):
            answer, _ = split_path_bound_transfers(planner.network, planner.build(config), planner.data_size // split)
            validate_answer(task, topology, answer, planner.network.raw)
            results[name] = replay_estimate(task, topology, answer, planner.network.raw)["total_us"]
            self.assertAlmostEqual(planner.split_aware_us(config), results[name], delta=results[name] * 1e-6)
        self.assertLess(results["hier"], 0.85 * results["plain"])


class TransferSplitTests(unittest.TestCase):
    @staticmethod
    def _loads(answer, planner):
        chunk = planner.data_size // answer["split"]
        rounds = []
        for round_data in answer["rounds"]:
            load = {}
            for transfer in round_data["transfers"]:
                for edge in zip(transfer["path"], transfer["path"][1:]):
                    load[edge] = load.get(edge, 0) + chunk * len(transfer["chunk_ranges"])
            rounds.append(load)
        return rounds

    @staticmethod
    def _sent(answer):
        return [sorted((t["src"], t["dst"], item) for t in r["transfers"] for item in t["chunk_ranges"])
                for r in answer["rounds"]]

    def test_split_keeps_loads_and_chunks_and_lowers_replay(self):
        for operator, raw in (("all-gather", {"algorithm": "doubling", "split": 8, "path_width": 1}),
                              ("all-reduce", {"algorithm": "rs_ag", "split": 8, "path_width": 1, "rs_dims": 2})):
            data = single_switch_input(operator)
            planner = Planner(data["task"], data["topology"], sum_notation=True)
            answer = planner.build(planner.normalize_config(raw))
            before = replay_estimate(data["task"], data["topology"], answer)
            split, info = split_path_bound_transfers(planner.network, answer, planner.data_size // 8, size_weight=0.0)
            after = replay_estimate(data["task"], data["topology"], split)
            with self.subTest(operator=operator):
                validate_answer(data["task"], data["topology"], split)
                self.assertLess(after["total_us"], before["total_us"])
                self.assertAlmostEqual(after["total_us"], info["after_us"], places=6)
                self.assertEqual(self._loads(split, planner), self._loads(answer, planner))
                self.assertEqual(self._sent(split), self._sent(answer))
                self.assertGreater(info["extra_transfers"], 0)
                for round_before, round_after in zip(before["rounds"], after["rounds"]):
                    self.assertLessEqual(round_after["time_us"], round_before["time_us"] + 1e-9)

    def test_no_split_when_size_cost_dominates_or_nothing_is_path_bound(self):
        data = single_switch_input()
        planner = Planner(data["task"], data["topology"], sum_notation=True)
        answer = planner.build(planner.normalize_config({"algorithm": "doubling", "split": 8, "path_width": 1}))
        same, info = split_path_bound_transfers(planner.network, answer, planner.data_size // 8, size_weight=1e9)
        self.assertIs(same, answer)
        self.assertEqual(info["extra_transfers"], 0)
        once, _ = split_path_bound_transfers(planner.network, answer, planner.data_size // 8, size_weight=0.0)
        twice, info = split_path_bound_transfers(planner.network, once, planner.data_size // 8, size_weight=0.0)
        self.assertIs(twice, once)

    def test_public_directed_ring_allgather_gets_faster(self):
        task, topology = public_case("directed_ring", "ag")
        planner = Planner(task, topology, sum_notation=True)
        answer = planner.build(planner.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 2}))
        before = replay_estimate(task, topology, answer, planner.network.raw)["total_us"]
        split, _ = split_path_bound_transfers(planner.network, answer, planner.data_size)
        validate_answer(task, topology, split, planner.network.raw)
        self.assertLess(replay_estimate(task, topology, split, planner.network.raw)["total_us"], 0.95 * before)


class RouteRefinementTests(unittest.TestCase):
    @staticmethod
    def congested_case(operator="all-gather", split=1, slow_plane=False):
        data = two_plane_input(operator, ranks=2, candidates=(1, 2))
        for edge in data["topology"]["edges"]:
            if edge["dst"] == "B":
                edge["bw"] = "5GB/s" if slow_plane else "10GB/s"
        data["topology"]["resources"][0]["capacity"] = "1.5GB/s" if slow_plane else "1GB/s"
        planner = Planner(data["task"], data["topology"], sum_notation=True)
        config = planner.normalize_config({"algorithm": "rs_ag" if operator == "all-reduce" else "doubling",
                                           "split": split, "path_width": 2})
        answer = planner.build(config)
        # A feasible but congested integer allocation, independent of the optimizer.
        for round_data in answer["rounds"]:
            merged = {}
            for transfer in round_data["transfers"]:
                pair = (transfer["src"], transfer["dst"])
                if pair not in merged:
                    merged[pair] = {**transfer, "path": [pair[0], "A", pair[1]], "chunk_ranges": []}
                merged[pair]["chunk_ranges"].extend(transfer["chunk_ranges"])
            round_data["transfers"] = list(merged.values())
        validate_answer(data["task"], data["topology"], answer)
        estimate = replay_estimate(data["task"], data["topology"], answer, planner.network.raw)
        return data, planner, config, answer, estimate

    def test_refinement_improves_and_preserves_other_candidates(self):
        data, planner, config, answer, estimate = self.congested_case()
        original = deepcopy(answer)
        cached_answer = planner.build(config)
        prediction = planner.predict(config)
        signature = planner.schedule_key(config)
        revised, measured, detail = refine_schedule_routes(planner, config, answer, estimate)
        self.assertTrue(detail["adopted"])
        self.assertEqual(detail["accepted_moves"], 2)
        self.assertAlmostEqual(measured["total_us"], 1.3 + 2 * 65536 / 1e10 * 1e6)
        self.assertEqual(answer, original)
        self.assertEqual(planner.build(config), cached_answer)
        self.assertEqual(planner.predict(config), prediction)
        self.assertEqual(planner.schedule_key(config), signature)
        validate_answer(data["task"], data["topology"], revised)
        self.assertEqual(len(revised["rounds"][0]["transfers"]), 2)
        self.assertTrue(all(t["path"][1] == "B" for t in revised["rounds"][0]["transfers"]))

    def test_repeated_overlapping_resources_match_complete_replay(self):
        data, _, config, answer, _ = self.congested_case()
        resource = data["topology"]["resources"][0]
        resource["edges"].extend({"src": "A", "dst": rank} for rank in data["task"]["ranks"])
        data["topology"]["resources"].append({"id": "overlap", "capacity": "2GB/s", "edges": resource["edges"][:2]})
        planner = Planner(data["task"], data["topology"])
        estimate = replay_estimate(data["task"], data["topology"], answer)
        self.assertAlmostEqual(estimate["total_us"], 1.3 + 4 * 65536 / 1e9 * 1e6)
        revised, measured, detail = refine_schedule_routes(planner, config, answer, estimate)
        self.assertTrue(detail["adopted"])
        for row in detail["rounds"]:
            actual = replay_estimate(data["task"], data["topology"], {**revised, "rounds": [revised["rounds"][row["round"] - 1]]})
            self.assertAlmostEqual(row["after_us"], actual["total_us"])
        validate_answer(data["task"], data["topology"], revised)
        self.assertLess(measured["total_us"], estimate["total_us"])

    def test_slower_direction_is_rejected(self):
        data, _, config, answer, _ = self.congested_case()
        for edge in data["topology"]["edges"]:
            if edge["dst"] == "B":
                edge["bidirectional"] = False
                data["topology"]["edges"].append({"src": "B", "dst": edge["src"], "bw": "1MB/s"})
        planner = Planner(data["task"], data["topology"])
        estimate = replay_estimate(data["task"], data["topology"], answer)
        revised, measured, detail = refine_schedule_routes(planner, config, answer, estimate)
        self.assertIs(revised, answer)
        self.assertFalse(detail["adopted"])
        self.assertGreater(detail["checked_moves"], 0)
        self.assertEqual(measured, estimate)

    def test_activating_a_long_path_can_outweigh_lower_resource_load(self):
        data = two_plane_input(ranks=2, candidates=(1,))
        topology = data["topology"]
        topology["switches"] = ["A", "B", "C"]
        topology["switch_processing_overhead_us"] = 100
        topology["edges"] = [
            {"src": "r0", "dst": "A", "bw": "10GB/s", "bidirectional": True},
            {"src": "r1", "dst": "B", "bw": "10GB/s", "bidirectional": True},
            {"src": "A", "dst": "B", "bw": "10GB/s", "bidirectional": True},
            {"src": "A", "dst": "C", "bw": "10GB/s", "bidirectional": True},
            {"src": "C", "dst": "B", "bw": "10GB/s", "bidirectional": True},
        ]
        topology["resources"] = [{"id": "core", "capacity": "2GB/s", "edges": [
            {"src": "A", "dst": "B"}, {"src": "B", "dst": "A"}]}]
        planner = Planner(data["task"], topology)
        config = planner.normalize_config({"algorithm": "doubling", "split": 1, "path_width": 2})
        answer = planner.build(config)
        for transfer in answer["rounds"][0]["transfers"]:
            transfer["path"] = ["r0", "A", "B", "r1"] if transfer["src"] == "r0" else ["r1", "B", "A", "r0"]
        estimate = replay_estimate(data["task"], topology, answer)
        revised, _, detail = refine_schedule_routes(planner, config, answer, estimate)
        self.assertGreater(detail["checked_moves"], 0)
        self.assertFalse(detail["adopted"])
        self.assertIs(revised, answer)

    def test_copy_and_reduce_chunks_keep_their_meaning(self):
        data, planner, config, answer, estimate = self.congested_case("all-reduce", split=2)
        revised, measured, detail = refine_schedule_routes(planner, config, answer, estimate)
        self.assertTrue(detail["adopted"])
        validate_answer(data["task"], data["topology"], revised)
        self.assertEqual(revised["round_count"], answer["round_count"])
        for before, after in zip(answer["rounds"], revised["rounds"]):
            semantic = lambda row: sorted((t["action"], t["src"], t["dst"], chunk)
                                          for t in row["transfers"] for chunk in t["chunk_ranges"])
            self.assertEqual(semantic(before), semantic(after))
        self.assertLess(measured["total_us"], estimate["total_us"])

    def test_zero_search_budget_keeps_original(self):
        _, planner, config, answer, estimate = self.congested_case()
        revised, measured, detail = refine_schedule_routes(planner, config, answer, estimate, budget_seconds=0)
        self.assertIs(revised, answer)
        self.assertIs(measured, estimate)
        self.assertEqual(detail["checked_moves"], 0)
        self.assertFalse(detail["adopted"])

    def test_response_size_can_reject_a_faster_split(self):
        data, planner, config, answer, estimate = self.congested_case(split=2, slow_plane=True)
        revised, measured, detail = refine_schedule_routes(planner, config, answer, estimate, size_weight=1e9)
        self.assertGreater(detail["accepted_moves"], 0)
        self.assertLess(detail["candidate_us"], estimate["total_us"])
        self.assertGreater(detail["candidate_response_mb"], detail["before_response_mb"])
        self.assertEqual(detail["rejection"], "complete_schedule_cost")
        self.assertIs(revised, answer)
        self.assertEqual(measured, estimate)

    def test_refinement_validation_errors_propagate(self):
        _, planner, config, answer, estimate = self.congested_case()
        with patch("agentcc_solver.validate_answer", side_effect=AgentCCError("refinement bug")):
            with self.assertRaisesRegex(AgentCCError, "refinement bug"):
                refine_schedule_routes(planner, config, answer, estimate)


class SandboxTests(unittest.TestCase):
    def test_dangerous_code_is_rejected(self):
        from strategy_sandbox import StrategyError, run_strategy

        cases = {
            "import": "import os\ndef strategy(api): return []",
            "dunder": "def strategy(api): return ().__class__",
            "open": 'def strategy(api): return open("x")',
            "private": "def strategy(api): return api._planner",
            "unknown api": "def strategy(api): return api.secret",
            "while": "def strategy(api):\n    while True: pass",
            "getattr": 'def strategy(api): return getattr(api, "x")',
            "top level": "x = 1\ndef strategy(api): return []",
            "format": 'def strategy(api): return "{0}".format(api)',
            "try": "def strategy(api):\n    try:\n        return 1\n    except Exception:\n        return 2",
        }
        for name, source in cases.items():
            with self.subTest(name=name), self.assertRaises(StrategyError):
                run_strategy(source, SimpleNamespace(), max_seconds=1.0)

    def test_runaway_code_hits_step_limit(self):
        from strategy_sandbox import StrategyError, run_strategy

        source = "def strategy(api):\n    x = 0\n    for i in range(10**9):\n        x += i\n    return x"
        with self.assertRaisesRegex(StrategyError, "exceeded"):
            run_strategy(source, SimpleNamespace(), max_steps=50_000, max_seconds=2.0)

    def test_runtime_error_reports_line(self):
        from strategy_sandbox import StrategyError, run_strategy

        with self.assertRaisesRegex(StrategyError, "ZeroDivisionError at line 3"):
            run_strategy("def strategy(api):\n    x = 1\n    return x / 0", SimpleNamespace())

    def test_strategy_api_measures_on_current_topology(self):
        from time import perf_counter

        from agentcc_solver import StrategyAPI
        from strategy_sandbox import run_strategy

        data = two_plane_input("all-reduce")
        planner = Planner(data["task"], data["topology"])
        api = StrategyAPI(planner, planner.dimension_profiles(4), 0.33, perf_counter() + 5, max_measures=2)
        returned = run_strategy(strategy_code(STRATEGY_REPLY), api)
        self.assertEqual(len(returned), 3)
        self.assertEqual(len(api.measured), 2)  # measure budget is enforced
        self.assertIn("measure budget exhausted", json.dumps(api.calls))
        for row in api.measured:
            validate_answer(data["task"], data["topology"], row["answer"])

    def test_per_round_widths_build_valid_schedules(self):
        for operator, algorithm in (("all-gather", "doubling"), ("all-reduce", "rs_ag"), ("all-reduce", "tree")):
            data = two_plane_input(operator)
            planner = Planner(data["task"], data["topology"])
            raw = {"algorithm": algorithm, "split": 8, "path_width": 2}
            rounds = planner.round_count(planner.normalize_config(raw))
            raw["widths"] = [1 + (index % 2) for index in range(rounds)]
            answer = planner.build(planner.normalize_config(raw))
            with self.subTest(operator=operator, algorithm=algorithm):
                validate_answer(data["task"], data["topology"], answer)
            with self.subTest(bad="widths length"), self.assertRaisesRegex(AgentCCError, "widths"):
                planner.normalize_config({**raw, "widths": [1]})


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
                content = pending.pop(0) if pending else "no code"
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

    def test_model_pick_competes_with_agent_portfolio(self):
        data = two_plane_input("all-reduce")
        result, audit, events = self._run(['{"c":[["rs_ag",8,2,3],["doubling",1,1,0]]}'], task_input=data)
        validate_answer(data["task"], data["topology"], result["answer"])
        self.assertEqual(len(events["requests"]), 1)
        self.assertEqual(len(audit["model_candidates"]), 2)
        sources = {tool["arguments"]["source"] for tool in events["tools"] if tool["name"] == "evaluate_schedule"}
        self.assertEqual(sources, {"agent", "model"})
        self.assertIn("metrics", result["answer"])
        self.assertIn("strategy", result["answer"])
        self.assertIn("model_improved_selection", audit)
        self.assertIn("route_refinement", audit)
        self.assertIn("refine_schedule_routes", [event["name"] for event in events["tools"]])

    def test_prompt_carries_measurements_and_agent_predictions(self):
        data = two_plane_input("all-gather")
        _, _, events = self._run(['{"c":[]}'], task_input=data)
        request = events["requests"][0]
        prompt = request["messages"][-1]["content"]
        self.assertIn("plane-A-ingress", prompt)
        self.assertIn("dims_us_per_MiB", prompt)
        self.assertLess(len(prompt) + len(request["messages"][0]["content"]), 1400)
        self.assertEqual(request["max_tokens"], 48)
        changed = deepcopy(data)
        changed["topology"]["resources"][0]["capacity"] = "1GB/s"
        _, _, events_changed = self._run(['{"c":[]}'], task_input=changed)
        self.assertNotEqual(prompt, events_changed["requests"][0]["messages"][-1]["content"])

    def test_full_search_overlaps_model_and_proposals_wait_for_cache_owner(self):
        for model_finishes_first in (True, False):
            with self.subTest(model_finishes_first=model_finishes_first):
                module = self._load_agent_module({"tools": [], "recovery": []})
                rank_entered, model_started = threading.Event(), threading.Event()
                rank_finished, model_finished = threading.Event(), threading.Event()
                original_rank, original_chat = module.rank_portfolio, module._chat
                original_proposals = module.Evaluator.run_proposals

                def rank(planner, candidates=None):
                    if candidates is not None:
                        self.assertTrue(all(raw["path_width"] == 1 for raw in candidates))
                        return original_rank(planner, candidates)
                    rank_entered.set()
                    self.assertTrue(model_started.wait(2), "full ranking blocked the model request")
                    if model_finishes_first:
                        self.assertTrue(model_finished.wait(2))
                    result = original_rank(planner)
                    rank_finished.set()
                    return result

                async def chat(*args):
                    model_started.set()
                    self.assertTrue(await asyncio.to_thread(rank_entered.wait, 2))
                    result = await original_chat(*args)
                    if not model_finishes_first:
                        self.assertTrue(await asyncio.to_thread(rank_finished.wait, 2))
                    model_finished.set()
                    return result

                def proposals(evaluator, *args):
                    self.assertTrue(rank_finished.is_set(), "model proposals raced with ranking")
                    return original_proposals(evaluator, *args)

                with patch.object(self, "_load_agent_module", return_value=module), \
                     patch.object(module, "rank_portfolio", side_effect=rank), \
                     patch.object(module, "_chat", side_effect=chat), \
                     patch.object(module.Evaluator, "run_proposals", proposals):
                    data = two_plane_input("all-reduce")
                    result, audit, events = self._run(['{"c":[["rs_ag",8,2,3]]}'], task_input=data)
                validate_answer(data["task"], data["topology"], result["answer"])
                self.assertEqual(len(events["requests"]), 1)
                self.assertNotIn("portfolio_error", audit)
                self.assertNotIn("model_error", {key: value for key, value in audit.items() if value})
                self.assertEqual(set(audit["timings_ms"]), {
                    "parse", "portfolio_rank", "portfolio_build",
                    "model", "portfolio_join", "model_candidates", "select_validate", "refine_validate",
                    "mapping_generation", "mapping_prepare", "rank_mapping",
                    "tree_prepare", "dual_tree_validate", "transfer_split", "blocks_prepare", "blocks_validate", "final_validate", "speculative_finalize", "hier_validate",
                })

    def test_initial_predictions_obey_current_task_constraints(self):
        cases = [two_plane_input("all-gather"), two_plane_input("all-reduce"),
                 two_plane_input("all-reduce"), two_plane_input("all-reduce")]
        cases[1]["task"]["allow_copy"] = False
        cases[2]["task"]["max_nonempty_rounds"] = 3
        cases[3]["task"]["data_size"]["per_rank_bytes"] = 40960
        for data in cases:
            with self.subTest(task=data["task"]):
                result, audit, events = self._run(['{"c":[]}'], task_input=data)
                validate_answer(data["task"], data["topology"], result["answer"])
                prompt = events["requests"][0]["messages"][-1]["content"]
                payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{"):])
                self.assertIn("before routing", payload["dims_scope"])
                self.assertEqual(payload["bytes_per_rank"], data["task"]["data_size"]["per_rank_bytes"])
                self.assertEqual(payload["alignment_bytes"], data["task"]["split_policy"]["alignment_bytes"])
                planner = Planner(data["task"], data["topology"], sum_notation=True)
                self.assertEqual([row[0] for row in payload["dims_us_per_MiB"]], planner.steps)
                self.assertTrue(all(row[1] > 0 for row in payload["dims_us_per_MiB"]))
                # The model request is sent before any route is generated.
                self.assertEqual(audit["planning_ms"] < audit["solve_ms"], True)

    def test_initial_prompt_exposes_reverse_bandwidth_changes_off_single_path(self):
        data = two_plane_input("all-reduce")
        changed = deepcopy(data)
        changed["topology"]["edges"][1]["bidirectional"] = False
        changed["topology"]["edges"].append({"src": "B", "dst": "r0", "bw": "1GB/s"})
        payloads = []
        for task_input in (data, changed):
            _, _, events = self._run(['{"c":[]}'], task_input=task_input)
            prompt = events["requests"][0]["messages"][-1]["content"]
            payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{"):])
            payloads.append(payload)
        original, modified = payloads
        self.assertEqual(original["links"]["unequal_reverse"], 0)
        self.assertEqual(modified["links"]["unequal_reverse"], 1)
        self.assertEqual(modified["links"]["one_way"], 0)
        self.assertEqual(modified["links"]["GBps_range"], [1.0, 10.0])

    def test_size_weight_prefers_smaller_response_when_time_is_close(self):
        module = self._load_agent_module({"tools": [], "recovery": []})
        data = two_plane_input("all-reduce")
        planner = module.Planner(data["task"], data["topology"])
        ranked = module.rank_portfolio(planner)
        self.assertTrue(ranked)
        costs = [cost for cost, _, _ in ranked]
        self.assertEqual(costs, sorted(costs))

    @patch.dict(os.environ, {"AGENTCC_RANK_MAPPING": "1"})
    def test_model_mapping_choice_changes_the_current_instance_schedule(self):
        task, topology = public_case("uneven_pods", "ar")
        outputs = []
        for choice in (0, 1):
            result, audit, events = self._run([json.dumps({"c": [], "m": choice})],
                                               task_input={"task": task, "topology": topology})
            validate_answer(task, topology, result["answer"])
            self.assertIsNone(audit["model_error"])
            self.assertEqual(len(events["requests"]), 1)
            self.assertEqual(audit["model_mapping_choice"], choice)
            outputs.append((result, audit))
        self.assertEqual(outputs[0][1]["rank_mapping"]["reason"], "model_kept_original")
        mapped = outputs[1][1]["rank_mapping"]
        self.assertTrue(mapped["adopted"])
        self.assertLess(mapped["after_us"], mapped["before_us"] * .97)
        self.assertEqual(outputs[1][1]["selected"]["config"]["mapping"], "locality")
        self.assertIn("model_mapping", outputs[1][1]["selected"]["sources"])
        self.assertEqual(sorted(mapped["rank_order"]), list(range(256)))

    @patch.dict(os.environ, {"AGENTCC_RANK_MAPPING": "1"})
    def test_mapping_tool_keeps_model_link_and_original_candidate(self):
        events = {"tools": [], "recovery": []}
        module = self._load_agent_module(events)
        task, topology = public_case("uneven_pods", "ar")
        planner = Planner(task, topology, sum_notation=True)
        evaluator = module.Evaluator(planner)
        evaluator.run_portfolio()
        best = evaluator.select()
        original = deepcopy(best["_answer"])
        self.assertEqual(evaluator.prepared_mapping["mapping"], "resource")
        with patch("agentcc_solver.optimize_round", side_effect=AssertionError("mapping prediction cache missed")):
            mapped, detail = evaluator.remap(best, 2, "model-map-123")
        self.assertTrue(detail["adopted"])
        self.assertEqual(best["_answer"], original)
        self.assertEqual(planner.build(best["config"]), original)
        tools = [event for event in events["tools"] if event["name"] in ("search_rank_mapping", "evaluate_schedule")]
        self.assertEqual(tools[-2]["name"], "search_rank_mapping")
        self.assertEqual(tools[-2]["parent_llm_call_id"], "model-map-123")
        self.assertEqual(tools[-1]["parent_llm_call_id"], "model-map-123")
        self.assertEqual(tools[-1]["arguments"]["source"], "model_mapping")
        refined, _ = evaluator.refine(mapped, "model-map-123")
        validate_answer(task, topology, refined["_answer"])
        self.assertIsNot(mapped["_planner"], planner)

    @patch.dict(os.environ, {"AGENTCC_RANK_MAPPING": "1"})
    def test_mapping_screen_rejects_a_slower_schedule_without_building_it(self):
        module = self._load_agent_module({"tools": [], "recovery": []})
        task, topology = public_case("mesh2d", "ar")
        evaluator = module.Evaluator(Planner(task, topology, sum_notation=True))
        evaluator.run_portfolio()
        best = evaluator.select()
        result, detail = evaluator.remap(best, 1, None)
        self.assertIs(result, best)
        self.assertFalse(detail["adopted"])
        self.assertEqual(detail["reason"], "integer_replay_screen")
        self.assertNotIn("candidate", detail)

    @patch.dict(os.environ, {"AGENTCC_RANK_MAPPING": "1"})
    def test_mapping_defers_validation_to_the_final_answer(self):
        module = self._load_agent_module({"tools": [], "recovery": []})
        task, topology = public_case("uneven_pods", "ar")
        evaluator = module.Evaluator(Planner(task, topology, sum_notation=True))
        evaluator.run_portfolio()
        best = evaluator.select()
        with patch.object(module, "validate_answer", side_effect=AssertionError("remap must not validate")):
            result, _ = evaluator.remap(best, 1, None)
        validate_answer(task, topology, result["_answer"])

    def test_final_validation_failure_falls_back_to_the_validated_selection(self):
        task, topology = public_case("uneven_pods", "ar")
        calls = []
        module_validate = validate_answer

        def second_call_fails(*args, **kwargs):
            calls.append(1)
            if len(calls) > 1:
                raise AgentCCError("final validation failed")
            return module_validate(*args, **kwargs)

        events = {"requests": [], "tools": [], "recovery": []}
        module = self._load_agent_module(events)
        with patch.object(module, "validate_answer", side_effect=second_call_fails),                 patch.object(AgentLoopTests, "_load_agent_module", staticmethod(lambda _events: module)):
            result, audit, _ = self._run(['{"c":[]}'], task_input={"task": task, "topology": topology})
        self.assertEqual(audit.get("final_validation_error"), "AgentCCError: final validation failed")
        validate_answer(task, topology, result["answer"])
        self.assertEqual(result["answer"]["round_count"], len(result["answer"]["rounds"]))

    def test_equivalent_widths_share_build_and_keep_model_evidence(self):
        events = {"tools": [], "recovery": []}
        module = self._load_agent_module(events)
        data = two_plane_input(ranks=4, candidates=(1, 2))
        data["topology"]["switches"] = ["A"]
        data["topology"]["edges"] = [edge for edge in data["topology"]["edges"] if edge["dst"] == "A"]
        planner = Planner(data["task"], data["topology"])
        evaluator = module.Evaluator(planner)
        with patch.object(planner, "build", wraps=planner.build) as build:
            rows, _ = evaluator.run_portfolio()
            self.assertEqual(build.call_count, 2)
            self.assertEqual(sum(row["status"] == "duplicate" for row in rows), 2)  # 2 splits x widths (8, 16)
            duplicate = evaluator.evaluate({"algorithm": "doubling", "split": 1, "path_width": 16},
                                           "model", "model-call-123")
            self.assertEqual(duplicate["status"], "duplicate")
            self.assertEqual(build.call_count, 2)
        self.assertEqual(events["tools"][-1]["parent_llm_call_id"], "model-call-123")
        matched = next(row for row in evaluator.results.values() if row["config"]["split"] == 1)
        self.assertEqual(matched["sources"], {"agent", "model"})
        self.assertIn("equivalent_config", duplicate)

    def test_schedule_key_preserves_distinct_structure_and_per_round_widths(self):
        data = two_plane_input("all-reduce")
        planner = Planner(data["task"], data["topology"])
        config = planner.normalize_config({"algorithm": "rs_ag", "split": 8, "path_width": 2})
        single = {**config, "widths": [1] * planner.round_count(config)}
        self.assertNotEqual(planner.schedule_key(config), planner.schedule_key(single))
        other = planner.normalize_config({"algorithm": "tree", "split": 8, "path_width": 2})
        self.assertNotEqual(planner.schedule_key(config), planner.schedule_key(other))

    def test_integer_prediction_matches_every_public_portfolio_candidate(self):
        module = self._load_agent_module({"tools": [], "recovery": []})
        for family in FAMILIES:
            for short in ("ag", "ar"):
                task, topology = public_case(family, short)
                planner = Planner(task, topology, sum_notation=True)
                for _, config, predicted in module.rank_portfolio(planner):
                    answer = planner.build(config)
                    actual = replay_estimate(task, topology, answer, planner.network.raw)["total_us"]
                    with self.subTest(family=family, op=short, config=config):
                        self.assertAlmostEqual(predicted["us"], actual, delta=max(1e-6, actual * 1e-9))

    def test_bad_model_output_never_fails_the_request(self):
        data = two_plane_input("all-reduce")
        for reply in [
            '{"c":[["tree",3,1,0],["nonsense",1,1,0]]}',
            '{"c":[["rs_ag",8,99,3]]}',
            "not json at all",
            "",
            '{"c":[["rs_ag",8,2,30]]}',
        ]:
            result, audit, _ = self._run([reply], task_input=data)
            with self.subTest(reply=reply):
                validate_answer(data["task"], data["topology"], result["answer"])
                self.assertIn("agent", audit["selected"]["sources"])

    def test_model_failure_and_timeout_return_valid_answer(self):
        error = openai.APIConnectionError(request=httpx.Request("POST", "http://model.test/v1/chat/completions"))
        for kwargs, reason in [
            ({"error": error}, "model_api_error:APIConnectionError"),
            ({"error": RuntimeError("gateway returned junk")}, "model_error:RuntimeError"),
            ({"delay": 0.5, "environment": {"AGENT_LLM_BUDGET_SECONDS": "0.2"}}, "model_timeout"),
        ]:
            result, audit, events = self._run([], **kwargs)
            with self.subTest(reason=reason):
                self.assertEqual(audit["model_error"], reason)
                # Agent-side steps (e.g. agent_blocks) may refine the pick; no model source may appear.
                self.assertIn("agent", audit["selected"]["sources"])
                self.assertFalse([s for s in audit["selected"]["sources"] if s.startswith("model")])
                self.assertIn("use_agent_portfolio", events["recovery"])
                data = two_plane_input("all-reduce")
                validate_answer(data["task"], data["topology"], result["answer"])

    def test_candidate_crash_is_contained(self):
        events = {"requests": [], "tools": [], "recovery": []}
        module = self._load_agent_module(events)
        data = two_plane_input("all-reduce")
        planner = module.Planner(data["task"], data["topology"])
        evaluator = module.Evaluator(planner)
        with patch.object(planner, "build", side_effect=TypeError("'<' not supported")):
            row = evaluator.evaluate({"algorithm": "rs_ag", "split": 8, "path_width": 2}, "model")
        self.assertEqual(row["status"], "error")
        self.assertIn("TypeError", row["error"])
        with self.assertRaises(AgentCCError):
            evaluator.select()
        emergency = evaluator.emergency()
        validate_answer(data["task"], data["topology"], emergency["_answer"])

    def test_rank_mapping_is_off_by_default(self):
        data = two_plane_input("all-reduce")
        result, audit, events = self._run(['{"c":[],"m":1}'], task_input=data)
        validate_answer(data["task"], data["topology"], result["answer"])
        self.assertEqual(audit["rank_mapping"]["reason"], "disabled")
        self.assertNotIn("mapping", audit["selected"]["config"])
        request = events["requests"][0]["messages"]
        self.assertNotIn('"m"', request[0]["content"])
        self.assertNotIn("rank_mapping", request[1]["content"])
        module = self._load_agent_module({"tools": [], "recovery": [], "requests": []})
        reduce = module.Planner(**{k: v for k, v in zip(("task", "topology"), data.values())})
        self.assertFalse(module.mappable(reduce, {"algorithm": "rs_ag"}))

    @patch.dict(os.environ, {"AGENTCC_RANK_MAPPING": "1"})
    def test_mappable_schedules(self):
        module = self._load_agent_module({"tools": [], "recovery": [], "requests": []})
        gather = module.Planner(**{k: v for k, v in zip(("task", "topology"), two_plane_input("all-gather").values())})
        reduce = module.Planner(**{k: v for k, v in zip(("task", "topology"), two_plane_input("all-reduce").values())})
        self.assertTrue(module.mappable(gather, {"algorithm": "doubling"}))
        self.assertTrue(module.mappable(reduce, {"algorithm": "rs_ag"}))
        self.assertFalse(module.mappable(reduce, {"algorithm": "doubling"}))
        self.assertFalse(module.mappable(reduce, {"algorithm": "tree"}))

    @patch.dict(os.environ, {"AGENTCC_RANK_MAPPING": "1"})
    def test_speculative_result_is_reused_only_without_model_choices(self):
        task, topology = public_case("uneven_pods", "ar")
        data = {"task": task, "topology": topology}
        reused, audit, _ = self._run(['{"c":[]}'], task_input=data)
        self.assertTrue(audit["speculative_reuse"])
        validate_answer(task, topology, reused["answer"])
        rerun, audit_m, _ = self._run(['{"c":[],"m":0}'], task_input=data)
        self.assertFalse(audit_m["speculative_reuse"])
        self.assertEqual(audit_m["rank_mapping"]["reason"], "model_kept_original")
        validate_answer(task, topology, rerun["answer"])

    def test_default_model_budget_is_two_seconds_up_to_three_while_searching(self):
        module = self._load_agent_module({"tools": [], "recovery": [], "requests": []})
        self.assertEqual(module.DEFAULT_MODEL_BUDGET_SECONDS, 2.0)
        self.assertEqual(module.DEFAULT_MODEL_BUDGET_SECONDS * module.MODEL_HARD_FACTOR, 3.0)

    def test_model_may_answer_after_soft_budget_while_search_runs(self):
        data = two_plane_input("all-reduce")
        loader = AgentLoopTests._load_agent_module

        def slow_search(events):
            module = loader(events)
            original = module.Evaluator.portfolio_and_speculate

            def delayed(self):
                result = original(self)
                time.sleep(0.6)
                return result

            module.Evaluator.portfolio_and_speculate = delayed
            return module

        with patch.object(AgentLoopTests, "_load_agent_module", staticmethod(slow_search)):
            result, audit, _ = self._run(['{"c":[["rs_ag",8,2,3]]}'], task_input=data, delay=0.3,
                                         environment={"AGENT_LLM_BUDGET_SECONDS": "0.1", "AGENT_LLM_HARD_SECONDS": "1.0"})
        self.assertEqual(audit["model_wait"], "search")
        self.assertIsNone(audit["model_error"])
        self.assertIn("model=ok", result["answer"]["strategy"])
        validate_answer(data["task"], data["topology"], result["answer"])

    def test_answer_records_model_outcome(self):
        data = two_plane_input("all-reduce")
        result, _, _ = self._run(['{"c":[["rs_ag",8,2,3]]}'], task_input=data)
        self.assertIn("model=ok", result["answer"]["strategy"])
        self.assertIn("llm_ms=", result["answer"]["strategy"])
        late, audit, events = self._run(
            [], task_input=data, delay=0.5, environment={"AGENT_LLM_BUDGET_SECONDS": "0.2"})
        self.assertIn("model=model_timeout", late["answer"]["strategy"])
        self.assertEqual(audit["model_error"], "model_timeout")
        self.assertEqual(len(events["requests"]), 1)
        validate_answer(data["task"], data["topology"], late["answer"])

    def test_sum_notation_switch(self):
        data = two_plane_input("all-reduce")
        _, audit_on, _ = self._run(['{"c":[]}'], task_input=data)
        self.assertTrue(audit_on["sum_notation"])
        result, audit_off, _ = self._run(['{"c":[]}'], task_input=data, environment={"AGENTCC_SUM_NOTATION": "0"})
        self.assertFalse(audit_off["sum_notation"])
        copies = [item for r in result["answer"]["rounds"] for t in r["transfers"] if t["action"] == "copy"
                  for item in t["chunk_ranges"]]
        self.assertTrue(copies and not any(item.endswith("_sum") for item in copies))

    def test_reply_parsing(self):
        module = self._load_agent_module({"tools": [], "recovery": []})
        proposals, mapping, tree, error = module.parse_proposals(
            '<think>["x",1,1]</think>```json\n{"c":[["rs_ag",32,4,5],["doubling",1,8,0]]}\n```')
        self.assertIsNone(error)
        self.assertIsNone(mapping)
        self.assertIsNone(tree)
        self.assertEqual(proposals, [
            {"algorithm": "rs_ag", "split": 32, "path_width": 4, "rs_dims": 5},
            {"algorithm": "doubling", "split": 1, "path_width": 8},
        ])
        self.assertEqual(module.parse_proposals(None)[3], "model_empty_response")
        self.assertEqual(module.parse_proposals(123)[3], "model_empty_response")
        self.assertEqual(len(module.parse_proposals('{"c":[["rs_ag",16,4,4],["dou')[0]), 1)
        self.assertEqual(module.parse_proposals('{"c":[],"m":2}'), ([], 2, None, None))
        self.assertEqual(module.parse_proposals('{"c":[],"t":3}'), ([], None, 3, None))
        for choice in (3, -1, True, "locality", None):
            self.assertEqual(module.parse_proposals(json.dumps({"c": [], "m": choice}))[3], "model_mapping_error")
        for choice in (4, -1, True, "equal", None):
            self.assertEqual(module.parse_proposals(json.dumps({"c": [], "t": choice}))[3], "model_tree_error")

    def test_model_tree_choice_changes_adoption_with_one_call_and_keeps_original(self):
        task, topology = public_case("fullmesh_clos", "ar")
        results = []
        loader = AgentLoopTests._load_agent_module

        def without_hierarchy(events):
            module = loader(events)
            module.Evaluator.try_hierarchy = lambda _self, best, _call: (best, {"adopted": False})
            return module

        for choice in (0, 1):
            with patch.object(AgentLoopTests, "_load_agent_module", staticmethod(without_hierarchy)):
                result, audit, events = self._run([json.dumps({"c": [], "m": 0, "t": choice})],
                                                   task_input={"task": task, "topology": topology})
            validate_answer(task, topology, result["answer"])
            self.assertEqual(len(events["requests"]), 1)
            self.assertEqual(audit["model_tree_choice"], choice)
            results.append((result, audit, events))
        self.assertEqual(results[0][1]["dual_tree"]["reason"], "model_disabled")
        detail = results[1][1]["dual_tree"]
        self.assertTrue(detail["adopted"])
        self.assertLess(detail["after_us"], detail["before_us"] * .97)
        self.assertLessEqual(detail["candidate_cost"], detail["before_cost"])
        self.assertEqual(results[1][1]["selected"]["config"]["algorithm"], "dual_tree")
        self.assertIn("model_dual_tree", results[1][1]["selected"]["sources"])
        self.assertEqual(results[0][2]["requests"][0], results[1][2]["requests"][0])
        self.assertIn("parent_llm_call_id", results[1][2]["tools"][-1])

    def test_hierarchy_is_adopted_on_public_allreduce_and_validated(self):
        task, topology = public_case("nvlink_like", "ar")
        result, audit, events = self._run([json.dumps({"c": [], "m": 0, "t": 0})],
                                           task_input={"task": task, "topology": topology})
        validate_answer(task, topology, result["answer"])
        detail = audit["hierarchy"]
        self.assertTrue(detail["adopted"])
        self.assertLess(detail["after_us"], detail["before_us"])
        self.assertIn("hier_k", audit["selected"]["config"])
        self.assertIn("agent_hierarchy", audit["selected"]["sources"])
        self.assertIn("hier_k=", result["answer"]["strategy"])
        self.assertTrue(any(tool["name"] == "search_hierarchy" for tool in events["tools"]))

    def test_tree_prediction_gate_and_deferred_validation(self):
        events = {"tools": [], "recovery": []}
        module = self._load_agent_module(events)
        task, topology = public_case("fullmesh_clos", "ar")
        evaluator = module.Evaluator(Planner(task, topology, sum_notation=True))
        evaluator.prepared_tree = prepare_dual_tree(evaluator.planner)
        original = {"estimate_us": 1.0, "score": 1.0, "sources": {"agent"}}
        with patch.object(module, "build_dual_tree", side_effect=AssertionError("slow candidate was built")):
            result, detail = evaluator.try_dual_tree(original, 1, "model-tree-123")
        self.assertIs(result, original)
        self.assertEqual(detail["reason"], "integer_replay_screen")
        original["estimate_us"] = 1000
        with patch.object(module, "validate_answer", side_effect=AssertionError("dual tree must not validate")):
            result, detail = evaluator.try_dual_tree(original, 1, "model-tree-123")
        self.assertEqual(events["tools"][-1]["parent_llm_call_id"], "model-tree-123")
        self.assertIs(result, original)
        self.assertEqual(detail["reason"], "complete_schedule_cost")


if __name__ == "__main__":
    unittest.main()
