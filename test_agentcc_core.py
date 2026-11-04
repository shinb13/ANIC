"""Hand-computed timing cases and adversarial causal/state transitions."""

from copy import deepcopy
import unittest

from agentcc_solver import AgentCCError, build_solution, estimate_schedule_time_us, validate_answer
from src.agentcc_network import Network
from src.agentcc_plan import (
    Plan, Problem, RoundVerifier, Transfer, build_baseline, estimate_plan_time_us,
    estimate_round, plan_from_answer, plan_to_answer, validate_plan,
)


def inputs(operator="all-gather", count=2, split=1):
    ranks = [f"r{i}" for i in range(count)]
    task = {
        "task_id": "unit", "operator": operator, "ranks": ranks,
        "data_size": {"per_rank_bytes": 1000 * split},
        "split_policy": {"candidates": [split], "alignment_bytes": 1},
        "max_nonempty_rounds": 16, "allow_copy": True,
    }
    topology = {
        "ranks": ranks, "switches": [], "communication_startup_time_us": 2,
        "switch_processing_overhead_us": 5,
        "edges": [{"src": src, "dst": dst, "bw": "1GB/s"} for src in ranks for dst in ranks if src != dst],
    }
    return task, topology


def ag(p, src, dst, mask):
    return Transfer("copy", src, dst, p.route(src, dst), chunks=mask)


def ar(p, src, dst, slot, mask, action="reduce"):
    return Transfer(action, src, dst, p.route(src, dst), fragments=((slot, mask),))


class ReplayTimingTests(unittest.TestCase):
    def test_direct_paths_have_no_switch_overhead(self):
        task, topology = inputs()
        topology["resources"] = [{"id": "shared", "capacity": "1GB/s", "edges": deepcopy(topology["edges"])}]
        p = Problem(task, topology, 1)
        timing = estimate_round(p, (ag(p, 0, 1, 1), ag(p, 1, 0, 2)))
        # 2 us startup + max(1 us per directed edge, 2 us pooled resource).
        self.assertEqual(timing.max_overhead_us, 2)
        self.assertEqual(timing.total_us, 4)

    def test_multihop_overhead_and_repeated_resource_traversals(self):
        task, topology = inputs()
        topology["switches"] = ["a", "b", "c"]
        nodes = ["r0", "a", "b", "c", "r1"]
        topology["edges"] = [{"src": a, "dst": b, "bw": "1GB/s", "bidirectional": True, "resource_id": "fabric"} for a, b in zip(nodes, nodes[1:])]
        topology["resources"] = [{"id": "fabric", "capacity": "1GB/s"}]
        p = Problem(task, topology, 1)
        timing = estimate_round(p, (ag(p, 0, 1, 1), ag(p, 1, 0, 2)))
        # 8 edge traversals * 1000 B; 2 + 3*5 startup/switch overhead.
        self.assertEqual(timing.resource_loads, ((0, 8000),))
        self.assertEqual(timing.max_path_us, 21)
        self.assertEqual(timing.total_us, 25)

    def test_round_uses_maximum_overhead_across_all_paths(self):
        task, topology = inputs(count=4)
        topology["switches"] = ["a", "b"]
        topology["edges"] = [
            {"src": "r0", "dst": "r1", "bw": "1MB/s"},
            {"src": "r2", "dst": "a", "bw": "1GB/s"},
            {"src": "a", "dst": "b", "bw": "1GB/s"},
            {"src": "b", "dst": "r3", "bw": "1GB/s"},
        ]
        p = Problem(task, topology, 1)
        timing = estimate_round(p, (ag(p, 0, 1, 1), ag(p, 2, 3, 4)))
        self.assertEqual(timing.max_path_us, 1002)
        self.assertEqual(timing.max_overhead_us, 12)
        self.assertEqual(timing.total_us, 1012)

    def test_duplicate_edges_add_capacity_and_memberships_deduplicate(self):
        task, topology = inputs()
        topology["edges"] = [
            {"src": "r0", "dst": "r1", "bw": "0.5GB/s", "resource_id": "a"},
            {"src": "r0", "dst": "r1", "bw": "4Gbps", "resource_id": "a"},
        ]
        edge = {"src": "r0", "dst": "r1"}
        topology["resources"] = [
            {"id": "a", "capacity": "0.5GB/s", "edges": [edge, edge]},
            {"id": "b", "bw": "0.25GB/s", "edges": [edge]},
        ]
        p = Problem(task, topology, 1)
        timing = estimate_round(p, (ag(p, 0, 1, 1),))
        self.assertEqual(p.network.edge_capacities, (1e9,))
        self.assertEqual(timing.resource_loads, ((0, 1000), (1, 1000)))
        self.assertEqual(timing.total_us, 6)

    def test_allreduce_counts_slots_not_contributor_tokens(self):
        task, topology = inputs("all-reduce", count=4, split=2)
        p = Problem(task, topology, 2)
        plan = build_baseline(p)
        # Second round: each expression has two contributors, but each slot
        # still carries exactly 1000 B. h=2; two slots => 4 us total.
        timing = estimate_round(p, plan.rounds[1])
        self.assertEqual(timing.max_path_us, 4)
        self.assertEqual(timing.total_us, 4)
        self.assertTrue(all(load == 2000 for _, load in timing.edge_loads))

    def test_empty_round_has_zero_cost_and_loads_reset(self):
        task, topology = inputs()
        p = Problem(task, topology, 1)
        transfer = ag(p, 0, 1, 1)
        self.assertEqual(estimate_round(p, ()).total_us, 0)
        self.assertEqual(estimate_plan_time_us(p, Plan(p.operator, 1, ((transfer,), (), (transfer,)))), 6)


class RoundVerificationTests(unittest.TestCase):
    def test_same_round_forwarding_rejected_and_failure_rolls_back(self):
        task, topology = inputs(count=4)
        p = Problem(task, topology, 1)
        verifier = RoundVerifier(p)
        state = verifier.possession
        first = ag(p, 0, 1, 1)
        issue = verifier.apply_round((first, ag(p, 1, 2, 1)))
        self.assertEqual((issue.code, issue.round_number, issue.transfer_index), ("SOURCE_STATE", 1, 1))
        self.assertEqual(verifier.possession, state)
        self.assertEqual(verifier.sent, {})
        self.assertEqual(verifier.round_number, 0)
        self.assertIsNone(verifier.apply_round((first,)))
        self.assertEqual(verifier.possession[1], 3)
        self.assertEqual(verifier.finish().code, "INCOMPLETE")

    def test_two_arrivals_of_same_chunk_rejected(self):
        task, topology = inputs(count=4)
        p = Problem(task, topology, 1)
        verifier = RoundVerifier(p)
        self.assertIsNone(verifier.apply_round((ag(p, 0, 1, 1),)))
        issue = verifier.apply_round((ag(p, 0, 2, 1), ag(p, 1, 2, 1)))
        self.assertEqual(issue.code, "DUPLICATE_RECEIVE")
        self.assertEqual(verifier.possession[2], 4)

    def test_partial_slot_reduce_and_completed_copy_are_legal(self):
        task, topology = inputs("all-reduce", split=2)
        p = Problem(task, topology, 2)
        plan = Plan(p.operator, 2, (
            (ar(p, 0, 1, 0, 1),),
            (ar(p, 1, 0, 0, 3, "copy"), ar(p, 1, 0, 1, 2)),
            (ar(p, 0, 1, 1, 3, "copy"),),
        ))
        validate_plan(p, plan)
        answer = plan_to_answer(p, plan)
        validate_answer(task, topology, answer)
        self.assertEqual(estimate_schedule_time_us(task, topology, answer), 10)

    def test_allreduce_same_round_completed_expression_rejected(self):
        task, topology = inputs("all-reduce")
        p = Problem(task, topology, 1)
        verifier = RoundVerifier(p)
        issue = verifier.apply_round((ar(p, 0, 1, 0, 1), ar(p, 1, 0, 0, 3, "copy")))
        self.assertEqual(issue.code, "SOURCE_STATE")
        self.assertEqual(verifier.partials, ((1,), (2,)))

    def test_overlapping_contributors_rejected(self):
        task, topology = inputs("all-reduce", count=4)
        p = Problem(task, topology, 1)
        verifier = RoundVerifier(p)
        self.assertIsNone(verifier.apply_round((ar(p, 1, 0, 0, 2), ar(p, 1, 2, 0, 2))))
        issue = verifier.apply_round((ar(p, 0, 2, 0, 3),))
        self.assertEqual((issue.code, issue.slot), ("DUPLICATE_CONTRIBUTOR", 0))

    def test_copy_requires_complete_reduction(self):
        task, topology = inputs("all-reduce")
        p = Problem(task, topology, 1)
        self.assertEqual(RoundVerifier(p).apply_round((ar(p, 0, 1, 0, 1, "copy"),)).code, "INCOMPLETE_COPY")

    def test_duplicate_completed_copy_rejected(self):
        task, topology = inputs("all-reduce", count=4)
        p = Problem(task, topology, 1)
        verifier = RoundVerifier(p)
        for transfers in build_baseline(p).rounds:
            self.assertIsNone(verifier.apply_round(transfers))
        # This pair has never sent before; the receiver already has the result.
        self.assertEqual(verifier.apply_round((ar(p, 2, 1, 0, 15, "copy"),)).code, "DUPLICATE_RECEIVE")

    def test_prefix_fork_does_not_modify_parent(self):
        task, topology = inputs(count=4)
        p = Problem(task, topology, 1)
        parent = RoundVerifier(p)
        branch = parent.fork()
        self.assertIsNone(branch.apply_round((ag(p, 0, 1, 1),)))
        self.assertEqual(parent.possession[1], 2)
        self.assertEqual(parent.sent, {})
        self.assertEqual(parent.round_number, 0)
        self.assertEqual(branch.possession[1], 3)

    def test_repeated_slot_in_one_transfer_rejected(self):
        task, topology = inputs("all-reduce", split=2)
        p = Problem(task, topology, 2)
        transfer = Transfer("reduce", 0, 1, p.route(0, 1), fragments=((0, 1), (0, 1)))
        self.assertEqual(RoundVerifier(p).apply_round((transfer,)).code, "PAYLOAD")

    def test_only_nonempty_rounds_consume_limit(self):
        task, topology = inputs()
        task["max_nonempty_rounds"] = 1
        p = Problem(task, topology, 1)
        verifier = RoundVerifier(p)
        for _ in range(3):
            self.assertIsNone(verifier.apply_round(()))
        self.assertIsNone(verifier.apply_round(build_baseline(p).rounds[0]))
        self.assertIsNone(verifier.apply_round(()))
        self.assertIsNone(verifier.finish())
        self.assertEqual(verifier.apply_round((ag(p, 0, 1, 1),)).code, "ROUND_LIMIT")

    def test_task_copy_policy_is_enforced(self):
        task, topology = inputs()
        task["allow_copy"] = False
        p = Problem(task, topology, 1)
        self.assertEqual(RoundVerifier(p).apply_round((ag(p, 0, 1, 1),)).code, "ACTION")


class PublicFormatTests(unittest.TestCase):
    def test_numeric_json_roundtrip_for_both_operators_and_multiple_splits(self):
        for operator in ("all-gather", "all-reduce"):
            for split in (1, 2, 4):
                with self.subTest(operator=operator, split=split):
                    task, topology = inputs(operator, count=4, split=split)
                    p = Problem(task, topology, split)
                    plan = build_baseline(p)
                    answer = plan_to_answer(p, plan)
                    self.assertEqual(plan_from_answer(p, answer), plan)
                    validate_answer(task, topology, answer)
                    self.assertEqual(estimate_plan_time_us(p, plan), estimate_schedule_time_us(task, topology, answer))

    def test_missing_final_chunks_are_rejected(self):
        task, topology = inputs()
        answer = build_solution(task, topology)
        answer["final_chunks"].pop()
        with self.assertRaises(AgentCCError) as caught:
            validate_answer(task, topology, answer)
        self.assertEqual(caught.exception.issue.code, "FINAL_CHUNKS")

    def test_expression_cannot_repeat_a_contributor(self):
        task, topology = inputs("all-reduce")
        answer = build_solution(task, topology)
        answer["rounds"][0]["transfers"][0]["chunk_ranges"] = ["r0_c0+r0_c0"]
        with self.assertRaises(AgentCCError) as caught:
            validate_answer(task, topology, answer)
        self.assertEqual(caught.exception.issue.round_number, 1)

    def test_direction_and_rank_intermediate_are_checked(self):
        task, topology = inputs(count=4)
        p = Problem(task, topology, 1)
        with self.assertRaises(AgentCCError):
            p.network.compile_route(["r0", "r1", "r2"])
        topology["edges"] = [{"src": "r0", "dst": "r1", "bw": "1GB/s"}]
        p = Problem(task, topology, 1)
        with self.assertRaises(AgentCCError):
            p.network.compile_route(["r1", "r0"])

    def test_invalid_resource_references_rejected(self):
        _, base = inputs()
        variants = [
            [{"id": "a", "capacity": "1GB/s"}, {"id": "a", "capacity": "1GB/s"}],
            [{"id": "a", "capacity": "1GB/s", "edges": [{"src": "r0", "dst": "r0"}]}],
        ]
        for resources in variants:
            topology = deepcopy(base)
            topology["resources"] = resources
            with self.assertRaises(AgentCCError):
                Network(topology)
        base["edges"][0]["resource_id"] = "missing"
        with self.assertRaises(AgentCCError):
            Network(base)


if __name__ == "__main__":
    unittest.main()
