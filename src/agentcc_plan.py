"""Numeric schedules, transactional round verification, and V22 Replay costs.

AllGather uses a bit per raw chunk (rank * split + slot). AllReduce uses a
contributor bit per rank for each slot. JSON strings exist only at the boundary.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable

from src.agentcc_network import AgentCCError, Network, Route, Router


def bit_indices(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def candidate_splits(task: dict[str, Any], preferred_split: int | None = None) -> list[int]:
    data_size = int(task["data_size"]["per_rank_bytes"])
    alignment = int(task["split_policy"]["alignment_bytes"])
    if data_size <= 0 or alignment <= 0:
        raise AgentCCError("data size and alignment must be positive")
    candidates = [int(value) for value in task["split_policy"]["candidates"]]
    preferred = task.get("split") if preferred_split is None else preferred_split
    choices = ([int(preferred)] if preferred in candidates else []) + candidates
    valid = list(dict.fromkeys(
        split for split in choices
        if split > 0 and data_size % split == 0 and (data_size // split) % alignment == 0
    ))
    if not valid:
        raise AgentCCError("no split candidate satisfies data and alignment constraints")
    return valid


class Problem:
    """Compile one request and one legal split; no fixture-derived assumptions."""

    def __init__(self, task: dict[str, Any], topology: dict[str, Any], split: int):
        if list(task["ranks"]) != list(topology["ranks"]):
            raise AgentCCError("task.ranks and topology.ranks must match")
        if type(split) is not int or split not in candidate_splits(task):
            raise AgentCCError("split is not a legal task candidate")
        self.operator = task["operator"]
        if self.operator not in ("all-gather", "all-reduce"):
            raise AgentCCError(f"unsupported operator: {self.operator}")
        self.network = Network(topology)
        self.ranks = self.network.ranks
        self.rank_ids = {rank: index for index, rank in enumerate(self.ranks)}
        self.split = split
        self.chunk_bytes = int(task["data_size"]["per_rank_bytes"]) // split
        self.max_rounds = int(task["max_nonempty_rounds"])
        self.allow_copy = task.get("allow_copy", topology.get("allow_copy", True))
        self.all_contributors = (1 << len(self.ranks)) - 1
        self.all_chunks = (1 << (len(self.ranks) * split)) - 1
        reference_bytes = self.chunk_bytes if self.operator == "all-gather" else self.chunk_bytes * split
        self.router = Router(self.network, reference_bytes)

    @cached_property
    def chunk_names(self) -> tuple[str, ...]:
        return tuple(f"{rank}_c{slot}" for rank in self.ranks for slot in range(self.split))

    @cached_property
    def chunk_ids(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.chunk_names)}

    def route(self, src: int, dst: int) -> Route:
        return self.router.route(src, dst)


@dataclass(frozen=True, slots=True)
class Transfer:
    action: str
    src: int
    dst: int
    route: Route
    chunks: int = 0  # AllGather: raw-chunk bitset.
    fragments: tuple[tuple[int, int], ...] = ()  # AllReduce: (slot, contributor bitset).


@dataclass(frozen=True, slots=True)
class Plan:
    operator: str
    split: int
    rounds: tuple[tuple[Transfer, ...], ...]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    round_number: int | None = None
    transfer_index: int | None = None
    src: int | None = None
    dst: int | None = None
    slot: int | None = None

    def raise_error(self):
        location = f"round {self.round_number}: " if self.round_number is not None else ""
        raise AgentCCError(f"{self.code}: {location}{self.message}", issue=self)


class RoundVerifier:
    """Commit a round only if all checks pass. Failed attempts leave no state.

    Senders always read the previous committed state. Receivers accumulate into
    a separate state, so same-round forwarding and duplicate arrivals are caught.
    Call ``finish`` only for a complete candidate; prefixes need not finish.
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.round_number = 0
        self.nonempty_rounds = 0
        self.possession = tuple(((1 << problem.split) - 1) << (rank * problem.split) for rank in range(len(problem.ranks)))
        self.partials = tuple(tuple(1 << rank for _ in range(problem.split)) for rank in range(len(problem.ranks)))
        self.sent: dict[tuple[int, int, int], int] = {}

    def fork(self) -> RoundVerifier:
        """Branch a prefix; immutable possession/state tuples can be shared."""
        branch = copy(self)
        branch.sent = self.sent.copy()
        return branch

    def apply_round(self, transfers: Iterable[Transfer]) -> ValidationIssue | None:
        transfers = tuple(transfers)
        p = self.problem
        round_number = self.round_number + 1
        if transfers and self.nonempty_rounds >= p.max_rounds:
            return ValidationIssue("ROUND_LIMIT", "too many nonempty rounds", round_number)
        possession = list(self.possession)
        partials = [list(row) for row in self.partials]
        sent_updates: dict[tuple[int, int, int], int] = {}
        received_slots: set[tuple[int, int]] = set()
        for index, transfer in enumerate(transfers):
            src, dst = transfer.src, transfer.dst

            def issue(code, message, slot=None):
                return ValidationIssue(code, message, round_number, index, src, dst, slot)

            if not (0 <= src < len(p.ranks) and 0 <= dst < len(p.ranks)) or src == dst:
                return issue("ENDPOINTS", "transfer endpoints must be distinct ranks")
            route = transfer.route
            if p.network.routes.get(route.nodes) is not route or route.nodes[0] != src or route.nodes[-1] != dst:
                return issue("PATH", "route must belong to this problem and match the endpoints")
            if transfer.action == "copy" and not p.allow_copy:
                return issue("ACTION", "task does not allow copy")
            if p.operator == "all-gather":
                mask = transfer.chunks
                if transfer.action != "copy":
                    return issue("ACTION", "AllGather only permits copy")
                if mask <= 0 or mask & ~p.all_chunks or transfer.fragments:
                    return issue("PAYLOAD", "invalid or empty AllGather chunk bitset")
                if mask & ~self.possession[src]:
                    return issue("SOURCE_STATE", "source lacks a chunk at round start")
                if mask & possession[dst]:
                    return issue("DUPLICATE_RECEIVE", "destination receives a duplicate chunk")
                key = (src, dst, -1)
                sent_mask = sent_updates.get(key, self.sent.get(key, 0))
                if mask & sent_mask:
                    return issue("DUPLICATE_SEND", "same chunk sent to the same pair twice")
                sent_updates[key] = sent_mask | mask
                possession[dst] |= mask
                continue

            if transfer.action not in ("reduce", "copy"):
                return issue("ACTION", "unsupported AllReduce action")
            if transfer.chunks or not transfer.fragments:
                return issue("PAYLOAD", "invalid or empty AllReduce fragments")
            slots = set()
            for slot, mask in transfer.fragments:
                if not 0 <= slot < p.split or slot in slots or mask <= 0 or mask & ~p.all_contributors:
                    return issue("PAYLOAD", "slots and contributors must be valid and unique", slot)
                slots.add(slot)
                if mask != self.partials[src][slot]:
                    return issue("SOURCE_STATE", "expression does not match source state at round start", slot)
                key = (src, dst, slot)
                sent_mask = sent_updates.get(key, self.sent.get(key, 0))
                if mask & sent_mask:
                    return issue("DUPLICATE_SEND", "same contribution sent to the same pair twice", slot)
                sent_updates[key] = sent_mask | mask
                if transfer.action == "reduce":
                    if mask & partials[dst][slot]:
                        return issue("DUPLICATE_CONTRIBUTOR", "reduce overlaps destination contributors", slot)
                    partials[dst][slot] |= mask
                else:
                    if mask != p.all_contributors:
                        return issue("INCOMPLETE_COPY", "copy requires a complete reduction", slot)
                    if partials[dst][slot] == mask or (dst, slot) in received_slots:
                        return issue("DUPLICATE_RECEIVE", "slot already complete or received this round", slot)
                    partials[dst][slot] = mask
                received_slots.add((dst, slot))
        self.possession = tuple(possession)
        self.partials = tuple(tuple(row) for row in partials)
        self.sent.update(sent_updates)
        self.round_number = round_number
        self.nonempty_rounds += bool(transfers)
        return None

    def finish(self) -> ValidationIssue | None:
        p = self.problem
        if p.operator == "all-gather":
            complete = all(mask == p.all_chunks for mask in self.possession)
        else:
            complete = all(mask == p.all_contributors for row in self.partials for mask in row)
        return None if complete else ValidationIssue("INCOMPLETE", "collective did not finish at every rank", self.round_number)


def validate_plan(problem: Problem, plan: Plan) -> None:
    if (plan.operator, plan.split) != (problem.operator, problem.split):
        ValidationIssue("HEADER", "plan operator/split does not match problem").raise_error()
    verifier = RoundVerifier(problem)
    for transfers in plan.rounds:
        issue = verifier.apply_round(transfers)
        if issue:
            issue.raise_error()
    issue = verifier.finish()
    if issue:
        issue.raise_error()


def build_baseline(problem: Problem) -> Plan:
    """Build and verify the existing recursive-doubling baseline in numeric form."""
    count = len(problem.ranks)
    if count == 0 or count & (count - 1):
        raise AgentCCError("the baseline requires a power-of-two rank count")
    verifier = RoundVerifier(problem)
    rounds = []
    step = 1
    while step < count:
        transfers = []
        for left in range(count):
            right = left ^ step
            if left >= right:
                continue
            for src, dst in ((left, right), (right, left)):
                route = problem.route(src, dst)
                if problem.operator == "all-gather":
                    mask = verifier.possession[src] & ~verifier.possession[dst]
                    transfers.append(Transfer("copy", src, dst, route, chunks=mask))
                else:
                    fragments = tuple(enumerate(verifier.partials[src]))
                    transfers.append(Transfer("reduce", src, dst, route, fragments=fragments))
        issue = verifier.apply_round(transfers)
        if issue:
            issue.raise_error()
        rounds.append(tuple(transfers))
        step <<= 1
    issue = verifier.finish()
    if issue:
        issue.raise_error()
    return Plan(problem.operator, problem.split, tuple(rounds))


@dataclass(frozen=True, slots=True)
class RoundTiming:
    total_us: float
    max_path_us: float
    max_overhead_us: float
    edge_busy_us: float
    resource_busy_us: float
    edge_loads: tuple[tuple[int, int], ...]
    resource_loads: tuple[tuple[int, int], ...]


def estimate_round(problem: Problem, transfers: Iterable[Transfer]) -> RoundTiming:
    network = problem.network
    edge_load: dict[int, int] = {}
    resource_load: dict[int, int] = {}
    max_path_us = max_overhead_us = 0.0
    for transfer in transfers:
        fragments = transfer.chunks.bit_count() if problem.operator == "all-gather" else len({slot for slot, _ in transfer.fragments})
        byte_count = problem.chunk_bytes * fragments
        overhead = network.startup_us + transfer.route.switch_count * network.switch_overhead_us
        path_time = overhead
        for edge in transfer.route.edges:
            path_time += 1e6 * byte_count / network.edge_capacities[edge]
            edge_load[edge] = edge_load.get(edge, 0) + byte_count
        for resource in transfer.route.resources:
            resource_load[resource] = resource_load.get(resource, 0) + byte_count
        max_path_us = max(max_path_us, path_time)
        max_overhead_us = max(max_overhead_us, overhead)
    edge_busy_us = max((1e6 * load / network.edge_capacities[edge] for edge, load in edge_load.items()), default=0.0)
    resource_busy_us = max((1e6 * load / network.resource_capacities[resource] for resource, load in resource_load.items()), default=0.0)
    total_us = max(max_path_us, max_overhead_us + max(edge_busy_us, resource_busy_us))
    return RoundTiming(total_us, max_path_us, max_overhead_us, edge_busy_us, resource_busy_us, tuple(edge_load.items()), tuple(resource_load.items()))


def estimate_plan_time_us(problem: Problem, plan: Plan) -> float:
    return sum(estimate_round(problem, transfers).total_us for transfers in plan.rounds)


def plan_to_answer(problem: Problem, plan: Plan) -> dict[str, Any]:
    """Serialize a verified winner once. Expression cache is request-local."""
    names = problem.chunk_names
    expressions: dict[tuple[int, int], str] = {}
    rounds = []
    for transfers in plan.rounds:
        if not transfers:
            continue
        output = []
        for transfer in transfers:
            if problem.operator == "all-gather":
                ranges = [names[index] for index in bit_indices(transfer.chunks)]
            else:
                ranges = []
                for slot, mask in transfer.fragments:
                    key = (slot, mask)
                    if key not in expressions:
                        expressions[key] = "+".join(names[rank * problem.split + slot] for rank in bit_indices(mask))
                    ranges.append(expressions[key])
            output.append({
                "action": transfer.action, "src": problem.ranks[transfer.src], "dst": problem.ranks[transfer.dst],
                "chunk_ranges": ranges, "path": [problem.network.nodes[node] for node in transfer.route.nodes],
            })
        rounds.append({"round": len(rounds) + 1, "transfers": output})
    return {
        "operator": plan.operator, "split": plan.split, "chunks": list(names),
        "initial_possession": {rank: list(names[index * problem.split:(index + 1) * problem.split]) for index, rank in enumerate(problem.ranks)},
        "final_chunks": list(names) if problem.operator == "all-gather" else [f"c{slot}_sum" for slot in range(problem.split)],
        "round_count": len(rounds), "rounds": rounds,
    }


def plan_from_answer(problem: Problem, answer: dict[str, Any]) -> Plan:
    """Parse the public explicit-path format without replaying its state yet."""
    p = problem
    expected_initial = {rank: list(p.chunk_names[index * p.split:(index + 1) * p.split]) for index, rank in enumerate(p.ranks)}
    expected_final = list(p.chunk_names) if p.operator == "all-gather" else [f"c{slot}_sum" for slot in range(p.split)]
    if answer["operator"] != p.operator or answer["split"] != p.split:
        ValidationIssue("HEADER", "operator/split mismatch").raise_error()
    if answer["chunks"] != list(p.chunk_names) or answer["initial_possession"] != expected_initial:
        ValidationIssue("HEADER", "chunks or initial_possession mismatch").raise_error()
    if answer["final_chunks"] != expected_final:
        ValidationIssue("FINAL_CHUNKS", "final_chunks must cover the collective result").raise_error()
    if answer["round_count"] != len(answer["rounds"]):
        ValidationIssue("ROUND_COUNT", "round_count mismatch").raise_error()
    rounds = []
    expression_cache: dict[str, tuple[int, int]] = {}
    for number, round_data in enumerate(answer["rounds"], start=1):
        if round_data["round"] != number:
            ValidationIssue("ROUND_NUMBER", "round numbers must be consecutive", number).raise_error()
        transfers = []
        for index, data in enumerate(round_data["transfers"]):
            try:
                src, dst = p.rank_ids[data["src"]], p.rank_ids[data["dst"]]
                route = p.network.compile_route(data["path"])
                mask = 0
                fragments = []
                for expression in data["chunk_ranges"]:
                    if p.operator == "all-gather":
                        bit = 1 << p.chunk_ids[expression]
                        if mask & bit:
                            raise AgentCCError("duplicate chunk in one transfer")
                        mask |= bit
                    else:
                        if expression not in expression_cache:
                            slot = None
                            contributors = 0
                            for token in expression.split("+"):
                                rank, token_slot = divmod(p.chunk_ids[token], p.split)
                                if (slot is not None and slot != token_slot) or contributors & (1 << rank):
                                    raise AgentCCError("expression requires unique contributors from one slot")
                                slot = token_slot
                                contributors |= 1 << rank
                            expression_cache[expression] = (slot, contributors)
                        fragments.append(expression_cache[expression])
                transfers.append(Transfer(data["action"], src, dst, route, mask, tuple(fragments)))
            except (KeyError, TypeError, AttributeError, AgentCCError) as error:
                ValidationIssue("TRANSFER_FORMAT", str(error), number, index).raise_error()
        rounds.append(tuple(transfers))
    return Plan(p.operator, p.split, tuple(rounds))
