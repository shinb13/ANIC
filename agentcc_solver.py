"""Small deterministic AgentCC baseline and local validator.

The solver works on the public AgentCC task/topology contract.  It uses
recursive doubling, which is a good correctness baseline for the official
256-rank power-of-two instances and for the public two-rank examples.
"""

from __future__ import annotations

from heapq import heappop, heappush
import json
from math import inf
import re
from pathlib import Path
from typing import Any


class AgentCCError(ValueError):
    """Raised when an input or generated solution violates the task contract."""


def _bandwidth_bytes_per_second(value: Any) -> float:
    if isinstance(value, (int, float)):
        if value <= 0:
            raise AgentCCError("bandwidth must be positive")
        return float(value)
    text = str(value).strip().upper().replace(" ", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(GB/S|GIB/S|GBPS|GBIT/S|MB/S|KB/S|B/S)", text)
    if not match:
        raise AgentCCError(f"unsupported bandwidth: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2)
    scale = {
        "B/S": 1,
        "KB/S": 10**3,
        "MB/S": 10**6,
        "GB/S": 10**9,
        "GIB/S": 2**30,
        "GBPS": 10**9 / 8,
        "GBIT/S": 10**9 / 8,
    }[unit]
    return amount * scale


def _topology_network(
    topology: dict[str, Any],
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], set[str]],
    dict[str, float],
    dict[str, list[str]],
]:
    """Return directed edge capacities, resource memberships, and adjacency."""

    edge_capacity: dict[tuple[str, str], float] = {}
    edge_resources: dict[tuple[str, str], set[str]] = {}
    adjacency: dict[str, list[str]] = {}

    def add_edge(src: str, dst: str, bandwidth: float) -> None:
        key = (src, dst)
        edge_capacity[key] = edge_capacity.get(key, 0.0) + bandwidth
        neighbours = adjacency.setdefault(src, [])
        if dst not in neighbours:
            neighbours.append(dst)

    for edge in topology["edges"]:
        src = edge["src"]
        dst = edge["dst"]
        bandwidth = _bandwidth_bytes_per_second(edge["bw"])
        add_edge(src, dst, bandwidth)
        resource_id = edge.get("resource_id")
        if resource_id is not None:
            edge_resources.setdefault((src, dst), set()).add(str(resource_id))
        if edge.get("bidirectional", False):
            add_edge(dst, src, bandwidth)
            if resource_id is not None:
                edge_resources.setdefault((dst, src), set()).add(str(resource_id))

    resource_capacity: dict[str, float] = {}
    for resource in topology.get("resources", []):
        resource_id = str(resource["id"])
        capacity = resource.get("capacity", resource.get("bw"))
        resource_capacity[resource_id] = _bandwidth_bytes_per_second(capacity)
        for edge in resource.get("edges", []):
            src = edge["src"]
            dst = edge["dst"]
            edge_resources.setdefault((src, dst), set()).add(resource_id)
            if edge.get("bidirectional", False):
                edge_resources.setdefault((dst, src), set()).add(resource_id)

    for neighbours in adjacency.values():
        neighbours.sort()
    return edge_capacity, edge_resources, resource_capacity, adjacency


def _edge_set(topology: dict[str, Any]) -> set[tuple[str, str]]:
    edge_capacity, _, _, _ = _topology_network(topology)
    return set(edge_capacity)


def _routes(
    task: dict[str, Any],
    topology: dict[str, Any],
    split: int,
) -> dict[tuple[str, str], list[str]]:
    """Find weighted switch-only-intermediate paths for every rank pair."""

    ranks = list(task["ranks"])
    switches = set(topology.get("switches", []))
    edge_capacity, edge_resources, resource_capacity, adjacency = _topology_network(topology)
    data_size = int(task["data_size"]["per_rank_bytes"])
    reference_bytes = data_size // split
    if task["operator"] == "all-reduce":
        reference_bytes = data_size
    switch_overhead = float(topology.get("switch_processing_overhead_us", 0))

    def edge_cost(src: str, dst: str) -> float:
        cost = 1e6 * reference_bytes / edge_capacity[(src, dst)]
        for resource_id in edge_resources.get((src, dst), ()):
            cost += 1e6 * reference_bytes / resource_capacity[resource_id]
        return cost

    routes: dict[tuple[str, str], list[str]] = {}
    rank_set = set(ranks)
    for source in ranks:
        distance: dict[str, float] = {source: 0.0}
        previous: dict[str, str | None] = {source: None}
        queue: list[tuple[float, str]] = [(0.0, source)]
        while queue:
            current_distance, node = heappop(queue)
            if current_distance != distance[node]:
                continue
            for neighbour in adjacency.get(node, []):
                if neighbour in rank_set and neighbour != source:
                    candidate = current_distance + edge_cost(node, neighbour)
                    if candidate < distance.get(neighbour, inf):
                        distance[neighbour] = candidate
                        previous[neighbour] = node
                    continue
                if neighbour in rank_set:
                    continue
                candidate = current_distance + edge_cost(node, neighbour) + switch_overhead
                if candidate < distance.get(neighbour, inf):
                    distance[neighbour] = candidate
                    previous[neighbour] = node
                    heappush(queue, (candidate, neighbour))

        for destination in ranks:
            if destination == source:
                continue
            if destination not in previous:
                raise AgentCCError(f"no legal route from {source} to {destination}")
            path: list[str] = []
            node: str | None = destination
            while node is not None:
                path.append(node)
                node = previous[node]
            path.reverse()
            if any(node not in switches for node in path[1:-1]):
                raise AgentCCError(f"route uses a rank as an intermediate node: {path}")
            routes[(source, destination)] = path
    return routes


def _candidate_splits(task: dict[str, Any], preferred_split: int | None = None) -> list[int]:
    data_size = int(task["data_size"]["per_rank_bytes"])
    alignment = int(task["split_policy"]["alignment_bytes"])
    candidates = [int(value) for value in task["split_policy"]["candidates"]]
    preferred = task.get("split") if preferred_split is None else preferred_split
    choices = ([int(preferred)] if preferred in candidates else []) + candidates
    seen: set[int] = set()
    valid: list[int] = []
    for split in choices:
        if split in seen:
            continue
        seen.add(split)
        if split > 0 and data_size % split == 0 and (data_size // split) % alignment == 0:
            valid.append(split)
    if not valid:
        raise AgentCCError("no split candidate satisfies data and alignment constraints")
    return valid


def _raw_chunk(rank: str, slot: int) -> str:
    return f"{rank}_c{slot}"


def _all_chunks(ranks: list[str], split: int) -> list[str]:
    return [_raw_chunk(rank, slot) for rank in ranks for slot in range(split)]


def _copy_possession(possession: dict[str, set[str]]) -> dict[str, set[str]]:
    return {rank: set(chunks) for rank, chunks in possession.items()}


def _ordered_chunks(chunks: set[str], order: dict[str, int]) -> list[str]:
    return sorted(chunks, key=lambda chunk: order[chunk])


def _pair_rounds(ranks: list[str]):
    count = len(ranks)
    if count == 0 or count & (count - 1):
        raise AgentCCError("the baseline requires a power-of-two rank count")
    step = 1
    while step < count:
        pairs = []
        for index in range(count):
            partner = index ^ step
            if index < partner:
                pairs.append((index, partner))
        yield pairs
        step <<= 1


def _allgather(task: dict[str, Any], topology: dict[str, Any], routes: dict[tuple[str, str], list[str]], split: int) -> dict[str, Any]:
    ranks = list(task["ranks"])
    if not task.get("allow_copy", True):
        raise AgentCCError("AllGather requires copy support")
    possession = {
        rank: {_raw_chunk(rank, slot) for slot in range(split)}
        for rank in ranks
    }
    chunk_order = {chunk: index for index, chunk in enumerate(_all_chunks(ranks, split))}
    rounds: list[dict[str, Any]] = []
    for pairs in _pair_rounds(ranks):
        start = _copy_possession(possession)
        transfers: list[dict[str, Any]] = []
        updates: dict[str, set[str]] = {rank: set() for rank in ranks}
        for left_index, right_index in pairs:
            left = ranks[left_index]
            right = ranks[right_index]
            left_chunks = _ordered_chunks(start[left] - start[right], chunk_order)
            right_chunks = _ordered_chunks(start[right] - start[left], chunk_order)
            if left_chunks:
                transfers.append({
                    "action": "copy",
                    "src": left,
                    "dst": right,
                    "chunk_ranges": left_chunks,
                    "path": routes[(left, right)],
                })
                updates[right].update(left_chunks)
            if right_chunks:
                transfers.append({
                    "action": "copy",
                    "src": right,
                    "dst": left,
                    "chunk_ranges": right_chunks,
                    "path": routes[(right, left)],
                })
                updates[left].update(right_chunks)
        for rank in ranks:
            possession[rank].update(updates[rank])
        rounds.append({"round": len(rounds) + 1, "transfers": transfers})

    chunks = _all_chunks(ranks, split)
    return {
        "operator": "all-gather",
        "split": split,
        "chunks": chunks,
        "initial_possession": {
            rank: [_raw_chunk(rank, slot) for slot in range(split)]
            for rank in ranks
        },
        "final_chunks": chunks,
        "round_count": len(rounds),
        "rounds": rounds,
    }


def _reduce_expression(contributors: set[int], ranks: list[str], slot: int) -> str:
    return "+".join(_raw_chunk(ranks[index], slot) for index in sorted(contributors))


def _allreduce(task: dict[str, Any], topology: dict[str, Any], routes: dict[tuple[str, str], list[str]], split: int) -> dict[str, Any]:
    ranks = list(task["ranks"])
    partial: dict[str, dict[int, set[int]]] = {
        rank: {slot: {index} for slot in range(split)}
        for index, rank in enumerate(ranks)
    }
    rounds: list[dict[str, Any]] = []
    for pairs in _pair_rounds(ranks):
        start = {
            rank: {slot: set(contributors) for slot, contributors in slots.items()}
            for rank, slots in partial.items()
        }
        updates = {
            rank: {slot: set(contributors) for slot, contributors in slots.items()}
            for rank, slots in start.items()
        }
        transfers: list[dict[str, Any]] = []
        for left_index, right_index in pairs:
            left = ranks[left_index]
            right = ranks[right_index]
            left_ranges = [_reduce_expression(start[left][slot], ranks, slot) for slot in range(split)]
            right_ranges = [_reduce_expression(start[right][slot], ranks, slot) for slot in range(split)]
            transfers.append({
                "action": "reduce",
                "src": left,
                "dst": right,
                "chunk_ranges": left_ranges,
                "path": routes[(left, right)],
            })
            transfers.append({
                "action": "reduce",
                "src": right,
                "dst": left,
                "chunk_ranges": right_ranges,
                "path": routes[(right, left)],
            })
            for slot in range(split):
                updates[left][slot].update(start[right][slot])
                updates[right][slot].update(start[left][slot])
        partial = updates
        rounds.append({"round": len(rounds) + 1, "transfers": transfers})

    chunks = _all_chunks(ranks, split)
    return {
        "operator": "all-reduce",
        "split": split,
        "chunks": chunks,
        "initial_possession": {
            rank: [_raw_chunk(rank, slot) for slot in range(split)]
            for rank in ranks
        },
        "final_chunks": [f"c{slot}_sum" for slot in range(split)],
        "round_count": len(rounds),
        "rounds": rounds,
    }


def estimate_schedule_time_us(task: dict[str, Any], topology: dict[str, Any], answer: dict[str, Any]) -> float:
    """Estimate the V22 Replay communication time for a generated answer."""

    edge_capacity, edge_resources, resource_capacity, _ = _topology_network(topology)
    chunk_bytes = int(task["data_size"]["per_rank_bytes"]) // int(answer["split"])
    startup_us = float(topology.get("communication_startup_time_us", 0))
    switch_overhead_us = float(topology.get("switch_processing_overhead_us", 0))
    total_us = 0.0

    for round_data in answer["rounds"]:
        edge_load: dict[tuple[str, str], int] = {}
        resource_load: dict[str, int] = {}
        transfer_times: list[float] = []
        for transfer in round_data["transfers"]:
            byte_count = chunk_bytes * len(transfer["chunk_ranges"])
            path = transfer["path"]
            path_time = startup_us + max(0, len(path) - 2) * switch_overhead_us
            for src, dst in zip(path, path[1:]):
                edge = (src, dst)
                path_time += 1e6 * byte_count / edge_capacity[edge]
                edge_load[edge] = edge_load.get(edge, 0) + byte_count
                for resource_id in edge_resources.get(edge, ()):
                    resource_load[resource_id] = resource_load.get(resource_id, 0) + byte_count
            transfer_times.append(path_time)

        if not transfer_times:
            continue
        edge_busy_us = max(
            (1e6 * byte_count / edge_capacity[edge] for edge, byte_count in edge_load.items()),
            default=0.0,
        )
        resource_busy_us = max(
            (1e6 * byte_count / resource_capacity[resource_id] for resource_id, byte_count in resource_load.items()),
            default=0.0,
        )
        contention_time = startup_us + switch_overhead_us + max(edge_busy_us, resource_busy_us)
        total_us += max(max(transfer_times), contention_time)
    return total_us


def build_solution(
    task: dict[str, Any],
    topology: dict[str, Any],
    preferred_split: int | None = None,
) -> dict[str, Any]:
    if list(task["ranks"]) != list(topology["ranks"]):
        raise AgentCCError("task.ranks and topology.ranks must match")
    split = min(_candidate_splits(task, preferred_split))
    routes = _routes(task, topology, split)
    if task["operator"] == "all-gather":
        answer = _allgather(task, topology, routes, split)
    elif task["operator"] == "all-reduce":
        answer = _allreduce(task, topology, routes, split)
    else:
        raise AgentCCError(f"unsupported operator: {task['operator']}")
    if answer["round_count"] > int(task["max_nonempty_rounds"]):
        raise AgentCCError("generated schedule exceeds max_nonempty_rounds")
    return answer


def solve_input(task_input: dict[str, Any], preferred_split: int | None = None) -> dict[str, Any]:
    """Return the business response expected from BaseAgent.solve()."""

    task = task_input["task"]
    topology = task_input["topology"]
    answer = build_solution(task, topology, preferred_split)
    validate_answer(task, topology, answer)
    return {"answer": answer}


def _parse_raw_chunk(
    chunk: str,
    ranks: list[str],
    split: int,
    rank_lookup: dict[str, int] | None = None,
) -> tuple[int, int]:
    if rank_lookup is None:
        rank_lookup = {rank: index for index, rank in enumerate(ranks)}
    rank, separator, slot_text = chunk.rpartition("_c")
    if separator and rank in rank_lookup and slot_text.isdigit() and int(slot_text) < split:
        return rank_lookup[rank], int(slot_text)
    raise AgentCCError(f"invalid chunk name: {chunk}")


def _parse_expression(
    expression: str,
    ranks: list[str],
    split: int,
    rank_lookup: dict[str, int],
) -> tuple[int, set[int]]:
    tokens = expression.split("+")
    if not tokens or any(not token for token in tokens):
        raise AgentCCError(f"invalid reduce expression: {expression}")
    parsed = [_parse_raw_chunk(token, ranks, split, rank_lookup) for token in tokens]
    slots = {slot for _, slot in parsed}
    if len(slots) != 1 or len({rank for rank, _ in parsed}) != len(parsed):
        raise AgentCCError(f"reduce expression must contain unique contributors from one slot: {expression}")
    return parsed[0][1], {rank for rank, _ in parsed}


def validate_answer(task: dict[str, Any], topology: dict[str, Any], answer: dict[str, Any]) -> None:
    """Validate the correctness invariants needed by the public format."""

    ranks = list(task["ranks"])
    split = int(answer["split"])
    data_size = int(task["data_size"]["per_rank_bytes"])
    alignment = int(task["split_policy"]["alignment_bytes"])
    if split not in {int(value) for value in task["split_policy"]["candidates"]}:
        raise AgentCCError("split is not one of the task candidates")
    if data_size % split != 0 or (data_size // split) % alignment != 0:
        raise AgentCCError("split violates data size or alignment")
    expected_chunks = _all_chunks(ranks, split)
    if answer["operator"] != task["operator"]:
        raise AgentCCError("operator mismatch")
    if answer["chunks"] != expected_chunks:
        raise AgentCCError("chunks are incomplete or out of order")
    expected_initial = {
        rank: [_raw_chunk(rank, slot) for slot in range(split)]
        for rank in ranks
    }
    if answer["initial_possession"] != expected_initial:
        raise AgentCCError("initial_possession mismatch")
    if answer["round_count"] != len(answer["rounds"]):
        raise AgentCCError("round_count mismatch")
    if answer["round_count"] > int(task["max_nonempty_rounds"]):
        raise AgentCCError("too many rounds")

    edge_set = _edge_set(topology)
    switches = set(topology.get("switches", []))
    rank_set = set(ranks)
    routes = set()
    for round_number, round_data in enumerate(answer["rounds"], start=1):
        if round_data["round"] != round_number:
            raise AgentCCError("round numbers must be consecutive")
        for transfer in round_data["transfers"]:
            src = transfer["src"]
            dst = transfer["dst"]
            path = transfer["path"]
            if src not in rank_set or dst not in rank_set or src == dst:
                raise AgentCCError("transfer endpoints must be distinct ranks")
            if path[0] != src or path[-1] != dst:
                raise AgentCCError("path endpoints do not match transfer")
            if any(node not in switches for node in path[1:-1]):
                raise AgentCCError("path contains a rank as an intermediate node")
            if any((left, right) not in edge_set for left, right in zip(path, path[1:])):
                raise AgentCCError(f"path contains an undeclared edge: {path}")
            routes.add((src, dst, round_number))

    if task["operator"] == "all-gather":
        _validate_allgather(task, answer, ranks, split)
    else:
        _validate_allreduce(task, answer, ranks, split)


def _validate_allgather(task: dict[str, Any], answer: dict[str, Any], ranks: list[str], split: int) -> None:
    possession = {
        rank: {_raw_chunk(rank, slot) for slot in range(split)}
        for rank in ranks
    }
    rank_lookup = {rank: index for index, rank in enumerate(ranks)}
    sent_pairs: set[tuple[str, str, str]] = set()
    for round_data in answer["rounds"]:
        start = _copy_possession(possession)
        updates = {rank: set() for rank in ranks}
        for transfer in round_data["transfers"]:
            if transfer["action"] != "copy":
                raise AgentCCError("AllGather only permits copy")
            src, dst = transfer["src"], transfer["dst"]
            for chunk in transfer["chunk_ranges"]:
                _parse_raw_chunk(chunk, ranks, split, rank_lookup)
                if chunk not in start[src]:
                    raise AgentCCError("source does not possess chunk at round start")
                if chunk in start[dst] or chunk in updates[dst]:
                    raise AgentCCError("destination receives a duplicate chunk")
                key = (chunk, src, dst)
                if key in sent_pairs:
                    raise AgentCCError("same chunk sent to the same pair twice")
                sent_pairs.add(key)
                updates[dst].add(chunk)
        for rank in ranks:
            possession[rank].update(updates[rank])

    expected = set(_all_chunks(ranks, split))
    if any(chunks != expected for chunks in possession.values()):
        raise AgentCCError("AllGather did not finish at every rank")


def _validate_allreduce(task: dict[str, Any], answer: dict[str, Any], ranks: list[str], split: int) -> None:
    partial: dict[str, dict[int, set[int]]] = {
        rank: {slot: {index} for slot in range(split)}
        for index, rank in enumerate(ranks)
    }
    rank_lookup = {rank: index for index, rank in enumerate(ranks)}
    sent_pairs: set[tuple[str, str, str]] = set()
    for round_data in answer["rounds"]:
        start = {
            rank: {slot: set(contributors) for slot, contributors in slots.items()}
            for rank, slots in partial.items()
        }
        updates = {
            rank: {slot: set(contributors) for slot, contributors in slots.items()}
            for rank, slots in start.items()
        }
        for transfer in round_data["transfers"]:
            action = transfer["action"]
            src, dst = transfer["src"], transfer["dst"]
            if len(transfer["chunk_ranges"]) != split:
                raise AgentCCError("AllReduce transfer must cover every slot")
            for expression in transfer["chunk_ranges"]:
                slot, contributors = _parse_expression(expression, ranks, split, rank_lookup)
                expected_source = start[src][slot]
                if contributors != expected_source:
                    raise AgentCCError("transfer expression does not match source state")
                key_prefix = (src, dst)
                for contributor in contributors:
                    key = (_raw_chunk(ranks[contributor], slot), src, dst)
                    if key in sent_pairs:
                        raise AgentCCError("same contribution sent to the same pair twice")
                    sent_pairs.add(key)
                if action == "reduce":
                    if updates[dst][slot] & contributors:
                        raise AgentCCError("reduce contains a duplicate contributor")
                    updates[dst][slot].update(contributors)
                elif action == "copy":
                    if contributors != set(range(len(ranks))):
                        raise AgentCCError("copy is only valid for a complete reduction")
                    if not updates[dst][slot].issubset(contributors):
                        raise AgentCCError("copy would discard a contributor")
                    updates[dst][slot] = set(contributors)
                else:
                    raise AgentCCError(f"unsupported AllReduce action: {action}")
        partial = updates

    all_contributors = set(range(len(ranks)))
    if any(partial[rank][slot] != all_contributors for rank in ranks for slot in range(split)):
        raise AgentCCError("AllReduce did not finish at every rank")


def validate_request(task: dict[str, Any], topology: dict[str, Any], response: dict[str, Any]) -> None:
    if response.get("status") != "success":
        raise AgentCCError("response status is not success")
    validate_answer(task, topology, response["output"]["answer"])


def solve_request(task: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any]:
    answer = build_solution(task, topology)
    response = {
        "task_id": task["task_id"],
        "status": "success",
        "output": {"answer": answer},
    }
    if "instance_id" in task:
        response["instance_id"] = task["instance_id"]
    validate_request(task, topology, response)
    return response


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("topology")
    args = parser.parse_args()
    print(json.dumps(solve_request(load_json(args.task), load_json(args.topology)), ensure_ascii=False, indent=2))
