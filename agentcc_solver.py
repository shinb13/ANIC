"""AgentCC schedule construction, topology-based optimization, and validation."""

from __future__ import annotations

from heapq import heappop, heappush
import json
from math import inf
import re
from pathlib import Path
from time import perf_counter
from typing import Any


class AgentCCError(ValueError):
    """Raised when an input or generated solution violates the task contract."""


ROUTE_POLICIES = {
    "baseline": 1.0,
    "bandwidth_first": 0.25,
    "resource_first": 2.0,
}

SCHEDULE_ALGORITHMS = {
    "all-gather": ("recursive_doubling", "ordered_doubling"),
    "all-reduce": ("recursive_doubling", "tree"),
}

BALANCE_MAX_PATH_SEARCHES = 64


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


def _edge_set(
    topology: dict[str, Any],
    network: tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], set[str]],
        dict[str, float],
        dict[str, list[str]],
    ] | None = None,
) -> set[tuple[str, str]]:
    edge_capacity, _, _, _ = network if network is not None else _topology_network(topology)
    return set(edge_capacity)


def _routes(
    task: dict[str, Any],
    topology: dict[str, Any],
    split: int,
    route_policy: str = "baseline",
    network: tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], set[str]],
        dict[str, float],
        dict[str, list[str]],
    ] | None = None,
) -> dict[tuple[str, str], list[str]]:
    """Find weighted paths for the XOR pairs used by all supported schedules."""

    ranks = list(task["ranks"])
    if not ranks or len(ranks) & (len(ranks) - 1):
        raise AgentCCError("the baseline requires a power-of-two rank count")
    switches = set(topology.get("switches", []))
    if route_policy not in ROUTE_POLICIES:
        raise AgentCCError(f"unknown route policy: {route_policy}")
    edge_capacity, edge_resources, resource_capacity, adjacency = (
        network if network is not None else _topology_network(topology)
    )
    data_size = int(task["data_size"]["per_rank_bytes"])
    reference_bytes = data_size // split
    if task["operator"] == "all-reduce":
        reference_bytes = data_size
    resource_penalty = ROUTE_POLICIES[route_policy]
    switch_overhead = float(topology.get("switch_processing_overhead_us", 0))

    edge_costs = {
        edge: 1e6 * reference_bytes / capacity
        + resource_penalty * sum(
            1e6 * reference_bytes / resource_capacity[resource_id]
            for resource_id in edge_resources.get(edge, ())
        )
        for edge, capacity in edge_capacity.items()
    }

    routes: dict[tuple[str, str], list[str]] = {}
    rank_set = set(ranks)
    for source_index, source in enumerate(ranks):
        targets = {ranks[source_index ^ (1 << bit)] for bit in range(len(ranks).bit_length() - 1)}
        distance: dict[str, float] = {source: 0.0}
        previous: dict[str, str | None] = {source: None}
        queue: list[tuple[float, str]] = [(0.0, source)]
        while queue:
            current_distance, node = heappop(queue)
            if current_distance != distance[node]:
                continue
            for neighbour in adjacency.get(node, []):
                if neighbour in rank_set and neighbour != source:
                    if neighbour not in targets:
                        continue
                    candidate = current_distance + edge_costs[(node, neighbour)]
                    if candidate < distance.get(neighbour, inf):
                        distance[neighbour] = candidate
                        previous[neighbour] = node
                    continue
                if neighbour in rank_set:
                    continue
                candidate = current_distance + edge_costs[(node, neighbour)] + switch_overhead
                if candidate < distance.get(neighbour, inf):
                    distance[neighbour] = candidate
                    previous[neighbour] = node
                    heappush(queue, (candidate, neighbour))

        for destination in sorted(targets):
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


def _pair_rounds(ranks: list[str], dimension_order: list[int] | None = None):
    count = len(ranks)
    if count == 0 or count & (count - 1):
        raise AgentCCError("the baseline requires a power-of-two rank count")
    steps = (
        [1 << bit for bit in range(count.bit_length() - 1)]
        if dimension_order is None else dimension_order
    )
    for step in steps:
        pairs = []
        for index in range(count):
            partner = index ^ step
            if index < partner:
                pairs.append((index, partner))
        yield pairs


def doubling_round_profiles(
    task: dict[str, Any],
    topology: dict[str, Any],
    routes: dict[tuple[str, str], list[str]],
    network: tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], set[str]],
        dict[str, float],
        dict[str, list[str]],
    ],
) -> list[dict[str, Any]]:
    """Measure each XOR dimension using the current paths and shared resources."""

    edge_capacity, edge_resources, resource_capacity, _ = network
    ranks = task["ranks"]
    data_size = int(task["data_size"]["per_rank_bytes"])
    dimension_count = len(ranks).bit_length() - 1
    payloads = (
        [data_size * (1 << position) for position in range(dimension_count)]
        if task["operator"] == "all-gather" else [data_size]
    )
    startup_us = float(topology.get("communication_startup_time_us", 0))
    switch_us = float(topology.get("switch_processing_overhead_us", 0))
    profiles = []
    for bit in range(dimension_count):
        step = 1 << bit
        edge_counts: dict[tuple[str, str], int] = {}
        resource_counts: dict[str, int] = {}
        path_costs = []
        paths = [routes[(source, ranks[index ^ step])] for index, source in enumerate(ranks)]
        for path in paths:
            overhead = startup_us + (len(path) - 2) * switch_us
            per_byte_us = 0.0
            for edge in zip(path, path[1:]):
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
                per_byte_us += 1e6 / edge_capacity[edge]
                for resource_id in edge_resources.get(edge, ()):
                    resource_counts[resource_id] = resource_counts.get(resource_id, 0) + 1
            path_costs.append((overhead, per_byte_us))

        bottleneck_edge = max(edge_counts, key=lambda edge: edge_counts[edge] / edge_capacity[edge])
        busy_per_byte_us = max(
            max(1e6 * count / edge_capacity[edge] for edge, count in edge_counts.items()),
            max(
                (1e6 * count / resource_capacity[rid] for rid, count in resource_counts.items()),
                default=0.0,
            ),
        )
        max_overhead = max(overhead for overhead, _ in path_costs)
        resource_evidence = None
        if resource_counts:
            rid = max(resource_counts, key=lambda item: resource_counts[item] / resource_capacity[item])
            used_edges = sorted(edge for edge in edge_counts if rid in edge_resources.get(edge, ()))
            resource_evidence = {
                "id": rid,
                "capacity_bytes_per_second": resource_capacity[rid],
                "path_edge_traversals": resource_counts[rid],
                "used_member_count": len(used_edges),
                "used_member_examples": [list(edge) for edge in used_edges[:4]],
            }
        src, dst = bottleneck_edge
        profiles.append({
            "step": step,
            "times_us": [
                max(
                    max(overhead + per_byte * size for overhead, per_byte in path_costs),
                    max_overhead + busy_per_byte_us * size,
                )
                for size in payloads
            ],
            "bottleneck_edge": {
                "src": src,
                "dst": dst,
                "bandwidth_bytes_per_second": edge_capacity[bottleneck_edge],
                "path_traversals": edge_counts[bottleneck_edge],
                "resource_ids": sorted(edge_resources.get(bottleneck_edge, ())),
            },
            "bottleneck_resource": resource_evidence,
            "example_path": paths[0],
        })
    return profiles


def optimal_doubling_order(profiles: list[dict[str, Any]]) -> list[int]:
    """Find the cheapest order for fixed XOR dimensions and fixed routes."""

    dimension_count = len(profiles)
    plans: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    for mask in range(1 << dimension_count):
        cost, order = plans[mask]
        position = mask.bit_count()
        for dimension, profile in enumerate(profiles):
            if mask & (1 << dimension):
                continue
            next_mask = mask | (1 << dimension)
            candidate = cost + profile["times_us"][position]
            if next_mask not in plans or candidate < plans[next_mask][0]:
                plans[next_mask] = candidate, order + [profile["step"]]
    return plans[(1 << dimension_count) - 1][1]


def _allgather(
    task: dict[str, Any],
    topology: dict[str, Any],
    routes: dict[tuple[str, str], list[str]],
    split: int,
    dimension_order: list[int] | None = None,
) -> dict[str, Any]:
    ranks = list(task["ranks"])
    if not task.get("allow_copy", True):
        raise AgentCCError("AllGather requires copy support")
    possession = {
        rank: {_raw_chunk(rank, slot) for slot in range(split)}
        for rank in ranks
    }
    chunk_order = {chunk: index for index, chunk in enumerate(_all_chunks(ranks, split))}
    rounds: list[dict[str, Any]] = []
    for pairs in _pair_rounds(ranks, dimension_order):
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


def _allreduce_tree(
    task: dict[str, Any],
    routes: dict[tuple[str, str], list[str]],
    split: int,
) -> dict[str, Any]:
    ranks = list(task["ranks"])
    if not task.get("allow_copy", True):
        raise AgentCCError("tree AllReduce requires copy support")
    contributors = {index: {index} for index in range(len(ranks))}
    active = set(contributors)
    stages = []
    rounds = []
    for pairs in _pair_rounds(ranks):
        transfers = []
        active_pairs = []
        for left, right in pairs:
            if left not in active:
                continue
            transfers.append({
                "action": "reduce",
                "src": ranks[right],
                "dst": ranks[left],
                "chunk_ranges": [
                    _reduce_expression(contributors[right], ranks, slot) for slot in range(split)
                ],
                "path": routes[(ranks[right], ranks[left])],
            })
            contributors[left].update(contributors[right])
            active.remove(right)
            active_pairs.append((left, right))
        stages.append(active_pairs)
        rounds.append({"round": len(rounds) + 1, "transfers": transfers})

    complete = [
        _reduce_expression(set(range(len(ranks))), ranks, slot) for slot in range(split)
    ]
    for pairs in reversed(stages):
        rounds.append({
            "round": len(rounds) + 1,
            "transfers": [
                {
                    "action": "copy",
                    "src": ranks[left],
                    "dst": ranks[right],
                    "chunk_ranges": complete,
                    "path": routes[(ranks[left], ranks[right])],
                }
                for left, right in pairs
            ],
        })
    return {
        "operator": "all-reduce",
        "split": split,
        "chunks": _all_chunks(ranks, split),
        "initial_possession": {
            rank: [_raw_chunk(rank, slot) for slot in range(split)] for rank in ranks
        },
        "final_chunks": [f"c{slot}_sum" for slot in range(split)],
        "round_count": len(rounds),
        "rounds": rounds,
    }


def _schedule_loads(
    task: dict[str, Any],
    topology: dict[str, Any],
    answer: dict[str, Any],
    network: tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], set[str]],
        dict[str, float],
        dict[str, list[str]],
    ] | None = None,
) -> list[tuple[dict[tuple[str, str], int], dict[str, int]]]:
    _, edge_resources, _, _ = network if network is not None else _topology_network(topology)
    chunk_bytes = int(task["data_size"]["per_rank_bytes"]) // int(answer["split"])
    round_loads: list[tuple[dict[tuple[str, str], int], dict[str, int]]] = []
    for round_data in answer["rounds"]:
        edge_load: dict[tuple[str, str], int] = {}
        resource_load: dict[str, int] = {}
        for transfer in round_data["transfers"]:
            byte_count = chunk_bytes * len(transfer["chunk_ranges"])
            for src, dst in zip(transfer["path"], transfer["path"][1:]):
                edge = (src, dst)
                edge_load[edge] = edge_load.get(edge, 0) + byte_count
                for resource_id in edge_resources.get(edge, ()):
                    resource_load[resource_id] = resource_load.get(resource_id, 0) + byte_count
        round_loads.append((edge_load, resource_load))
    return round_loads


def estimate_schedule_time_us(
    task: dict[str, Any],
    topology: dict[str, Any],
    answer: dict[str, Any],
    network: tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], set[str]],
        dict[str, float],
        dict[str, list[str]],
    ] | None = None,
) -> float:
    """Estimate the V22 Replay communication time for a generated answer."""

    edge_capacity, _, resource_capacity, _ = network if network is not None else _topology_network(topology)
    round_loads = _schedule_loads(task, topology, answer, network)
    chunk_bytes = int(task["data_size"]["per_rank_bytes"]) // int(answer["split"])
    startup_us = float(topology.get("communication_startup_time_us", 0))
    switch_overhead_us = float(topology.get("switch_processing_overhead_us", 0))
    total_us = 0.0

    for round_index, round_data in enumerate(answer["rounds"]):
        edge_load, resource_load = round_loads[round_index]
        transfer_times: list[float] = []
        max_startup_and_switch_us = 0.0
        for transfer in round_data["transfers"]:
            byte_count = chunk_bytes * len(transfer["chunk_ranges"])
            path = transfer["path"]
            startup_and_switch_us = startup_us + max(0, len(path) - 2) * switch_overhead_us
            max_startup_and_switch_us = max(max_startup_and_switch_us, startup_and_switch_us)
            path_time = startup_and_switch_us
            for src, dst in zip(path, path[1:]):
                edge = (src, dst)
                path_time += 1e6 * byte_count / edge_capacity[edge]
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
        contention_time = max_startup_and_switch_us + max(edge_busy_us, resource_busy_us)
        total_us += max(max(transfer_times), contention_time)
    return total_us


def _least_loaded_path(source, destination, adjacency, edge_load, resource_load, byte_count, deadline):
    """Minimize incremental squared utilization; the caller checks Replay cost."""

    distance = {source: 0.0}
    previous = {}
    queue = [(0.0, source)]
    while queue:
        if perf_counter() >= deadline:
            return None
        cost, node = heappop(queue)
        if cost != distance[node]:
            continue
        if node == destination:
            path = [destination]
            while path[-1] != source:
                path.append(previous[path[-1]])
            return list(reversed(path))
        for neighbour, edge, per_byte_us, resources, is_switch in adjacency.get(node, ()):
            if not is_switch and neighbour != destination:
                continue
            weight = (2 * edge_load.get(edge, 0) + byte_count) * per_byte_us**2
            weight += sum(
                (2 * resource_load.get(rid, 0) + byte_count) * resource_us**2
                for rid, resource_us in resources
            )
            candidate = cost + weight
            if candidate < distance.get(neighbour, inf):
                distance[neighbour] = candidate
                previous[neighbour] = node
                heappush(queue, (candidate, neighbour))
    raise AgentCCError(f"no legal route from {source} to {destination}")


def balance_schedule_routes(task, topology, answer, network, *, deadline, max_searches=BALANCE_MAX_PATH_SEARCHES):
    """Refine paths in place without increasing any round's Replay estimate."""

    started = perf_counter()
    edge_capacity, edge_resources, resource_capacity, adjacency = network
    switches = set(topology.get("switches", []))
    weighted_adjacency = {
        node: [
            (
                neighbour, (node, neighbour), 1e6 / edge_capacity[(node, neighbour)],
                [(rid, 1e6 / resource_capacity[rid]) for rid in sorted(edge_resources.get((node, neighbour), ()))],
                neighbour in switches,
            )
            for neighbour in neighbours
        ]
        for node, neighbours in adjacency.items()
    }
    chunk_bytes = int(task["data_size"]["per_rank_bytes"]) // int(answer["split"])
    loads = _schedule_loads(task, topology, answer, network)
    round_times = [
        estimate_schedule_time_us(task, topology, {"split": answer["split"], "rounds": [round_data]}, network)
        for round_data in answer["rounds"]
    ]
    initial_time = sum(round_times)
    searches = changed = 0
    stop_reason = "candidates_exhausted"
    for round_index in sorted(range(len(round_times)), key=lambda index: (-round_times[index], index)):
        round_data = answer["rounds"][round_index]
        edge_load, resource_load = loads[round_index]
        energy = sum((value * 1e6 / edge_capacity[edge])**2 for edge, value in edge_load.items())
        energy += sum((value * 1e6 / resource_capacity[rid])**2 for rid, value in resource_load.items())
        ranked_transfers = sorted(
            round_data["transfers"],
            key=lambda transfer: -max(
                max(
                    [edge_load[edge] / edge_capacity[edge]]
                    + [resource_load[rid] / resource_capacity[rid] for rid in edge_resources.get(edge, ())]
                )
                for edge in zip(transfer["path"], transfer["path"][1:])
            ),
        )
        for transfer in ranked_transfers[:16]:
            if perf_counter() >= deadline or searches >= max_searches:
                stop_reason = "deadline" if perf_counter() >= deadline else "search_limit"
                break
            old_path = transfer["path"]
            byte_count = chunk_bytes * len(transfer["chunk_ranges"])
            other_edges, other_resources = dict(edge_load), dict(resource_load)
            for edge in zip(old_path, old_path[1:]):
                other_edges[edge] -= byte_count
                for rid in edge_resources.get(edge, ()):
                    other_resources[rid] -= byte_count
            searches += 1
            path = _least_loaded_path(
                transfer["src"], transfer["dst"], weighted_adjacency,
                other_edges, other_resources, byte_count, deadline,
            )
            if path is None:
                stop_reason = "deadline"
                break
            if path == old_path:
                continue
            transfer["path"] = path
            single_round = {"split": answer["split"], "rounds": [round_data]}
            new_time = estimate_schedule_time_us(task, topology, single_round, network)
            new_edges, new_resources = _schedule_loads(task, topology, single_round, network)[0]
            new_energy = sum((value * 1e6 / edge_capacity[edge])**2 for edge, value in new_edges.items())
            new_energy += sum((value * 1e6 / resource_capacity[rid])**2 for rid, value in new_resources.items())
            if new_time < round_times[round_index] or (new_time == round_times[round_index] and new_energy < energy):
                round_times[round_index] = new_time
                edge_load, resource_load = new_edges, new_resources
                energy = new_energy
                changed += 1
            else:
                transfer["path"] = old_path
        if stop_reason != "candidates_exhausted":
            break
    return {
        "path_searches": searches,
        "changed_paths": changed,
        "initial_estimate_us": initial_time,
        "final_estimate_us": sum(round_times),
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        "stop_reason": stop_reason,
    }


def build_solution(
    task: dict[str, Any],
    topology: dict[str, Any],
    preferred_split: int | None = None,
    route_policy: str = "baseline",
    network: tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], set[str]],
        dict[str, float],
        dict[str, list[str]],
    ] | None = None,
    *,
    algorithm: str = "recursive_doubling",
    routes: dict[tuple[str, str], list[str]] | None = None,
    dimension_order: list[int] | None = None,
) -> dict[str, Any]:
    if list(task["ranks"]) != list(topology["ranks"]):
        raise AgentCCError("task.ranks and topology.ranks must match")
    if network is None:
        network = _topology_network(topology)
    operator = task["operator"]
    if operator not in SCHEDULE_ALGORITHMS:
        raise AgentCCError(f"unsupported operator: {operator}")
    if algorithm not in SCHEDULE_ALGORITHMS[operator]:
        raise AgentCCError(f"unsupported algorithm for {operator}: {algorithm}")
    split = min(_candidate_splits(task, preferred_split))
    if routes is None:
        routes = _routes(task, topology, split, route_policy, network)
    if algorithm == "ordered_doubling":
        if dimension_order is None:
            dimension_order = optimal_doubling_order(
                doubling_round_profiles(task, topology, routes, network)
            )
        answer = _allgather(task, topology, routes, split, dimension_order)
    elif algorithm == "tree":
        answer = _allreduce_tree(task, routes, split)
    elif operator == "all-gather":
        answer = _allgather(task, topology, routes, split)
    else:
        answer = _allreduce(task, topology, routes, split)
    if answer["round_count"] > int(task["max_nonempty_rounds"]):
        raise AgentCCError("generated schedule exceeds max_nonempty_rounds")
    return answer


def solve_input(
    task_input: dict[str, Any],
    preferred_split: int | None = None,
    route_policy: str = "baseline",
    network: tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], set[str]],
        dict[str, float],
        dict[str, list[str]],
    ] | None = None,
    *,
    algorithm: str = "recursive_doubling",
    routes: dict[tuple[str, str], list[str]] | None = None,
    dimension_order: list[int] | None = None,
) -> dict[str, Any]:
    """Return the business response expected from BaseAgent.solve()."""

    task = task_input["task"]
    topology = task_input["topology"]
    if network is None:
        network = _topology_network(topology)
    answer = build_solution(
        task, topology, preferred_split, route_policy, network,
        algorithm=algorithm, routes=routes, dimension_order=dimension_order,
    )
    validate_answer(task, topology, answer, network)
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


def validate_answer(
    task: dict[str, Any],
    topology: dict[str, Any],
    answer: dict[str, Any],
    network: tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], set[str]],
        dict[str, float],
        dict[str, list[str]],
    ] | None = None,
) -> None:
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

    edge_set = _edge_set(topology, network)
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


def validate_request(
    task: dict[str, Any],
    topology: dict[str, Any],
    response: dict[str, Any],
    network: tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], set[str]],
        dict[str, float],
        dict[str, list[str]],
    ] | None = None,
) -> None:
    if response.get("status") != "success":
        raise AgentCCError("response status is not success")
    validate_answer(task, topology, response["output"]["answer"], network)


def solve_request(task: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any]:
    network = _topology_network(topology)
    answer = build_solution(task, topology, network=network)
    response = {
        "task_id": task["task_id"],
        "status": "success",
        "output": {"answer": answer},
    }
    if "instance_id" in task:
        response["instance_id"] = task["instance_id"]
    validate_request(task, topology, response, network)
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
