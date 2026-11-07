"""AgentCC topology analysis, multipath schedule construction, Replay estimate and validation.

The module is deterministic and has no model dependency.  ``src.my_agent``
uses it as the tool layer: the model proposes schedule parameters, this module
builds each proposal from the current task and topology, estimates it with the
published Replay formula, and validates the result before it may be returned.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from copy import copy
from dataclasses import dataclass
from math import inf
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


class AgentCCError(ValueError):
    """Raised when an input or generated solution violates the task contract."""


ALGORITHMS = {
    "all-gather": ("doubling",),
    "all-reduce": ("rs_ag", "doubling", "tree"),
}
MAX_PATH_WIDTH = 16
MAX_BLOCK_DIMS = 3
# Block rounds carry 3-7x more pairs than an XOR round; fewer Frank-Wolfe steps keep the
# search inside the model wait.  Predictions stay exact: they price the plan actually emitted.
BLOCK_OPTIMIZER_ITERATIONS = 24
OPTIMIZER_ITERATIONS = 48
PROFILE_BYTES = 1 << 20

NetworkTuple = tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], set[str]],
    dict[str, float],
    dict[str, list[str]],
]


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

_BANDWIDTH_UNITS = {
    "B/S": 1,
    "KB/S": 10**3,
    "MB/S": 10**6,
    "GB/S": 10**9,
    "TB/S": 10**12,
    "KIB/S": 2**10,
    "MIB/S": 2**20,
    "GIB/S": 2**30,
    "KBPS": 10**3 / 8,
    "MBPS": 10**6 / 8,
    "GBPS": 10**9 / 8,
    "TBPS": 10**12 / 8,
    "KBIT/S": 10**3 / 8,
    "MBIT/S": 10**6 / 8,
    "GBIT/S": 10**9 / 8,
    "TBIT/S": 10**12 / 8,
}


def _bandwidth_bytes_per_second(value: Any) -> float:
    if isinstance(value, bool):
        raise AgentCCError(f"unsupported bandwidth: {value!r}")
    if isinstance(value, (int, float)):
        if not value > 0:
            raise AgentCCError("bandwidth must be positive")
        return float(value)
    text = str(value).strip().upper().replace(" ", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?(?:E[+-]?[0-9]+)?)([A-Z/]*)", text)
    if not match:
        raise AgentCCError(f"unsupported bandwidth: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2) or "B/S"
    if unit not in _BANDWIDTH_UNITS:
        raise AgentCCError(f"unsupported bandwidth unit: {value!r}")
    if not amount > 0:
        raise AgentCCError("bandwidth must be positive")
    return amount * _BANDWIDTH_UNITS[unit]


def _topology_network(topology: dict[str, Any]) -> NetworkTuple:
    """Return directed edge capacities, resource memberships, and adjacency.

    Invalid inputs named by the public specification are rejected instead of
    being completed silently: undeclared endpoints, overlapping rank and switch
    sets, self loops, duplicate resource IDs, references to undefined resources,
    and resource references to directed edges that do not exist.
    """

    ranks = [str(rank) for rank in topology.get("ranks", [])]
    switches = [str(switch) for switch in topology.get("switches", [])]
    rank_set, switch_set = set(ranks), set(switches)
    if len(rank_set) != len(ranks) or len(switch_set) != len(switches):
        raise AgentCCError("topology ranks and switches must be unique")
    if rank_set & switch_set:
        raise AgentCCError("topology ranks and switches must not overlap")
    nodes = rank_set | switch_set

    edge_capacity: dict[tuple[str, str], float] = {}
    edge_resources: dict[tuple[str, str], set[str]] = {}
    adjacency: dict[str, list[str]] = {}
    direct_references: list[tuple[tuple[str, str], str]] = []

    def add_edge(src: str, dst: str, bandwidth: float) -> None:
        key = (src, dst)
        edge_capacity[key] = edge_capacity.get(key, 0.0) + bandwidth
        neighbours = adjacency.setdefault(src, [])
        if dst not in neighbours:
            neighbours.append(dst)

    for edge in topology["edges"]:
        src = edge["src"]
        dst = edge["dst"]
        if src not in nodes or dst not in nodes:
            raise AgentCCError(f"edge endpoint is not a declared rank or switch: {src}->{dst}")
        if src == dst:
            raise AgentCCError(f"self-loop edge is not allowed: {src}")
        bandwidth = _bandwidth_bytes_per_second(edge["bw"])
        add_edge(src, dst, bandwidth)
        resource_id = edge.get("resource_id")
        if resource_id is not None:
            direct_references.append(((src, dst), str(resource_id)))
        if edge.get("bidirectional", False):
            add_edge(dst, src, bandwidth)
            if resource_id is not None:
                direct_references.append(((dst, src), str(resource_id)))

    resource_capacity: dict[str, float] = {}
    for resource in topology.get("resources", []) or []:
        resource_id = str(resource["id"])
        if resource_id in resource_capacity:
            raise AgentCCError(f"duplicate resource id: {resource_id}")
        capacity = resource.get("capacity", resource.get("bw"))
        if capacity is None:
            raise AgentCCError(f"resource {resource_id} has no capacity")
        resource_capacity[resource_id] = _bandwidth_bytes_per_second(capacity)
        for edge in resource.get("edges", []) or []:
            keys = [(edge["src"], edge["dst"])]
            if edge.get("bidirectional", False):
                keys.append((edge["dst"], edge["src"]))
            for key in keys:
                if key not in edge_capacity:
                    raise AgentCCError(
                        f"resource {resource_id} references an undeclared edge: {key[0]}->{key[1]}"
                    )
                edge_resources.setdefault(key, set()).add(resource_id)

    for key, resource_id in direct_references:
        if resource_id not in resource_capacity:
            raise AgentCCError(f"edge {key[0]}->{key[1]} references undefined resource {resource_id}")
        edge_resources.setdefault(key, set()).add(resource_id)

    for neighbours in adjacency.values():
        neighbours.sort()
    return edge_capacity, edge_resources, resource_capacity, adjacency


def valid_splits(task: dict[str, Any]) -> list[int]:
    """Return every candidate split that satisfies divisibility and alignment."""

    data_size = int(task["data_size"]["per_rank_bytes"])
    alignment = int(task["split_policy"].get("alignment_bytes", 1) or 1)
    valid = sorted({
        int(value) for value in task["split_policy"]["candidates"]
        if int(value) > 0 and data_size % int(value) == 0 and (data_size // int(value)) % alignment == 0
    })
    if not valid:
        raise AgentCCError("no split candidate satisfies data and alignment constraints")
    return valid


def _candidate_splits(task: dict[str, Any], preferred_split: int | None = None) -> list[int]:
    """Compatibility helper: valid splits, preferred first."""

    valid = valid_splits(task)
    preferred = task.get("split") if preferred_split is None else preferred_split
    return ([int(preferred)] if preferred in valid else []) + [s for s in valid if s != preferred]


def validate_task_input(task: dict[str, Any], topology: dict[str, Any]) -> None:
    """Reject task inputs the schedules cannot satisfy, with a stated reason."""

    if task.get("operator") not in ALGORITHMS:
        raise AgentCCError(f"unsupported operator: {task.get('operator')}")
    ranks = list(task["ranks"])
    if ranks != list(topology["ranks"]):
        raise AgentCCError("task.ranks and topology.ranks must match")
    if len(set(ranks)) != len(ranks):
        raise AgentCCError("task.ranks contains duplicates")
    if len(ranks) < 2 or len(ranks) & (len(ranks) - 1):
        raise AgentCCError("the XOR schedules require a power-of-two rank count of at least 2")
    if int(task["data_size"]["per_rank_bytes"]) <= 0:
        raise AgentCCError("data_size.per_rank_bytes must be positive")
    if int(task["max_nonempty_rounds"]) <= 0:
        raise AgentCCError("max_nonempty_rounds must be positive")
    if task["operator"] == "all-gather" and not task.get("allow_copy", True):
        raise AgentCCError("AllGather requires copy support")
    valid_splits(task)


# ---------------------------------------------------------------------------
# Network model and candidate paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    nodes: tuple[str, ...]
    elements: np.ndarray  # edge ids followed by resource ids offset by edge count
    per_byte_us: float
    overhead_us: float

    @property
    def path(self) -> list[str]:
        return list(self.nodes)


class Network:
    """Indexed view of the current topology with switch-only candidate routes."""

    def __init__(self, task: dict[str, Any], topology: dict[str, Any], raw: NetworkTuple | None = None):
        self.raw = raw if raw is not None else _topology_network(topology)
        capacity, edge_resources, resource_capacity, adjacency = self.raw
        self.ranks = list(task["ranks"])
        self.rank_set = set(self.ranks)
        self.switch_names = sorted(set(topology.get("switches", [])))
        self.switch_index = {name: index for index, name in enumerate(self.switch_names)}
        self.edges = list(capacity)
        self.edge_index = {edge: index for index, edge in enumerate(self.edges)}
        self.edge_capacity = np.array([capacity[edge] for edge in self.edges], dtype=float)
        self.resource_names = sorted(resource_capacity)
        self.resource_index = {name: index for index, name in enumerate(self.resource_names)}
        self.resource_capacity = np.array([resource_capacity[name] for name in self.resource_names], dtype=float)
        self.edge_resource_ids = [
            tuple(sorted(self.resource_index[rid] for rid in edge_resources.get(edge, ())))
            for edge in self.edges
        ]
        self.element_inv = np.concatenate([1e6 / self.edge_capacity, 1e6 / self.resource_capacity])
        self.edge_count = len(self.edges)
        self.startup_us = float(topology.get("communication_startup_time_us", 0))
        self.switch_us = float(topology.get("switch_processing_overhead_us", 0))
        self.allow_detour = bool(task.get("allow_detour", topology.get("allow_detour", True)))
        self.adjacency = adjacency
        # Static ranking cost: edge time plus the time of every bound resource.
        self.static_cost = np.array([
            1e6 / capacity[edge] + sum(1e6 / resource_capacity[self.resource_names[r]] for r in rids)
            for edge, rids in zip(self.edges, self.edge_resource_ids)
        ])
        self._route_cache: dict[tuple[int, int], list[Route]] = {}
        self._segment_cache: dict[tuple[int, int], list[tuple[tuple[str, ...], float]]] = {}
        self._route_objects: dict[tuple[str, ...], Route] = {}
        self._prepare_switch_graph()

    def _prepare_switch_graph(self) -> None:
        count = len(self.switch_names)
        self.rank_out: dict[str, list[tuple[int, float]]] = {rank: [] for rank in self.ranks}
        self.rank_in: dict[str, list[tuple[int, float]]] = {rank: [] for rank in self.ranks}
        self.switch_out: list[dict[int, float]] = [dict() for _ in range(count)]
        self.switch_in: list[dict[int, float]] = [dict() for _ in range(count)]
        distance = np.full((count, count), inf)
        following = np.full((count, count), -1, dtype=np.int64)
        for edge, cost in zip(self.edges, self.static_cost):
            src, dst = edge
            if src in self.rank_set and dst in self.switch_index:
                self.rank_out[src].append((self.switch_index[dst], float(cost)))
            elif src in self.switch_index and dst in self.rank_set:
                self.rank_in[dst].append((self.switch_index[src], float(cost)))
            elif src in self.switch_index and dst in self.switch_index:
                u, v = self.switch_index[src], self.switch_index[dst]
                self.switch_out[u][v] = float(cost)
                self.switch_in[v][u] = float(cost)
                hop_cost = float(cost) + 1e-12
                if hop_cost < distance[u, v]:
                    distance[u, v] = hop_cost
                    following[u, v] = v
        for index in range(count):
            distance[index, index] = 0.0
            following[index, index] = index
        for k in range(count):
            through = distance[:, k:k + 1] + distance[k:k + 1, :]
            better = through < distance
            if better.any():
                distance = np.where(better, through, distance)
                following = np.where(better, following[:, k:k + 1], following)
        self.switch_distance = distance
        self.switch_following = following

    def _switch_path(self, u: int, v: int) -> tuple[str, ...]:
        nodes = [u]
        while nodes[-1] != v:
            nxt = int(self.switch_following[nodes[-1], v])
            if nxt < 0 or nxt in nodes:
                raise AgentCCError("switch path reconstruction failed")
            nodes.append(nxt)
        return tuple(self.switch_names[node] for node in nodes)

    def _segments(self, u: int, v: int, limit: int = 4) -> list[tuple[tuple[str, ...], float]]:
        key = (u, v)
        cached = self._segment_cache.get(key)
        if cached is not None:
            return cached
        options: dict[tuple[str, ...], float] = {}
        names = self.switch_names
        if v in self.switch_out[u]:
            options[(names[u], names[v])] = self.switch_out[u][v]
        out_u, in_v = self.switch_out[u], self.switch_in[v]
        smaller, larger = (out_u, in_v) if len(out_u) <= len(in_v) else (in_v, out_u)
        for w in smaller:
            if w in larger and w != u and w != v:
                options[(names[u], names[w], names[v])] = out_u[w] + in_v[w]
        if np.isfinite(self.switch_distance[u, v]):
            nodes = self._switch_path(u, v)
            options.setdefault(nodes, float(self.switch_distance[u, v]))
        result = sorted(options.items(), key=lambda item: (item[1], len(item[0]), item[0]))[:limit]
        self._segment_cache[key] = result
        return result

    def _route(self, nodes: tuple[str, ...]) -> Route:
        route = self._route_objects.get(nodes)
        if route is not None:
            return route
        edge_ids = [self.edge_index[edge] for edge in zip(nodes, nodes[1:])]
        resources = [self.edge_count + r for edge_id in edge_ids for r in self.edge_resource_ids[edge_id]]
        route = Route(
            nodes=nodes,
            elements=np.array(edge_ids + resources, dtype=np.int64),
            per_byte_us=float(sum(1e6 / self.edge_capacity[edge_id] for edge_id in edge_ids)),
            overhead_us=self.startup_us + (len(nodes) - 2) * self.switch_us,
        )
        self._route_objects[nodes] = route
        return route

    def routes(self, source: int, destination: int) -> list[Route]:
        """Diverse switch-only candidate routes, ordered for use as the width grows."""

        key = (source, destination)
        cached = self._route_cache.get(key)
        if cached is not None:
            return cached
        src, dst = self.ranks[source], self.ranks[destination]
        candidates: dict[tuple[str, ...], float] = {}
        if (src, dst) in self.edge_index:
            candidates[(src, dst)] = float(self.static_cost[self.edge_index[(src, dst)]])
        out_list, in_list = self.rank_out[src], self.rank_in[dst]
        if out_list and in_list:
            ui = np.array([u for u, _ in out_list])
            vi = np.array([v for v, _ in in_list])
            first = np.array([c for _, c in out_list])
            last = np.array([c for _, c in in_list])
            total = first[:, None] + self.switch_distance[np.ix_(ui, vi)] + last[None, :]
            flat_total = total.ravel()
            limit = min(2 * MAX_PATH_WIDTH, flat_total.size)
            if limit < flat_total.size:
                head = np.argpartition(flat_total, limit - 1)[:limit]
                head = head[np.argsort(flat_total[head], kind="stable")]
            else:
                head = np.argsort(flat_total, kind="stable")
            columns = len(vi)
            names = self.switch_names
            for flat in head.tolist():
                cost_ab = flat_total[flat]
                if not np.isfinite(cost_ab):
                    break
                a, b = divmod(flat, columns)
                u, v = int(ui[a]), int(vi[b])
                if u == v:
                    candidates.setdefault((src, names[u], dst), float(cost_ab))
                    continue
                edge_cost = float(first[a] + last[b])
                # Two-hop segments are only expanded for the best few switch pairs.
                for segment, cost in (self._segments(u, v) if len(candidates) < MAX_PATH_WIDTH else
                                      self._segments(u, v)[:1]):
                    candidates.setdefault((src,) + segment + (dst,), edge_cost + cost)
        if not candidates:
            raise AgentCCError(f"no legal route from {src} to {dst}")
        ordered = sorted(candidates.items(), key=lambda item: (item[1], len(item[0])))
        if not self.allow_detour:
            fewest = min(len(nodes) for nodes, _ in ordered)
            ordered = [item for item in ordered if len(item[0]) == fewest]
        # Diversity first: spread first and last hops before reusing them.
        selected: list[tuple[str, ...]] = []
        chosen: set[tuple[str, ...]] = set()
        for cap in (1, 2, MAX_PATH_WIDTH):
            first_use: Counter = Counter(nodes[:2] for nodes in selected)
            last_use: Counter = Counter(nodes[-2:] for nodes in selected)
            for nodes, _ in ordered:
                if len(selected) >= MAX_PATH_WIDTH:
                    break
                if nodes in chosen or first_use[nodes[:2]] >= cap or last_use[nodes[-2:]] >= cap:
                    continue
                selected.append(nodes)
                chosen.add(nodes)
                first_use[nodes[:2]] += 1
                last_use[nodes[-2:]] += 1
        result = [self._route(nodes) for nodes in selected]
        self._route_cache[key] = result
        return result

    def element_name(self, element: int) -> dict[str, Any]:
        if element < self.edge_count:
            src, dst = self.edges[element]
            return {"type": "edge", "id": f"{src}->{dst}", "capacity_Bps": float(self.edge_capacity[element])}
        name = self.resource_names[element - self.edge_count]
        return {"type": "resource", "id": name, "capacity_Bps": float(self.resource_capacity[element - self.edge_count])}

    def rank_costs(self, weights: np.ndarray) -> np.ndarray:
        """Directed rank-pair prices with switches as the only intermediates."""

        n, switches = len(self.ranks), len(self.switch_names)
        lookup = {rank: index for index, rank in enumerate(self.ranks)}
        out = np.full((n, switches), inf)
        incoming = np.full((switches, n), inf)
        distance = np.full((switches, switches), inf)
        costs = np.full((n, n), inf)
        np.fill_diagonal(distance, 0.0)
        for (src, dst), weight in zip(self.edges, weights):
            if src in lookup and dst in lookup:
                costs[lookup[src], lookup[dst]] = weight
            elif src in lookup:
                out[lookup[src], self.switch_index[dst]] = weight
            elif dst in lookup:
                incoming[self.switch_index[src], lookup[dst]] = weight
            else:
                distance[self.switch_index[src], self.switch_index[dst]] = weight
        for switch in range(switches):
            np.minimum(distance, distance[:, switch:switch + 1] + distance[switch:switch + 1, :], out=distance)
        reach = np.full((n, switches), inf)
        for switch in range(switches):
            np.minimum(reach, out[:, switch:switch + 1] + distance[switch:switch + 1, :], out=reach)
        for switch in range(switches):
            np.minimum(costs, reach[:, switch:switch + 1] + incoming[switch:switch + 1, :], out=costs)
        return costs

    def rank_orders(self) -> dict[str, dict[str, Any]]:
        """Binary groupings from bidirectional path costs, with/without resource prices."""

        n = len(self.ranks)
        result = {}
        for name, weights in (("locality", 1e6 / self.edge_capacity), ("resource", self.static_cost)):
            directed = self.rank_costs(weights)
            costs = np.maximum(directed, directed.T)
            groups = [[index] for index in range(n)]
            while len(groups) > 1:
                merged = []
                remaining = list(groups)
                while remaining:
                    first = remaining.pop(0)
                    options = np.array(remaining)
                    forward = costs[np.array(first)[None, :], options].sum(axis=1)
                    reverse = costs[np.array(first)[None, :], options[:, ::-1]].sum(axis=1)
                    selected = int(np.argmin(np.minimum(forward, reverse)))
                    second = remaining.pop(selected)
                    if reverse[selected] < forward[selected]:
                        second = second[::-1]
                    merged.append(first + second)
                groups = merged
            order = np.array(groups[0])
            steps = [1 << bit for bit in range(n.bit_length() - 1)]
            original = np.array([costs[np.arange(n), np.arange(n) ^ step] for step in steps])
            mapped = np.array([costs[order, order[np.arange(n) ^ step]] for step in steps])
            result[name] = {
                "directed_costs": directed,
                "rank_order": order.tolist(),
                "mapped_max": mapped.max(axis=1).tolist(),
                "original_mean": original.mean(axis=1).tolist(), "mapped_mean": mapped.mean(axis=1).tolist(),
            }
        return result


# ---------------------------------------------------------------------------
# Round optimizer
# ---------------------------------------------------------------------------


@dataclass
class RoundPlan:
    routes: list[list[Route]]
    fractions: np.ndarray  # demands x width, rows sum to one
    busy_per_byte_us: float
    bottleneck: dict[str, Any]
    elapsed_ms: float
    entry_element: np.ndarray
    entry_path: np.ndarray
    path_per_byte_us: np.ndarray
    path_overhead_us: np.ndarray


def optimize_round(
    network: Network,
    pairs: list[tuple[int, int]],
    width: int,
    iterations: int = OPTIMIZER_ITERATIONS,
) -> RoundPlan:
    """Split equal demands over candidate routes to minimize the Replay round time.

    Each pair carries one byte.  Elements are directed edges, shared resources,
    and one private element per route that models the transfer's own path time
    (bytes times the sum of per-edge times).  A Frank-Wolfe iteration on a
    smoothed maximum moves demand towards routes with the lowest marginal cost.
    The result scales linearly with the bytes carried by every pair.
    """

    started = perf_counter()
    route_lists = [network.routes(src, dst)[:max(1, width)] for src, dst in pairs]
    demand_count = len(route_lists)
    width_max = max(len(routes) for routes in route_lists)
    index = np.full((demand_count, width_max), -1, dtype=np.int64)
    flat_routes: list[Route] = []
    for demand, routes in enumerate(route_lists):
        for slot, route in enumerate(routes):
            index[demand, slot] = len(flat_routes)
            flat_routes.append(route)
    path_count = len(flat_routes)
    lengths = np.array([len(route.elements) for route in flat_routes], dtype=np.int64)
    entry_element = np.concatenate([route.elements for route in flat_routes]) if path_count else np.zeros(0, np.int64)
    entry_path = np.repeat(np.arange(path_count), lengths)
    per_byte = np.array([route.per_byte_us for route in flat_routes])
    element_inv = network.element_inv
    element_count = len(element_inv)
    valid = index >= 0
    safe_index = np.where(valid, index, 0)

    # Start from a time-balanced split of each demand.
    weights = np.where(valid, 1.0 / per_byte[safe_index], 0.0)
    fractions_matrix = weights / weights.sum(axis=1, keepdims=True)
    fraction = np.zeros(path_count)
    fraction[index[valid]] = fractions_matrix[valid]

    def evaluate(current: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        load = np.bincount(entry_element, weights=current[entry_path], minlength=element_count)
        busy = load * element_inv
        private = current * per_byte
        return busy, private, float(max(busy.max(initial=0.0), private.max(initial=0.0)))

    best_fraction = fraction.copy()
    busy, private, best_value = evaluate(fraction)
    rows = np.arange(demand_count)
    if width_max > 1:
        for step in range(iterations):
            busy, private, value = evaluate(fraction)
            if value < best_value:
                best_value, best_fraction = value, fraction.copy()
            power = 8.0 + 56.0 * step / max(1, iterations - 1)
            scale = value if value > 0 else 1.0
            element_grad = (busy / scale) ** (power - 1.0) * element_inv
            path_grad = np.bincount(entry_path, weights=element_grad[entry_element], minlength=path_count)
            path_grad += (private / scale) ** (power - 1.0) * per_byte
            candidate_grad = np.where(valid, path_grad[safe_index], inf)
            target = index[rows, np.argmin(candidate_grad, axis=1)]
            gamma = 2.0 / (step + 3.0)
            fraction *= 1.0 - gamma
            np.add.at(fraction, target, gamma)
        busy, private, value = evaluate(fraction)
        if value < best_value:
            best_value, best_fraction = value, fraction.copy()
    busy, private, best_value = evaluate(best_fraction)
    if private.max(initial=0.0) >= busy.max(initial=0.0):
        bottleneck = {"type": "path_time", "id": "->".join(flat_routes[int(np.argmax(private))].nodes)}
    else:
        bottleneck = network.element_name(int(np.argmax(busy)))
    matrix = np.where(valid, best_fraction[safe_index], 0.0)
    matrix /= matrix.sum(axis=1, keepdims=True)
    return RoundPlan(
        routes=route_lists,
        fractions=matrix,
        busy_per_byte_us=best_value,
        bottleneck=bottleneck,
        elapsed_ms=(perf_counter() - started) * 1000,
        entry_element=entry_element,
        entry_path=entry_path,
        path_per_byte_us=per_byte,
        path_overhead_us=np.array([route.overhead_us for route in flat_routes]),
    )


_GOLDEN = 0.6180339887498949


def _discretize(fractions: np.ndarray, units: int) -> np.ndarray:
    """Round each row to ``units`` integer pieces with per-row offsets.

    Plain largest-remainder rounding makes every symmetric pair round towards
    the same routes and concentrates load.  Systematic sampling with a
    low-discrepancy offset per row keeps the aggregate load of each element
    close to the fractional optimum while every row still sums to ``units``.
    """

    rows = fractions.shape[0]
    offsets = (np.arange(rows) * _GOLDEN) % 1.0
    cumulative = np.cumsum(fractions, axis=1) * units
    cumulative[:, -1] = units
    marks = np.floor(cumulative + offsets[:, None] - 1e-9)
    previous = np.concatenate([np.floor(offsets[:, None] - 1e-9), marks[:, :-1]], axis=1)
    counts = (marks - previous).astype(np.int64)
    return counts


# ---------------------------------------------------------------------------
# Planner: dimension profiles, configurations and schedule construction
# ---------------------------------------------------------------------------


class Planner:
    """Per-request state: current network, cached round plans and chunk names."""

    def __init__(self, task: dict[str, Any], topology: dict[str, Any], *, sum_notation: bool = False):
        validate_task_input(task, topology)
        # When true, copies of a completely reduced slot are written as
        # ``c{slot}_sum`` (the notation the output format defines for complete
        # slots) instead of listing all contributors.
        self.sum_notation = bool(sum_notation)
        self.task = task
        self.topology = topology
        self.network = Network(task, topology)
        self.operator = task["operator"]
        self.ranks = list(task["ranks"])
        self.rank_order = tuple(range(len(self.ranks)))
        self.rank_count = len(self.ranks)
        self.steps = [1 << bit for bit in range(self.rank_count.bit_length() - 1)]
        self.splits = valid_splits(task)
        self.data_size = int(task["data_size"]["per_rank_bytes"])
        self.max_rounds = int(task["max_nonempty_rounds"])
        self.allow_copy = bool(task.get("allow_copy", True))
        self._plans: dict[tuple, RoundPlan] = {}
        self._names: dict[int, list[list[str]]] = {}
        self._allocations: dict[tuple[int, int], np.ndarray] = {}
        self._allocation_keys: dict[tuple[int, int], tuple] = {}
        self._integer_times: dict[tuple[int, int, int], float] = {}
        self._rank_orders: dict[str, dict[str, Any]] | None = None

    def with_rank_order(self, order: list[int]) -> "Planner":
        """Use logical ranks internally, sharing only the physical network/route cache."""

        mapped = copy(self)
        mapped.rank_order = tuple(order)
        mapped.ranks = [self.network.ranks[index] for index in order]
        mapped._plans, mapped._names = {}, {}
        mapped._allocations, mapped._allocation_keys, mapped._integer_times = {}, {}, {}
        return mapped

    def mapping_candidates(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Cheap path-work screening; these scores are not Replay predictions."""

        if self._rank_orders is None:
            self._rank_orders = self.network.rank_orders()
        if config["algorithm"] == "doubling" and self.operator == "all-gather":
            # 0.2.11y: AllGather payload doubles every round and the agent puts the
            # costliest dimension first, so the cheapest dimension carries the most.
            weights = [1 << position for position in reversed(range(len(self.steps)))]
        else:
            split, h = config["split"], config["rs_dims"]
            weights = [split >> position for position in range(h)] + [split >> h] * (len(self.steps) - h)
        candidates = []
        for name, info in self._rank_orders.items():
            original = sorted(info["original_mean"])
            mapped = sorted(info["mapped_mean"])
            reachable = all(math.isfinite(value) for value in info["mapped_max"])
            before = sum(cost * weight for cost, weight in zip(original, weights))
            after = sum(cost * weight for cost, weight in zip(mapped, weights))
            candidates.append({"mapping": name, "rank_order": info["rank_order"], "reachable": reachable,
                               "proxy_gain": 1.0 - after / before if reachable else None})
        return candidates

    # -- round plans -------------------------------------------------------
    def exchange_pairs(self, step: int) -> list[tuple[int, int]]:
        return [(index, index ^ step) for index in range(self.rank_count)]

    def round_plan(self, key: tuple, pairs: list[tuple[int, int]], width: int,
                   iterations: int = OPTIMIZER_ITERATIONS) -> RoundPlan:
        cache_key = key + (width,)
        plan = self._plans.get(cache_key)
        if plan is None:
            physical_pairs = [(self.rank_order[src], self.rank_order[dst]) for src, dst in pairs]
            plan = optimize_round(self.network, physical_pairs, width, iterations)
            self._plans[cache_key] = plan
        return plan

    def exchange_plan(self, step: int, width: int) -> RoundPlan:
        return self.round_plan(("xor", step), self.exchange_pairs(step), width)

    def integer_allocation(self, plan: RoundPlan, units: int) -> np.ndarray:
        """The same integer route counts are used for prediction and emission."""

        key = (id(plan), units)
        counts = self._allocations.get(key)
        if counts is None:
            counts = _discretize(plan.fractions, units)
            self._allocations[key] = counts
        return counts

    def integer_round_time_us(self, plan: RoundPlan, units: int, chunk_bytes: int) -> float:
        """Replay cost of the emitted integer allocation, without chunk expressions."""

        key = (id(plan), units, chunk_bytes)
        if key not in self._integer_times:
            counts = self.integer_allocation(plan, units)
            sizes = np.concatenate([row[:len(routes)] for row, routes in zip(counts, plan.routes)]) * chunk_bytes
            active = sizes > 0
            loads = np.bincount(
                plan.entry_element, weights=sizes[plan.entry_path], minlength=len(self.network.element_inv),
            )
            overhead = float(plan.path_overhead_us[active].max(initial=0.0))
            path_time = (plan.path_overhead_us[active] + sizes[active] * plan.path_per_byte_us[active]).max(initial=0.0)
            contention = overhead + float((loads * self.network.element_inv).max(initial=0.0))
            self._integer_times[key] = float(max(path_time, contention))
        return self._integer_times[key]

    def dimension_profiles(self, width: int = MAX_PATH_WIDTH) -> list[dict[str, Any]]:
        """Per-XOR-dimension Replay cost per MiB on single and optimized multipath routes."""

        profiles = []
        for step in self.steps:
            multi = self.exchange_plan(step, width)
            single = self.exchange_plan(step, 1)
            used = (multi.fractions > 1e-3).sum(axis=1)
            profiles.append({
                "step": step,
                "us_per_MiB_multipath": round(multi.busy_per_byte_us * PROFILE_BYTES, 3),
                "us_per_MiB_single_path": round(single.busy_per_byte_us * PROFILE_BYTES, 3),
                "multipath_bottleneck": multi.bottleneck,
                "single_path_bottleneck": single.bottleneck,
                "candidate_routes_per_pair": round(float(np.mean([len(r) for r in multi.routes])), 2),
                "routes_used_per_pair": round(float(used.mean()), 2),
                "max_hops": max(len(route.nodes) - 1 for routes in multi.routes for route in routes),
            })
        return profiles

    # -- configurations -----------------------------------------------------
    def round_count(self, config: dict[str, Any]) -> int:
        dims = len(self.steps)
        if config["algorithm"] == "rs_ag":
            return 2 * config["rs_dims"] + dims - config["rs_dims"]
        if config["algorithm"] == "tree":
            return 2 * dims
        if config.get("blocks"):
            return len(config["blocks"])
        return dims

    @staticmethod
    def round_width(config: dict[str, Any], index: int) -> int:
        widths = config.get("widths")
        return widths[index] if widths else config["path_width"]

    def tree_stages(self, order: list[int]) -> list[tuple[tuple, list[tuple[int, int]], str]]:
        """(plan key, pairs, action) for every tree round, reduce rounds first."""

        active = set(range(self.rank_count))
        up = []
        for position, step in enumerate(order):
            pairs = [(index ^ step, index) for index in sorted(active) if not index & step]
            up.append((("tree-up", tuple(order[:position + 1])), pairs))
            for src, _ in pairs:
                active.discard(src)
        down = [(("tree-down", key[1]), [(dst, src) for src, dst in pairs]) for key, pairs in reversed(up)]
        return [(key, pairs, "reduce") for key, pairs in up] + [(key, pairs, "copy") for key, pairs in down]

    def normalize_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Check a proposed configuration and fill in agent-derived values."""

        if not isinstance(raw, dict):
            raise AgentCCError("candidate must be an object")
        algorithm = raw.get("algorithm")
        allowed = list(ALGORITHMS[self.operator])
        if self.operator == "all-reduce" and not self.allow_copy:
            allowed = ["doubling"]
        if algorithm not in allowed:
            raise AgentCCError(f"algorithm must be one of {allowed}")
        split = raw.get("split")
        if isinstance(split, bool) or not isinstance(split, int) or split not in self.splits:
            raise AgentCCError(f"split must be one of {self.splits}")
        width = raw.get("path_width")
        if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= MAX_PATH_WIDTH:
            raise AgentCCError(f"path_width must be an integer in [1, {MAX_PATH_WIDTH}]")
        order = raw.get("order", "auto")
        config: dict[str, Any] = {"algorithm": algorithm, "split": split, "path_width": width}
        widths = raw.get("widths")
        if algorithm == "rs_ag":
            max_dims = min(split.bit_length() - 1, len(self.steps))
            if max_dims < 1:
                raise AgentCCError("rs_ag requires split >= 2")
            rs_dims = raw.get("rs_dims", max_dims)
            if isinstance(rs_dims, bool) or not isinstance(rs_dims, int) or not 1 <= rs_dims <= max_dims:
                raise AgentCCError(f"rs_dims must be an integer in [1, {max_dims}] for split {split}")
            config["rs_dims"] = rs_dims
        if order == "auto":
            config["order"] = self.auto_order(config)
            config["order_source"] = "auto"
        else:
            if not isinstance(order, list) or sorted(order) != self.steps or any(isinstance(s, bool) for s in order):
                raise AgentCCError(f"order must be 'auto' or a permutation of {self.steps}")
            config["order"] = [int(step) for step in order]
            config["order_source"] = "proposed"
        blocks = raw.get("blocks")
        if blocks is not None:
            if self.operator != "all-gather" or algorithm != "doubling":
                raise AgentCCError("blocks apply to all-gather doubling only")
            if (not isinstance(blocks, list) or not blocks
                    or any(not isinstance(block, list) or not 1 <= len(block) <= MAX_BLOCK_DIMS for block in blocks)
                    or any(isinstance(step, bool) or not isinstance(step, int) for block in blocks for step in block)
                    or sorted(step for block in blocks for step in block) != self.steps):
                raise AgentCCError(f"blocks must partition {self.steps} into lists of 1-{MAX_BLOCK_DIMS} steps")
            config["blocks"] = [[int(step) for step in block] for block in blocks]
            config["order"] = [step for block in config["blocks"] for step in block]
            config["order_source"] = "blocks"
        rounds = self.round_count(config)
        if rounds > self.max_rounds:
            raise AgentCCError(
                f"{algorithm} needs {rounds} rounds, above max_nonempty_rounds {self.max_rounds}"
            )
        if widths is not None:
            if (not isinstance(widths, list) or len(widths) != rounds
                    or any(isinstance(w, bool) or not isinstance(w, int) or not 1 <= w <= MAX_PATH_WIDTH for w in widths)):
                raise AgentCCError(f"widths must list {rounds} integers in [1, {MAX_PATH_WIDTH}], one per round")
            config["widths"] = [int(w) for w in widths]
        return config

    # -- fast prediction -------------------------------------------------------
    def _mean_name_length(self, split: int) -> float:
        names = self.chunk_names(split)
        return sum(len(name) for rank_names in names for name in rank_names) / (self.rank_count * split)

    def round_specs(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Per round: optimized plan, chunks per pair and contributors per chunk."""

        split, order, n = config["split"], config["order"], self.rank_count
        specs = []
        if config["algorithm"] == "tree":
            for index, (key, pairs, action) in enumerate(self.tree_stages(order)):
                plan = self.round_plan(key, pairs, self.round_width(config, index))
                group = n if action == "copy" else 1 << index
                specs.append({"plan": plan, "pairs": len(pairs), "chunks": split, "group": group,
                              "complete": action == "copy"})
            return specs
        if self.operator == "all-gather" and config.get("blocks"):
            held = split
            for index, block in enumerate(config["blocks"]):
                plan = self.block_plan(block, self.round_width(config, index))
                specs.append({"plan": plan, "pairs": len(plan.routes), "chunks": held, "group": 1,
                              "complete": False})
                held <<= len(block)
            return specs
        for index, (step, chunks) in enumerate(self.round_payloads(config)):
            plan = self.exchange_plan(step, self.round_width(config, index))
            if self.operator == "all-gather":
                group = 1
            elif config["algorithm"] == "rs_ag" and index >= len(self.steps):
                group = n
            else:
                group = 1 << index
            complete = self.operator == "all-reduce" and config["algorithm"] == "rs_ag" and index >= len(self.steps)
            specs.append({"plan": plan, "pairs": n, "chunks": chunks, "group": group, "complete": complete})
        return specs

    def predict(self, config: dict[str, Any]) -> dict[str, float]:
        """Integer Replay cost (us) and approximate response size (MB), without a build."""

        chunk_bytes = self.data_size // config["split"]
        name = self._mean_name_length(config["split"])
        total_us, size = 0.0, 0.0
        for spec in self.round_specs(config):
            plan = spec["plan"]
            total_us += self.integer_round_time_us(plan, spec["chunks"], chunk_bytes)
            counts = self.integer_allocation(plan, spec["chunks"])
            used = float((counts > 0).sum(axis=1).mean())
            if spec.get("complete") and self.sum_notation:
                entry = len(f"c{config['split']}_sum") + 3
            else:
                entry = spec["group"] * (name + 1) + 2
            size += spec["pairs"] * (spec["chunks"] * entry + used * 110)
        size += 2 * self.rank_count * config["split"] * (name + 3)
        return {"us": total_us, "mb": size / 1e6}

    def schedule_key(self, config: dict[str, Any]) -> tuple:
        """Merge only identical integer allocations of the same communication structure."""

        rounds = []
        for spec in self.round_specs(config):
            plan = spec["plan"]
            key = (id(plan), spec["chunks"])
            allocation_key = self._allocation_keys.get(key)
            if allocation_key is None:
                counts = self.integer_allocation(plan, spec["chunks"])
                allocation_key = tuple(
                    tuple((route.nodes, int(count)) for route, count in zip(routes, row) if count > 0)
                    for routes, row in zip(plan.routes, counts)
                )
                self._allocation_keys[key] = allocation_key
            rounds.append(allocation_key)
        blocks = tuple(tuple(block) for block in config.get("blocks") or ())
        return (self.operator, config["algorithm"], config["split"], config.get("rs_dims"),
                self.rank_order, tuple(config["order"]), blocks, tuple(rounds))

    def normalize_config_safe(self, raw: Any) -> dict[str, Any] | None:
        """normalize_config that returns None instead of raising."""

        try:
            if not isinstance(raw, dict):
                return None
            clean = {key: raw[key] for key in ("algorithm", "split", "path_width", "rs_dims", "order", "widths", "blocks")
                     if key in raw}
            return self.normalize_config(clean)
        except (AgentCCError, TypeError, AttributeError):
            return None

    def auto_order(self, config: dict[str, Any]) -> list[int]:
        """Rearrangement order from multipath profiles at the configured width."""

        cost = {step: self.exchange_plan(step, config["path_width"]).busy_per_byte_us for step in self.steps}
        if config["algorithm"] == "doubling" and self.operator == "all-gather":
            # Payload doubles every round: most expensive dimension first.
            return sorted(self.steps, key=lambda step: (-cost[step], step))
        if config["algorithm"] == "rs_ag":
            # Reduce-scatter payload halves: cheapest dimensions carry the most bytes.
            return sorted(self.steps, key=lambda step: (cost[step], step))
        return list(self.steps)

    def predict_us(self, config: dict[str, Any]) -> float | None:
        """Continuous multipath model: overhead plus per-byte cost times payload."""

        if config["algorithm"] == "tree":
            return None
        chunk = self.data_size / config["split"]
        total = 0.0
        for step, chunks in self.round_payloads(config):
            plan = self.exchange_plan(step, config["path_width"])
            overhead = max(route.overhead_us for routes in plan.routes for route in routes)
            total += overhead + plan.busy_per_byte_us * chunk * chunks
        return total

    def round_payloads(self, config: dict[str, Any]) -> list[tuple[int, int]]:
        """(step, chunks per pair) for each round of an exchange schedule."""

        order, split = config["order"], config["split"]
        if self.operator == "all-gather":
            return [(step, split * (1 << position)) for position, step in enumerate(order)]
        if config["algorithm"] == "doubling":
            return [(step, split) for step in order]
        h = config["rs_dims"]
        rs = [(order[p], split >> (p + 1)) for p in range(h)]
        middle = [(step, split >> h) for step in order[h:]]
        return rs + middle + list(reversed(rs))

    # -- construction --------------------------------------------------------
    def chunk_names(self, split: int) -> list[list[str]]:
        names = self._names.get(split)
        if names is None:
            names = [[f"{rank}_c{slot}" for slot in range(split)] for rank in self.ranks]
            self._names[split] = names
        return names

    def build(self, config: dict[str, Any]) -> dict[str, Any]:
        if self.operator == "all-gather":
            rounds = self._build_allgather(config)
        elif config["algorithm"] == "rs_ag":
            rounds = self._build_rs_ag(config)
        elif config["algorithm"] == "tree":
            rounds = self._build_tree(config)
        else:
            rounds = self._build_allreduce_doubling(config)
        split = config["split"]
        names = self.chunk_names(split)
        by_rank = dict(zip(self.ranks, names))
        chunks = [name for rank in self.task["ranks"] for name in by_rank[rank]]
        answer = {
            "operator": self.operator,
            "split": split,
            "chunks": chunks,
            "initial_possession": {rank: list(by_rank[rank]) for rank in self.task["ranks"]},
            "final_chunks": chunks if self.operator == "all-gather" else [f"c{slot}_sum" for slot in range(split)],
            "round_count": len(rounds),
            "rounds": [{"round": number, "transfers": transfers} for number, transfers in enumerate(rounds, 1)],
        }
        if answer["round_count"] > self.max_rounds:
            raise AgentCCError("generated schedule exceeds max_nonempty_rounds")
        return answer

    def _emit(self, plan: RoundPlan, pairs: list[tuple[int, int]], units: list[list[Any]], action: str, render):
        """Split each pair's ordered units over its routes and render transfers."""

        transfers = []
        sizes = {len(item) for item in units}
        counts_by_size = {size: self.integer_allocation(plan, size) for size in sizes}
        for demand, ((src, dst), items) in enumerate(zip(pairs, units)):
            if not items:
                continue
            counts = counts_by_size[len(items)][demand]
            offset = 0
            for route, count in zip(plan.routes[demand], counts):
                if count <= 0:
                    continue
                piece = items[offset:offset + int(count)]
                offset += int(count)
                transfers.append({
                    "action": action,
                    "src": self.ranks[src],
                    "dst": self.ranks[dst],
                    "chunk_ranges": render(src, piece),
                    "path": route.path,
                })
        return transfers

    def _build_allgather(self, config: dict[str, Any]) -> list[list[dict[str, Any]]]:
        if config.get("blocks"):
            return self._build_allgather_blocks(config)
        split = config["split"]
        names = self.chunk_names(split)
        members = [[index] for index in range(self.rank_count)]
        rounds = []
        for index_round, step in enumerate(config["order"]):
            pairs = self.exchange_pairs(step)
            plan = self.exchange_plan(step, self.round_width(config, index_round))
            held = [[name for member in group for name in names[member]] for group in members]
            units = [held[src] for src, _ in pairs]
            rounds.append(self._emit(plan, pairs, units, "copy", lambda _src, piece: piece))
            members = [sorted(members[index] + members[index ^ step]) for index in range(self.rank_count)]
        return rounds

    def _build_allgather_blocks(self, config: dict[str, Any]) -> list[list[dict[str, Any]]]:
        """One round per block: every rank sends all it holds to each XOR combination of the block."""

        names = self.chunk_names(config["split"])
        members = [[index] for index in range(self.rank_count)]
        rounds = []
        for index_round, block in enumerate(config["blocks"]):
            masks = block_masks(block)
            pairs = [(index, index ^ mask) for mask in masks for index in range(self.rank_count)]
            plan = self.block_plan(block, self.round_width(config, index_round))
            held = [[name for member in group for name in names[member]] for group in members]
            rounds.append(self._emit(plan, pairs, [held[src] for src, _ in pairs], "copy", lambda _src, piece: piece))
            members = [sorted(set(members[index]).union(*(members[index ^ mask] for mask in masks)))
                       for index in range(self.rank_count)]
        return rounds

    def block_plan(self, block: list[int], width: int) -> RoundPlan:
        """Round plan for one block; a single dimension reuses the plain XOR exchange plan."""

        if len(block) == 1:
            return self.exchange_plan(block[0], width)
        masks = block_masks(block)
        pairs = [(index, index ^ mask) for mask in masks for index in range(self.rank_count)]
        return self.round_plan(("block", tuple(sorted(block))), pairs, width, BLOCK_OPTIMIZER_ITERATIONS)

    def split_aware_round_us(self, plan: RoundPlan, units: int, chunk_bytes: int) -> float:
        """Integer round cost once path-bound transfers are split (see split_path_bound_transfers)."""

        counts = self.integer_allocation(plan, units)
        sizes = np.concatenate([row[:len(routes)] for row, routes in zip(counts, plan.routes)]) * chunk_bytes
        active = sizes > 0
        loads = np.bincount(plan.entry_element, weights=sizes[plan.entry_path], minlength=len(self.network.element_inv))
        overhead = float(plan.path_overhead_us[active].max(initial=0.0))
        single = (plan.path_overhead_us[active] + chunk_bytes * plan.path_per_byte_us[active]).max(initial=0.0)
        return float(max(single, overhead + float((loads * self.network.element_inv).max(initial=0.0))))

    def allgather_blocks(self, config: dict[str, Any], max_block: int = MAX_BLOCK_DIMS
                         ) -> tuple[list[list[int]], float, float]:
        """Best partition of the AllGather order into consecutive blocks (dynamic programme).

        Keeps the configured dimension order (costliest first, smallest payload)
        and only decides which neighbouring dimensions share a round.  Returns
        (blocks, predicted us, predicted us with one dimension per round), both
        priced with split-aware integer round costs.
        """

        order, width = config["order"], config["path_width"]
        chunk_bytes = self.data_size // config["split"]
        count = len(order)
        best = [math.inf] * (count + 1)
        choice = [0] * (count + 1)
        best[0] = 0.0
        singles = 0.0
        for start in range(count):
            units = config["split"] << start
            for end in range(start + 1, min(count, start + max_block) + 1):
                cost = self.split_aware_round_us(self.block_plan(order[start:end], width), units, chunk_bytes)
                if end == start + 1:
                    singles += cost
                if best[start] + cost < best[end]:
                    best[end], choice[end] = best[start] + cost, start
        blocks, end = [], count
        while end:
            blocks.append(order[choice[end]:end])
            end = choice[end]
        return blocks[::-1], best[count], singles

    def _complete(self, expression, complete: tuple[int, ...], slot: int) -> str:
        return f"c{slot}_sum" if self.sum_notation else expression(complete, slot)

    def _expression_cache(self, split: int):
        names = self.chunk_names(split)
        cache: dict[tuple[tuple[int, ...], int], str] = {}

        def expression(group: tuple[int, ...], slot: int) -> str:
            key = (group, slot)
            text = cache.get(key)
            if text is None:
                text = "+".join(names[member][slot] for member in group)
                cache[key] = text
            return text

        return expression

    def _build_allreduce_doubling(self, config: dict[str, Any]) -> list[list[dict[str, Any]]]:
        split = config["split"]
        expression = self._expression_cache(split)
        groups = [(index,) for index in range(self.rank_count)]
        rounds = []
        for index_round, step in enumerate(config["order"]):
            pairs = self.exchange_pairs(step)
            plan = self.exchange_plan(step, self.round_width(config, index_round))
            units = [list(range(split)) for _ in pairs]
            current = groups
            rounds.append(self._emit(
                plan, pairs, units, "reduce",
                lambda src, piece: [expression(current[src], slot) for slot in piece],
            ))
            groups = [tuple(sorted(groups[index] + groups[index ^ step])) for index in range(self.rank_count)]
        return rounds

    def _build_rs_ag(self, config: dict[str, Any]) -> list[list[dict[str, Any]]]:
        split, h = config["split"], config["rs_dims"]
        order = config["order"]
        expression = self._expression_cache(split)
        complete = tuple(range(self.rank_count))
        groups = [(index,) for index in range(self.rank_count)]
        owned = [list(range(split)) for _ in range(self.rank_count)]
        rounds = []
        for step in order[:h]:
            pairs = self.exchange_pairs(step)
            plan = self.exchange_plan(step, self.round_width(config, len(rounds)))
            keep, give = [], []
            for index in range(self.rank_count):
                slots = owned[index]
                half = len(slots) // 2
                low, high = slots[:half], slots[half:]
                if index & step:
                    keep.append(high)
                    give.append(low)
                else:
                    keep.append(low)
                    give.append(high)
            current = groups
            rounds.append(self._emit(
                plan, pairs, give, "reduce",
                lambda src, piece: [expression(current[src], slot) for slot in piece],
            ))
            owned = keep
            groups = [tuple(sorted(groups[index] + groups[index ^ step])) for index in range(self.rank_count)]
        for step in order[h:]:
            pairs = self.exchange_pairs(step)
            plan = self.exchange_plan(step, self.round_width(config, len(rounds)))
            current = groups
            rounds.append(self._emit(
                plan, pairs, [list(owned[src]) for src, _ in pairs], "reduce",
                lambda src, piece: [expression(current[src], slot) for slot in piece],
            ))
            groups = [tuple(sorted(groups[index] + groups[index ^ step])) for index in range(self.rank_count)]
        if any(group != complete for group in groups):
            raise AgentCCError("reduce phase did not cover every contributor")
        for step in reversed(order[:h]):
            pairs = self.exchange_pairs(step)
            plan = self.exchange_plan(step, self.round_width(config, len(rounds)))
            rounds.append(self._emit(
                plan, pairs, [list(owned[src]) for src, _ in pairs], "copy",
                lambda _src, piece: [self._complete(expression, complete, slot) for slot in piece],
            ))
            owned = [sorted(owned[index] + owned[index ^ step]) for index in range(self.rank_count)]
        return rounds

    def _build_tree(self, config: dict[str, Any]) -> list[list[dict[str, Any]]]:
        if not self.allow_copy:
            raise AgentCCError("tree AllReduce requires copy support")
        split = config["split"]
        expression = self._expression_cache(split)
        groups = {index: (index,) for index in range(self.rank_count)}
        complete = tuple(range(self.rank_count))
        rounds = []
        for index_round, (key, pairs, action) in enumerate(self.tree_stages(config["order"])):
            plan = self.round_plan(key, pairs, self.round_width(config, index_round))
            if action == "reduce":
                current = dict(groups)
                rounds.append(self._emit(
                    plan, pairs, [list(range(split)) for _ in pairs], "reduce",
                    lambda src, piece, current=current: [expression(current[src], slot) for slot in piece],
                ))
                for src, dst in pairs:
                    groups[dst] = tuple(sorted(groups[dst] + groups[src]))
            else:
                rounds.append(self._emit(
                    plan, pairs, [list(range(split)) for _ in pairs], "copy",
                    lambda _src, piece: [self._complete(expression, complete, slot) for slot in piece],
                ))
        if groups[0] != complete:
            raise AgentCCError("tree reduction did not reach rank 0 with every contributor")
        return rounds


def block_masks(block: list[int]) -> list[int]:
    """Every non-empty XOR combination of the block's dimensions."""

    return [sum(step for bit, step in enumerate(block) if mask >> bit & 1) for mask in range(1, 1 << len(block))]


def prepare_dual_tree(planner: Planner) -> dict[str, Any]:
    """Two four-way reduction trees, jointly routed and priced at integer slot counts."""

    started = perf_counter()
    depth = (len(planner.steps) + 1) // 2
    if (planner.operator != "all-reduce" or not planner.allow_copy or planner.splits[-1] < 4
            or 2 * depth > planner.max_rounds):
        return {"reason": "ineligible"}
    network, n, split = planner.network, planner.rank_count, planner.splits[-1]
    if planner._rank_orders is None:
        planner._rank_orders = network.rank_orders()
    grouping = planner._rank_orders["resource"]
    inverse = {physical: logical for logical, physical in enumerate(planner.rank_order)}
    order = [inverse[index] for index in grouping["rank_order"]]
    index = np.ix_(planner.rank_order, planner.rank_order)
    original_costs = grouping["directed_costs"][index]
    trees = []
    for tree in range(2):
        active, levels = list(order), []
        for level in range(depth):
            costs = original_costs
            if tree:
                load = np.zeros(len(network.element_inv))
                for src, dst in trees[0][level]:
                    for a, b in ((src, dst), (dst, src)):
                        routes = network.routes(planner.rank_order[a], planner.rank_order[b])[:4]
                        route = min(routes, key=lambda item: float(network.element_inv[item.elements].sum()))
                        np.add.at(load, route.elements, 1)
                busy = load * network.element_inv
                prices = network.element_inv * (1 + 8 * (busy / busy.max()) ** 3)
                edge_prices = np.array([
                    prices[eid] + sum(prices[network.edge_count + rid] for rid in members)
                    for eid, members in enumerate(network.edge_resource_ids)
                ])
                costs = network.rank_costs(edge_prices)[index]
            pairs, next_active = [], []
            for start in range(0, len(active), 4):
                group = active[start:start + 4]
                matrix = costs[np.ix_(group, group)].copy()
                np.fill_diagonal(matrix, 0)
                scores = matrix.sum(axis=0) + matrix.sum(axis=1)
                parent_index = int(np.argmin(scores))
                if not math.isfinite(float(scores[parent_index])):
                    raise AgentCCError("dual tree group has no bidirectionally reachable parent")
                parent = group[parent_index]
                next_active.append(parent)
                pairs.extend((child, parent) for child in group if child != parent)
            levels.append(pairs)
            active = next_active
        trees.append(levels)
    tree_finished = perf_counter()
    stages = []
    sizes = np.ones((2, n), dtype=np.int64)
    for action in ("reduce", "copy"):
        for level in (range(depth) if action == "reduce" else reversed(range(depth))):
            pairs, ids, contributors = [], [], []
            state = sizes.copy()
            for tree in range(2):
                for src, dst in trees[tree][level]:
                    pairs.append((src, dst) if action == "reduce" else (dst, src))
                    ids.append(tree)
                    contributors.append(int(state[tree, src]) if action == "reduce" else n)
                    if action == "reduce":
                        sizes[tree, dst] += state[tree, src]
            plan = planner.round_plan(("dual-tree", action, tuple(pairs)), pairs, 4)
            stages.append({"pairs": pairs, "ids": np.array(ids), "action": action,
                           "contributors": np.array(contributors), "plan": plan})
    routes_finished = perf_counter()
    options, allocations = [], {}
    chunk_bytes = planner.data_size // split
    name = planner._mean_name_length(split)
    for choice, first in ((1, split // 2), (2, split // 4), (3, 3 * split // 4)):
        total_us, size = 0.0, 2 * n * split * (name + 3)
        counts_by_round = []
        for stage in stages:
            plan = stage["plan"]
            counts = np.where(stage["ids"][:, None] == 0,
                              planner.integer_allocation(plan, first),
                              planner.integer_allocation(plan, split - first))
            counts_by_round.append(counts)
            amounts = np.concatenate([row[:len(routes)] for row, routes in zip(counts, plan.routes)]) * chunk_bytes
            active = amounts > 0
            load = np.bincount(plan.entry_element, weights=amounts[plan.entry_path],
                               minlength=len(network.element_inv))
            path_us = np.where(active, plan.path_overhead_us + amounts * plan.path_per_byte_us, 0)
            total_us += max(float(path_us.max(initial=0)),
                            float(plan.path_overhead_us[active].max(initial=0)
                                  + (load * network.element_inv).max(initial=0)))
            entries = stage["contributors"] * (name + 1) + 2
            if stage["action"] == "copy" and planner.sum_notation:
                entries = np.full(len(stage["pairs"]), len(f"c{split}_sum") + 3)
            size += float((counts.sum(axis=1) * entries + (counts > 0).sum(axis=1) * 110).sum())
        options.append({"choice": choice, "first_slots": first, "us": total_us, "mb": size / 1e6})
        allocations[choice] = counts_by_round
    return {
        "split": split, "path_width": 4, "round_count": 2 * depth, "options": options,
        "trees": [{"reduce_levels": [[[planner.ranks[src], planner.ranks[dst]] for src, dst in level]
                                      for level in tree]} for tree in trees],
        "timings_ms": {"trees": (tree_finished - started) * 1000,
                       "routes": (routes_finished - tree_finished) * 1000,
                       "predictions": (perf_counter() - routes_finished) * 1000},
        "_stages": stages, "_allocations": allocations,
    }


def build_dual_tree(planner: Planner, prepared: dict[str, Any], choice: int) -> dict[str, Any]:
    """Emit the chosen slot partition; tree contributions remain independent."""

    split = prepared["split"]
    first = next(option["first_slots"] for option in prepared["options"] if option["choice"] == choice)
    groups = [[(rank,) for rank in range(planner.rank_count)] for _ in range(2)]
    complete = tuple(range(planner.rank_count))
    expression = planner._expression_cache(split)
    rounds = []
    for stage, counts in zip(prepared["_stages"], prepared["_allocations"][choice]):
        state, transfers = [list(row) for row in groups], []
        for demand, ((src, dst), tree) in enumerate(zip(stage["pairs"], stage["ids"])):
            slots = list(range(first)) if tree == 0 else list(range(first, split))
            offset = 0
            for route, amount in zip(stage["plan"].routes[demand], counts[demand]):
                if amount == 0:
                    continue
                piece = slots[offset:offset + int(amount)]
                offset += int(amount)
                values = [expression(state[tree][src], slot) if stage["action"] == "reduce"
                          else planner._complete(expression, complete, slot) for slot in piece]
                transfers.append({"action": stage["action"], "src": planner.ranks[src], "dst": planner.ranks[dst],
                                  "chunk_ranges": values, "path": route.path})
            if stage["action"] == "reduce":
                groups[tree][dst] = tuple(sorted(groups[tree][dst] + state[tree][src]))
        rounds.append({"round": len(rounds) + 1, "transfers": transfers})
    by_rank = dict(zip(planner.ranks, planner.chunk_names(split)))
    return {
        "operator": "all-reduce", "split": split,
        "chunks": [name for rank in planner.task["ranks"] for name in by_rank[rank]],
        "initial_possession": {rank: list(by_rank[rank]) for rank in planner.task["ranks"]},
        "final_chunks": [f"c{slot}_sum" for slot in range(split)], "round_count": len(rounds), "rounds": rounds,
    }


# ---------------------------------------------------------------------------
# Replay estimate (independent of the optimizer's internal arrays)
# ---------------------------------------------------------------------------


def repair_round_routes(
    network: Network, plan: RoundPlan, round_data: dict[str, Any], chunk_bytes: int,
    deadline: float, max_pairs: int = 16,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move whole chunks between a pair's existing routes on private round state."""

    by_pair: dict[tuple[str, str], dict[tuple[str, ...], dict[str, Any]]] = {}
    for transfer in round_data["transfers"]:
        by_pair.setdefault((transfer["src"], transfer["dst"]), {})[tuple(transfer["path"])] = transfer
    flat_routes = [route for routes in plan.routes for route in routes]
    pair_index = np.repeat(np.arange(len(plan.routes)), [len(routes) for routes in plan.routes])
    counts = np.array([
        len(by_pair[(route.nodes[0], route.nodes[-1])][route.nodes]["chunk_ranges"])
        if route.nodes in by_pair[(route.nodes[0], route.nodes[-1])] else 0
        for route in flat_routes
    ], dtype=np.int64)
    loads = np.bincount(plan.entry_element, weights=counts[plan.entry_path], minlength=len(network.element_inv))
    offsets = np.cumsum([0] + [len(routes) for routes in plan.routes])

    def evaluate() -> tuple[float, np.ndarray, np.ndarray]:
        active = counts > 0
        path_times = np.where(active, plan.path_overhead_us + counts * chunk_bytes * plan.path_per_byte_us, 0.0)
        busy = loads * chunk_bytes * network.element_inv
        overhead = plan.path_overhead_us[active].max(initial=0.0)
        return float(max(path_times.max(initial=0.0), overhead + busy.max(initial=0.0))), path_times, busy

    before, path_times, busy = evaluate()
    current = before
    moves = []
    dirty_pairs = set()
    checked = 0
    while perf_counter() < deadline:
        active_overhead = plan.path_overhead_us[counts > 0].max(initial=0.0)
        if path_times.max(initial=0.0) >= active_overhead + busy.max(initial=0.0):
            sources = np.flatnonzero(path_times >= current - max(1e-9, current * 1e-10)).tolist()
        else:
            element = int(np.argmax(busy))
            multiplicity = np.bincount(plan.entry_path[plan.entry_element == element], minlength=len(counts))
            sources = np.flatnonzero((counts > 0) & (multiplicity > 0)).tolist()
            sources.sort(key=lambda index: (-counts[index] * multiplicity[index], index))
        pairs = set()
        best_move = None
        best_time = current
        tolerance = max(1e-9, current * 1e-10)
        for source in sources:
            pair = int(pair_index[source])
            if offsets[pair + 1] - offsets[pair] < 2:
                continue
            if pair not in pairs and len(pairs) == max_pairs:
                continue
            pairs.add(pair)
            for destination in range(offsets[pair], offsets[pair + 1]):
                if perf_counter() >= deadline:
                    break
                if destination == source:
                    continue
                counts[source] -= 1
                counts[destination] += 1
                np.add.at(loads, flat_routes[source].elements, -1)
                np.add.at(loads, flat_routes[destination].elements, 1)
                candidate_time, _, _ = evaluate()
                checked += 1
                if candidate_time < best_time - tolerance:
                    best_time = candidate_time
                    best_move = (source, destination)
                counts[source] += 1
                counts[destination] -= 1
                np.add.at(loads, flat_routes[source].elements, 1)
                np.add.at(loads, flat_routes[destination].elements, -1)
            if perf_counter() >= deadline:
                break
        if best_move is None:
            break
        source, destination = best_move
        counts[source] -= 1
        counts[destination] += 1
        np.add.at(loads, flat_routes[source].elements, -1)
        np.add.at(loads, flat_routes[destination].elements, 1)
        moves.append({"src": flat_routes[source].nodes[0], "dst": flat_routes[source].nodes[-1],
                      "from_path": list(flat_routes[source].nodes), "to_path": list(flat_routes[destination].nodes),
                      "before_us": current, "after_us": best_time})
        dirty_pairs.add(int(pair_index[source]))
        current, path_times, busy = evaluate()

    info = {"before_us": before, "after_us": current, "checked_moves": checked,
            "accepted_moves": len(moves), "moves": moves}
    if not moves:
        return round_data, info
    changed = {(plan.routes[pair][0].nodes[0], plan.routes[pair][0].nodes[-1]) for pair in dirty_pairs}
    transfers = [transfer for transfer in round_data["transfers"] if (transfer["src"], transfer["dst"]) not in changed]
    for pair in sorted(dirty_pairs):
        routes = plan.routes[pair]
        original = by_pair[(routes[0].nodes[0], routes[0].nodes[-1])]
        template = next(iter(original.values()))
        chunks = [chunk for transfer in original.values() for chunk in transfer["chunk_ranges"]]
        offset = 0
        for route, count in zip(routes, counts[offsets[pair]:offsets[pair + 1]]):
            if count > 0:
                transfers.append({**template, "path": route.path, "chunk_ranges": chunks[offset:offset + int(count)]})
                offset += int(count)
    return {**round_data, "transfers": transfers}, info


def refine_schedule_routes(
    planner: Planner, config: dict[str, Any], answer: dict[str, Any], estimate: dict[str, Any],
    *, size_weight: float = 0.10, budget_seconds: float = 0.1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Repair up to three slow rounds; validate and adopt only a better complete schedule."""

    started = perf_counter()
    deadline = started + budget_seconds
    rounds = list(answer["rounds"])
    info: dict[str, Any] = {"budget_ms": budget_seconds * 1000, "before_us": estimate["total_us"],
                            "adopted": False, "rounds": []}
    specs = planner.round_specs(config)
    search_started = perf_counter()
    for index in sorted(range(len(rounds)), key=lambda i: -estimate["rounds"][i]["time_us"]):
        if len(info["rounds"]) == 3:
            break
        if perf_counter() >= deadline:
            break
        if all(len(routes) == 1 for routes in specs[index]["plan"].routes):
            continue
        rounds[index], detail = repair_round_routes(
            planner.network, specs[index]["plan"], rounds[index], planner.data_size // config["split"], deadline,
        )
        info["rounds"].append({"round": index + 1, **detail})
    info["search_ms"] = (perf_counter() - search_started) * 1000
    info["accepted_moves"] = sum(row["accepted_moves"] for row in info["rounds"])
    info["checked_moves"] = sum(row["checked_moves"] for row in info["rounds"])
    info["before_response_mb"] = answer_size_estimate(answer) / 1e6
    result, result_estimate = answer, estimate
    if info["accepted_moves"]:
        candidate = {**answer, "rounds": rounds}
        measured = replay_estimate(planner.task, planner.topology, candidate, planner.network.raw)
        validate_answer(planner.task, planner.topology, candidate, planner.network.raw)
        candidate_mb = answer_size_estimate(candidate) / 1e6
        before_cost = math.log(estimate["total_us"]) + size_weight * info["before_response_mb"]
        candidate_cost = math.log(measured["total_us"]) + size_weight * candidate_mb
        info.update(candidate_us=measured["total_us"], candidate_response_mb=candidate_mb,
                    before_cost=before_cost, candidate_cost=candidate_cost)
        if measured["total_us"] < estimate["total_us"] and candidate_cost <= before_cost:
            result, result_estimate = candidate, measured
            info["adopted"] = True
        else:
            info["rejection"] = "complete_schedule_cost"
    info["after_us"] = result_estimate["total_us"]
    info["after_response_mb"] = answer_size_estimate(result) / 1e6
    info["elapsed_ms"] = (perf_counter() - started) * 1000
    return result, result_estimate, info


def split_path_bound_transfers(
    network: Network, answer: dict[str, Any], chunk_bytes: int, *, size_weight: float = 0.10,
    max_targets: int = 24,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split transfers whose own path time is above the round's contention time.

    Replay charges a transfer its path time, startup + switch overhead + bytes
    times the sum of per-edge times, while edge and resource busy times depend
    only on the total bytes on each element.  Sending a pair's chunks as several
    transfers on the same path therefore leaves every load unchanged and only
    shortens the longest path time.  Each round is split down to the target
    time that minimises ln(total Replay time) + size_weight * response MB, so
    rounds where the saving is small next to the extra transfer records stay as
    they are.  Chunk lists stay in order and no chunk is sent twice.
    """

    started = perf_counter()
    plans = []
    for round_data in answer["rounds"]:
        transfers = round_data["transfers"]
        if not transfers:
            plans.append(None)
            continue
        routes = [network._route(tuple(transfer["path"])) for transfer in transfers]
        counts = np.array([len(transfer["chunk_ranges"]) for transfer in transfers], dtype=np.int64)
        overhead = np.array([route.overhead_us for route in routes])
        unit = np.array([route.per_byte_us for route in routes]) * chunk_bytes
        lengths = [len(route.elements) for route in routes]
        load = np.bincount(np.concatenate([route.elements for route in routes]),
                           weights=np.repeat(counts * chunk_bytes, lengths), minlength=len(network.element_inv))
        contention = float(overhead.max() + (load * network.element_inv).max())
        path = overhead + counts * unit
        record = 80 + np.array([sum(len(node) + 3 for node in transfer["path"]) for transfer in transfers])
        plans.append({"contention": contention, "path": path, "counts": counts, "overhead": overhead,
                      "unit": unit, "record": record, "time": float(max(contention, path.max()))})
    total = sum(plan["time"] for plan in plans if plan is not None)

    def pieces_for(plan: dict[str, Any], target: float) -> np.ndarray:
        room = target - plan["overhead"]
        per_piece = np.where(room > plan["unit"], np.floor(room / plan["unit"]), 1).astype(np.int64)
        return np.where(plan["path"] > target, -(-plan["counts"] // np.maximum(per_piece, 1)), 1)

    def evaluate(plan: dict[str, Any], target: float) -> tuple[float, float, np.ndarray]:
        pieces = pieces_for(plan, target)
        size = -(-plan["counts"] // pieces)
        time = float(max(plan["contention"], (plan["overhead"] + size * plan["unit"]).max()))
        extra_mb = float(((pieces - 1) * plan["record"]).sum()) / 1e6
        return time, extra_mb, pieces

    info: dict[str, Any] = {"before_us": total, "rounds": []}
    chosen: list[np.ndarray | None] = []
    current = total
    for index, plan in enumerate(plans):
        if plan is None or plan["path"].max() <= plan["contention"] * (1 + 1e-12):
            chosen.append(None)
            continue
        above = np.unique(plan["path"][plan["path"] > plan["contention"]])[::-1]
        targets = [plan["contention"]] + list(above[1:max_targets])
        best = (0.0, plan["time"], 0.0, None)
        for target in targets:
            time, extra_mb, pieces = evaluate(plan, float(target))
            gain = math.log((current - plan["time"] + time) / current) + size_weight * extra_mb
            if gain < best[0]:
                best = (gain, time, extra_mb, pieces)
        if best[3] is None:
            chosen.append(None)
            continue
        info["rounds"].append({"round": index + 1, "before_us": plan["time"], "after_us": best[1],
                               "extra_transfers": int((best[3] - 1).sum()), "extra_mb": best[2]})
        current += best[1] - plan["time"]
        chosen.append(best[3])
    info.update(after_us=current, extra_transfers=sum(row["extra_transfers"] for row in info["rounds"]),
                elapsed_ms=(perf_counter() - started) * 1000)
    if not info["rounds"]:
        return answer, info
    rounds = []
    for round_data, pieces in zip(answer["rounds"], chosen):
        if pieces is None:
            rounds.append(round_data)
            continue
        transfers = []
        for transfer, count in zip(round_data["transfers"], pieces.tolist()):
            items = transfer["chunk_ranges"]
            if count <= 1:
                transfers.append(transfer)
                continue
            step = -(-len(items) // count)
            transfers.extend({**transfer, "chunk_ranges": items[start:start + step]}
                             for start in range(0, len(items), step))
        rounds.append({**round_data, "transfers": transfers})
    return {**answer, "rounds": rounds}, info


def replay_estimate(
    task: dict[str, Any],
    topology: dict[str, Any],
    answer: dict[str, Any],
    network: NetworkTuple | None = None,
) -> dict[str, Any]:
    """Estimate the published Replay time and report each round's bottleneck."""

    edge_capacity, edge_resources, resource_capacity, _ = network if network is not None else _topology_network(topology)
    chunk_bytes = int(task["data_size"]["per_rank_bytes"]) // int(answer["split"])
    startup_us = float(topology.get("communication_startup_time_us", 0))
    switch_us = float(topology.get("switch_processing_overhead_us", 0))
    inverse = {edge: 1e6 / capacity for edge, capacity in edge_capacity.items()}
    rounds = []
    total = 0.0
    for round_data in answer["rounds"]:
        transfers = round_data["transfers"]
        if not transfers:
            continue
        edge_load: dict[tuple[str, str], int] = {}
        resource_load: dict[str, int] = {}
        max_path = 0.0
        max_overhead = 0.0
        for transfer in transfers:
            size = chunk_bytes * len(transfer["chunk_ranges"])
            path = transfer["path"]
            overhead = startup_us + max(0, len(path) - 2) * switch_us
            max_overhead = max(max_overhead, overhead)
            path_time = overhead
            for edge in zip(path, path[1:]):
                edge_load[edge] = edge_load.get(edge, 0) + size
                path_time += size * inverse[edge]
                for resource_id in edge_resources.get(edge, ()):
                    resource_load[resource_id] = resource_load.get(resource_id, 0) + size
            max_path = max(max_path, path_time)
        edge_key = max(edge_load, key=lambda edge: edge_load[edge] * inverse[edge])
        edge_busy = edge_load[edge_key] * inverse[edge_key]
        resource_busy, resource_key = 0.0, None
        if resource_load:
            resource_key = max(resource_load, key=lambda rid: resource_load[rid] / resource_capacity[rid])
            resource_busy = 1e6 * resource_load[resource_key] / resource_capacity[resource_key]
        contention = max_overhead + max(edge_busy, resource_busy)
        time_us = max(max_path, contention)
        if max_path >= contention:
            limiter = "path_time"
        elif resource_busy > edge_busy:
            limiter = f"resource:{resource_key}"
        else:
            limiter = f"edge:{edge_key[0]}->{edge_key[1]}"
        rounds.append({
            "time_us": time_us,
            "max_path_us": max_path,
            "max_edge_busy_us": edge_busy,
            "max_resource_busy_us": resource_busy,
            "limiter": limiter,
            "transfers": len(transfers),
        })
        total += time_us
    return {"total_us": total, "rounds": rounds}


def estimate_schedule_time_us(
    task: dict[str, Any],
    topology: dict[str, Any],
    answer: dict[str, Any],
    network: NetworkTuple | None = None,
) -> float:
    """Estimate the V22 Replay communication time for a generated answer."""

    return replay_estimate(task, topology, answer, network)["total_us"]


def attach_metrics(answer: dict[str, Any], estimate: dict[str, Any]) -> None:
    """Self-reported metrics in seconds, as in the published examples."""

    answer["metrics"] = {
        "total_time": estimate["total_us"] / 1e6,
        "round_times": [row["time_us"] / 1e6 for row in estimate["rounds"]],
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _raw_chunk(rank: str, slot: int) -> str:
    return f"{rank}_c{slot}"


def _all_chunks(ranks: list[str], split: int) -> list[str]:
    return [_raw_chunk(rank, slot) for rank in ranks for slot in range(split)]


def _parse_raw_chunk(chunk: str, split: int, rank_lookup: dict[str, int]) -> tuple[int, int]:
    rank, separator, slot_text = chunk.rpartition("_c")
    if separator and rank in rank_lookup and slot_text.isdigit() and int(slot_text) < split:
        return rank_lookup[rank], int(slot_text)
    raise AgentCCError(f"invalid chunk name: {chunk}")


_SUM_PATTERN = re.compile(r"c(\d+)_sum")


def _parse_expression(expression: str, split: int, rank_lookup: dict[str, int], rank_count: int) -> tuple[int, frozenset[int]]:
    match = _SUM_PATTERN.fullmatch(expression)
    if match:
        slot = int(match.group(1))
        if slot >= split:
            raise AgentCCError(f"invalid complete slot: {expression}")
        return slot, frozenset(range(rank_count))
    tokens = expression.split("+")
    if any(not token for token in tokens):
        raise AgentCCError(f"invalid reduce expression: {expression}")
    parsed = [_parse_raw_chunk(token, split, rank_lookup) for token in tokens]
    slots = {slot for _, slot in parsed}
    contributors = frozenset(rank for rank, _ in parsed)
    if len(slots) != 1 or len(contributors) != len(parsed):
        raise AgentCCError(f"reduce expression must contain unique contributors from one slot: {expression}")
    return parsed[0][1], contributors


def validate_answer(
    task: dict[str, Any],
    topology: dict[str, Any],
    answer: dict[str, Any],
    network: NetworkTuple | None = None,
) -> None:
    """Validate the correctness invariants of the public solution format."""

    ranks = list(task["ranks"])
    split = int(answer["split"])
    data_size = int(task["data_size"]["per_rank_bytes"])
    alignment = int(task["split_policy"].get("alignment_bytes", 1) or 1)
    if split not in {int(value) for value in task["split_policy"]["candidates"]}:
        raise AgentCCError("split is not one of the task candidates")
    if data_size % split != 0 or (data_size // split) % alignment != 0:
        raise AgentCCError("split violates data size or alignment")
    if answer["operator"] != task["operator"]:
        raise AgentCCError("operator mismatch")
    if answer["chunks"] != _all_chunks(ranks, split):
        raise AgentCCError("chunks are incomplete or out of order")
    expected_initial = {rank: [_raw_chunk(rank, slot) for slot in range(split)] for rank in ranks}
    if answer["initial_possession"] != expected_initial:
        raise AgentCCError("initial_possession mismatch")
    if answer["round_count"] != len(answer["rounds"]):
        raise AgentCCError("round_count mismatch")
    nonempty = sum(1 for round_data in answer["rounds"] if round_data["transfers"])
    if nonempty > int(task["max_nonempty_rounds"]):
        raise AgentCCError("too many rounds")

    edge_capacity = (network if network is not None else _topology_network(topology))[0]
    switches = set(topology.get("switches", []))
    rank_set = set(ranks)
    for round_number, round_data in enumerate(answer["rounds"], start=1):
        if round_data["round"] != round_number:
            raise AgentCCError("round numbers must be consecutive")
        if not isinstance(round_data["transfers"], list):
            raise AgentCCError("transfers must be an array")
        for transfer in round_data["transfers"]:
            src, dst, path = transfer["src"], transfer["dst"], transfer["path"]
            if src not in rank_set or dst not in rank_set or src == dst:
                raise AgentCCError("transfer endpoints must be distinct ranks")
            if len(path) < 2 or path[0] != src or path[-1] != dst:
                raise AgentCCError("path endpoints do not match transfer")
            if len(set(path)) != len(path):
                raise AgentCCError(f"path repeats a node: {path}")
            if any(node not in switches for node in path[1:-1]):
                raise AgentCCError("path contains a rank as an intermediate node")
            if any(edge not in edge_capacity for edge in zip(path, path[1:])):
                raise AgentCCError(f"path contains an undeclared edge: {path}")
            if not transfer["chunk_ranges"]:
                raise AgentCCError("transfer has no chunks")

    if task["operator"] == "all-gather":
        _validate_allgather(answer, ranks, split)
    else:
        _validate_allreduce(task, answer, ranks, split)


def _validate_allgather(answer: dict[str, Any], ranks: list[str], split: int) -> None:
    rank_lookup = {rank: index for index, rank in enumerate(ranks)}
    possession = [set(_raw_chunk(rank, slot) for slot in range(split)) for rank in ranks]
    sent: set[tuple[str, str, str]] = set()
    for round_data in answer["rounds"]:
        received: list[set[str]] = [set() for _ in ranks]
        for transfer in round_data["transfers"]:
            if transfer["action"] != "copy":
                raise AgentCCError("AllGather only permits copy")
            src, dst = rank_lookup[transfer["src"]], rank_lookup[transfer["dst"]]
            source_held, target_held, target_new = possession[src], possession[dst], received[dst]
            for chunk in transfer["chunk_ranges"]:
                if chunk not in source_held:
                    _parse_raw_chunk(chunk, split, rank_lookup)
                    raise AgentCCError("source does not possess chunk at round start")
                if chunk in target_held or chunk in target_new:
                    raise AgentCCError("destination receives a duplicate chunk")
                key = (chunk, transfer["src"], transfer["dst"])
                if key in sent:
                    raise AgentCCError("same chunk sent to the same pair twice")
                sent.add(key)
                target_new.add(chunk)
        for index, chunks in enumerate(received):
            possession[index] |= chunks
    expected = set(_all_chunks(ranks, split))
    if any(chunks != expected for chunks in possession):
        raise AgentCCError("AllGather did not finish at every rank")
    if answer["final_chunks"] != _all_chunks(ranks, split):
        raise AgentCCError("final_chunks must list every original chunk")


def _validate_allreduce(task: dict[str, Any], answer: dict[str, Any], ranks: list[str], split: int) -> None:
    rank_lookup = {rank: index for index, rank in enumerate(ranks)}
    count = len(ranks)
    everyone = frozenset(range(count))
    partial: list[list[frozenset[int]]] = [[frozenset({index})] * split for index in range(count)]
    sent: set[tuple[int, frozenset[int], str, str]] = set()
    parse_cache: dict[str, tuple[int, frozenset[int]]] = {}
    copy_allowed = task.get("allow_copy", True)
    for round_data in answer["rounds"]:
        start = [list(slots) for slots in partial]
        updates = [list(slots) for slots in partial]
        for transfer in round_data["transfers"]:
            action = transfer["action"]
            src, dst = rank_lookup[transfer["src"]], rank_lookup[transfer["dst"]]
            seen_slots: set[int] = set()
            for expression in transfer["chunk_ranges"]:
                parsed = parse_cache.get(expression)
                if parsed is None:
                    parsed = _parse_expression(expression, split, rank_lookup, count)
                    parse_cache[expression] = parsed
                slot, contributors = parsed
                if slot in seen_slots:
                    raise AgentCCError("transfer repeats a slot")
                seen_slots.add(slot)
                if contributors != start[src][slot]:
                    raise AgentCCError("transfer expression does not match source state")
                key = (slot, contributors, transfer["src"], transfer["dst"])
                if key in sent:
                    raise AgentCCError("same chunk sent to the same pair twice")
                sent.add(key)
                if action == "reduce":
                    if updates[dst][slot] & contributors:
                        raise AgentCCError("reduce contains a duplicate contributor")
                    updates[dst][slot] = updates[dst][slot] | contributors
                elif action == "copy":
                    if not copy_allowed:
                        raise AgentCCError("copy is disabled for this task")
                    if contributors != everyone:
                        raise AgentCCError("copy is only valid for a complete reduction")
                    if start[dst][slot] == everyone or updates[dst][slot] == everyone:
                        raise AgentCCError("destination receives a duplicate complete chunk")
                    updates[dst][slot] = everyone
                else:
                    raise AgentCCError(f"unsupported AllReduce action: {action}")
        partial = updates
    if any(slot_set != everyone for slots in partial for slot_set in slots):
        raise AgentCCError("AllReduce did not finish at every rank")
    if answer["final_chunks"] != [f"c{slot}_sum" for slot in range(split)]:
        raise AgentCCError("final_chunks must list every complete slot")


def validate_request(
    task: dict[str, Any],
    topology: dict[str, Any],
    response: dict[str, Any],
    network: NetworkTuple | None = None,
) -> None:
    if response.get("status") != "success":
        raise AgentCCError("response status is not success")
    validate_answer(task, topology, response["output"]["answer"], network)


# ---------------------------------------------------------------------------
# Deterministic reference configuration and CLI
# ---------------------------------------------------------------------------


class StrategyAPI:
    """The only object a model-written strategy can use.

    Facts are copies of current-instance data.  ``predict`` is a fast
    integer cost model (no build); ``measure`` builds the schedule and applies
    the published Replay formula.  Every call is logged for the audit, and
    measured candidates are kept so the agent can choose among them.
    """

    def __init__(self, planner: Planner, profiles: list[dict[str, Any]], size_weight: float,
                 deadline: float, max_measures: int = 6):
        self._planner = planner
        self._deadline = deadline
        self._max_measures = max_measures
        self.operator = planner.operator
        self.ranks = planner.rank_count
        self.splits = list(planner.splits)
        self.max_rounds = planner.max_rounds
        self.max_width = MAX_PATH_WIDTH
        self.algorithms = list(ALGORITHMS[planner.operator]) if planner.allow_copy or planner.operator == "all-gather" \
            else ["doubling"]
        self.steps = list(planner.steps)
        self.bytes_per_rank = planner.data_size
        self.size_weight = size_weight
        self.dims = [
            {"step": row["step"], "us_per_MiB_w1": row["us_per_MiB_single_path"],
             "us_per_MiB_w4": row["us_per_MiB_multipath"]}
            for row in profiles
        ]
        self.calls: list[dict[str, Any]] = []
        self.measured: list[dict[str, Any]] = []
        self.notes: list[str] = []

    def time_left(self) -> float:
        return max(0.0, self._deadline - perf_counter())

    def log(self, text: Any) -> None:
        if len(self.notes) < 20:
            self.notes.append(str(text)[:200])

    def _config(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise AgentCCError("config must be a dict")
        clean = {key: raw[key] for key in ("algorithm", "split", "path_width", "rs_dims", "order", "widths") if key in raw}
        return self._planner.normalize_config(clean)

    def predict(self, raw: Any) -> dict[str, Any] | None:
        """{'us', 'mb', 'cost'} or None if the config is not allowed."""

        try:
            config = self._config(raw)
            result = self._planner.predict(config)
        except AgentCCError as exc:
            self.calls.append({"tool": "predict", "config": raw, "error": str(exc)})
            return None
        cost = math.log(max(result["us"], 1e-9)) + self.size_weight * result["mb"]
        out = {"us": round(result["us"], 3), "mb": round(result["mb"], 3), "cost": round(cost, 5)}
        if len(self.calls) < 400:
            self.calls.append({"tool": "predict", "config": raw, **out})
        return out

    def measure(self, raw: Any) -> dict[str, Any] | None:
        """Build and Replay-estimate; at most ``max_measures`` per strategy run."""

        if len(self.measured) >= self._max_measures or self.time_left() <= 0:
            self.calls.append({"tool": "measure", "config": raw, "error": "measure budget exhausted"})
            return None
        try:
            config = self._config(raw)
            answer = self._planner.build(config)
        except AgentCCError as exc:
            self.calls.append({"tool": "measure", "config": raw, "error": str(exc)})
            return None
        estimate = replay_estimate(self._planner.task, self._planner.topology, answer, self._planner.network.raw)
        mb = answer_size_estimate(answer) / 1e6
        cost = math.log(max(estimate["total_us"], 1e-9)) + self.size_weight * mb
        limiter = max(estimate["rounds"], key=lambda item: item["time_us"])["limiter"]
        out = {"us": round(estimate["total_us"], 3), "mb": round(mb, 3), "cost": round(cost, 5), "slowest": limiter}
        self.measured.append({"config": config, "answer": answer, "estimate": estimate, **out})
        self.calls.append({"tool": "measure", "config": raw, **out})
        return out

    def best(self) -> dict[str, Any] | None:
        if not self.measured:
            return None
        row = min(self.measured, key=lambda item: item["cost"])
        return {"us": row["us"], "mb": row["mb"], "cost": row["cost"]}

    def sweep(self, algorithms: Any = None, splits: Any = None, widths: Any = (2, 4, 8, 16),
              rs_dims: Any = None, limit: int = 8) -> list[dict[str, Any]]:
        """Predict the cross product the strategy asks for; return the cheapest first.

        Each item is {"cfg": cfg, "us", "mb", "cost"}.  ``rs_dims=None`` tries
        every legal value for rs_ag.
        """

        algorithms = list(algorithms) if algorithms else list(self.algorithms)
        splits = list(splits) if splits else list(self.splits)
        rows = []
        for algorithm in algorithms:
            for split in splits:
                for width in widths:
                    options = [None]
                    if algorithm == "rs_ag":
                        options = list(rs_dims) if rs_dims else list(range(1, int(split).bit_length()))
                    for h in options:
                        cfg = {"algorithm": algorithm, "split": split, "path_width": width}
                        if h is not None:
                            cfg["rs_dims"] = h
                        result = self.predict(cfg)
                        if result is not None:
                            rows.append({"cfg": cfg, **result})
        rows.sort(key=lambda item: item["cost"])
        return rows[:max(1, int(limit))]


def answer_size_estimate(answer: dict[str, Any]) -> int:
    """Approximate serialized size in bytes without building the JSON string."""

    total = 0
    for round_data in answer["rounds"]:
        for transfer in round_data["transfers"]:
            total += 80 + sum(len(item) + 3 for item in transfer["chunk_ranges"])
            total += sum(len(node) + 3 for node in transfer["path"])
    return total + 16 * len(answer["chunks"])


def portfolio_configs(planner: Planner) -> list[dict[str, Any]]:
    """Agent-side candidates evaluated while the model is consulted.

    The set is small and derived from the current input: the smallest split
    for AllGather at three path widths, and for AllReduce reduce-scatter at
    the largest and a middle split.  Anything the round limit or copy rules
    forbid is skipped.
    """

    raw: list[dict[str, Any]] = []
    if planner.operator == "all-gather":
        for width in (2, 8, 16):
            raw.append({"algorithm": "doubling", "split": planner.splits[0], "path_width": width})
    else:
        large = planner.splits[-1]
        middle = [split for split in planner.splits if 2 <= split < large]
        for split, width in [(large, 2), (large, 4)] + ([(middle[-1], 4)] if middle else []):
            raw.append({"algorithm": "rs_ag", "split": split, "path_width": width})
        raw.append({"algorithm": "tree", "split": planner.splits[0], "path_width": 4})
    configs = []
    for item in raw:
        try:
            configs.append(planner.normalize_config(item))
        except AgentCCError:
            continue
    if not configs:
        configs.append(reference_config(planner))
    return configs


def reference_config(planner: Planner) -> dict[str, Any]:
    """Conservative agent-side configuration used when no model plan is usable."""

    if planner.operator == "all-gather":
        raw = {"algorithm": "doubling", "split": planner.splits[0], "path_width": 4, "order": "auto"}
    elif planner.allow_copy and planner.splits[-1] >= 2:
        raw = {"algorithm": "rs_ag", "split": planner.splits[-1], "path_width": 4, "order": "auto"}
    else:
        raw = {"algorithm": "doubling", "split": planner.splits[0], "path_width": 4, "order": "auto"}
    try:
        return planner.normalize_config(raw)
    except AgentCCError:
        raw.update({"algorithm": "doubling", "split": planner.splits[0]})
        raw.pop("rs_dims", None)
        return planner.normalize_config(raw)


def solve_request(task: dict[str, Any], topology: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    planner = Planner(task, topology)
    config = reference_config(planner) if config is None else planner.normalize_config(config)
    answer = planner.build(config)
    validate_answer(task, topology, answer, planner.network.raw)
    attach_metrics(answer, replay_estimate(task, topology, answer, planner.network.raw))
    response = {"task_id": task["task_id"], "status": "success", "output": {"answer": answer}}
    if "instance_id" in task:
        response["instance_id"] = task["instance_id"]
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
