"""Compile an input topology once; paths contain numeric node/edge IDs."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf, isfinite
import re
from typing import Any


class AgentCCError(ValueError):
    """Invalid input or schedule; ``issue`` optionally locates a violation."""

    def __init__(self, message: str, *, issue=None):
        super().__init__(message)
        self.issue = issue


def bandwidth_bytes_per_second(value: Any) -> float:
    if isinstance(value, (int, float)):
        bandwidth = float(value)
    else:
        text = str(value).strip().upper().replace(" ", "")
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(GB/S|GIB/S|GBPS|GBIT/S|MB/S|KB/S|B/S)", text)
        if not match:
            raise AgentCCError(f"unsupported bandwidth: {value!r}")
        scale = {
            "B/S": 1, "KB/S": 10**3, "MB/S": 10**6, "GB/S": 10**9,
            "GIB/S": 2**30, "GBPS": 10**9 / 8, "GBIT/S": 10**9 / 8,
        }[match.group(2)]
        bandwidth = float(match.group(1)) * scale
    if not isfinite(bandwidth) or bandwidth <= 0:
        raise AgentCCError("bandwidth must be finite and positive")
    return bandwidth


@dataclass(frozen=True, slots=True)
class Route:
    nodes: tuple[int, ...]
    edges: tuple[int, ...]
    # A resource occurs once per bound edge traversal, NOT once per path.
    resources: tuple[int, ...]
    switch_count: int


class Network:
    def __init__(self, topology: dict[str, Any]):
        self.ranks = tuple(topology["ranks"])
        self.nodes = self.ranks + tuple(topology.get("switches", []))
        self.node_ids = {node: index for index, node in enumerate(self.nodes)}
        if len(self.nodes) != len(self.node_ids):
            raise AgentCCError("rank and switch names must be distinct")
        self.startup_us = float(topology.get("communication_startup_time_us", 0))
        self.switch_overhead_us = float(topology.get("switch_processing_overhead_us", 0))

        capacities: dict[tuple[int, int], float] = {}
        memberships: dict[tuple[int, int], set[str]] = {}

        def edge_key(src, dst):
            try:
                return self.node_ids[src], self.node_ids[dst]
            except KeyError as error:
                raise AgentCCError(f"edge references an undeclared node: {error.args[0]}") from error

        for edge in topology["edges"]:
            key = edge_key(edge["src"], edge["dst"])
            bandwidth = bandwidth_bytes_per_second(edge["bw"])
            directions = (key, key[::-1]) if edge.get("bidirectional", False) else (key,)
            for direction in directions:
                capacities[direction] = capacities.get(direction, 0.0) + bandwidth
                if edge.get("resource_id") is not None:
                    memberships.setdefault(direction, set()).add(str(edge["resource_id"]))

        resource_capacity: dict[str, float] = {}
        for resource in topology.get("resources", []):
            name = str(resource["id"])
            if name in resource_capacity:
                raise AgentCCError(f"duplicate resource ID: {name}")
            resource_capacity[name] = bandwidth_bytes_per_second(resource.get("capacity", resource.get("bw")))
            for edge in resource.get("edges", []):
                key = edge_key(edge["src"], edge["dst"])
                directions = (key, key[::-1]) if edge.get("bidirectional", False) else (key,)
                for direction in directions:
                    memberships.setdefault(direction, set()).add(name)

        self.resource_names = tuple(resource_capacity)
        resource_ids = {name: index for index, name in enumerate(self.resource_names)}
        self.resource_capacities = tuple(resource_capacity.values())
        for edge, resources in memberships.items():
            if edge not in capacities:
                raise AgentCCError("resource references an undeclared directed edge")
            if resources - resource_ids.keys():
                raise AgentCCError("edge references an undefined resource")
        self.edge_nodes = tuple(capacities)
        self.edge_ids = {edge: index for index, edge in enumerate(self.edge_nodes)}
        self.edge_capacities = tuple(capacities.values())
        self.edge_resources = tuple(
            tuple(resource_ids[name] for name in sorted(memberships.get(edge, ())))
            for edge in self.edge_nodes
        )
        self.adjacency: list[list[tuple[int, int]]] = [[] for _ in self.nodes]
        for edge_id, (src, dst) in enumerate(self.edge_nodes):
            self.adjacency[src].append((dst, edge_id))
        for neighbours in self.adjacency:
            neighbours.sort(key=lambda pair: self.nodes[pair[0]])
        self.routes: dict[tuple[int, ...], Route] = {}

    def compile_route(self, path) -> Route:
        try:
            nodes = tuple(self.node_ids[node] for node in path)
        except (KeyError, TypeError) as error:
            raise AgentCCError("path contains an undeclared node") from error
        if len(nodes) < 2 or nodes[0] == nodes[-1]:
            raise AgentCCError("path needs distinct rank endpoints")
        if nodes[0] >= len(self.ranks) or nodes[-1] >= len(self.ranks):
            raise AgentCCError("path endpoints must be ranks")
        if any(node < len(self.ranks) for node in nodes[1:-1]):
            raise AgentCCError("path contains a rank as an intermediate node")
        cached = self.routes.get(nodes)
        if cached is not None:
            return cached
        try:
            edges = tuple(self.edge_ids[pair] for pair in zip(nodes, nodes[1:]))
        except KeyError as error:
            raise AgentCCError("path contains an undeclared directed edge") from error
        route = Route(nodes, edges, tuple(r for e in edges for r in self.edge_resources[e]), len(nodes) - 2)
        self.routes[nodes] = route
        return route


class Router:
    """Keep the baseline's weighted paths, materializing only requested pairs."""

    def __init__(self, network: Network, reference_bytes: int):
        self.network = network
        self.costs = []
        for edge, capacity in enumerate(network.edge_capacities):
            cost = 1e6 * reference_bytes / capacity
            for resource in network.edge_resources[edge]:
                cost += 1e6 * reference_bytes / network.resource_capacities[resource]
            self.costs.append(cost)
        self.previous: dict[int, dict[int, int | None]] = {}

    def route(self, src: int, dst: int) -> Route:
        network = self.network
        if src not in self.previous:
            distance = {src: 0.0}
            previous: dict[int, int | None] = {src: None}
            queue = [(0.0, network.nodes[src], src)]
            while queue:
                current, _, node = heappop(queue)
                if current != distance[node]:
                    continue
                for neighbour, edge in network.adjacency[node]:
                    if neighbour == src:
                        continue
                    terminal = neighbour < len(network.ranks)
                    candidate = current + self.costs[edge]
                    if not terminal:
                        candidate += network.switch_overhead_us
                    if candidate < distance.get(neighbour, inf):
                        distance[neighbour] = candidate
                        previous[neighbour] = node
                        if not terminal:
                            heappush(queue, (candidate, network.nodes[neighbour], neighbour))
            self.previous[src] = previous
        previous = self.previous[src]
        if dst not in previous:
            raise AgentCCError(f"no legal route from {network.nodes[src]} to {network.nodes[dst]}")
        path = []
        node: int | None = dst
        while node is not None:
            path.append(network.nodes[node])
            node = previous[node]
        return network.compile_route(reversed(path))
