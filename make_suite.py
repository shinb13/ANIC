"""Build a local "V22-style" test set by perturbing the public V21 examples.

The platform's 14 instances (V22) change link bandwidths, shared-resource
capacities and some edges relative to the public examples, so results on the
public files alone transfer poorly.  Each variant here keeps the structure of a
public example and, with a fixed seed:

* scales every directed link bandwidth by a random factor in [0.5, 1.5]
  (both directions of a bidirectional link independently, so reverse
  bandwidths can differ),
* scales every shared-resource capacity by a random factor in [0.5, 1.5],
* removes a few switch-switch links (only if every rank can still reach every
  other rank) and adds a few new switch-switch links with bandwidths drawn
  from the existing ones.

These are local stand-ins, not the official instances.

    python make_suite.py --out suite --seeds 5
"""

import argparse
import copy
import json
import random
from collections import deque
from pathlib import Path

FAMILIES = ["uneven_pods", "directed_ring", "sparse_fabric", "cloudmatrix", "fullmesh_clos", "mesh2d", "nvlink_like"]
EXAMPLES = Path(__file__).parent / "agentcc_zip_tmp" / "examples" / "256-rank"


def parse_bw(text) -> float:
    if isinstance(text, (int, float)):
        return float(text)
    value = text.strip()
    for unit, scale in (("GB/s", 1e9), ("MB/s", 1e6), ("KB/s", 1e3), ("Gbps", 1e9 / 8), ("Gbit/s", 1e9 / 8), ("B/s", 1.0)):
        if value.endswith(unit):
            return float(value[: -len(unit)]) * scale
    return float(value)


def fmt_bw(value: float) -> str:
    return f"{value / 1e9:.3f}GB/s"


def directed_edges(edges):
    for edge in edges:
        yield edge["src"], edge["dst"]
        if edge.get("bidirectional"):
            yield edge["dst"], edge["src"]


def connected(ranks, edges) -> bool:
    adjacency = {}
    for src, dst in directed_edges(edges):
        adjacency.setdefault(src, set()).add(dst)
    start = ranks[0]
    for direction in ("forward", "reverse"):
        graph = adjacency if direction == "forward" else {}
        if direction == "reverse":
            for src, targets in adjacency.items():
                for dst in targets:
                    graph.setdefault(dst, set()).add(src)
        seen, queue = {start}, deque([start])
        while queue:
            node = queue.popleft()
            for nxt in graph.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        if any(rank not in seen for rank in ranks):
            return False
    return True


def perturb(task, topology, seed):
    rng = random.Random(seed)
    topo = copy.deepcopy(topology)
    ranks = list(task["ranks"])
    rank_set = set(ranks)
    switch_edges = [edge for edge in topo["edges"] if edge["src"] not in rank_set and edge["dst"] not in rank_set]
    # Remove a few switch-switch links while every rank stays reachable.
    removable = list(switch_edges)
    rng.shuffle(removable)
    removed = set()
    budget = max(1, len(switch_edges) // 30)
    for edge in removable:
        if len(removed) >= budget:
            break
        trial = [item for item in topo["edges"] if item is not edge and id(item) not in removed]
        if connected(ranks, trial):
            removed.add(id(edge))
    gone = {(edge["src"], edge["dst"]) for edge in topo["edges"] if id(edge) in removed}
    gone |= {(dst, src) for edge in topo["edges"] if id(edge) in removed and edge.get("bidirectional")
             for src, dst in [(edge["src"], edge["dst"])]}
    topo["edges"] = [edge for edge in topo["edges"] if id(edge) not in removed]
    # Add a few new switch-switch links.
    switches = sorted({node for edge in topo["edges"] for node in (edge["src"], edge["dst"])} - rank_set)
    existing = set(directed_edges(topo["edges"]))
    bws = [parse_bw(edge["bw"]) for edge in switch_edges] or [parse_bw(topo["edges"][0]["bw"])]
    for _ in range(max(1, len(switch_edges) // 30) if len(switches) >= 2 else 0):
        for _attempt in range(50):
            a, b = rng.sample(switches, 2)
            if (a, b) not in existing and (b, a) not in existing:
                topo["edges"].append({"src": a, "dst": b, "bw": fmt_bw(rng.choice(bws)), "bidirectional": True})
                existing |= {(a, b), (b, a)}
                break
    # Scale link bandwidths per direction.
    new_edges = []
    for edge in topo["edges"]:
        bw = parse_bw(edge["bw"])
        if edge.get("bidirectional"):
            for src, dst in ((edge["src"], edge["dst"]), (edge["dst"], edge["src"])):
                item = {key: value for key, value in edge.items() if key not in ("src", "dst", "bw", "bidirectional")}
                new_edges.append({"src": src, "dst": dst, "bw": fmt_bw(bw * rng.uniform(0.5, 1.5)),
                                  "bidirectional": False, **item})
        else:
            new_edges.append({**edge, "bw": fmt_bw(bw * rng.uniform(0.5, 1.5))})
    topo["edges"] = new_edges
    present = {(edge["src"], edge["dst"]) for edge in new_edges}
    # Scale shared resources and drop references to removed links.
    resources = []
    for resource in topo.get("resources") or []:
        item = copy.deepcopy(resource)
        capacity = parse_bw(item.get("capacity", item.get("bw")))
        item.pop("bw", None)
        item["capacity"] = fmt_bw(capacity * rng.uniform(0.5, 1.5))
        refs = []
        for ref in item.get("edges") or []:
            pairs = [(ref["src"], ref["dst"])] + ([(ref["dst"], ref["src"])] if ref.get("bidirectional") else [])
            for src, dst in pairs:
                if (src, dst) in present and (src, dst) not in gone:
                    refs.append({"src": src, "dst": dst})
        item["edges"] = refs
        if refs or not resource.get("edges"):
            resources.append(item)
    used = {resource["id"] for resource in resources}
    for edge in topo["edges"]:
        if edge.get("resource_id") and edge["resource_id"] not in used:
            edge.pop("resource_id")
    topo["resources"] = resources
    new_task = copy.deepcopy(task)
    new_task["task_id"] = f"{task['task_id']}_s{seed}"
    new_task["instance_id"] = new_task["task_id"]
    return new_task, topo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="suite")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.out)
    from agentcc_solver import Planner  # validates every variant

    for family in FAMILIES:
        for seed in range(1, args.seeds + 1):
            for short in ("ag", "ar"):
                base = EXAMPLES / family / short
                task = json.loads((base / "task.json").read_text(encoding="utf-8"))
                topology = json.loads((base / "topology.json").read_text(encoding="utf-8"))
                new_task, new_topo = perturb(task, topology, seed * 1000 + FAMILIES.index(family))
                Planner(new_task, new_topo)
                target = out / f"{family}_s{seed}" / short
                target.mkdir(parents=True, exist_ok=True)
                (target / "task.json").write_text(json.dumps(new_task), encoding="utf-8")
                (target / "topology.json").write_text(json.dumps(new_topo), encoding="utf-8")
        print(family, "ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
