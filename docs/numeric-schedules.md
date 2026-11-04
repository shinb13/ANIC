# Numeric schedules and round verification

This step supplies the cost model and correctness state machine for later bounded
search. The production baseline still chooses the minimum legal split and uses
recursive doubling. It does not implement tree packing, topology sketches, or a
new AllReduce algorithm, and does not claim a communication-score improvement.

## Representation

`Problem(task, topology, split)` compiles only the request's input. Directed edge
and resource capacities are arrays indexed by integer IDs. Duplicate directed
edge declarations add capacity; each edge/resource membership is deduplicated.
Undefined resources, duplicate resource IDs and references to missing edges fail.

`Route` stores numeric node/edge IDs and the resources traversed. The same resource
can appear more than once when multiple edges on a path bind it. `Router` retains
the baseline's bandwidth/resource-weighted Dijkstra policy, caches predecessor
maps per source, and materializes only the paths requested by the schedule.

`Plan` and `Transfer` are immutable. AllGather raw chunk ID is `rank_id * split +
slot`; its payload is an integer bitset. AllReduce payload is a tuple of
`(slot, contributor_mask)` pairs. One contributor bit represents one input rank.
Only included slots are transferred; there is no requirement to send every slot.
Partial-slot support is a verifier capability, not a new baseline selection rule.

## Validation feedback

```python
from src.agentcc_plan import Problem, RoundVerifier

problem = Problem(task, topology, split=1)
prefix = RoundVerifier(problem)
candidate = prefix.fork()
issue = candidate.apply_round(transfers)
if issue is None:
    prefix = candidate
# Call prefix.finish() when evaluating a complete schedule.
```

`apply_round` reads sender state at the beginning of the round, accumulates
receiver updates separately, and commits only when every transfer passes. An
invalid attempt leaves possession, sent-contribution records and round counters
unchanged. `fork()` shares immutable state tuples and copies sent records, allowing
alternative prefixes without modifying the parent. Problem/routes are shared;
their caches are request-local, not a thread-safe concurrent search API.

`ValidationIssue` contains `code`, `message`, a 1-based `round_number`, a 0-based
`transfer_index`, numeric `src`/`dst`, and an optional `slot`. Numeric ranks map
through `problem.ranks`. Examples include `SOURCE_STATE`, `DUPLICATE_RECEIVE`,
`DUPLICATE_SEND`, `DUPLICATE_CONTRIBUTOR`, `INCOMPLETE_COPY`, `ROUND_LIMIT` and
`INCOMPLETE`. Public validation raises `AgentCCError` and exposes the issue as
`error.issue`. JSON parsing errors also include a round/transfer when available.

AllGather forbids receiving an already held/pending chunk. AllReduce checks exact
source expressions, disjoint reduce contributors, and complete-result copies.
An already complete destination cannot receive the result again. Empty rounds
consume no nonempty-round budget; serialization removes them and renumbers.

`build_baseline` verifies each round as it builds. `plan_to_answer` serializes a
verified plan once, caches reduction expressions within the request, and generates
the canonical public `chunks`, `initial_possession` and `final_chunks` fields.
`plan_from_answer` reads this canonical explicit-path format; it is not a complete
implementation of optional official aliases such as `route_id` or nested
`solution` wrappers. `validate_answer` additionally replays the parsed plan.

## Replay model

For each transfer, let `B` be its byte count and `h = startup_us + switch_count *
switch_overhead_us`. AllGather counts raw chunks; AllReduce counts unique slots,
irrespective of the number of contributors in an expression.

```text
path_time = h + sum(1e6 * B / edge_capacity)
edge_busy = max(1e6 * accumulated_edge_bytes / edge_capacity)
resource_busy = max(1e6 * accumulated_resource_bytes / resource_capacity)
round_time = max(max(path_time), max(h) + max(edge_busy, resource_busy))
```

The previous estimator always added one switch overhead to the contention term.
The new estimator uses the actual maximum per-transfer overhead: zero switches
on a direct path, and every intermediate switch on a multi-switch path. Resource
load counts every bound edge traversal; overlapping resources each receive the
full load. Loads reset per round. Capacities affect time, not schedule legality.

`estimate_round` returns `RoundTiming`, including these timing components and
sparse `(id, bytes)` edge/resource loads. IDs map through `problem.network`.
`estimate_plan_time_us` evaluates numeric candidates without constructing JSON;
call it only after verification. `estimate_schedule_time_us` is the public JSON
adapter. The official Replay remains authoritative.

## Verification and benchmarking

Run `python -m unittest -v` and `python benchmark_agentcc.py --repeat 3`.
The benchmark reports separate build, external-validation, local-estimation and
JSON-serialization times. `build_s` includes topology compilation, routing,
numeric construction, incremental validation and output-dictionary creation.
It excludes file reads and model/SDK/HTTP latency. Do not compare it directly to
the platform's end-to-end solve-time score.
