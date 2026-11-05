# AgentCC scheduling agent

This repository contains a model-guided AgentCC scheduling agent with
deterministic schedule construction and constraint validation.

For each request, the agent first builds and validates the `baseline` plan.
It then calls the configured OpenAI-compatible model once to select a scheduling
algorithm and route policy. Only the selected extra candidate is constructed,
validated, and accepted when its local communication estimate is strictly lower
than the baseline. Model API failures, timeouts, invalid decisions, exhausted
budgets, and rejected candidates retain the validated baseline; unexpected
programming and configuration errors propagate.

The agent reads the request's `task` and
`topology`, including directed edge bandwidth and shared resources, builds
resource-aware switch-only-intermediate routes, and selects from:

- `recursive_doubling`: the original baseline for either operator.
- `ordered_doubling`: AllGather with a topology-derived XOR dimension order.
  Dynamic programming minimizes the published round-cost model for fixed
  routes and dimensions; 256 ranks still require eight rounds.
- `tree`: AllReduce reduction followed by broadcast, using sixteen rounds for
  256 ranks. The tree sends fewer total logical bytes but is retained only when
  its estimated communication time improves on the baseline.

The minimum legal split is used for all candidates. The agent does not load
fixed public topology data or select schedules by instance names.
Path weights are precomputed once per static route policy. Only XOR exchange
pairs used by the supported schedules are materialized: 2,048 paths for 256
ranks, rather than 65,280 paths. Other ranks remain forbidden as path interiors.
Baseline routes and the AllGather cost table are reused for `baseline` and
`load_balanced`.

The local estimator includes directed edge loads, overlapping shared resources,
path transmission, startup, and switch overhead. The official platform Replay
remains the scoring authority.

The public examples are local fixtures only. Official evaluation data arrives
inside each `/solve` request.

## Build and run

The Dockerfile uses the downloaded platform base image `agenthack/base:1.1.1`.
At runtime the platform must provide `PLATFORM_LLM_URL`, `PLATFORM_API_KEY`,
and `AGENT_MODEL_NAME`; these values are never stored in the repository or
image.

```powershell
docker build --platform linux/amd64 -t team-001-agent:0.1.5 .
docker run --rm -p 127.0.0.1:18080:8080 `
  -e PLATFORM_LLM_URL -e PLATFORM_API_KEY -e AGENT_MODEL_NAME `
  team-001-agent:0.1.5
```

The runtime exposes `GET /health` and `POST /solve` on port 8080. The request
body follows the AgentRuntime contract: `task_id` plus an `input` object with
the current AgentCC `task` and `topology`.

The model request uses `AGENT_LLM_BUDGET_SECONDS` (default `2`) as an outer
wall-clock deadline and sets the OpenAI client retry count to zero. The solve
soft deadline is `AGENT_SOLVE_BUDGET_SECONDS` (default `25`); a 250 ms return
reserve is kept before building an extra candidate.

The model receives actual task constraints, per-dimension timing estimates,
directed bottleneck edges and bandwidths, shared-resource capacities and used
member examples, and representative current paths. This is a bounded view of
the current graph; it does not contain the full edge list. The prompt states
that resource member examples are partial.

The model returns exactly an `algorithm` and `policy`, for example:

```json
{"algorithm":"ordered_doubling","policy":"load_balanced"}
```

Available algorithms are restricted by operator, copy support, and round limit.
Route policies `baseline`, `bandwidth_first`, and `resource_first` apply shared
resource penalties of 1.0, 0.25, and 2.0 respectively during path construction.
The additional `load_balanced` policy starts from baseline paths and refines
individual transfers using concurrent directed-edge and shared-resource loads.
It searches at most 64 paths, at most 16 transfers per round, with a 250 ms
search budget bounded by the remaining solve deadline. Higher-cost rounds are
visited first. Squared utilization guides path proposals; the complete local
Replay formula decides acceptance. No round estimate may increase. Equal-time
moves require lower squared utilization, allowing progress across tied
bottlenecks. Repeated traversals and overlapping resource memberships are
counted during acceptance. The final candidate is validated before comparison
with the baseline. This policy is unavailable when detours are disallowed.

Each solve logs the exact model prompt and response, linked to a SHA-256 hash
of the current task and topology. The metrics record includes `baseline_ms`,
`planning_ms`, `llm_ms`, `candidate_ms`, `solve_ms`, `response_bytes`,
`selected_algorithm`, `selected_policy`, `model_plan`, both communication
estimates, `candidate_validation`, `route_balancing`, and `fallback_reason`. The
balancing record includes attempted and changed paths, initial and final
estimates, elapsed time, and the stop reason. API credentials are
not included in these records.

`response_bytes` measures the compact business answer inside the agent. For
the complete `/solve` path, record `http_total_ms` and the exact response length
in a local client; SDK wrapping, telemetry flushing, and HTTP transmission are
outside the agent's `solve_ms`.

## Local verification

```powershell
python -m unittest -v test_agentcc.py
```

The suite covers both operators on all fourteen public 256-rank examples,
exhaustive dimension-order comparisons, overlapping resources and directional
bandwidths, input-capacity changes, alignment, tree contribution correctness,
model-driven adoption and rejection using real estimates, and request
cancellation, needed-pair routing, route balancing under repeated/overlapping
resources, rejection of slower detours, and exhausted search deadlines.
Public examples and local estimates do not establish an official
Replay score.

Version 0.1.5 passes 21 unit tests. A paired container run compared the preserved
0.1.4 image with 0.1.5 through the actual SDK HTTP interface, using a local mock
model, 1 CPU, 1024 MiB, and 256 pids. All 28 evaluated requests were locally
valid and recorded one model API call each. All 28 input/output/metrics audit
chains were checked. The mock selects `load_balanced` for 0.1.5 and `baseline`
for 0.1.4, with the same scheduling algorithm for each operator.

Relative to 0.1.4 on these public fixtures, nine communication estimates
improved and five were unchanged. AllGather improvements were 0.79–5.66%;
Uneven Pods and Mesh2D AllReduce improved by 36.27% and 30.04%. Other AllReduce
estimates were unchanged. In this single paired run after one warmup per image,
mean HTTP duration changed from 897.581 to 680.501 ms; 0.1.5 requests took
417.560–1650.020 ms. Planning plus candidate work took 147.137–459.591 ms,
including 9.973–250.027 ms for load balancing. See
[the 0.1.5 validation report](optimization-validation-0.1.5.json) for per-case
data and artifact checksums. These timings exclude real model latency, and
the communication values are local estimates rather than official scores.

Version 0.1.4 previously passed all fourteen public examples through the actual
AgentRuntime HTTP interface with a local mock model service, using 1 CPU,
1024 MiB, and 256 pids. Each request recorded one model API call, accepted
the selected optimization, and produced a locally valid answer. The additional
planning and candidate work measured 226–436 ms in this single run.
The model prompt, response, input hash, and acceptance records were verified
for every request. See [the validation report](optimization-validation-0.1.4.json)
for timings, image identity, source hashes, and the exported archive checksum.
This run does not measure real-model latency or official Replay performance.

## Platform evaluation reported by the user

The platform report completed at 2026-11-05 13:08 records 14/14 feasible
instances, 128 rounds, and accepted execution evidence. Means calculated from
the displayed case scores are 15.000 for correctness, 38.840 for communication,
34.484 for solve time, and 88.324 overall. The earlier user-provided report
averaged 81.583 overall. Seven Replay times improved and seven were unchanged;
the displayed report does not expose the model decisions or fallback reasons.
See [the platform evaluation summary](PLATFORM_EVALUATION.md) and
[per-instance data](platform-evaluation-0.1.5.json). The measurements above in
Local verification remain separate mock-model results.

## Container entry point

The image starts `AgentRuntime(AgentCCAgent()).run()` and exposes the SDK's
`/health` and `/solve` endpoints.
