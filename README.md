# AgentCC scheduling agent (0.2.0)

The agent searches AgentCC AllGather/AllReduce schedules with the model in
the loop. Every request is planned from its own `task` and `topology`. The
agent loads no fixed instance data and never selects by instance name.

## Decision loop

1. **Parse and validate.** The agent rejects invalid inputs with a reason:
   undeclared edge endpoints, rank/switch overlap, duplicate or undefined
   resources, resource references to missing edges, mismatched ranks, a
   non-power-of-two rank count, or no split that satisfies alignment.
2. **Profile.** For every XOR exchange dimension, the agent optimizes the round
   on 1 path and on up to 16 switch-only paths per pair. It reports the cost
   per MiB and the bottleneck, which is a directed edge, a shared resource, or
   path time.
3. **Model round 1.** The model receives the constraints, the valid splits,
   the tightest shared resources and the profiles. It proposes up to 4
   configurations: `algorithm`, `split`, `path_width`, `rs_dims` and `order`.
4. **Tool `evaluate_schedule`.** Each proposal is normalized, built, checked
   against every correctness constraint, and estimated with the published
   Replay formula. Each evaluation is recorded as an SDK `tool_call` linked to
   the model call that produced it.
5. **Model round 2.** The model receives the measured times and the slowest
   rounds' limiters, then proposes refinements, which are evaluated the same
   way.
6. **Select.** The fastest validated candidate is returned with
   `metrics.total_time`/`round_times` in seconds and a `strategy` string. If
   the model fails or times out, or every proposal is invalid, a validated
   agent-side reference is returned and recorded as `agent_reference`.

Algorithms:
- AllGather `doubling`.
- AllReduce `rs_ag`: reduce-scatter, then a reduce exchange, then all-gather.
  It moves about 2x the data instead of 8x.
- AllReduce `doubling` and `tree`.

Each pair's chunks are split over several paths by a Frank-Wolfe optimizer
that works on edge, resource and path-time loads.

## Evidence

Each solve logs one `agentcc.audit` JSON line. It contains:
- the input SHA-256;
- per model round: the model content, analysis, finish reason and usage;
- every tool result (status, estimate, slowest rounds and limiters);
- the selection, selected config and timings.

API credentials are not logged.

## Configuration

The platform injects `PLATFORM_LLM_URL`, `PLATFORM_API_KEY` and
`AGENT_MODEL_NAME`; none are stored in the image. Optional settings:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `AGENT_LLM_BUDGET_SECONDS` | 15 | per model call |
| `AGENT_LLM_TOTAL_SECONDS` | 25 | all model calls in one solve |
| `AGENT_SOLVE_BUDGET_SECONDS` | 60 | soft solve deadline (3 s return reserve) |
| `AGENT_MODEL_ROUNDS` | 2 | propose/feedback rounds |
| `AGENT_MAX_OUTPUT_TOKENS` | 1024 | model `max_tokens` |

## Build, run, test

```powershell
docker build --platform linux/amd64 -t team-001-agent:0.2.0 .
docker run --rm -p 127.0.0.1:18080:8080 -e PLATFORM_LLM_URL -e PLATFORM_API_KEY -e AGENT_MODEL_NAME team-001-agent:0.2.0
python -m unittest -v test_agentcc.py
```

The entry point is `AgentRuntime(AgentCCAgent()).run()` ([src/__main__.py](src/__main__.py)).
It exposes only the SDK's `/health` and `/solve`.

## Local results (public examples, mock model, not official Replay)

All 14 public 256-rank cases were run through the real SDK HTTP runtime with a
local mock model. Every case returned a validated answer and made 2 model
calls. HTTP time was 3.5–7.6 s, including about 1.2 s of mock latency per call.

| Case | 0.1.6 estimate (us) | 0.2.0 estimate (us) |
| --- | ---: | ---: |
| uneven_pods AG / AR | 37525 / 5483 | 26192 / 592 |
| directed_ring AG / AR | 29270 / 2039 | 29270 / 630 |
| sparse_fabric AG / AR | 20803 / 1669 | 12871 / 299 |
| cloudmatrix AG / AR | 29385 / 1506 | 8877 / 182 |
| fullmesh_clos AG / AR | 8278 / 1045 | 3415 / 208 |
| mesh2d AG / AR | 58851 / 1265 | 7004 / 312 |
| nvlink_like AG / AR | 10736 / 1193 | 2036 / 140 |

The real model's latency, its proposals, and the official Replay result are
not yet measured. The platform's hidden instances use different bandwidths.
