# AgentCC scheduling agent (0.2.1)

The agent searches AgentCC AllGather/AllReduce schedules with the model in
the loop. Every request is planned from its own `task` and `topology`. The
agent loads no fixed instance data and never selects by instance name.

## Decision loop

1. **Parse and validate.** Invalid inputs are rejected with a reason
   (undeclared edge endpoints, rank/switch overlap, duplicate or undefined
   resources, resource references to missing edges, mismatched ranks,
   non-power-of-two rank count, no split satisfying alignment).
2. **Profile.** Every XOR exchange dimension is optimized at path width 1 and
   4; cost per MiB and the limiting edge, shared resource or path are measured.
3. **One model call.** The model receives the constraints, valid splits,
   tightest shared resources and the profiles (about 1.5 KB), and replies with
   up to 3 compact picks `[algorithm, split, width, rs_dims]` (160 output
   tokens, 8 s cap). While it thinks, the agent evaluates its own small
   portfolio in a worker thread.
4. **Tool `evaluate_schedule`.** Model picks and portfolio entries are built on
   the current topology and Replay-estimated. Each is an SDK `tool_call`; model
   picks are linked to the producing model call.
5. **Select.** Candidates are ranked by log(estimate) plus a small
   response-size term; the best is fully validated (next one on failure) and
   returned with `metrics` and `strategy`. The audit records whether the
   winner came from the model, the agent, or both, and the model's gain over
   the agent portfolio. Model failures or timeouts use the portfolio.

0.2.0 used two model rounds and scored 44.7 communication but 19.9 solve time
on the platform (see [platform-evaluation-0.2.0.json](platform-evaluation-0.2.0.json)).
0.2.1 keeps the schedule quality and cuts the model to one short call.
A feedback round remains available through `AGENT_MODEL_ROUNDS=2`.

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
| `AGENT_LLM_BUDGET_SECONDS` | 8 | per model call |
| `AGENT_MODEL_ROUNDS` | 1 | model rounds (2 adds a feedback round) |
| `AGENT_MAX_OUTPUT_TOKENS` | 160 | model `max_tokens` |

## Build, run, test

```powershell
docker build --platform linux/amd64 -t team-001-agent:0.2.1 .
docker run --rm -p 127.0.0.1:18080:8080 -e PLATFORM_LLM_URL -e PLATFORM_API_KEY -e AGENT_MODEL_NAME team-001-agent:0.2.1
python -m unittest -v test_agentcc.py
```

### Local test with a real model

Use your own development API, never the platform credentials. In PowerShell:

```powershell
docker load -i team-001-agent-0.2.1.tar        # only if the image is not loaded yet
$env:PLATFORM_LLM_URL = "https://your-dev-gateway/v1"
$env:AGENT_MODEL_NAME = "your-model"
$env:PLATFORM_API_KEY = Read-Host "dev API key" -MaskInput
docker run --rm --name agentcc-test --cpus 1 --memory 1024m --pids-limit 256 `
  -p 127.0.0.1:18080:8080 -e PLATFORM_LLM_URL -e PLATFORM_API_KEY -e AGENT_MODEL_NAME `
  -e AGENTHACK_TELEMETRY_REQUIRED=false team-001-agent:0.2.1
```

In a second terminal: `python local_test.py`. Model decisions are in
`docker logs agentcc-test | Select-String agentcc.audit`. Stop with
`docker stop agentcc-test`, then `Remove-Item Env:PLATFORM_API_KEY`.

The entry point is `AgentRuntime(AgentCCAgent()).run()` ([src/__main__.py](src/__main__.py)).
It exposes only the SDK's `/health` and `/solve`.

## Local results (public examples, mock model, not official Replay)

In the 0.2.1 container under 1 CPU / 1024 MiB / 256 pids, with a mock model
answering in 1 s:
- All 14 cases were valid, with 1 model call each.
- Solve times were 1.4–2.4 s inside the agent and 1.5–3.1 s over HTTP.
- Memory peaked at 167 MiB.
- Communication estimates matched 0.2.0. In two cases a model-only pick won
  (mesh2d AR 26.6% and nvlink_like AR 8.7% below the agent portfolio).

With a 10 s mock latency, the 8 s cap triggered and the portfolio answer was
returned in about 9 s. Real model latency on the platform is not measured
locally.
