# AgentCC scheduling agent (0.2.4)

0.2.4 is 0.2.3 with a shorter model wait and less planning work. Schedules are
unchanged on the public examples. Every request is planned from its own `task`
and `topology`. The agent loads no fixed instance data and never selects by
instance name.

## Loop

1. **Analyse.** The agent parses and validates the input (invalid inputs are
   rejected with a reason). It profiles every XOR dimension at path width 1,
   then ranks a portfolio of configurations with the fast predictor.
2. **Model call.** One short call (about 1.3 KB prompt, 120 output tokens,
   3 s cap). The model receives the bottlenecks and the agent's own
   predicted time/size trade-offs, and proposes up to 3 configurations. It
   may add picks the agent did not list. The agent builds its top 3 in
   parallel while the model is thinking.
3. **Tool `evaluate_schedule`.** Candidates are built on the current topology
   and Replay-estimated. A model pick whose prediction already loses to a built
   candidate is answered from the prediction instead. Each evaluation is an SDK
   `tool_call`, linked to the model call for model picks.
4. **Select.** Candidates are ranked by `ln(Replay us) + 0.10 × response MB`.
   The first one that passes full validation is returned. The audit records
   whether the model's pick won outright.

## Changes in 0.2.4: solve time

- **Model cap 3 s instead of 6 s.** A reply after the cap is dropped, the agent's
  portfolio is used, and the fallback is recorded as a recovery. Set
  `AGENT_LLM_BUDGET_SECONDS` to change the cap.
- **Less planning work.** The agent ranks `tree` only when `rs_ag` is not
  available, and builds its top 3 instead of 4. The model can still propose
  `tree`.
- **Prediction filter for model picks.** A pick is not built when its predicted
  cost is more than 0.02 above the best built candidate. Across 989
  configurations on the public examples this never rejected a winner.
- **Evidence in the answer.** `answer.strategy` now ends with `llm_ms` and the
  model outcome, because the platform report does not return logs.

Why these changes: on the public examples, a full sweep of the configurations
the model can name found nothing worth more than an estimated 0.02 points over
the agent's own pick. Local computation takes 0.1–0.7 s, so the model wait is
the largest part of the solve time that the agent controls.

The real model's latency on the platform is unknown. If it often exceeds 3 s, the
model's picks will rarely be used; `model=model_timeout` in `answer.strategy`
shows this.

### Tried and not adopted

All on the public examples with the local Replay estimate. The estimated value
of one second of solve time is about an 18% change in Replay time.

| Idea | Result |
| --- | --- |
| One-way shift schedules for AllGather | 77–628% slower than XOR |
| Local search over dimension order | no change |
| Topology-aware rank relabelling | helps `uneven_pods` only (AG −14%, AR −24%); `mesh2d` gets much slower; about +0.06 points overall |
| Load-aware rounding of chunks to routes | mean −2.5%; the extra build time costs more than it gains |
| Larger response size weight | selection almost unchanged |

## Changes in 0.2.3: `c{slot}_sum` in copies

In the AllReduce all-gather phase, each copy of a completely reduced slot
is written as `c{slot}_sum`. The output format defines this notation for
complete slots. Before 0.2.3, every copy listed all 256 contributors. With
the short form, AllReduce responses fell from 9–18 MB to 1.0–1.8 MB. Replay
estimates did not change, and the best-communication `rs_ag` configurations
are selected again.

0.2.3 scored 92.7 on the platform according to a team screenshot (no report is
archived here), so the official Replay accepted this notation inside transfers.
Set `AGENTCC_SUM_NOTATION=0` to restore full contributor expressions.

## Changes from 0.2.1 (0.2.2)

- **Always returns an answer.** Model, parse, candidate and selection errors
  are all contained. A last-resort reference configuration is built and
  validated if nothing else survives. In 0.3.0, fullmesh_clos AR returned no
  result.
- **Response size counts.** AllReduce responses fell from 9–18 MB toward
  2–9 MB. On the platform, solve time grew about 1.46 s per MB. The weight
  0.10 was chosen by sweeping slope assumptions from 0.7 to 2.0 s/MB under
  the new 25/50/25 scoring; it kept rs_ag wherever its communication margin
  was large.
- **Lower fixed cost.** The agent profiles at width 1 only (width-4 plans are
  built later, only when used). The model cap is 6 s instead of 8 s, and
  the model output limit is 120 tokens instead of 160.
- 0.3.0's code-generation sandbox (`strategy_sandbox.py`, `StrategyAPI`)
  stays in the repository but is not on the default path.

## Evidence

Each solve logs one `agentcc.audit` JSON line. Per round it records:
- the prompt and the model reply;
- the extracted code and any error;
- every `predict` and `measure` call with its result, and the strategy's log
  lines;
- the run time.

The line also records the selection and its origin. In the SDK trace, each
code execution is a `run_strategy` tool call linked to the model call that
produced the code. Error-driven retries are marked with `recovery`. API
credentials are not logged.

## Configuration

The platform injects `PLATFORM_LLM_URL`, `PLATFORM_API_KEY` and
`AGENT_MODEL_NAME`; none are stored in the image. Optional settings:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `AGENT_LLM_BUDGET_SECONDS` | 3 | model call cap |
| `AGENT_MAX_OUTPUT_TOKENS` | 120 | model `max_tokens` |
| `AGENTCC_SUM_NOTATION` | 1 | write complete-slot copies as `c{slot}_sum` |

## Build, run, test

```powershell
docker build --platform linux/amd64 -t team-001-agent:0.2.4 .
python -m unittest -v test_agentcc.py
```

### Local test with a real model

Use your own development API, never the platform credentials. In PowerShell:

```powershell
docker load -i team-001-agent-0.2.4.tar        # only if the image is not loaded yet
$env:PLATFORM_LLM_URL = "https://your-dev-gateway/v1"
$env:AGENT_MODEL_NAME = "your-model"
$env:PLATFORM_API_KEY = Read-Host "dev API key" -MaskInput
docker run --rm --name agentcc-test --cpus 1 --memory 1024m --pids-limit 256 `
  -p 127.0.0.1:18080:8080 -e PLATFORM_LLM_URL -e PLATFORM_API_KEY -e AGENT_MODEL_NAME `
  -e AGENTHACK_TELEMETRY_REQUIRED=false team-001-agent:0.2.4
```

In a second terminal, run `python local_test.py`. The code the model wrote is
in `docker logs agentcc-test | Select-String agentcc.audit`. To stop, run
`docker stop agentcc-test`, then `Remove-Item Env:PLATFORM_API_KEY`.

The entry point is `AgentRuntime(AgentCCAgent()).run()` ([src/__main__.py](src/__main__.py)).

## Local results (public examples, mock model, not official Replay)

0.2.3 and 0.2.4 ran with `bench_local.py` under 1 CPU / 1024 MiB / 256 pids. This
used a `python:3.11-slim` test image and a stub SDK, not the competition base
image, and called `solve` directly rather than over HTTP. Figures are medians of
3 runs per case; the data is in
[optimization-validation-0.2.4.json](optimization-validation-0.2.4.json).

| Mock model delay | Mean solve, 0.2.3 | Mean solve, 0.2.4 | Model reply used by 0.2.4 |
| ---: | ---: | ---: | --- |
| 1 s | 1.35 s | 1.31 s | yes |
| 5.5 s | 5.85 s | 3.30 s | no, timed out at 3 s |

- All 14 cases were valid in every run.
- Replay estimates and response sizes were identical in 0.2.3 and 0.2.4.
- Responses were 0.84–1.78 MB, and memory peaked at 119 MiB.

These figures do not include the real model, the real SDK or platform transfer
time. To repeat them, run `python bench_local.py --help`.

### Container HTTP check (real SDK, mock model)

Both images were built on `agenthack/base:1.1.1` (SDK 1.1.1, Python 3.11.16,
numpy 1.26.4) from the committed source. `local_test.py` sent all 14 cases over
HTTP under 1 CPU / 1024 MiB / 256 pids. An OpenAI-compatible mock model answered
after a fixed delay.

| Mock model delay | 0.2.3 per case | 0.2.4 per case | Valid | Peak memory |
| ---: | ---: | ---: | --- | ---: |
| 1 s | 1.1–1.7 s | 1.1–1.6 s | 14/14 both | 139 / 177 MiB |
| 5.5 s | 5.6–6.2 s | 3.1–3.6 s | 14/14 both | 142 / 131 MiB |

The SDK recorded one model call per case. When 0.2.4 timed out at 3 s, the call
was recorded with 0 tokens. Replay estimates were the same in both images.

The delivered image is `yzh-agentcc-opt-01.tar` (`team-001-agent:0.2.4`). Its
checksum is in [yzh-agentcc-opt-01.tar.sha256](yzh-agentcc-opt-01.tar.sha256).

The local validator accepts `c{slot}_sum` only as the complete set of
contributors for that slot. A copy of an incomplete slot written this way is
rejected.
