# AgentCC scheduling agent (0.2.3)

0.2.2 is built on 0.2.1, the best-scoring version so far (88.2 under the old
rules). It keeps 0.2.1's communication quality and aims at solve time and
reliability. Every request is planned from its own `task` and `topology`. The
agent loads no fixed instance data and never selects by instance name.

## Loop

1. **Analyse.** The agent parses and validates the input (invalid inputs are
   rejected with a reason). It profiles every XOR dimension at path width 1,
   then ranks a portfolio of configurations with the fast predictor.
2. **Model call.** One short call (about 1.3 KB prompt, 120 output tokens,
   6 s cap). The model receives the bottlenecks and the agent's own
   predicted time/size trade-offs, and proposes up to 3 configurations. It
   may add picks the agent did not list. The agent builds its top 4 in
   parallel while the model is thinking.
3. **Tool `evaluate_schedule`.** Every candidate is built on the current
   topology and Replay-estimated. Each evaluation is an SDK `tool_call`,
   linked to the model call for model picks.
4. **Select.** Candidates are ranked by `ln(Replay us) + 0.10 × response MB`.
   The first one that passes full validation is returned. The audit records
   whether the model's pick won outright.

## Changes in 0.2.3: `c{slot}_sum` in copies

In the AllReduce all-gather phase, each copy of a completely reduced slot
is written as `c{slot}_sum`. The output format defines this notation for
complete slots. Before 0.2.3, every copy listed all 256 contributors. With
the short form, AllReduce responses fell from 9–18 MB to 1.0–1.8 MB. Replay
estimates did not change, and the best-communication `rs_ag` configurations
are selected again.

Whether the official Replay accepts this notation inside transfers has not
been confirmed. Set `AGENTCC_SUM_NOTATION=0` to restore full contributor
expressions; 0.2.2 is the same code with that behaviour.

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
| `AGENT_LLM_BUDGET_SECONDS` | 6 | model call cap |
| `AGENT_MAX_OUTPUT_TOKENS` | 120 | model `max_tokens` |
| `AGENTCC_SUM_NOTATION` | 1 | write complete-slot copies as `c{slot}_sum` |

## Build, run, test

```powershell
docker build --platform linux/amd64 -t team-001-agent:0.2.3 .
python -m unittest -v test_agentcc.py
```

### Local test with a real model

Use your own development API, never the platform credentials. In PowerShell:

```powershell
docker load -i team-001-agent-0.2.3.tar        # only if the image is not loaded yet
$env:PLATFORM_LLM_URL = "https://your-dev-gateway/v1"
$env:AGENT_MODEL_NAME = "your-model"
$env:PLATFORM_API_KEY = Read-Host "dev API key" -MaskInput
docker run --rm --name agentcc-test --cpus 1 --memory 1024m --pids-limit 256 `
  -p 127.0.0.1:18080:8080 -e PLATFORM_LLM_URL -e PLATFORM_API_KEY -e AGENT_MODEL_NAME `
  -e AGENTHACK_TELEMETRY_REQUIRED=false team-001-agent:0.2.3
```

In a second terminal, run `python local_test.py`. The code the model wrote is
in `docker logs agentcc-test | Select-String agentcc.audit`. To stop, run
`docker stop agentcc-test`, then `Remove-Item Env:PLATFORM_API_KEY`.

The entry point is `AgentRuntime(AgentCCAgent()).run()` ([src/__main__.py](src/__main__.py)).

## Local results (public examples, mock model, not official Replay)

These runs used the 0.2.3 container under 1 CPU / 1024 MiB / 256 pids, with
a mock model answering in 1 s:
- All 14 cases were valid, each taking 1.3–2.0 s over HTTP.
- AllReduce responses were 1.0–1.8 MB with the same Replay estimates as
  0.2.1.
- Memory peaked at 140 MiB.

The local validator accepts `c{slot}_sum` only as the complete set of
contributors for that slot. A copy of an incomplete slot written this way is
rejected.
