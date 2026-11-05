# AgentCC scheduling agent (0.3.0)

The model writes and revises the search code that drives the solver. Every
request is planned from its own `task` and `topology`. The agent loads no
fixed instance data and never selects by instance name.

## Loop: analyse → generate code → execute → iterate

1. **Analyse.** The agent parses and validates the input and rejects invalid
   inputs with a reason. It then measures every XOR exchange dimension at
   path width 1 and 4: Replay cost per MiB and the limiting edge, shared
   resource or path.
2. **Generate.** The model receives four things:
   - the measurements;
   - the tool API ([API_DOC in src/my_agent.py](src/my_agent.py));
   - a working baseline strategy;
   - an instruction to revise that baseline for this instance.

   It writes a Python function `strategy(api)` that decides which
   algorithms, splits, `rs_dims`, per-round path widths and orders to
   explore. The function ranks them with `api.predict` / `api.sweep` and
   builds the most promising with `api.measure`.
3. **Execute.** [strategy_sandbox.py](strategy_sandbox.py) checks the code
   statically and runs it under a step and time limit. The checks reject
   imports, dunder or private access, file, network or process access,
   `while`, `try` and non-whitelisted names and attributes. The strategy
   receives whitelisted builtins only. Each `measure` builds a full schedule
   on the current topology and applies the published Replay formula; at
   most 5 measures run per strategy run.
4. **Iterate.** The measured candidates, slowest-round limiters, the model's
   log lines and any error or traceback line go back to the model. The model
   revises its code, and the revision runs the same way. The default is 2
   model rounds.
5. **Select.** Candidates are ranked by `ln(Replay us) + 0.15 × response MB`.
   The first one that passes full validation is returned with `metrics` and a
   `strategy` string naming the producing round. If the model or its code
   fails in every round, the baseline strategy runs (`agent_fallback`).

The size term comes from the 0.2.1 platform run. There, solve time grew about
1.46 s per response MB (r = 0.97), and the 9–18 MB AllReduce answers cost 7–9
solve-time points each
([platform-evaluation-0.2.1.json](platform-evaluation-0.2.1.json)).

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
| `AGENT_MODEL_ROUNDS` | 2 | generate + revise rounds |
| `AGENT_LLM_BUDGET_SECONDS` | 25 | per model call |
| `AGENT_MAX_OUTPUT_TOKENS` | 900 | model `max_tokens` |

## Build, run, test

```powershell
docker build --platform linux/amd64 -t team-001-agent:0.3.0 .
python -m unittest -v test_agentcc.py
```

### Local test with a real model

Use your own development API, never the platform credentials. In PowerShell:

```powershell
docker load -i team-001-agent-0.3.0.tar        # only if the image is not loaded yet
$env:PLATFORM_LLM_URL = "https://your-dev-gateway/v1"
$env:AGENT_MODEL_NAME = "your-model"
$env:PLATFORM_API_KEY = Read-Host "dev API key" -MaskInput
docker run --rm --name agentcc-test --cpus 1 --memory 1024m --pids-limit 256 `
  -p 127.0.0.1:18080:8080 -e PLATFORM_LLM_URL -e PLATFORM_API_KEY -e AGENT_MODEL_NAME `
  -e AGENTHACK_TELEMETRY_REQUIRED=false team-001-agent:0.3.0
```

In a second terminal, run `python local_test.py`. The code the model wrote is
in `docker logs agentcc-test | Select-String agentcc.audit`. To stop, run
`docker stop agentcc-test`, then `Remove-Item Env:PLATFORM_API_KEY`.

The entry point is `AgentRuntime(AgentCCAgent()).run()` ([src/__main__.py](src/__main__.py)).

## Local results (public examples, mock model, not official Replay)

The run used the 0.3.0 container under 1 CPU / 1024 MiB / 256 pids. The mock
model answered in 2 s and wrote real strategy code for both rounds:
- All 14 cases were valid, with 2 model calls each.
- Each case took 4.5–6.3 s over HTTP and used about 1.65k tokens.
- Memory peaked at 147 MiB.
- AllReduce responses were 0.6–4.5 MB, down from 9–18 MB in 0.2.1.

On these public examples the AllReduce Replay estimates are higher than in
0.2.1, because larger responses are now penalized. An estimate from the
0.2.1 platform scoring curves puts the net change near +2.9 total points.
That figure is not verified.

A deliberately broken first-round strategy (`api.secret()`) was rejected.
Its error was sent back to the model, and the second-round code produced the
answer.

Real deepseek_v4 code quality and latency are not measured locally. Two
25-second calls could take most of the solve-time score.
