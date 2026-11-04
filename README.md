# AgentCC baseline

This repository contains an AgentCC scheduling baseline with a model-guided
split decision and deterministic schedule construction.

The agent calls the configured OpenAI-compatible model once per request to
choose a legal split candidate. It then reads the request's `task` and
`topology`, including directed edge bandwidth and shared resources, builds
resource-aware switch-only-intermediate routes, and uses recursive doubling
for AllGather and AllReduce. It does not use fixed public topology data.

The local estimator follows the published Replay timing terms to compare split
candidates. The official platform Replay remains the scoring authority.

The public examples are local fixtures only. Official evaluation data arrives
inside each `/solve` request.

## Build and run

The Dockerfile uses the downloaded platform base image `agenthack/base:1.1.1`.
At runtime the platform must provide `PLATFORM_LLM_URL`, `PLATFORM_API_KEY`,
and `AGENT_MODEL_NAME`; these values are never stored in the repository or
image.

```powershell
docker build --platform linux/amd64 -t team-001-agent:0.1.0 .
docker run --rm -p 18080:8080 team-001-agent:0.1.0
```

The runtime exposes `GET /health` and `POST /solve` on port 8080. The request
body follows the AgentRuntime contract: `task_id` plus an `input` object with
the current AgentCC `task` and `topology`.

## Local verification

```powershell
python -m unittest -v test_agentcc.py
```

## Container entry point

The image starts `AgentRuntime(AgentCCAgent()).run()` and exposes the SDK's
`/health` and `/solve` endpoints.
