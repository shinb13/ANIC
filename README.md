# AgentCC scheduling agent (0.2.8)

0.2.8 adds a resource-priced two-tree AllReduce candidate to 0.2.7's
topology-derived logical rank orders, 0.2.6's overlapping model/search pipeline
and 0.2.5's integer-route repair.
The user-reported 0.2.3 evaluation scored 92.672 on average,
with all 14 cases feasible under the 25/50/25 rules. Every request is planned
from its own `task` and `topology`. The
agent loads no fixed instance data and never selects by instance name.

## Loop

1. **Analyse.** The agent parses and validates the input (invalid inputs are
   rejected with a reason). It profiles every XOR dimension at path width 1,
   then predicts one AllGather seed or up to three AllReduce seeds at width 1
   using their integer path allocations and the published Replay formula.
   Illegal seeds are excluded by the same task constraints as other candidates.
   For a legal `rs_ag` seed it also generates two binary rank groupings and
   reports their approximate path-work gains, explicitly separate from Replay.
2. **Model call.** One short call (120 output tokens, 6 s cap). The model
   receives the bottlenecks and the agent's own
   initial time/size trade-offs, and proposes up to 3 configurations. The
   prompt explicitly identifies these as initial seeds, not a complete search.
   It may add picks the agent did not list. While the model is thinking, one
   worker ranks the complete original portfolio and builds up to 4 distinct
   schedules, then prepares one promising mapping prediction and the eligible
   two-tree predictions. Model proposals
   are evaluated after that worker finishes. Optional model field `m` keeps
   the original order (0), chooses locality (1), or resource pricing (2);
   omitting `m` lets the agent choose by the path-work proxy. Optional `t`
   disables the two-tree trial (0), chooses equal slots (1), or gives the first
   tree one quarter (2) or three quarters (3). Omitting it selects the lowest
   predicted two-tree cost.
3. **Tool `evaluate_schedule`.** Each distinct candidate is built on the
   current topology and Replay-estimated. Equivalent candidates reuse an
   existing build. Each evaluation is an SDK `tool_call`, linked to the
   model call for model picks, including duplicate proposals.
4. **Select.** Candidates are ranked by `ln(Replay us) + 0.10 × response MB`.
   The first one that passes full validation is selected. The audit records
   whether the model's pick won outright.
5. **Remap.** If the selected algorithm is `rs_ag`, the preferred mapping
   passes a path-work screen and predicts at least 3% less communication, at
   most one additional schedule is built and fully validated. It replaces
   the original only with lower communication time and non-increasing cost.
6. **Refine.** The selected schedule receives up to 100 ms of route-repair
   search on at most three slow rounds with alternative routes. Whole chunks
   move only between routes within the selected width for the same sender and
   receiver. A repaired schedule is fully re-estimated and validated, then
   adopted only if communication time falls and the selection cost does not rise.
7. **Try two trees.** Integer predictions for three slot partitions account for
   the combined per-round edge and shared-resource loads of both trees. If the
   selected partition predicts at least 3% less time than the refined original,
   one complete schedule is built and validated. Adoption requires lower time
   and non-increasing cost. This candidate has its own routing plans and does
   not enter the original schedule's route-repair representation.

## Changes in 0.2.8: shared-resource-priced two-tree AllReduce

- Two trees are generated from the current topology, without instance-name
  dispatch. Each level merges at most four rank groups. Candidate parents are
  priced using independently reachable directed reduction and broadcast paths;
  intermediate switches only forward. The second tree reprices each level's
  first-tree edge and overlapping shared-resource loads to seek a different
  bottleneck distribution. This is a heuristic, not a disjointness guarantee.
- The candidate uses the largest legal split, width 4 and three whole-slot
  partitions: 1:1, 1:3 and 3:1. The original split/alignment rules and round limit
  determine eligibility. Each slot is reduced and broadcast by exactly one tree.
  Contributor state is read at the start of each round and updates are applied
  for the next round. Complete-copy notation remains configurable.
- Both trees' transfers share each round's route optimizer. Its equal-demand
  fractions are rounded to each tree's integer slot count. Unequal partitions
  do not receive a separate weighted fractional optimization; every partition
  is nevertheless scored using its actual integer counts and combined loads.
- Tree generation and predictions run in the portfolio worker during the one
  model call. They reuse the request's rank-cost and routing data. This can still
  extend the critical path; a model disabling the trial can discard prepared
  work. At most one new complete two-tree schedule is built.
- The model sees the current topology profiles, constraints and legal tree
  partitions, explicitly marked as not yet measured. Optional `t` materially
  controls the evaluated partition. `prepare_dual_tree` records both tree edge
  sets, predictions and timings; `evaluate_dual_tree` links the model call to the
  selected partition, complete cost and adoption. Full validation failures
  propagate; unroutable construction is recorded as a rejected candidate.
- The approach borrows tree packing from
  [ForestColl](https://www.usenix.org/system/files/nsdi26-zhao-liangyu.pdf) and
  congestion pricing from [DualColl](https://arxiv.org/html/2610.15704v1).
  It does not implement either paper's full algorithm or claim its optimality
  guarantees under the competition's directed, overlapping-resource,
  integer-slot and synchronous-round rules.

Validation passed 54 tests and 76 evaluated container HTTP requests with a
controlled 1 s mock-model delay. Two public repeats retained every original
built candidate and mapping result. Fullmesh Clos AR changed from 207.583 to
129.588 us (37.57%), with 8 rounds instead of 12; other public two-tree trials
were screened out. Identical-input model controls received identical model
requests: `t=0` retained the original, while `t=1/2/3` adopted different valid
slot partitions. This verifies decision mechanics, not real-model quality.

Mean public HTTP time increased from 2355.7 to 2393.3 ms (+37.6 ms), meeting
the experimental 100 ms mean-overhead target in this run. P95 increased from
3370.5 to 3804.1 ms (+433.6 ms), so the mean does not describe every case.
The unchanged wall-clock route repair also varied on Mesh2D AR and Cloudmatrix
AG; those changes are separate from two-tree gains. No official Replay or
real-model evaluation was run. See
[the 0.2.8 validation report](optimization-validation-0.2.8.json).

## Changes in 0.2.7: topology-derived rank pairing

- Logical index `i` refers to physical rank `P[i]`. XOR communication, shard
  ownership and contributor expressions share that logical order. Route
  endpoints, input possession and output chunk ordering retain the original
  task identities. Task/topology data and original plan/allocation caches are
  unchanged; mapped planners share the physical network and route cache.
- Locality grouping uses reciprocal edge bandwidths; resource grouping adds
  reciprocal capacities for every shared-resource membership. Shortest paths
  allow only switch intermediates. The worse direction prices each pair;
  greedy binary merges consider forward and reversed group alignment.
- Weighted mean path costs screen the proposed layout at 0.25% proxy gain.
  This excludes aggregate contention and startup cost and is not a Replay
  prediction. Integer allocations and the complete Replay estimator decide
  whether the candidate merits a build and adoption. The original schedule
  remains available when a mapping loses.
- One mapping prediction is prepared during the model wait. Its planner and
  allocation caches are reused for the final trial when the model choice and
  parameters agree. A model override may require another prediction; choosing
  the original can discard preparation. Only one extra full schedule is built.
- `prepare_rank_mapping` and `search_rank_mapping` tool evidence records
  current configurations, proxy scores, the permutation, predictions and
  adoption. Model-directed search/evaluation links to the model call. Audit
  timings separate mapping generation, preparation and final search.
- The communication-structure approach is inspired by
  [TACCL](https://www.usenix.org/conference/nsdi23/presentation/shah);
  this implementation uses two generated permutations rather than its full
  algorithm synthesizer. No topology-name dispatch or stored instance answer
  is added.

Validation passed 48 tests and 68 evaluated container HTTP requests with a
controlled 1 s mock-model delay. On two public repeats, Uneven Pods AR fell
from 586.200 to 416.387 us (28.97%) and Sparse Fabric AR from 294.492 to
275.089 us (6.59%). Mean per-request relative communication reduction across
all public cases was 2.53%. Original built candidates matched in all 34 pairs;
separate identical-input controls verified that model `m=0` keeps the original
while `m=1` adopts a different, faster current-instance schedule.

Mean HTTP time increased from 1853.9 to 2006.3 ms, adding 152.4 ms; this
slightly exceeds the proposed 150 ms mean-overhead target. P95 rose from
2439.5 to 2683.3 ms. Mapping did not improve Mesh2D AR, NVLink-like AR or
Fullmesh Clos AR in this run. Two non-mapped results regressed by at most
0.232% because the unchanged wall-clock repair searched a different number
of moves. This is a submission candidate; real-model choices and official
total scores remain unmeasured. See
[the 0.2.7 validation report](optimization-validation-0.2.7.json).

## Changes in 0.2.6: earlier model request

- Full candidate ranking moves out of the pre-model critical path. Initial
  seeds use the width-1 exchange profiles already computed for this input;
  AllReduce also measures a width-1 tree when allowed. The complete portfolio,
  build budget, model-call budget and final selection cost retain 0.2.5 behaviour.
- The prompt includes bytes per rank and alignment as well as legal splits,
  the round limit and topology-derived bottlenecks. Link statistics expose
  one-way arcs, unequal reverse bandwidths and the current bandwidth range,
  including changes outside the initial single-path routes. It no longer presents the
  top six full-search configurations. Real-model selection quality therefore
  needs a platform comparison; fixed mock replies cannot establish that.
- Planning caches have one computing owner at a time. Model requests consume
  a serialized prompt; model-proposed candidate evaluation waits for portfolio
  ranking/building. No additional model call or compute worker is introduced.
- Audit timings separate parsing, dimension profiles, initial predictions,
  portfolio ranking/building, model waiting, portfolio join waiting, model
  candidates, selection/validation and refinement/validation. `llm_started_ms`
  measures invocation from solve start, not the network's receive time.

Validation passed 41 tests and 64 evaluated HTTP requests with a controlled
1 s mock-model delay. Two public repeats reversed version order per case.
Mean HTTP time fell from 2162.6 to 1897.9 ms (12.24%); P95 fell from 3084.0
to 2797.7 ms (9.28%). The mock received model requests after 689.1 ms on
average in 0.2.5 and 357.0 ms in 0.2.6, measured from client HTTP start.
All 32 pairs had identical built candidates and pre-refinement selections.
Final communication estimates matched in 26 of 28 public pairs; one Mesh2D
AG and one AR result increased by 0.340% and 0.248% respectively, with fewer
repair trials inside the unchanged wall-clock search budget. Every repair
still improved on its own pre-repair schedule. These are local measurements;
real-model proposal quality and official total scores remain unmeasured.
See [the 0.2.6 validation report](optimization-validation-0.2.6.json).

## Changes in 0.2.5: integer-route congestion repair

- Each pass examines at most 16 bottleneck-related sender/receiver pairs and
  chooses the best strictly improving one-chunk move among those evaluated.
  Directed edges, overlapping resources, repeated resource membership and
  startup/processing overhead when activating or retiring a route all count.
- Counts and loads are private to the repair. Other candidates, integer
  prediction caches and the original answer remain unchanged. Modified
  transfers are merged by pair and path, with empty transfers removed.
- The SDK `refine_schedule_routes` call records the selected configuration,
  original sources, model call link, trial/accepted moves, before/after costs
  and timing. Implementation and validation errors propagate.
- The 100 ms deadline bounds search, not the complete refinement call;
  rebuilding changed transfers, full estimation and validation add time.

In one paired container run with a controlled 1 s mock-model delay, all 36
evaluated requests were valid. Eleven of 14 public cases improved and three
were unchanged. The mean per-case relative communication-time reduction was
0.786%; mean HTTP time increased from 2024.8 to 2184.4 ms. Mesh2D AR improved
2.139%, NVLink-like AR 1.534%, and Fullmesh Clos AR stayed unchanged. These
are local estimates, not official Replay scores. A separate 20 vs 100 ms
budget comparison retained less improvement at 20 ms without a clear total
time saving, so the default stays at 100 ms. See
[the 0.2.5 validation report](optimization-validation-0.2.5.json).

The complete suite has 38 passing tests, including prediction agreement on
318 public configurations, repair isolation, resource accounting, slower
reverse paths, long-path startup, AllReduce semantics, zero search budget,
response-size rejection and error propagation.

## Changes in 0.2.4: integer prediction and build reuse

- The predictor uses exactly the integer route counts later used by the
  emitter. It accounts for used-path startup/processing overhead, directed
  edge loads, every overlapping resource and repeated resource membership
  along a path. Response size remains approximate; the size weight is 0.10.
- Route counts, integer round costs and allocation identities are cached
  within a request. The portfolio ranking computed before the model call
  is reused by the build worker.
- Candidates merge only when their algorithm, split, reduce-scatter depth,
  dimension order and actual per-round path/count allocations match. The
  requested configuration, matched configuration and model call link remain
  in the tool evidence.
- The continuous optimizer, chunk rounding, communication algorithms, model
  timeout and output-token budget retain their 0.2.3 behavior.

Validation covers 30 tests and 318 configurations across the 14 public
examples. Every configuration produced the same complete schedule as 0.2.3
and passed validation; integer prediction matched full Replay estimation
within floating-point tolerance. See [the validation report](optimization-validation-0.2.4.json)
for controlled HTTP comparisons and their timing limits. These are local
tests with a mock model, not official Replay results.

## Changes in 0.2.3: `c{slot}_sum` in copies

In the AllReduce all-gather phase, each copy of a completely reduced slot
is written as `c{slot}_sum`. The output format defines this notation for
complete slots. Before 0.2.3, every copy listed all 256 contributors. With
the short form, AllReduce responses fell from 9–18 MB to 1.0–1.8 MB. Replay
estimates did not change, and the best-communication `rs_ag` configurations
are selected again.

The user-reported 0.2.3 platform evaluation accepted all 14 cases. Set
`AGENTCC_SUM_NOTATION=0` to restore full contributor
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

Each solve logs one `agentcc.audit` JSON line with the current input hash,
model reply and usage metadata, candidate configurations, integer predictions,
full Replay estimates, equivalent-build matches, selected sources and timing.
The `route_refinement` record adds accepted moves, adoption, response-size
changes and separate search/total timing.
The `timings_ms` record separates the compute and wait stages; overlapping
stages should not be summed. The SDK captures the model request;
model-proposed `evaluate_schedule` tool
calls reference its model call ID. Duplicate proposals still add their source
to the reused candidate. API credentials are not logged.

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
docker build --platform linux/amd64 -t team-001-agent:0.2.8 .
python -m unittest -v test_agentcc.py
```

### Local test with a real model

Use your own development API, never the platform credentials. In PowerShell:

```powershell
docker load -i team-001-agent-0.2.8.tar        # only if the image is not loaded yet
$env:PLATFORM_LLM_URL = "https://your-dev-gateway/v1"
$env:AGENT_MODEL_NAME = "your-model"
$env:PLATFORM_API_KEY = Read-Host "dev API key" -MaskInput
docker run --rm --name agentcc-test --cpus 1 --memory 1024m --pids-limit 256 `
  -p 127.0.0.1:18080:8080 -e PLATFORM_LLM_URL -e PLATFORM_API_KEY -e AGENT_MODEL_NAME `
  -e AGENTHACK_TELEMETRY_REQUIRED=false team-001-agent:0.2.8
```

In a second terminal, run `python local_test.py`. The model's proposals and decisions are
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
