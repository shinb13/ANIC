# Platform evaluation: submission 0.1.5

The user supplied this platform report, completed at **2026-11-05 13:08**, and
identified it as the last submitted version. The displayed report contains no
image digest. The local submission archive and its checksum are recorded in
[the structured report](platform-evaluation-0.1.5.json). This is separate from
[local mock-model validation](optimization-validation-0.1.5.json).

All 14 instances were feasible. The platform reported 14 Replay calls, 128
communication rounds, 14 model calls, 46,514 tokens, verified result archives,
and accepted execution evidence.

| Metric | Previous report, 10:53 | Latest report, 13:08 | Change |
| --- | ---: | ---: | ---: |
| Correctness / 15 | 15.000 | 15.000 | +0.000 |
| Communication / 50 | 33.2975 | 38.840 | +5.542 |
| Solve time / 35 | 33.285 | 34.484 | +1.199 |
| Total / 100 | 81.583 | 88.324 | +6.741 |

Means above are calculated from the displayed per-instance values; their
rounding can differ slightly from adding the separately rounded component
scores. The latest platform summary displays 38.8 for communication and 34.5
for solve time. Token use increased from 9,736 to 46,514 (4.78 times).

| Instance | Rounds | Replay, microseconds | Replay reduction vs 10:53 | Communication / 50 | Solve, seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| uneven_pods AG | 8 | 43364.061 | 90.01% | 44.620 | 3.206 |
| uneven_pods AR | 16 | 3173.629 | 56.88% | 41.180 | 2.976 |
| directed_ring AG | 8 | 38564.951 | 93.50% | 45.648 | 3.342 |
| directed_ring AR | 8 | 9571.634 | 0.00% | 36.166 | 3.149 |
| sparse_fabric AG | 8 | 241076.032 | 0.00% | 40.890 | 2.418 |
| sparse_fabric AR | 8 | 6653.219 | 0.00% | 38.340 | 2.440 |
| cloudmatrix AG | 8 | 30992.001 | 79.62% | 42.709 | 3.623 |
| cloudmatrix AR | 8 | 3094.397 | 0.00% | 35.444 | 3.570 |
| fullmesh_clos AG | 8 | 178686.559 | 0.00% | 35.401 | 2.721 |
| fullmesh_clos AR | 16 | 1098.141 | 66.78% | 41.871 | 2.541 |
| mesh2d AG | 8 | 37851.116 | 76.52% | 43.110 | 2.972 |
| mesh2d AR | 8 | 7561.407 | 0.00% | 29.566 | 2.971 |
| nvlink_like AG | 8 | 10114.774 | 93.53% | 43.967 | 3.444 |
| nvlink_like AR | 8 | 2697.161 | 0.00% | 24.843 | 2.827 |

Seven Replay times improved and seven exactly match the previous displayed
values. Only Uneven Pods AR and Fullmesh Clos AR used sixteen rounds. Given
the current candidate algorithms, their round counts are consistent with tree
AllReduce, but the summary does not expose the selected policy or route changes.

The unchanged values do not establish whether the model selected the baseline,
the model request timed out or failed validation, or a constructed candidate was
rejected. The report exposes no raw decisions, fallback reasons, input hashes,
or candidate estimates. These causes must remain unresolved. Both reports use
the same displayed instance identifiers, but their complete hidden inputs are
unavailable for an independent equality check.

Before adding more algorithms, investigate real-model candidate selection and
timeout/adoption behavior in a controlled run. Keep the existing current-input
model evidence and correctness checks. Instance names and the measurements in
this report must not be used to select precomputed schedules.
