# AgentCC 输出格式

Agent 对每个 `/solve` 请求返回当前任务的 JSON 响应。完整小例子见
`examples/shared-resource/all-gather-solution.json` 和
`all-reduce-solution.json`；它们只针对同目录的两 Rank 演示任务。

```json
{
  "task_id": "public-shared-ag",
  "status": "success",
  "output": {
    "answer": {
      "operator": "all-gather",
      "chunks": ["r0_c0", "r1_c0"],
      "initial_possession": {"r0": ["r0_c0"], "r1": ["r1_c0"]},
      "final_chunks": ["r0_c0", "r1_c0"],
      "round_count": 1,
      "rounds": [{
        "round": 1,
        "transfers": [
          {"action": "copy", "src": "r0", "dst": "r1", "chunk_ranges": ["r0_c0"], "path": ["r0", "S0", "r1"]},
          {"action": "copy", "src": "r1", "dst": "r0", "chunk_ranges": ["r1_c0"], "path": ["r1", "S0", "r0"]}
        ]
      }]
    }
  }
}
```

- `task_id` 必须对应当前任务；若提供 `instance_id`，也须对应当前实例。
  `status` 必须为 `success`。`output.answer` 内还可再包一层 `solution` 对象，
  但不应依赖多余包装层。
- `chunks` 必须完整列出所有 `rank × split` 的原始分片，命名为
  `r{rank}_c{slot}`；不能省略。`initial_possession` 中每个 Rank
  仅持有自己的全部槽位。选定 split 从当前 `task.split_policy.candidates`
  获取，并满足每 Rank 字节数整除及 `alignment_bytes`。若输出包含
  `solution.split`，其值必须与 `chunks` 的槽位数一致。
- `round_count` 与 `rounds[].round` 一致；每轮 `transfers` 是数组。
  空轮会被删除，非空轮不得超过平台给定的安全上限。
- 每个 transfer 的 `src` 和 `dst` 是不同 Rank，`path` 以它们为端点，
  中间只能是已声明交换节点；每一步须是当前拓扑中的有向边。
  可用 `route`/`route_id` 表示路径，但显式 `path` 数组最不含糊。
- 同轮只能发送轮开始前持有的分片。AllGather 只能 `copy`；
  `final_chunks` 为全部原始分片。AllReduce 通过同槽位 `reduce`
  合并互不重复的贡献，完成后才能 `copy`；`final_chunks` 须覆盖
  所有完整规约槽位，可写为 `c{slot}_sum`。表达式如
  `r0_c0+r1_c0` 必须与发送 Rank 实际已持有的贡献者集合一致。
- 不得重复接收已有分片或重复发送相同 `(chunk,src,dst)`。
  方案自报的 `metrics.total_time` 不取代官方 Replay；返回前最好
  用当前任务和拓扑重算，不要从拓扑名称推断带宽或共享资源。
