# 共享带宽资源规范

修订：2026-11-02（V22 资源成员与容量逐例变化；输入协议语义不改）。

本说明与 V22 公开判题器的 `Topology.from_data` 和 Replay 规则一致。
**一句话理解：`resources` 是多条有向边共同争用的带宽池。**

V22 固定 14 例中，每例的资源集合、`capacity` 和 `edges` 成员均以当前拓扑输入为准；新增的入向/出向共享资源与已有资源可以重叠。旧版同名资源或拓扑族名称不保证容量与成员不变。资源引用不增加新边，若某条旧边未出现在当前 `topology.edges`，任何旧路径都不能继续使用。
`edges[].bw` 限制单条有向边；`resources[].capacity` 限制绑定到该资源的
所有有向边合计可用的带宽。资源不是新增连接或交换节点，不决定可达性；
它使同轮经过不同边的传输也可能相互争用。

`topology.resources` 是可选数组，缺失或空数组表示没有额外共享带宽
约束。当前 V22 的 14 个正式实例均有非空资源数组，但 Agent 应以当前
`/solve` 输入为准，不能按拓扑名称推断资源。

| 输入字段 | 含义 |
| --- | --- |
| `resources[].id` | 必需的唯一字符串 ID；名称本身不规定方向。 |
| `resources[].capacity` | 严格大于 0 的共享总带宽。 |
| `resources[].bw` | 容量兼容字段；与 `capacity` 同时出现时以后者为准。 |
| `resources[].edges[]` | 可选的现有有向边引用列表；每项写 `src`、`dst`，可选 `bidirectional`。 |
| `edges[].resource_id` | 可选的资源 ID 引用；直接把基础边绑定到该资源。 |

数字带宽按 B/s 解析；`GB/s` 为十进制 `10^9 B/s`，
`Gbps`/`Gbit/s` 为比特率并除以 8。例：`8Gbps = 1GB/s`。

## 声明与方向

```json
{
  "edges": [
    {"src": "r0", "dst": "S0", "bw": "1GB/s", "bidirectional": true},
    {"src": "r1", "dst": "S0", "bw": "2GB/s", "bidirectional": true}
  ],
  "resources": [
    {
      "id": "ingress",
      "capacity": "1GB/s",
      "edges": [{"src": "r0", "dst": "S0"}, {"src": "r1", "dst": "S0"}]
    },
    {
      "id": "egress",
      "capacity": "1GB/s",
      "edges": [{"src": "S0", "dst": "r0"}, {"src": "S0", "dst": "r1"}]
    }
  ]
}
```

- 基础 `edges[].bidirectional: true` 创建两个有向边，各自有 `bw` 带宽。
- `resources[].edges[]` 默认只绑定所写的 `{src,dst}`。想绑定反方向，
  必须另写反向项，或在该资源引用上设 `bidirectional: true`，且反向基础边必须存在。
- 基础边上的 `resource_id` 也可引用资源；基础边若为双向，这种直接引用
  同时绑定两个方向。资源 ID 必须在 `resources` 中定义，重复 ID 或引用未知边无效。
- 同一有向边可以属于多个资源。路径上的每条边将传输字节数分别计入
  每个绑定资源；不会把负载在资源之间分摊。一条路径若经过同一资源的
  两条边，该资源计两次字节量。
- 若 `edges` 多次声明同一有向节点对，Replay 会把带宽相加为该方向的
  可用带宽；各声明的 `resource_id` 关联到该合并方向，不能把重复声明
  理解为完全隔离的独立路径。
- 同一资源重复引用同一条有向边只建立一次绑定。`resources[].id`
  重复、引用不存在的有向边、或 `resource_id` 指向未定义资源，
  都是无效拓扑输入。
- `capacity`（兼容字段 `bw`）可为正数字 B/s，或带单位的带宽字符串。
  `GB/s` 为十进制 `10^9 B/s`，`Gbps`/`Gbit/s` 为比特率并除以 8。
  资源名称如 `ingress`、`egress` 只是标识，方向只由边引用决定。

## Replay 计费

一次传输在 AllGather 中的负载为 `per_rank_bytes / split × chunk_ranges 数`；
AllReduce 每个唯一规约槽位计一个分片大小。对每一轮，分别累计所有传输
经过的有向边负载 `B_e`，以及每个资源所绑定的边负载 `B_r`。

设有向边带宽为 `C_e`，资源容量为 `C_r`，则各自忙碌时间为
`10^6 × B_e / C_e` 和 `10^6 × B_r / C_r` 微秒。每条传输的路径时间是
`startup_us + 中间交换节点数 × switch_overhead_us +
Σ(10^6 × 该传输字节数 / 路径有向边带宽)`。非空轮次的 Replay 耗时为：

```text
max(
  各传输路径时间的最大值,
  本轮最大启动及交换开销 + max(各有向边忙碌时间, 各共享资源忙碌时间)
)
```

总通信耗时为各非空轮次之和。各轮的资源负载不跨轮累计。带宽容量影响
**耗时**，不是每轮字节硬上限；只因负载高于容量，不会把方案判为非法。
非法情况包括路径经过未声明有向边、资源引用未知边或未定义 ID。

`examples/shared-resource/` 的 AllGather 一轮双向传输各为 32768 B。
两条 ingress 边合计 65536 B，两条 egress 边也合计 65536 B；
每个共享资源的忙碌时间为 `65.536 μs`。加上 `1 μs` 启动与
`0.3 μs` 交换开销后，该轮 Replay 为 `66.836 μs`，高于单条
传输路径的 `50.452 μs`。这说明两条边即使分别有足够带宽，
它们共享的 1 GB/s 仍会成为本轮瓶颈。完整输入字段逐项解释见
`TOPOLOGY_INPUT_GUIDE.md` 第 6 节。
