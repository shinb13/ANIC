# 赛题三输入拓扑说明

版本：2026-11-02（V22 固定 14 例连接、带宽和共享资源说明）  
适用协议：`agentcc.joint-test.v1` / `agentcc-official.v4`

## V22 固定题库变化与字段兼容

本版仍为七类拓扑 × AllGather/AllReduce 的固定 14 例，每例 256 rank。
`task`、`topology` 和 `solution` 的字段名、类型、含义以及判题规则沿用 V21；
不增加选手必须解析的新字段。`split_policy.candidates` 和
`alignment_bytes` 的语义不变，具体数值仍须逐例读取。

**变化发生在字段的值和集合内容中：**`topology.edges` 中的连线会新增，
部分旧连线会消失；各边的 `bw` 重新设置；`topology.resources` 中共享容量、
资源成员边和资源数量也会变化。`src`/`dst` 定义实际可走的有向边，
`bidirectional=true` 才额外声明反方向。`resources[].edges` 只引用已声明的
有向边，不创建连接；一条边可同时受到多项资源容量约束。

七类拓扑的名称只表示结构族，不保证固定的边列表、带宽表或资源 ID。
选手应以本次 `/solve` 请求内的完整 `input.task` 和 `input.topology` 构图、
计算负载并生成 solution。公开 ZIP 中的样例是格式示范，不是正式固定 14 例。

## 1. 输入对象

平台对每个实例调用一次选手 Agent 的 `POST /solve`。`solve()` 接收到的业务输入包含当前实例的任务对象和完整拓扑对象：

```json
{
  "challenge": "agentcc",
  "instance_id": "v22_256_uneven_pods_ag",
  "schema_version": "agentcc.joint-test.v1",
  "task": { "...": "当前实例的任务参数" },
  "topology": { "...": "当前实例的完整拓扑" }
}
```

选手应逐实例读取 `task` 和 `topology`。拓扑名称只用于识别实例，不能替代 `edges` 中的连接关系和带宽。当前每次评测运行 7 类 256-rank 拓扑各一题 AllGather 和 AllReduce，共 14 例。

## 2. `task` 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `task_id` | string | 当前任务 ID。 |
| `instance_id` | string | 当前实例 ID；返回结果中的实例 ID 应与此值一致。 |
| `operator` | string | `all-gather` 或 `all-reduce`。 |
| `topology_kind` | string | 基础拓扑和带宽形式的组合标识。 |
| `base_topology` | string | 七类拓扑族之一，以当前任务字段为准。 |
| `bandwidth_profile` | string | 当前为 `heterogeneous-v22`；具体边带宽只从拓扑输入读取。 |
| `topology_file` | string | 官方实例文件名。正式调用已经把对应内容放入同级 `topology` 对象，选手无需自行读取该路径。 |
| `ranks` | string[] | 参加本次集合通信的 Rank，顺序与拓扑中的 `ranks` 一致。 |
| `npu` | string[] | `ranks` 的兼容字段；当前实例中两者内容相同。 |
| `data_size.per_rank_bytes` | integer | 每个 Rank 初始持有的数据字节数；当前 AllGather 为 `1048576`、AllReduce 为 `2097152`。 |
| `split_policy.mode` | string | 当前为 `search_allowed`，允许在候选集合中选择切分数。 |
| `split_policy.candidates` | integer[] | 依实例为 `[1, 2, 4, 8, 16]` 或 `[1, 2, 4, 8, 16, 32]`。 |
| `split_policy.alignment_bytes` | integer | 每个原始分片的字节数必须是此值的整数倍；当前 v22 为 `4096` 字节。选定的切分数还必须能整除 `data_size.per_rank_bytes`。 |
| `split` | integer | 该实例给出的建议切分数。`search_allowed` 模式下，方案可以选择候选集合中的其他值，但方案中的 `chunks` 和所有传输必须与实际选择一致。 |
| `chunk_size` | integer | 按任务建议切分数计算的分片字节数，等于 `per_rank_bytes / split`。若方案改用其他候选切分数，应按实际切分数构造分片。 |
| `max_nonempty_rounds` | integer | 非空通信轮次上限，当前为 `64`。空轮会在 Replay 前删除。 |
| `allow_copy` | boolean | 是否允许复制已持有分片。当前为 `true`。 |
| `allow_detour` | boolean | 是否允许经过额外交换节点绕行。当前为 `true`。Rank 不能作为同一条路径的中间节点；Rank 中继需要拆成不同轮次的传输。 |
| `bidirectional` | boolean | 当前实例的链路为双向链路提示；实际方向仍以每条 `edge` 为准。 |
| `communication_constraints` | object | 分片接收、持有关系、重复发送和交换机转发规则。 |
| `difficulty` | string | 当前实例均标为 `hard`；v22 实际采用 14 例等权评分。 |
| `description` | string | 便于阅读的实例说明，不参与连接关系判定。 |

切分自检：设每 Rank 数据量为 `D = data_size.per_rank_bytes`，实际选定切分数为 `S`，对齐值为 `A = split_policy.alignment_bytes`。必须同时满足 `D % S == 0` 和 `(D / S) % A == 0`；等价于 `D % (S × A) == 0`。例如 AllGather 的 `D=1048576` 字节选 `S=32` 时，每片为 `32768` 字节（32 KiB），可被 `4096` 整除；AllReduce 的 `D=2097152` 字节选 `S=32` 时，每片为 `65536` 字节（64 KiB），也满足对齐。`task.chunk_size` 是按任务建议切分数计算的，改选 `S` 后应自行重算每片字节数。判题器在检查完整 `chunks` 集合时验证此约束。

当前实例中的约束字段为：

```json
{
  "receive_unique_chunk_once": true,
  "send_only_possessed_chunks": true,
  "send_unique_chunk_pair_once": true,
  "switches_forward_same_round_without_storage": true
}
```

含义如下：

- 同一 Rank 不能重复接收已经持有的同一分片。
- Rank 只能发送本轮开始时已经持有的分片或规约结果。
- 同一 `(chunk, src, dst)` 组合不能重复发送。
- 交换机只在当前轮沿路径转发，不保存分片到下一轮。

AllGather 的每个 Rank 初始持有自己的 `split` 个分片，目标是让每个 Rank 获得全部 Rank 的全部分片。AllReduce 需要先合并相同槽位中来自全部 Rank 的贡献，再把完整规约结果传播到全部 Rank。

## 3. `topology` 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `description` | string | 基础拓扑名称。 |
| `operator` | string[] | 该拓扑可支持的算子集合。当前七类拓扑均支持 AllGather 和 AllReduce。 |
| `ranks` | string[] | 全部计算节点 ID。 |
| `switches` | string[] | 全部交换节点 ID。Rank 与交换节点集合不能重叠。 |
| `data_size.per_rank_bytes` | integer 或带单位字符串 | 拓扑默认数据量。正式任务以 `task.data_size` 为准。当前 `1MB` 按 1 MiB 解析。 |
| `communication_startup_time_us` | number | 一次 Rank 逻辑发送的启动开销，单位为微秒。当前为 `1`。同轮并行发送不按全网发送数顺序累加。 |
| `switch_processing_overhead_us` | number | 路径每经过一个交换节点增加的处理开销，单位为微秒。当前为 `0.3`。 |
| `split_policy` | object | 拓扑默认切分候选。正式任务以 `task.split_policy` 为准。 |
| `allow_copy` | boolean | 拓扑默认复制规则。正式任务可覆盖该值。 |
| `allow_detour` | boolean | 拓扑默认绕行规则。正式任务可覆盖该值。 |
| `communication_constraints` | object | 拓扑默认通信约束。正式任务可覆盖该值。 |
| `edges` | object[] | 连接清单。未出现在清单中的节点对视为不相连。 |
| `resources` | object[] | 可选。每个对象定义一个由多条有向边共同争用的带宽池；完整字段、方向和 Replay 计费见第 6 节。 |

当前 V22 的 7 类拓扑、双算子共 14 个正式实例都提供非空 `resources`。
输入协议仍将该字段定义为可选；Agent 应逐例读取当前请求，不能从拓扑名
推断字段是否存在，也不能将本版的资源配置固化到后续版本。

## 4. `edges` 和带宽

每条边采用以下格式：

```json
{
  "src": "r0",
  "dst": "L1-N0-S0",
  "bw": "50GB/s",
  "bidirectional": true
}
```

`src` 和 `dst` 必须属于 `ranks` 或 `switches`。`bidirectional: true` 会生成两个可用方向；两个方向各自拥有 `bw` 给出的容量。`bw` 是单方向容量，`GB/s` 按十进制字节率换算，`1 GB/s = 10^9 B/s`。不要把双向容量相加后当作单方向带宽。

方案中的每次 Rank 到 Rank 传输必须给出一条由 `edges` 组成的连续路径。路径起点和终点是不同 Rank，中间节点只能是 `switches`。同轮中经过同一有向边的传输会累计该方向上的负载；Replay 根据累计负载、路径传输时间、通信启动开销和交换机处理开销计算该轮通信耗时。

## 5. 七类 256-rank 拓扑

| 基础拓扑 | 计算节点 | 交换节点 | 逻辑边 | 连接结构 |
| --- | ---: | ---: | ---: | --- |
| `uneven_pods` | 256 | 36 | 576 | 不等规模 Pod 和核心容量。 |
| `directed_ring` | 256 | 34 | 345 | 方向不对称的环、弦和中心路径。 |
| `sparse_fabric` | 256 | 40 | 591 | 稀疏、不规则 leaf-spine。 |
| `cloudmatrix` | 256 | 336 | 5376 | CloudMatrix 风格的合成规模泛化。 |
| `fullmesh_clos` | 256 | 36 | 3072 | FullMesh/Clos 风格的合成规模泛化。 |
| `mesh2d` | 256 | 8 | 3840 | 四个 64-rank 网格域。 |
| `nvlink_like` | 256 | 89 | 4496 | NVLink 风格的合成规模泛化。 |

表中的“边”是一条 JSON 逻辑边；其中 `bidirectional: true` 由 Replay 按两个方向处理。`directed_ring` 中还包含方向性单边。交换机数是图节点数，不代表实物设备台数；`nvlink_like` 不声称是实际 NVL72 的 256-rank 硬件规格。精确节点 ID 和连接关系应从当前实例输入读取。

## 6. 带宽与共享资源

### 6.1 直观含义

把 `resources` 理解为**几条链路共同争用的带宽池**，例如同一个上行口或
交换平面。`edges[].bw` 是一条有向边自身的带宽；
`resources[].capacity` 是绑定到该资源的**所有有向边合计**可使用的带宽。
它不是新的图节点或连接，不会自动生成边，也不会改变哪些路径可达。

在同一轮，若两个传输分别走两条边，而两条边绑定同一个资源，Replay 会把
两次经过这些边的字节数相加，再除以资源容量计算拥塞耗时。因此即使每条
边本身很快，也不能认为两次传输可以毫无争用地并行。容量只影响通信耗时，
**不是每轮最多可传多少字节的硬限制**。

### 6.2 输入字段

`topology.resources` 缺失或为 `[]` 时，没有额外共享带宽约束，仍按
`edges`、路径和其他规则判定。当前 V22 的 14 个正式实例均提供非空数组，
Agent 仍应按当前输入解析，不要按拓扑名称写死。

| 字段 | 类型 | 必需性与含义 |
| --- | --- | --- |
| `topology.resources` | object[] | 可选；每项定义一个独立的共享带宽资源。 |
| `resources[].id` | string | 必需；本拓扑内唯一的资源标识。名称没有内置方向或物理含义。 |
| `resources[].capacity` | number 或带单位字符串 | 通常使用；严格大于 0 的总带宽，单位见下文。 |
| `resources[].bw` | number 或带单位字符串 | `capacity` 的兼容字段；两者同时出现时以 `capacity` 为准。 |
| `resources[].edges` | object[] | 可选的有向边引用列表；也可在基础边上用 `resource_id` 建立关联。 |
| `resources[].edges[].src` / `.dst` | string | 引用一条**已经存在**的有向边，不创建新边。 |
| `resources[].edges[].bidirectional` | boolean | 可选，默认为单方向；为 `true` 时也绑定已存在的反向边。 |
| `topology.edges[].resource_id` | string | 可选，引用 `resources[].id`；基础边若为双向，该直接引用绑定两个方向。 |

带宽数字按 B/s（字节/秒）解析；`GB/s` 是十进制 `10^9 B/s`，
`Gbps`、`Gbit/s` 是比特/秒，须除以 8 换算成 B/s。
例如 `1GB/s = 10^9 B/s`，`8Gbps = 10^9 B/s`。资源名如 `ingress`
或 `egress` 只是标识，**方向仅由引用的有向边决定**。

下面的 `r0`、`r1` 是 Rank，`S0` 是已声明的交换节点；这是说明字段的
非正式示例，不是 V22 正式实例：

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

### 6.3 方向、重复边与多资源

- 基础边的 `bidirectional: true` 创建两个有向方向，两个方向各有
  `bw`。但在 `resources[].edges[]` 中写 `{"src":"r0","dst":"S0"}`
  **只绑定 `r0→S0`**，不会因基础边是双向就自动绑定 `S0→r0`。
  要绑定反方向，另列反向项，或在该资源引用上设
  `"bidirectional": true`；反向基础边必须存在。
- 若直接在基础边上写 `"resource_id": "ingress"`，该边是双向时
  两个方向都会绑定 `ingress`。这与上一条资源列表中的单方向引用不同。
- 同一有向边可同时绑定多个资源：经过它的全部字节数会**分别**计入
  每个资源，不是在资源之间分摊。同一资源重复引用同一方向只建立
  一次关联；但一条路径若经过该资源绑定的两条不同边，负载会计两次。
- 若 `edges` 多次声明相同的有向节点对，Replay 将这些声明的 `bw`
  相加为该方向的带宽，关联的资源 ID 合并；不能把它们当作可分别
  绕行的独立路径。
- 资源引用不存在的有向边、`resource_id` 指向未定义资源、重复的
  `resources[].id`，均是无效的拓扑输入。选手输出的路径也必须只经过
  实际存在的有向边。

### 6.4 Replay 怎样计算耗时

对每个非空通信轮次，Replay 为每条有向边 `e` 累计经过它的传输字节数
`B_e`，为每个共享资源 `r` 累计其绑定边上的传输字节数 `B_r`。
一条传输穿过同一资源绑定的两条边时，其字节数计入 `B_r` 两次。
设边带宽为 `C_e`、资源容量为 `C_r`，忙碌时间分别是
`10^6 × B_e/C_e` 与 `10^6 × B_r/C_r` 微秒。

单次传输路径耗时 = 启动开销 + 中间交换节点处理开销之和 +
该传输沿途每条有向边的传输时间之和。**本轮耗时**取以下两项的较大值：

1. 本轮所有单次传输路径耗时的最大值；
2. 本轮最大启动与交换开销 + 全部有向边及共享资源忙碌时间的最大值。

总通信耗时是各非空轮次耗时之和；不同轮次的负载不合并。资源容量是
计时带宽，并非每轮字节硬上限；传得多只会增加耗时，不会仅因此使
方案无效。

以上例的两 Rank AllGather 为例：同一轮 `r0→S0→r1` 与
`r1→S0→r0` 各传 32768 B，`ingress` 绑定的两条边合计
65536 B，`egress` 亦然；在 1 GB/s 下各需 65.536 μs。
取样例中的 1 μs 启动和 0.3 μs 交换开销，本轮 Replay 为
**66.836 μs**。只分别看两条单边的 `bw` 会低估此轮耗时。
完整的双算子输入和输出在 ZIP 的 `examples/shared-resource/`；
独立公式说明见 `SHARED_RESOURCES.md`。

新版“拓扑与输出规范 ZIP”还提供了覆盖七类拓扑、双算子和共享资源的
**非正式样例**。样例的拓扑结构用于练习解析；带宽和资源容量特意与
冻结正式实例不同，不能据此推断正式题目的参数或评分锚点。

## 7. 实例池和计分

当前实例池为 7 类拓扑 × 2 种算子 = 14 例，每次评测运行全部 14 例。
14 例等权，单例最高最终贡献 `100/14` 分；不再采用普通/复杂 40/60
的抽样权重。正确性、通信和求解时间分别占单例 15/50/35 分，详细公式
以 v22 赛题说明为准。

当前任务建议切分数为 `16` 或 `32`，但在 `search_allowed` 模式下可以从
每例 `split_policy.candidates` 选择其他值，所有分片和传输须与实际选择一致。

## 8. 读取和自检要点

- 使用当前请求中的完整 `topology`，不要缓存上一实例的边或带宽。
- 检查 `task.ranks` 与 `topology.ranks` 是否一致。
- 从 `split_policy.candidates` 选择切分数，并保证每 Rank 数据量能整除该值、每片字节数是 `split_policy.alignment_bytes` 的整数倍。
- 构造完整的 `rank × split` 初始分片集合，不使用省略号或未声明分片。
- 路径只使用当前 `edges`；Rank 中继拆成不同轮次。
- 同轮发送只能使用轮次开始前已持有的数据。
- 删除无传输的空轮后，非空轮次不得超过 64。
- 返回的 `instance_id` 和 `operator` 应与当前任务一致。

ZIP 的 `legacy/` 目录保留旧版 128/144-rank 图示资料，仅供历史参考；
旧切分候选和旧题量不适用于 v22。正式判题只读取平台随当前实例下发的
完整任务与拓扑。
