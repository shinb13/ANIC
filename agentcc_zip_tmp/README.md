# 赛题三拓扑与输出规范 V22

文档修订：2026-11-02（共享资源解释澄清，样例与判题规则不变）。

本包对应 `agentcc.joint-test.v1` 和 7 类拓扑、双算子的 14 例 V22 试测输入格式。
正式评测每次通过 `/solve` 请求中的 `input.task` 和 `input.topology` 下发完整参数。
**本包不是正式题库**：`examples/256-rank/` 的带宽和共享容量与冻结实例不同，
不包含评分锚点、隐藏答案或可用于替代当前请求的固定参数。

V22 正式 14 例会调整边连接、链路带宽和共享资源容量/成员；公开样例保留独立的示意参数，不是正式题库。输入字段及输出规范与 V21 兼容。

## 文件

- `TOPOLOGY_INPUT_GUIDE.md`：V22 任务、拓扑、切分与路径规则。
- `SHARED_RESOURCES.md`：共享资源的方向、引用、负载累计与 Replay 公式。
- `OUTPUT_FORMAT.md`：Agent 响应及 AllGather/AllReduce solution 字段。
- `examples/shared-resource/`：两个 Rank 的完整 task、topology 与双算子有效输出。
- `examples/256-rank/<family>/<ag|ar>/`：七类拓扑各自的非正式 256-rank 输入样例。
- `legacy/topology-output-spec-20261006.zip`：旧版图示与可编辑幻灯片，仅供历史参考。

## 七类拓扑

| family | 结构提示 | 可选切分上限 |
| --- | --- | ---: |
| `uneven_pods` | 不等规模 Pod、leaf 与 core | 32 |
| `directed_ring` | 方向不对称的环、弦与 hub | 32 |
| `sparse_fabric` | 不规则 leaf-spine | 32 |
| `cloudmatrix` | 多层交换平面 | 32 |
| `fullmesh_clos` | FullMesh/Clos | 16 |
| `mesh2d` | 二维网格域 | 16 |
| `nvlink_like` | NVLink 风格的多级结构 | 16 |

样例采用 256 个 Rank，用于验证解析器能处理新增族、方向性边、异构带宽和
`resources`。正式实例的节点、边和容量应以本次 `/solve` 输入为准；不能
缓存样例或按族名推断未声明的连线。`task.split` 是建议值，
`split_policy.mode=search_allowed` 时可从 `candidates` 中另选值，
但 `chunks`、初始持有和每次传输都必须与实际选择一致。

当前 V22 的 14 个正式实例均提供非空 `topology.resources`；协议层面该字段
仍可选。Agent 应兼容缺失或空列表的输入，以收到的拓扑为准。
