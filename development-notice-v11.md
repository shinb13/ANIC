# 选手 Agent 开发规范

| 项目 | 约定 |
| --- | --- |
| 文档版本 | 1.1 |
| 适用 SDK / 基础镜像 | `agenthack-sdk==1.1.0` / `agenthack/base:1.1.0` |
| 读者 | 选手、框架适配开发者、提交审核人员 |
| 配套文档 | [埋点接口补充](guide.md)、[服务器部署与采集指南](server-deployment-guide.md)、[字段采集对照](characteristics-field-mapping.md) |

## 1. 职责与适用范围

选手负责业务实现、规定接口和业务语义埋点。选手不需要在自己的镜像内安装抓包工具或部署 Jaeger。

业务不存在的行为不要求虚构埋点。

**仅更换基础镜像不能完成所有采集。** 必须通过 `AgentRuntime` 运行，同时标注实际的 Turn、状态和工具执行。SDK 不会从函数名或网络流量推断这些业务含义。

### 开发、提交与 API 使用边界

以下三条是提交验收的硬性要求：

1. **必须使用平台交付的 `agenthack/base:1.1.0` 镜像作为 Dockerfile 的 `FROM`。** 不得改用 `python`、`ubuntu` 或其他自建基础镜像，也不得在镜像内替换或禁用 SDK。
2. **开发和自测使用选手自己的 API。** 选手应使用自己申请的开发环境 API 地址和 Key，避免占用平台评测配额；开发 Key 只能通过环境变量传入，不能提交到代码或镜像。
3. **提交评测必须使用平台提供的 API。** 平台会在运行时注入 `PLATFORM_LLM_URL`、`PLATFORM_API_KEY` 等配置；提交镜像不得硬编码选手 API、第三方 API 或固定的 API Key，也不得绕过平台网关直连外部模型服务。

`PLATFORM_LLM_URL` 和 `PLATFORM_API_KEY` 是运行时配置接口名称，不表示开发阶段必须使用平台凭据：本地开发时传入选手自己的值，平台评测时由平台替换为提供的值。

## 2. 必须遵守的开发要求

| 编号 | 要求 |
| --- | --- |
| DEV-01 | 基于平台交付的 `agenthack/base:1.1.0` 构建，使用平台指定 CPU 架构，不替换或禁用 SDK。 |
| DEV-02 | 继承 `BaseAgent`，实现 `async solve(self, task_input: dict) -> dict`，返回 JSON 可序列化字典且包含 `answer`。 |
| DEV-03 | 使用 `AgentRuntime(MyAgent()).run()` 启动，监听 `0.0.0.0:8080`，保留 `/health` 和 `/solve`。 |
| DEV-04 | 按真实执行路径标注 Turn、语义阶段、Tool、业务重试及恢复。 |
| DEV-05 | 从环境变量读取模型服务 URL、Key 和平台配置；不把 Key、评测数据或私有答案写入镜像。 |
| DEV-06 | 以非 root 用户运行，不请求特权、宿主机网络、Docker socket 或抓包能力。 |
| DEV-07 | 从 `/data` 只读读取平台挂载的数据，不尝试修改数据集或读取挂载范围以外的评测数据。 |
| DEV-08 | 使用可取消的异步调用，等待任务内所有子任务完成或取消，正确关闭连接和流。 |
| DEV-09 | 保留 SDK 注入的关联请求头及平台身份变量，不伪造 Trace、usage、耗时或业务状态。 |
| DEV-10 | 提交镜像版本、依赖清单、启动说明、自测请求，以及任何需要额外适配的客户端或框架信息。 |
| DEV-11 | Dockerfile 必须以平台交付的 `agenthack/base:1.1.0` 为 `FROM`；不得使用其他基础镜像或替换 SDK。 |
| DEV-12 | 开发/自测使用选手自己的 API；提交评测只使用平台注入的 `PLATFORM_LLM_URL` 和 `PLATFORM_API_KEY`，不得硬编码或携带任何 API 凭据。 |

## 3. 项目结构和 Dockerfile

```text
my-agent/
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── README.md
└── src/
    ├── __init__.py
    ├── __main__.py
    └── my_agent.py
```

Dockerfile 示例（构建时可安装依赖，最终恢复非 root 运行）。第一行镜像必须保持为平台交付的 `agenthack/base:1.1.0`：

```dockerfile
FROM agenthack/base:1.1.0
USER root
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip check
COPY --chown=agent:agent src/ ./src/
USER agent
LABEL org.agenthack.team_id="team-001"
LABEL org.agenthack.version="0.1.0"
EXPOSE 8080
CMD ["python", "-m", "src"]
```

`org.agenthack.version` 表示选手业务版本，不是 SDK 版本。默认继承基础镜像的健康检查。`.dockerignore` 应排除 `.env`、密钥文件、本地数据、日志和缓存。基础镜像已经包含 OpenAI 1.35.0、HTTPX 0.27.0 和 Requests 2.32.0；不要无约束升级这些依赖。

`src/__main__.py`：

```python
from agenthack import AgentRuntime
from src.my_agent import MyAgent

if __name__ == "__main__":
    AgentRuntime(MyAgent()).run()
```

## 4. 输入输出接口

### 4.1 健康检查

```http
GET /health
```

```json
{"status":"healthy","telemetry_configured":true}
```

`telemetry_configured` 仅表示导出器已配置，**不代表 Collector 可达或 Jaeger 已入库**。本地未配置采集地址时可为 `false`；正式评测由平台设置 `AGENTHACK_TELEMETRY_REQUIRED=true`。

### 4.2 任务请求

```json
{
  "task_id": "case-001",
  "task_type": "qa",
  "input": {"question": "请简要回答问题。"},
  "constraints": {"timeout_ms": 60000},
  "session_id": "platform-generated-unique-session-id"
}
```

`task_id` 和 `input` 必填；`task_type` 默认 `qa`；`timeout_ms` 默认 30000，必须大于零。`session_id` 可省略，由运行时生成。每个 `/solve` 生成一个 Session Span。身份相关字段由平台填写。

**1.1.0 实现边界：** `solve()` 只接收请求的 `input`，不接收外层 `constraints`。运行时执行异步超时控制，但 `constraints.max_tokens` 当前只是被解析，不会自动限制模型 Token。需要 Token 上限时，由平台与选手约定 `input.max_output_tokens` 等业务字段，并在模型请求里实际设置；最终配额由模型网关落实。

### 4.3 任务响应

```json
{
  "task_id": "case-001",
  "status": "success",
  "output": {"answer": "回答内容"},
  "metadata": {
    "trace_id": "32位十六进制Trace ID",
    "session_id": "运行时会话ID",
    "llm_calls": 1,
    "tools_used": [],
    "total_tokens": 42,
    "duration_ms": 1234.5
  }
}
```

上面是结构示意，不是实测数据。`metadata` 由运行时生成，不由业务伪造。正常完成、超时、异常分别返回 `success`、`timeout`、`error`；业务失败也可能返回 HTTP 200，因此必须检查 JSON `status`。`success` 仅表示执行完成，不代表答案正确。

## 5. 最小模型调用示例

`src/my_agent.py`：

```python
import os
import openai
from agenthack import BaseAgent, state, turn

class MyAgent(BaseAgent):
    async def solve(self, task_input: dict) -> dict:
        question = str(task_input["question"])
        max_output_tokens = int(task_input.get("max_output_tokens", 256))
        if not 1 <= max_output_tokens <= 4096:
            raise ValueError("invalid max_output_tokens")

        with turn(trigger="user"):
            with state("reasoning"):
                async with openai.AsyncOpenAI(
                    base_url=os.environ["PLATFORM_LLM_URL"],
                    api_key=os.environ["PLATFORM_API_KEY"],
                    timeout=40,
                    max_retries=1,
                ) as client:
                    response = await client.chat.completions.create(
                        model=os.environ["AGENT_MODEL_NAME"],
                        messages=[{"role": "user", "content": question}],
                        max_tokens=max_output_tokens,
                    )
            with state("final_answer"):
                return {"answer": response.choices[0].message.content or ""}
```

`AGENT_MODEL_NAME` 是本示例与部署文档约定的模型变量，不是 SDK 强制字段。选手可以采用其他模型配置方式，但必须在 README 说明。任务没有真实规划过程时，不要添加空的 `planning` 阶段来补齐字段。

### 5.1 按阶段配置 API

下面的变量名在两个阶段保持不变，但值的来源不同：

| 阶段 | `PLATFORM_LLM_URL` / `PLATFORM_API_KEY` 的值 | 规则 |
| --- | --- | --- |
| 开发、自测、联调 | 选手自己的开发 API 地址和 Key | 只用于本地或选手控制的测试环境；不得写入仓库、镜像或提交材料。 |
| 正式提交、平台评测 | 平台提供并在运行时注入的 API 地址和 Key | 只能通过环境变量读取；不得覆盖、替换、转发到其他模型服务。 |

开发时可以这样传入自己的凭据（示例值必须替换为选手自己的配置）：

```bash
export PLATFORM_LLM_URL='https://your-development-gateway.example/v1'
read -r -s -p '开发 API key: ' PLATFORM_API_KEY
export PLATFORM_API_KEY
```

提交镜像时不要执行 `ENV PLATFORM_API_KEY=...`、把 Key 写进代码，或把开发网关地址写死。平台启动容器时会注入正式 API 配置。

## 6. 业务语义埋点规范

| 行为 | 必须如何标注 | 备注 |
| --- | --- | --- |
| 一轮 Agent 决策/执行 | `with turn(trigger=...)` | 单轮直接回答也标一轮；后续轮次按实际因果填写 `parent_turn_id`。 |
| 真实业务阶段 | `with state("reasoning")` 等 | 建议使用 `planning`、`reasoning`、`tool_execution`、`reflection`、`final_answer`。 |
| 普通工具 | `@tool(...)` 或 `with tool_call(...)` | 二选一，避免同一次工具执行重复记录。 |
| 模型触发工具 | `tool_call(..., parent_llm_call_id=...)` | 使用产生指令的响应 `_agenthack_llm_call_id`，不是模型返回的工具调用 ID。 |
| MCP 工具 | `tool_type="MCP"` 并填写 MCP 元信息 | `stdio` MCP 无 TCP 包，不虚构网络连接。 |
| 自己实现的重试/恢复 | `recovery("retry")` 等 | 仅重新规划/中止而未发生重试时使用 `count=0`；不重复统计 SDK 自动重试。 |

普通工具示例，函数必须在运行时的 `solve()` 上下文内执行：

```python
from agenthack import tool

@tool(name="lookup", tool_type="database", version="1.0")
async def lookup(key: str) -> dict:
    return {"key": key, "found": False}
```

固定工具可以直接使用 `@tool`。装饰器会自动创建 Tool Span，记录参数大小、结果大小和异常；不需要手写 `call.result()`：

```python
from agenthack import BaseAgent, state, tool, turn


@tool(name="lookup", tool_type="database", version="1.0")
async def lookup(key: str) -> dict:
    # 固定工具的实际实现；不要返回密钥或敏感原文。
    return {"key": key, "found": key == "example"}


class MyAgent(BaseAgent):
    async def solve(self, task_input: dict) -> dict:
        with turn(trigger="user"):
            with state("tool_execution"):
                result = await lookup(str(task_input["lookup_key"]))
            with state("final_answer"):
                return {"answer": "found=" + str(result["found"])}
```

工具函数必须在 `AgentRuntime` 管理的 `solve()` 上下文中调用。工具由模型在运行时动态选择，或需要显式记录 `parent_llm_call_id`、MCP 元信息时，使用下面的 `with tool_call(...)`；同一次执行不要同时套用 `@tool` 和 `tool_call`，否则会重复记录。

模型触发工具时，在解析并验证真实工具指令后执行以下模式；`response` 和 `dispatch_tool` 由业务实现：

```python
from agenthack import state, tool_call

with state("tool_execution"):
    with tool_call(
        "search", "API", arguments=validated_arguments,
        tool_version="1.0",
        parent_llm_call_id=response._agenthack_llm_call_id,
    ) as call:
        result = await dispatch_tool(validated_arguments)
        call.result(result)
```

上下文写法必须调用 `call.result(result)` 才能记录结果大小；装饰器会自动记录。业务错误如果仅返回 `{"error": ...}` 而没有抛出异常，SDK 仍可能将工具执行标为成功；应在工具包装层验证结果并抛出合适异常。工具名称、类型和版本不得嵌入密钥或用户身份。

### 6.1 MCP 工具规范用例

MCP 工具必须记录服务、工具、资源和操作标识；stdio MCP 没有 TCP 流量，不要虚构网络连接：

```python
from agenthack import tool_call

with tool_call(
    "search", "MCP", arguments={"query": "example"},
    tool_version="1.0",
    parent_llm_call_id=llm_response._agenthack_llm_call_id,
    mcp_server="internal-search", mcp_tool="search",
    mcp_resource="papers", mcp_operation="tools/call",
) as call:
    result = await mcp_client.call_tool("search", {"query": "example"})
    call.result(result)
```

`response._agenthack_llm_call_id` 是 SDK 的 LLM 调用标识，不是供应商返回的 tool-call ID。MCP 工具必须在实际收到模型工具指令并完成参数校验后执行，不得为了填充字段而固定调用。

### 6.2 流式调用规范用例

流必须完整消费或在上下文管理器中关闭；只有真实 tokenizer 或缓存统计才能填写 Prefix/Append Token：

```python
from agenthack import llm_context

with llm_context(prefix_tokens=prefix_count, append_tokens=append_count,
                 provider="platform-gateway"):
    stream = await client.chat.completions.create(
        model=model, messages=messages, stream=True,
        stream_options={"include_usage": True},
    )
    async with stream:
        async for chunk in stream:
            consume_chunk(chunk)
```

示例中的 Token 数值必须来自真实统计；没有 usage 或首 Token 观测值时留空，不按文本长度估算。

### 6.3 非 OpenAI 客户端规范用例

不在自动适配范围的模型客户端使用 `llm_call()` 补充模型语义；HTTP 网络采集仍需对应的传输适配：

```python
from agenthack import llm_call

with llm_call(model="custom", provider="internal",
              endpoint="http://gateway/v1") as call:
    response = await custom_client.generate(...)
    call.set(input_tokens=response.input_tokens,
             output_tokens=response.output_tokens,
             finish_reason=response.finish_reason)
```

更多字段说明、框架回调和上下文传播见[埋点接口补充](guide.md)。

## 7. 自动采集范围与缺失字段

| 数据 | SDK 1.1.0 行为 |
| --- | --- |
| Session、Trace ID、队伍与实例标签 | 经 `AgentRuntime` 自动生成或读取平台配置。 |
| OpenAI Chat Completions | 支持同步/异步、流式/非流式；本基础镜像使用 OpenAI 1.35.0。 |
| HTTPX / Requests 网络调用 | 在任务上下文内采集 HTTP Span；尽可能读取实际 socket。 |
| Token usage | 使用模型服务返回值；无 usage 时留空，不按文本长度推算。 |
| 首 Token 延迟 | 只对流式可观察增量记录；非流式不伪造。 |
| Prefix/Append Token | 由真实 tokenizer 或业务缓存统计通过 `llm_context()` 补充。 |
| 任意其他协议、子进程、自建 SDK | 不保证自动采集，必须申报并完成适配验收。 |

Responses API、aiohttp、原生 socket、子进程 curl、自定义 transport 和非 Python 程序不属于当前自动 LLM 适配范围。使用 `llm_call()` 可以补充模型语义，但不能因此自动获得不支持的传输层 socket 信息。LangChain 等框架的内部 Turn/Tool 也需要回调适配。

流式调用必须消费完成或用上下文管理器关闭；模型服务支持时请求 `stream_options={"include_usage": True}`。缺失字段表示“未观测或不适用”，不是零。会话 Token 汇总只包含已知 usage。不要为了让 86 项字段全部有值而填入虚构数据。

## 8. 并发、异常与数据要求

1. `async solve()` 内优先使用异步 I/O；阻塞任务通过 `asyncio.to_thread` 等方式执行。异步超时无法强制终止已经运行的线程，平台还会执行容器级硬超时。
2. 用局部变量保存每题状态，避免多个 `/solve` 共用可变全局列表、历史或临时文件。任务结束前等待子任务退出，不留下脱离会话的后台调用。
3. `asyncio.to_thread` 可传播上下文；手工线程池需显式复制 `contextvars`。跨进程须传递 W3C 上下文并在子进程安装相应采集。
4. 数据目录由 `DATA_PATH` 指定，正式评测为 `/data:ro`。临时文件放 `/tmp`；需要持久产物时由平台约定独立输出目录。
5. 不关闭 SDK instrumentation、不修改采样、不同时叠加另一套 OpenAI/HTTPX 自动埋点。
6. SDK 默认不导出 Prompt、答案、工具原文或 Authorization。选手日志、自定义 Span 属性以及平台原始 pcap 不受这一保证覆盖，禁止主动记录密钥。

## 9. 配置归属

| 变量 | 提供方 | 选手处理方式 |
| --- | --- | --- |
| `PLATFORM_LLM_URL` / `PLATFORM_API_KEY` | 平台 | 构建客户端时读取；不得硬编码。 |
| `AGENT_MODEL_NAME` | 本示例由平台注入 | 用作模型请求的 `model`。 |
| `DATA_PATH` | 平台 | 作为数据根目录。 |
| `TEAM_ID` / `AGENT_ID` / `AGENT_VERSION` / `AGENT_FRAMEWORK` | 平台 | 不覆盖，不用业务输入替换。 |
| `CONTAINER_ID` | 平台 | 实际容器 ID；未提供时保留 `unknown`，不填写实例名冒充。 |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | 平台 | 由 SDK 使用，不自行更改。 |
| `AGENTHACK_TELEMETRY_REQUIRED` / `AGENTHACK_FLUSH_ON_SOLVE` | 平台 | 正式评测为 `true`。 |
| `AGENTHACK_NETWORK_INTERFACE` | 平台 | 表示平台抓包范围，不是 SDK 自动探测结果。 |

## 10. 构建、自测与提交

在自己的项目目录使用 Bash 执行。以下 URL 和 Key **必须使用选手自己的开发 API**，不得使用平台正式评测凭据；只绑定本机端口。选手自测不强制部署采集服务。

```bash
docker build --platform linux/amd64 -t team-001-agent:0.1.0 .
docker run --rm --entrypoint python team-001-agent:0.1.0 \
  -c 'import agenthack; assert agenthack.__version__ == "1.1.0"'

mkdir -p test-data
export PLATFORM_LLM_URL='https://your-development-gateway.example/v1'
export AGENT_MODEL_NAME='your-model'
read -r -s -p '选手自己的开发 API key: ' PLATFORM_API_KEY
export PLATFORM_API_KEY

docker run -d --name my-agent-test \
  -p 127.0.0.1:18080:8080 \
  -v "$PWD/test-data:/data:ro" \
  -e PLATFORM_LLM_URL -e PLATFORM_API_KEY -e AGENT_MODEL_NAME \
  -e DATA_PATH=/data -e TEAM_ID=local-test \
  -e AGENTHACK_TELEMETRY_REQUIRED=false \
  team-001-agent:0.1.0

curl -fsS http://127.0.0.1:18080/health
curl --max-time 75 -fsS http://127.0.0.1:18080/solve \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"local-001","input":{"question":"请回复测试成功"},"constraints":{"timeout_ms":60000}}'

docker stop --time 15 my-agent-test
docker rm my-agent-test
unset PLATFORM_API_KEY
```

正式提交前确认：

- [ ] 镜像架构正确，SDK 为 1.1.0，非 root 启动且 `/health` 可用。
- [ ] Dockerfile 使用 `agenthack/base:1.1.0`；镜像中没有开发 API、平台 API 或其他凭据。
- [ ] `/solve` 成功、失败、超时响应均符合接口；不会因并发串题。
- [ ] 所有真实 Turn、Tool 和业务恢复路径均已埋点，没有重复计数。
- [ ] 流正确关闭；不支持自动采集的调用已声明并完成适配。
- [ ] 镜像不包含 Key、评测私有数据或测试答案。
- [ ] README 说明模型变量、所需数据、输出目录、依赖服务与支持的任务输入。
- [ ] 随镜像提供版本/摘要、依赖清单、至少一份成功请求与响应。
- [ ] 已确认开发/自测使用选手自己的 API，正式提交由平台注入 API；代码不会覆盖平台配置。

平台另行验证 Jaeger 入库、tshark 抓包、身份对应关系和时间对齐；这些不要求选手自行部署完成。
