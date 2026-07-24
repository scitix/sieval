# Code Evaluator

## 项目简介

多语言代码执行与测试服务。支持对模型生成代码进行直接运行或依据测试用例评估。

## 支持数据集

- HumanEval（多语言：python / javascript / typescript）
- LiveCodeBench（仅 python）
- SciCode（仅 python）
说明：HumanEval / SciCode 直接运行提交代码；LiveCodeBench 需函数名与输入输出用例比对。SciCode 的程序自带内联测试用例，运行无异常即判通过。

## 环境准备

### Python 环境

仅需 HumanEval：

```sh
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

需要同时支持 LiveCodeBench：

```sh
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements/livecodebench.txt
```

需要支持 SciCode（科学计算栈，要求 Python ≥ 3.11）：

```sh
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements/scicode.txt
```

### Node.js (JS / TS)

需要 Node.js (推荐版本 >= 20)：

```sh
npm install -g ts-node
```

### Docker

```sh
docker build -f docker/Dockerfile.python -t code-evaluator-py .
docker build -f docker/Dockerfile.javascript -t code-evaluator-js .
docker build -f docker/Dockerfile.typescript -t code-evaluator-ts .
docker build -f docker/Dockerfile.scicode -t code-evaluator-scicode .   # Python 3.11
```

## 启动服务

```sh
fastapi run app/server.py --port 11451
```

## API

### 健康检查

GET /health → `{"status": true, "msg": "healthy"}`

### 评估接口

POST /evaluations
字段：

- uuid
- source: "human-eval" | "mbpp" | "livecodebench" | "scicode"
- lang: python | javascript | typescript
- code: 代码字符串
- test: LiveCodeBench 专用测试描述（含 fn_name / inputs / outputs）
- timeout: float (可选，单位秒，默认值见下文)
- memory_limit: int (可选，单位 MB，默认 1024)

示例 HumanEval (Python) - 自定义超时与内存：

```json
{ 
  "uuid":"h1",
  "source":"human-eval",
  "lang":"python",
  "code":"print(1+2)",
  "timeout": 5.0,
  "memory_limit": 512
}
```

示例 LiveCodeBench：

```json
{
  "uuid":"lc1",
  "source":"livecodebench",
  "lang":"python",
  "code":"def add(a,b): return a+b",
  "test":{"fn_name":"add","inputs":["[1,2]","[3,4]"],"outputs":["3","7"]}
}
```

返回：

```json
{ "status": true, "msg": "" }
```

失败时 msg 包含错误原因。

## 调用示例

```sh
curl -X POST http://localhost:11451/evaluations \
  -H 'Content-Type: application/json' \
  -d '{"uuid":"demo","source":"human-eval","lang":"python","code":"print(42)","memory_limit":1024}'
```

## 资源限制与默认值

### 超时 (Timeout)

若请求中未指定 `timeout`，将使用以下默认值：

- python/js: 3s
- typescript: 5s
- livecodebench: 6s + 2s * 用例数

### 内存 (Memory Limit)

若请求中未指定 memory_limit，默认值为 1024 MB。

Python: 使用 `resource.setrlimit` 限制进程地址空间。
Node.js (JS/TS): 使用 `--max-old-space-size` 限制 V8 堆内存。

## 目录

- app/: 服务与执行逻辑
- docker/: 各语言镜像文件
- requirements/: 附加依赖（`livecodebench.txt` / `scicode.txt`）
- README.md: 文档

## 备注

未加强隔离，勿运行不可信高风险代码。
