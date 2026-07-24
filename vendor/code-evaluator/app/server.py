import atexit
import os
import sys
from typing import Any, Generic, TypeVar

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from .exec_js import execute_code as exec_js
from .exec_py_code import execute_code as exec_py_code
from .exec_py_test import execute_test as exec_py_test
from .exec_ts import execute_code as exec_ts

# logger
logger.configure(
    handlers=[
        {
            "sink": sys.stdout,
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "enqueue": True,
        }
    ],
)


# collect non-server loggers
@atexit.register
def exit_handler():
    logger.remove()


app = FastAPI()


# Generic type for response data
T = TypeVar("T")


class BasicResponse(BaseModel, Generic[T]):
    status: bool
    msg: str
    data: T | None = None


class ResourceMetrics(BaseModel):
    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_memory_mb: float
    peak_memory_mb: float


@app.get("/health")
async def check_health() -> BasicResponse[None]:
    return BasicResponse(status=True, msg="healthy")


class LiveCodeBenchTest(BaseModel):
    inputs: list[str]
    outputs: list[str]
    fn_name: str | None = None


class Sample(BaseModel):
    uuid: str
    source: str
    code: str
    test: LiveCodeBenchTest | None = None
    lang: str = "python"
    timeout: float | None = None
    memory_limit: int = 1024  # MB
    kwargs: dict[str, Any] | None = None


@app.post("/evaluations")
async def evaluate(sample: Sample) -> BasicResponse[ResourceMetrics]:
    if sample.source in {"human-eval", "mbpp", "scicode"}:
        # 'human-eval' / 'mbpp' / 'scicode': run the submitted code directly.
        # scicode sends a self-contained program (inlined targets + test cases),
        # so a clean run == pass, same as human-eval.
        logger.debug(f"code to exec:\n{sample.code}")

        CODE_EXECUTOR_MAP = {
            "javascript": (exec_js, 3.0),
            "python": (exec_py_code, 3.0),
            "typescript": (exec_ts, 5.0),
        }
        if sample.lang in CODE_EXECUTOR_MAP:
            fn, default_timeout = CODE_EXECUTOR_MAP[sample.lang]
            timeout = sample.timeout if sample.timeout is not None else default_timeout
            ok, msg, stats = await fn(
                code=sample.code, timeout=timeout, memory_limit=sample.memory_limit
            )
        else:
            ok, msg = False, f"not supported language: {sample.lang}"
            stats = None

        logger.info(
            f"evaluate sample '{sample.uuid}' from '{sample.source}', "
            f"language: {sample.lang}, timeout: {timeout}, memory_limit: {sample.memory_limit}, "
            f"kwargs: {sample.kwargs}, status: {ok}, msg: {msg}, "
            f"avg_cpu: {stats.cpu_percent if stats else 0:.2f}%, "
            f"peak_cpu: {stats.peak_cpu_percent if stats else 0:.2f}%, "
            f"avg_memory: {stats.memory_mb if stats else 0:.2f}MB, "
            f"peak_memory: {stats.peak_memory_mb if stats else 0:.2f}MB"
        )
        return BasicResponse(
            status=ok,
            msg=msg,
            data=(
                ResourceMetrics(
                    avg_cpu_percent=stats.cpu_percent,
                    peak_cpu_percent=stats.peak_cpu_percent,
                    avg_memory_mb=stats.memory_mb,
                    peak_memory_mb=stats.peak_memory_mb,
                )
                if stats
                else None
            ),
        )
    elif sample.source == "livecodebench":
        # 'livecodebench' use tests to eval the code
        logger.debug(f"code to exec:\n{sample.code}")

        if sample.lang != "python":
            return BasicResponse(
                status=False, msg=f"not supported language: {sample.lang}", data=None
            )

        if sample.test is None:
            timeout = sample.timeout if sample.timeout is not None else 3.0
            ok, msg, stats = await exec_py_code(
                code=sample.code, timeout=timeout, memory_limit=sample.memory_limit
            )
        else:
            default_timeout = 6.0 + len(sample.test.inputs) * 2.0
            timeout = sample.timeout if sample.timeout is not None else default_timeout
            ok, msg, stats = await exec_py_test(
                code=sample.code,
                inputs=sample.test.inputs,
                expect_outputs=sample.test.outputs,
                fn_name=sample.test.fn_name,
                timeout=timeout,
                memory_limit=sample.memory_limit,
            )

        logger.info(
            f"evaluate sample '{sample.uuid}' from '{sample.source}', "
            f"language: {sample.lang}, timeout: {timeout}, memory_limit: {sample.memory_limit}, "
            f"kwargs: {sample.kwargs}, status: {ok}, msg: {msg}, "
            f"avg_cpu: {stats.cpu_percent:.2f}%, "
            f"peak_cpu: {stats.peak_cpu_percent:.2f}%, "
            f"avg_memory: {stats.memory_mb:.2f}MB, "
            f"peak_memory: {stats.peak_memory_mb:.2f}MB"
        )
        return BasicResponse(
            status=ok,
            msg=msg,
            data=ResourceMetrics(
                avg_cpu_percent=stats.cpu_percent,
                peak_cpu_percent=stats.peak_cpu_percent,
                avg_memory_mb=stats.memory_mb,
                peak_memory_mb=stats.peak_memory_mb,
            ),
        )
    else:
        logger.error(f"not supported data source: {sample.source}")
        return BasicResponse(
            status=False, msg=f"not supported data source: {sample.source}", data=None
        )
