# Evaluating an external system

Most SiEval tasks evaluate a `Model` that SiEval calls through its dialects.
Some tasks measure a service or API that the task itself calls. Those tasks can
provide an `EvalSubject` instead of configuring a placeholder `Model`.

`EvalSubject` is the stable identity boundary between a task and the system it
measures. Its `subject_meta()` is written as the result's model identity,
`request_args()` contributes to inference identity, and `serving()` records
serving facts when they are known. Changing the subject identity or request
arguments refuses resume at inference. The default subject for existing tasks
is still their `Model`, so their recorded identity remains unchanged.

## Declare a model-less task

Set `requires_model = False` explicitly and implement `subject`. The task
constructor still accepts `model` because the CLI uses the same constructor
contract; it will pass `None` for a model-less task.

```python
import httpx

from sieval.core.models import (
    EvalSubject,
    Model,
    RecordingAsyncClient,
    ServingFingerprint,
    SubjectMeta,
    subject_call,
    subject_call_key,
)
from sieval.core.tasks import Task
from sieval.core.types import JSONValue


class RemoteService:
    def subject_meta(self) -> SubjectMeta:
        # Include stable identity only. Never put credentials here.
        return {"model": "answer-service", "revision": "2026-11"}

    def call_key(self) -> str:
        return subject_call_key(self.subject_meta())

    def request_args(self) -> JSONValue:
        return {"protocol": "answer-api-v1", "region": "us-east"}

    def serving(self) -> ServingFingerprint | None:
        return None  # Return a fingerprint when deployment facts are known.


class RemoteAnswerTask(Task):
    requires_model = False

    def __init__(self, dataset, model: Model | None, name: str | None = None):
        super().__init__(dataset=dataset, model=model, name=name)
        self._subject = RemoteService()

    @property
    def subject(self) -> EvalSubject:
        return self._subject

    async def infer(self, preprocessed, ctx):
        payload = {"question": preprocessed}
        with subject_call(self.subject) as call:
            async with RecordingAsyncClient(
                base_url="https://answers.example",
                max_retries=2,
            ) as client:
                response = await client.post("/v1/answer", json=payload)
            response.raise_for_status()
            result = response.json()

            # Report request details and usage from the response when present.
            # Every HTTP attempt made by RecordingAsyncClient is also recorded.
            call.record(
                usage=result.get("usage"),
                request_params={"body": payload},
                response_model=result.get("model"),
            )
        return result["answer"]
```

The subject may return any JSON-compatible value from `request_args()`. Keep
values that affect inference there so resume detects a change. `call_key()` is a
stable, non-empty string defined by the plugin and recorded on its calls.
`subject_call_key()` is an optional helper that derives it from `subject_meta()`.
The runner resolves and validates the subject after `setup()`, so a task may
resolve the service version there. Keep that subject stable while the run executes.
When setup cannot reach a checked service that has not started, a fresh dry-run
can count pending samples without resolving its subject. When setup succeeds,
the fresh plan validates the subject. A resume plan still requires a resolved
subject before comparing the recorded inference identity.

A task may bind a helper `Model` and provide an independent subject. Weights,
context windows, numerics and parsers from that binding describe the helper;
the subject supplies the measured system's identity and serving facts. Include
any helper details that affect inference in the subject's `request_args()` or
the task's `inference_args()`.
Both `TaskRunner.plan_run()` and `task_resume_plan()` apply this rule. Call
`task_resume_plan()` after setup has resolved the subject.

`subject_call()` opens one recorded logical call. Requests made through
`RecordingAsyncClient` inside it become attempts in `requests.jsonl`; its retry
loop is shared with SiEval's model clients. Call `call.record()` with the
request parameters and usage to populate `model_calls`, the request digest, and
token profiling. Calls outside an active task stage are not recorded.
`call.record()` takes a recursive JSON snapshot of the request parameters, so
later changes to a reused request body do not change the recorded call.

The runner holds task and stage concurrency limits while `infer()` runs. A
model-less task is responsible for its own provider protocol, validation,
capabilities, authentication, and any client configuration. SiEval does not
reconcile a dialect or model capability contract for it. Keep secrets out of
`subject_meta()`, `request_args()`, and `call.record()`.

## Configuration

Omit the `models:` section and the task's `model:` field:

```yaml
datasets:
  questions:
    class: my_plugin.QuestionDataset
tasks:
  remote:
    class: my_plugin.RemoteAnswerTask
    dataset: questions
```

Static validation and `--dry-run` identify the task as model-less and explain
that its `EvalSubject` owns the protocol contract. A task with the default
`requires_model = True` must still bind a `Model`; a missing model is never
inferred from a task's `infer()` implementation.
