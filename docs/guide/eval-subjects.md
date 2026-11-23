# Evaluating an external system

A task that measures a service it calls itself can provide an `EvalSubject`
instead of a placeholder `Model`. `subject_meta()` becomes the result's model
identity; `call_key()` and `request_args()` join the inference identity, so a
change to either refuses resume at `inference`; `serving()` records serving
facts when known. Existing tasks keep their `Model` as the subject, and their
identities are unchanged.

## Declare a model-less task

Set `requires_model = False` and implement `subject`. The constructor still
takes `model`; the CLI passes `None`.

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

- **Identity.** `request_args()` may return any JSON value; put everything
  that affects inference there. `call_key()` is a stable, non-empty string;
  `subject_call_key()` derives one from `subject_meta()`. Keep secrets out of
  all of these and out of `call.record()`.
- **Resolution.** The runner validates the subject after `setup()`, so setup
  may resolve a service version; keep it stable for the run. Call
  `task_resume_plan()` after setup too.
- **Recording.** `subject_call()` opens one logical call. Each attempt made by
  `RecordingAsyncClient` inside it goes to `requests.jsonl`, with SiEval's
  retry loop. `call.record()` snapshots the request parameters and usage into
  `model_calls`, the request digest and token profiling. Calls outside a task
  stage are not recorded.
- **Protocol.** The task owns the provider protocol, authentication and
  response validation; no dialect or capability contract is reconciled.
- **Helper `Model`.** A task may also bind a `Model` that it calls beside an
  independent subject. The run records no weights or serving for it, but its
  name, weights, request arguments, context window, numerics and parsers join
  the inference identity, so swapping it refuses resume. A context overflow
  recorded by such a task is inferred again rather than replayed.

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

Validation and `--dry-run` report the task as model-less. A task with the
default `requires_model = True` must still bind a `Model`.
