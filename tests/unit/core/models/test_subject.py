"""Contracts for evaluation subject identity.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

from types import MappingProxyType
from typing import Any

import pytest

from sieval.core.models.subject import (
    EvalSubject,
    ServingFingerprint,
    SubjectMeta,
    subject_call_key,
    validate_eval_subject,
)
from sieval.core.tasks.run_meta import digest, request_key


class _Subject:
    def __init__(
        self,
        *,
        metadata: Any = None,
        key: Any = None,
        request_args: Any = None,
        serving: Any = None,
    ):
        self.metadata = (
            {"model": "remote", "revision": "v1"} if metadata is None else metadata
        )
        self.key = key
        self.args = {"protocol": "v1"} if request_args is None else request_args
        self.serving_fingerprint = serving

    def subject_meta(self) -> SubjectMeta:
        return self.metadata

    def call_key(self) -> str:
        return subject_call_key(self.metadata) if self.key is None else self.key

    def request_args(self) -> Any:
        return self.args

    def serving(self) -> ServingFingerprint | None:
        return self.serving_fingerprint


def test_subject_call_key_is_canonical_and_excludes_endpoint_provenance() -> None:
    first = {
        "model": "remote",
        "revision": "v1",
        "api_base": "https://one.example",
        "provenance": {"run": "first"},
    }
    second = {
        "provenance": {"run": "second"},
        "api_base": "https://two.example",
        "revision": "v1",
        "model": "remote",
    }

    assert subject_call_key(first) == subject_call_key(second)


def test_validate_eval_subject_accepts_json_request_args_and_optional_serving() -> None:
    subject = _Subject(
        request_args=MappingProxyType({"route": "endpoint-v1", "version": 3}),
        serving=None,
    )

    assert validate_eval_subject(subject) is subject
    assert isinstance(subject, EvalSubject)


def test_request_key_normalizes_mapping_request_arguments() -> None:
    arguments = MappingProxyType(
        {
            "params": MappingProxyType({"temperature": 0.2, "stream": False}),
            "extra": {"private_extension": True},
            "dialect": "remote-v1",
        }
    )
    subject = _Subject(request_args=arguments)

    key = request_key(model=subject, task_args={})

    assert key == digest(
        {
            "model_args": {
                "params": {"temperature": 0.2},
                "dialect": "remote-v1",
            },
            "task_args": {},
        }
    )
    assert "extra" in arguments


def test_validate_eval_subject_requires_protocol() -> None:
    with pytest.raises(TypeError, match="must implement EvalSubject"):
        validate_eval_subject(object())


@pytest.mark.parametrize("metadata", [{}, {"model": ""}, {"model": 3}])
def test_validate_eval_subject_requires_nonempty_model_name(metadata: dict) -> None:
    with pytest.raises(ValueError, match="non-empty 'model'"):
        validate_eval_subject(_Subject(metadata=metadata))


def test_validate_eval_subject_accepts_a_plugin_defined_call_key() -> None:
    subject = _Subject(key="remote:v1")

    assert validate_eval_subject(subject) is subject


def test_validate_eval_subject_rejects_nonstring_call_key() -> None:
    with pytest.raises(TypeError, match=r"call_key\(\) must return"):
        validate_eval_subject(_Subject(key=42))


def test_validate_eval_subject_requires_json_request_args() -> None:
    with pytest.raises(TypeError, match="JSON-compatible value"):
        validate_eval_subject(_Subject(request_args=object()))


def test_validate_eval_subject_requires_serving_mapping() -> None:
    with pytest.raises(TypeError, match=r"serving\(\) must return a mapping"):
        validate_eval_subject(_Subject(serving="engine"))
