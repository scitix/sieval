"""OpenRouter POST /api/alpha/decisions adapter.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

from collections.abc import Mapping

from sieval.core.types import JSONValue

from ..client import (
    DecisionRequest,
    DecisionResponse,
    HTTPDecisionsClient,
    response_record,
)
from ..content import ImageEvidence, JsonPart
from ._wire import (
    mapped_question,
    named_decision_result,
    normalize_usage,
)


class OpenRouterDecisionsClient(HTTPDecisionsClient):
    provider = "openrouter"
    default_base_url = "https://openrouter.ai/api"
    default_api_key_env = "OPENROUTER_API_KEY"
    allowed_options = frozenset({"provider", "session_id", "user"})

    def endpoint(self) -> str:
        base = self._base_url.removesuffix("/v1")
        return base + "/alpha/decisions"

    def build_request(self, request: DecisionRequest) -> dict[str, JSONValue]:
        parts: list[JSONValue] = []
        for part in request.context.parts:
            if isinstance(part, ImageEvidence):
                url = part.url or f"data:{part.media_type};base64,{part.data}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif isinstance(part, JsonPart):
                parts.append(part.value)
            else:
                parts.append(part.text)
        return {
            **self._request_options(),
            "model": self.model_name,
            "state": parts[0]
            if len(parts) == 1 and isinstance(parts[0], str)
            else parts,
            "questions": {
                q.question_id: mapped_question(q) for q in request.decisions.questions
            },
        }

    def parse_response(
        self, data: Mapping[str, object], request: DecisionRequest
    ) -> DecisionResponse:
        result = named_decision_result(
            data.get("answers"), request.decisions.questions, allow_refusal=True
        )
        return response_record(
            data,
            result,
            data.get("model"),
            normalize_usage(data.get("usage")),
        )
