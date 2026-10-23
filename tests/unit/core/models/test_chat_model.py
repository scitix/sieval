"""Shell tests: backend selector wiring for ChatModel.

RFC #25 moved the chat-completions wire logic (streaming accumulation,
reasoning extraction, logprob parsing) into ``OpenAIChatDialect``; the
former ``_agenerate_impl`` / ``_alogprobs_impl`` coverage moved with it to
tests/unit/core/models/dialects/test_openai_chat.py, and the request-builder
validation (n/stream types, alogprobs n=1) lives on ``Model`` in
tests/unit/core/models/test_model.py. What remains here is the selector
contract: ChatModel pairs the shared client with that transport, which
supplies the model's capabilities.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import pytest

from sieval.core.models import (
    Capability,
    ChatInput,
    ChatMessage,
    ChatModel,
    TextPart,
)
from sieval.core.models.dialects.openai_chat import OpenAIChatDialect


class TestDefaultTransport:
    def test_builds_openai_chat_transport(self):
        m = ChatModel(model="m", api_key="k")
        assert isinstance(m._transport, OpenAIChatDialect)
        assert Capability.Chat in m.capabilities

    def test_transport_bound_to_shared_client_and_model(self):
        m = ChatModel(model="m", api_key="k")
        assert isinstance(m._transport, OpenAIChatDialect)
        assert m._transport._client is m._client
        assert m._transport._requested_model_id == "m"


class TestCompatibilityBranches:
    def test_coerce_input_preserves_chat_ir_and_rejects_mapping(self):
        model = ChatModel(model="m", api_key="k")
        prompt = ChatInput((ChatMessage("user", (TextPart("hi"),)),))

        assert model._coerce_input(prompt) is prompt
        with pytest.raises(TypeError, match="text, ChatInput, or messages"):
            model._coerce_input({"role": "user", "content": "hi"})
