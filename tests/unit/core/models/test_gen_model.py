"""Shell tests: backend selector wiring for GenModel.

RFC #25 moved the completions wire logic (streaming accumulation, echo split,
``_completion_top_logprobs`` sanitizing) into ``OpenAICompletionsDialect``;
the former ``_agenerate_impl`` / ``_alogprobs_impl`` coverage moved with it to
tests/unit/core/models/dialects/test_openai_completions.py, and the
request-builder validation (n/stream types, alogprobs n=1) lives on ``Model``
in tests/unit/core/models/test_model.py. What remains here is the selector
contract: GenModel pairs the shared client with that transport, which supplies
the model's capabilities.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from sieval.core.models import Capability, CompletionInput, GenModel
from sieval.core.models.dialects.openai_completions import OpenAICompletionsDialect


class TestDefaultTransport:
    def test_builds_openai_completions_transport(self):
        m = GenModel(model="m", api_key="k")
        assert isinstance(m._transport, OpenAICompletionsDialect)
        assert Capability.InputScoring in m.capabilities

    def test_transport_bound_to_shared_client_and_model(self):
        m = GenModel(model="m", api_key="k")
        assert isinstance(m._transport, OpenAICompletionsDialect)
        assert m._transport._client is m._client
        assert m._transport._model == "m"


class TestCompatibilityBranches:
    def test_coerce_input_preserves_completion_ir(self):
        model = GenModel(model="m", api_key="k")
        prompt = CompletionInput("prompt")

        assert model._coerce_input(prompt) is prompt
