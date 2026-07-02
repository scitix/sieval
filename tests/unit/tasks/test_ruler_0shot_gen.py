"""Test RULER unified implementation supporting all model scenarios."""

from unittest.mock import Mock

from sieval.datasets.ruler._shared import thinking_prefill, tokens_to_generate
from sieval.tasks.ruler_0shot_gen import _ChatGenBase


class TestTokensToGenerate:
    """Test token budget calculation for all model scenarios."""

    def test_qwen3_with_thinking(self):
        """Qwen3 + thinking: overhead + budget + base."""
        # 4 (overhead) + 5000 (budget) + 128 (base)
        result = tokens_to_generate(
            "niah",
            enable_thinking=True,
            think_budget=5000,
            model_name="Qwen3-8b",
        )
        assert result == 5132

    def test_qwen3_without_thinking(self):
        """Qwen3 without thinking: overhead + 1 (minimum) + base."""
        # 4 (overhead) + 1 (minimum) + 128 (base)
        result = tokens_to_generate(
            "niah",
            enable_thinking=False,
            think_budget=0,
            model_name="Qwen3-8b",
        )
        assert result == 133

    def test_other_model_with_thinking(self):
        """Non-Qwen3 with thinking: budget + base (no overhead)."""
        # 3000 (budget) + 128 (base)
        result = tokens_to_generate(
            "niah",
            enable_thinking=True,
            think_budget=3000,
            model_name="gpt-4",
        )
        assert result == 3128

    def test_other_model_without_thinking(self):
        """Non-Qwen3 without thinking: just base."""
        # 128 (base)
        result = tokens_to_generate(
            "niah",
            enable_thinking=False,
            think_budget=0,
            model_name="gpt-4",
        )
        assert result == 128

    def test_case_insensitive_model_detection(self):
        """Model name detection is case-insensitive."""
        # QWEN3 (uppercase) should also work
        result = tokens_to_generate(
            "niah",
            enable_thinking=False,
            think_budget=0,
            model_name="QWEN3-8B",
        )
        assert result == 133  # Still includes Qwen3 overhead


class TestThinkingPrefill:
    """Test thinking placeholder generation for message patterns."""

    def test_qwen3_without_thinking_returns_empty_block(self):
        """Qwen3 without thinking returns empty block to skip reasoning."""
        result = thinking_prefill("Qwen3-8b", enable_thinking=False)
        assert result == "<think>\n\n</think>\n\n"

    def test_qwen3_with_thinking_returns_empty_string(self):
        """Qwen3 with thinking returns empty (model continues in block)."""
        result = thinking_prefill("Qwen3-8b", enable_thinking=True)
        assert result == ""

    def test_other_model_returns_empty_string(self):
        """Non-Qwen3 models always return empty string."""
        for model in ["gpt-4", "llama-3", "claude-3"]:
            assert thinking_prefill(model, enable_thinking=True) == ""
            assert thinking_prefill(model, enable_thinking=False) == ""

    def test_case_insensitive_model_detection_in_prefill(self):
        """Model detection is case-insensitive in prefill."""
        result = thinking_prefill("QWEN3-8B", enable_thinking=False)
        assert result == "<think>\n\n</think>\n\n"


class TestMessageModes:
    """Test automatic message mode detection and construction."""

    def test_user_message_mode_default(self):
        """Default mode: answer_prefix appended to user message."""
        import asyncio

        task = Mock(spec=_ChatGenBase)
        task.model = Mock()
        task.model._model = "Qwen3-8b"
        task.model._kwargs = {
            "extra_body": {
                "continue_final_message": False,
                "add_generation_prompt": True,
            }
        }

        raw = {"input": "Context here.", "answer_prefix": "Answer: "}

        messages = asyncio.run(_ChatGenBase.preprocess(task, raw, None))

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Context here.Answer: "

    def test_assistant_message_mode_with_thinking_disabled(self):
        """Assistant mode: prefilled assistant turn with thinking_prefill."""
        import asyncio

        task = Mock(spec=_ChatGenBase)
        task.model = Mock()
        task.model._model = "Qwen3-8b"
        task.model._kwargs = {
            "extra_body": {
                "enable_thinking": False,
                "continue_final_message": True,
                "add_generation_prompt": False,
            }
        }

        raw = {"input": "Context here.", "answer_prefix": "Answer: "}

        messages = asyncio.run(_ChatGenBase.preprocess(task, raw, None))

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Context here."
        assert messages[1]["role"] == "assistant"
        # Should include thinking_prefill + answer_prefix
        assert messages[1]["content"] == "<think>\n\n</think>\n\nAnswer: "

    def test_assistant_message_mode_with_thinking_enabled(self):
        """Assistant mode with thinking: prefill returns empty string."""
        import asyncio

        task = Mock(spec=_ChatGenBase)
        task.model = Mock()
        task.model._model = "Qwen3-8b"
        task.model._kwargs = {
            "extra_body": {
                "enable_thinking": True,
                "continue_final_message": True,
                "add_generation_prompt": False,
            }
        }

        raw = {"input": "Context here.", "answer_prefix": "Answer: "}

        messages = asyncio.run(_ChatGenBase.preprocess(task, raw, None))

        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        # Empty prefill + answer_prefix
        assert messages[1]["content"] == "Answer: "

    def test_default_extra_body_missing(self):
        """Default behavior when extra_body is missing."""
        import asyncio

        task = Mock(spec=_ChatGenBase)
        task.model = Mock()
        task.model._model = "gpt-4"
        task.model._kwargs = {}  # No extra_body

        raw = {"input": "Context.", "answer_prefix": "Q: "}

        messages = asyncio.run(_ChatGenBase.preprocess(task, raw, None))

        # Should default to user-message mode
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Context.Q: "


class TestScenarios:
    """Test complete scenarios covering all use cases."""

    def test_scenario_qwen3_thinking(self):
        """Qwen3 with thinking budget."""
        tokens = tokens_to_generate(
            "niah",
            enable_thinking=True,
            think_budget=5000,
            model_name="Qwen3-8b",
        )
        prefill = thinking_prefill("Qwen3-8b", enable_thinking=True)

        assert tokens == 5132
        assert prefill == ""

    def test_scenario_qwen3_no_thinking(self):
        """Qwen3 without thinking."""
        tokens = tokens_to_generate(
            "niah",
            enable_thinking=False,
            think_budget=0,
            model_name="Qwen3-8b",
        )
        prefill = thinking_prefill("Qwen3-8b", enable_thinking=False)

        assert tokens == 133
        assert prefill == "<think>\n\n</think>\n\n"

    def test_scenario_gpt4_thinking(self):
        """GPT-4 with thinking."""
        tokens = tokens_to_generate(
            "niah",
            enable_thinking=True,
            think_budget=3000,
            model_name="gpt-4",
        )
        prefill = thinking_prefill("gpt-4", enable_thinking=True)

        assert tokens == 3128
        assert prefill == ""

    def test_scenario_gpt4_no_thinking(self):
        """GPT-4 without thinking."""
        tokens = tokens_to_generate(
            "niah",
            enable_thinking=False,
            think_budget=0,
            model_name="gpt-4",
        )
        prefill = thinking_prefill("gpt-4", enable_thinking=False)

        assert tokens == 128
        assert prefill == ""
