"""Test RULER unified implementation supporting all model scenarios."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from sieval.datasets.ruler._shared import thinking_prefill, tokens_to_generate
from sieval.tasks.ruler_0shot_gen import RulerZeroShotGenTask


def _final(context_length, subtask, prediction, references):
    """Build a minimal `finals` entry carrying only what report() reads."""
    return SimpleNamespace(
        feedback_result={
            "prediction": prediction,
            "references": references,
            "subtask": subtask,
            "context_length": context_length,
        }
    )


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

        task = Mock(spec=RulerZeroShotGenTask)
        task.model = Mock()
        task.model.meta.return_value = {
            "model": "Qwen3-8b",
            "default_params": {
                "extra_body": {
                    "continue_final_message": False,
                    "add_generation_prompt": True,
                }
            },
        }

        raw = {"input": "Context here.", "answer_prefix": "Answer: "}

        messages = asyncio.run(RulerZeroShotGenTask.preprocess(task, raw, None))

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Context here.Answer: "

    def test_assistant_message_mode_with_thinking_disabled(self):
        """Assistant mode: prefilled assistant turn with thinking_prefill."""
        import asyncio

        task = Mock(spec=RulerZeroShotGenTask)
        task.model = Mock()
        task.model.meta.return_value = {
            "model": "Qwen3-8b",
            "default_params": {
                "extra_body": {
                    "enable_thinking": False,
                    "continue_final_message": True,
                    "add_generation_prompt": False,
                }
            },
        }

        raw = {"input": "Context here.", "answer_prefix": "Answer: "}

        messages = asyncio.run(RulerZeroShotGenTask.preprocess(task, raw, None))

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Context here."
        assert messages[1]["role"] == "assistant"
        # Should include thinking_prefill + answer_prefix
        assert messages[1]["content"] == "<think>\n\n</think>\n\nAnswer: "

    def test_assistant_message_mode_with_thinking_enabled(self):
        """Assistant mode with thinking: prefill returns empty string."""
        import asyncio

        task = Mock(spec=RulerZeroShotGenTask)
        task.model = Mock()
        task.model.meta.return_value = {
            "model": "Qwen3-8b",
            "default_params": {
                "extra_body": {
                    "enable_thinking": True,
                    "continue_final_message": True,
                    "add_generation_prompt": False,
                }
            },
        }

        raw = {"input": "Context here.", "answer_prefix": "Answer: "}

        messages = asyncio.run(RulerZeroShotGenTask.preprocess(task, raw, None))

        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        # Empty prefill + answer_prefix
        assert messages[1]["content"] == "Answer: "

    def test_default_extra_body_missing(self):
        """Default behavior when extra_body is missing."""
        import asyncio

        task = Mock(spec=RulerZeroShotGenTask)
        task.model = Mock()
        task.model.meta.return_value = {
            "model": "gpt-4",
            "default_params": {},  # No extra_body
        }

        raw = {"input": "Context.", "answer_prefix": "Q: "}

        messages = asyncio.run(RulerZeroShotGenTask.preprocess(task, raw, None))

        # Should default to user-message mode
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Context.Q: "


class TestInferCap:
    """infer() caps generation at the sample's per-subtask gen_budget."""

    def test_infer_passes_gen_budget_as_max_tokens(self):
        from unittest.mock import AsyncMock

        task = Mock(spec=RulerZeroShotGenTask)
        task.model = Mock()
        task.model.agenerate = AsyncMock(return_value="out")
        ctx = SimpleNamespace(raw_sample={"gen_budget": 32})

        result = asyncio.run(RulerZeroShotGenTask.infer(task, ["msg"], ctx))

        assert result == "out"
        task.model.agenerate.assert_awaited_once_with(["msg"], max_tokens=32)


class TestReport:
    """Test report() score aggregation: cell → per-length mean → mean-of-means."""

    def test_mean_of_means_weights_lengths_equally(self):
        """Headline `score` averages per-length means, not raw cells/samples.

        Layout (one sample per cell, string_match_all → 100 hit / 0 miss):
          - 4k: niah=100, vt=0  → length mean 50.0
          - 8k: niah=0          → length mean 0.0
        Mean-of-means = (50 + 0) / 2 = 25.0. A flat mean over the three cells
        would be (100 + 0 + 0) / 3 = 33.33 — so 25.0 discriminates the two.
        """
        finals = [
            _final(4096, "niah_single_1", "the answer is cat", ["cat"]),
            _final(4096, "vt", "wrong", ["dog"]),
            _final(8192, "niah_single_1", "wrong", ["cat"]),
        ]

        result = asyncio.run(RulerZeroShotGenTask.report(Mock(), finals, []))

        assert result["score"] == 25.0
        assert result["score_4k"] == 50.0
        assert result["score_8k"] == 0.0
        assert result["score_niah_single_1_4k"] == 100.0
        assert result["score_vt_4k"] == 0.0
        assert result["score_niah_single_1_8k"] == 0.0
        assert result["fails"] == 0

    def test_qa_subtasks_use_part_match(self):
        """QA cells score with string_match_part (any ref hit), not _all.

        Prediction contains one of two references:
          - string_match_part → max(1, 0) = 100.0
          - string_match_all  → (1 + 0) / 2 = 50.0
        A single QA cell makes `score` equal the cell score, so 100.0 proves
        the part-match branch is taken for QA subtasks.
        """
        finals = [
            _final(4096, "qa_squad", "the answer is paris", ["paris", "france"]),
        ]

        result = asyncio.run(RulerZeroShotGenTask.report(Mock(), finals, []))

        assert result["score"] == 100.0
        assert result["score_qa_squad_4k"] == 100.0

    def test_fails_are_counted_not_scored(self):
        """`fails` reflects the fails list length; only finals feed scoring."""
        finals = [_final(4096, "niah_single_1", "cat", ["cat"])]

        result = asyncio.run(RulerZeroShotGenTask.report(Mock(), finals, [1, 2, 3]))

        assert result["score"] == 100.0
        assert result["fails"] == 3

    def test_empty_finals_yield_zero_score(self):
        """No finals → overall 0.0 without ZeroDivisionError."""
        result = asyncio.run(RulerZeroShotGenTask.report(Mock(), [], []))

        assert result["score"] == 0.0
        assert result["fails"] == 0


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
