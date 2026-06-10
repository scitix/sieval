"""Shared helpers for the RULER synthetic dataset family.

All RULER loaders measure prompt length against a tokenizer to fill a target
``max_seq_length``. They use the same builder: tiktoken for the ``gpt-4``
default, otherwise a HuggingFace ``AutoTokenizer`` for the model under test.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""


def build_tokenizer(tokenizer_model: str):
    """Return a token encoder exposing ``.encode(str) -> list[int]``.

    ``gpt-4`` (the RULER default) maps to a tiktoken encoding; any other value
    is treated as a HuggingFace model id loaded via ``AutoTokenizer``.
    """
    if tokenizer_model == "gpt-4":
        import tiktoken

        return tiktoken.encoding_for_model(tokenizer_model)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
