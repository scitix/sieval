"""The `MODEL_CONFIG_MAPPING` global upstream's `ast_checker` expects.

`convert_func_name` reads `MODEL_CONFIG_MAPPING[model].underscore_to_dot` to
decide whether a gold function name's dots must be rewritten to underscores
before comparison. Upstream carries a 2044-line registry of its own evaluated
models to answer that one question. sieval's models are not in it, so a verbatim
copy would `KeyError`.

Rather than delete the call (removing a helper is as much a deviation as adding
one), supply the global the file expects. The flag is a property of the
*transport*, not of a model: it is True exactly when the function schemas
travelled through an API whose tool names cannot contain `.`.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _ModelEntry:
    underscore_to_dot: bool


class _ModelConfigMapping:
    """Answers for any model name, because the flag does not depend on one."""

    def __init__(self) -> None:
        self._underscore_to_dot = False

    def set_underscore_to_dot(self, enabled: bool) -> None:
        self._underscore_to_dot = enabled

    def __getitem__(self, _model_name: str) -> _ModelEntry:
        return _ModelEntry(underscore_to_dot=self._underscore_to_dot)


MODEL_CONFIG_MAPPING = _ModelConfigMapping()


def set_underscore_to_dot(enabled: bool) -> None:
    """Set the flag for the protocol about to be graded.

    Prompt protocol → False: schemas reach the model verbatim, dots survive.
    FC protocol → True: `convert_to_tool` rewrote `.`→`_` on the way out, so
    gold must be rewritten to match.
    """
    MODEL_CONFIG_MAPPING.set_underscore_to_dot(enabled)
