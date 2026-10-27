"""The one reader of an evaluation config file.

``sieval run``, ``sieval eval``, their ``--dry-run`` and ``EvalSession`` all
open the config through :func:`load_eval_config`, so a file is accepted or
refused the same way on every entry point. A repeated mapping key is refused:
PyYAML keeps the last value silently, so a second ``api_base`` or
``capabilities:`` would otherwise decide the run unseen.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from pathlib import Path

import yaml

_MERGE_TAG = "tag:yaml.org,2002:merge"


class DuplicateKeyError(ValueError):
    """The config repeats a key within one mapping."""


class _EvalConfigLoader(yaml.SafeLoader):
    """``SafeLoader`` that records each repeated key; one instance per load."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self.duplicates: list[str] = []

    def construct_mapping(self, node, deep=False):
        # Before ``flatten_mapping``: keys a ``<<`` merge brings in may be
        # overridden by the mapping's own keys, as YAML intends.
        if isinstance(node, yaml.MappingNode):
            first_line: dict[object, int] = {}
            for key_node, _ in node.value:
                if key_node.tag == _MERGE_TAG:
                    continue
                key = self.construct_object(key_node, deep=False)
                line = key_node.start_mark.line + 1
                try:
                    seen = key in first_line
                except TypeError:
                    continue  # unhashable; SafeLoader refuses it itself
                if seen:
                    self.duplicates.append(
                        f"duplicate key {key!r} at line {line} "
                        f"(first at line {first_line[key]})"
                    )
                else:
                    first_line[key] = line
        return super().construct_mapping(node, deep=deep)


def load_eval_config(path: str | Path) -> object:
    """Parse the evaluation config at *path*, as ``yaml.safe_load`` would.

    Raises:
        DuplicateKeyError: a mapping repeats a key; names every repeat.
        yaml.YAMLError: the file is not valid YAML.
    """
    loader = _EvalConfigLoader(Path(path).read_text(encoding="utf-8"))
    try:
        data = loader.get_single_data()
    finally:
        loader.dispose()
    if loader.duplicates:
        raise DuplicateKeyError(f"{path}: " + "; ".join(loader.duplicates))
    return data
