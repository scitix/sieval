import gzip
import json
import os
import re

_NOISE_HAYSTACK = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)

_CORPUS_FILE = "PaulGrahamEssays.json.gz"

_NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."

def _build_haystack(name_or_path: str, type_haystack: str):
    if type_haystack == "essay":
        path = os.path.join(name_or_path, _CORPUS_FILE)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            text = json.load(f)["text"]
        return re.sub(r"\s+", " ", text).split(" ")
    if type_haystack == "noise":
        return _NOISE_HAYSTACK
    if type_haystack == "needle":
        return _NEEDLE
    else:
        raise NotImplementedError(f"{type_haystack} is not implemented.")
