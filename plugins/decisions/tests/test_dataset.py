"""JSONL decision datasets use UTF-8 independently of the process locale.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_utf8_jsonl_loads_with_ascii_locale(tmp_path):
    sample = {
        "input": "商品屏幕破损。",
        "questions": [
            {
                "id": "queue",
                "type": "choice",
                "instructions": "请选择处理部门。",
                "options": [
                    {"value": "returns", "guidance": "退换货"},
                    {"value": "billing", "guidance": "账单"},
                ],
            },
            {
                "id": "priority",
                "type": "score",
                "instructions": "紧急程度？",
                "levels": ["低", "高"],
            },
        ],
        "reference": {"queue": "returns", "priority": 1},
    }
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import json
import locale
import sys

from sieval_decisions.dataset import DecisionJSONLDataset
from sieval_decisions.samples import request_from_sample

assert not sys.flags.utf8_mode
assert locale.getencoding() in {"ANSI_X3.4-1968", "US-ASCII", "ASCII"}
dataset = DecisionJSONLDataset(sys.argv[1])
assert dataset.test_set is not None
sample = json.loads(dataset.test_set[0]["record_json"])
request = request_from_sample(sample)
assert len(request.decisions.questions) == 2
print(json.dumps(sample, ensure_ascii=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env={
            **os.environ,
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONUTF8": "0",
            "PYTHONPATH": os.pathsep.join([str(source), *sys.path]),
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == sample
