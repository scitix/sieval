"""AI-Generated Code - Claude Sonnet 4.6 (Anthropic)"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Import sieval from THIS checkout, which is also where the index is written.
# `python scripts/x.py` puts scripts/ on sys.path[0] but never the repo root, so
# `import sieval` otherwise resolves through the editable install — from a
# worktree that is a different checkout, whose registry would be written here.
sys.path.insert(0, str(ROOT))

from sieval.core.datasets.meta import (  # noqa: E402
    dataset_meta_to_dict,
    import_all_datasets,
    iter_dataset_metas,
)
from sieval.core.tasks.meta import (  # noqa: E402
    import_all_tasks,
    iter_task_metas,
    task_meta_to_dict,
)

INDEX_PATH = ROOT / "sieval" / "meta" / "index.json"


def render_payload() -> str:
    import_all_datasets()
    import_all_tasks()
    datasets = sorted(
        (dataset_meta_to_dict(m) for m in iter_dataset_metas()),
        key=lambda d: d["name"],
    )
    tasks = sorted(
        (task_meta_to_dict(m) for m in iter_task_metas()),
        key=lambda d: d["name"],
    )
    payload = {
        "schema_version": 1,
        "datasets": datasets,
        "tasks": tasks,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate sieval/meta/index.json from the task + dataset registries."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether the committed index is up-to-date without writing files.",
    )
    args = parser.parse_args()

    rendered = render_payload()
    current = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else ""

    if args.check:
        if current == rendered:
            return 0
        raise SystemExit(
            "sieval/meta/index.json is out of date. "
            "Run `python scripts/sync_meta_index.py` to regenerate."
        )

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(rendered, encoding="utf-8")
    os.replace(tmp, INDEX_PATH)
    payload = json.loads(rendered)
    relative = INDEX_PATH.relative_to(ROOT)
    print(  # noqa: T201
        f"Wrote {len(payload['datasets'])} dataset and "
        f"{len(payload['tasks'])} task entries to {relative}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
