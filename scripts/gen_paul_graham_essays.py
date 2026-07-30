#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""Regenerate RULER's Paul Graham Essays haystack corpus (bring-your-own).

Adapted from NVIDIA RULER's ``scripts/data/synthetic/json/
download_paulgraham_essay.py`` (Apache-2.0). Fetches ~218 essays (paulgraham.com
HTML + gkamradt's needle-haystack repo text), concatenates them into a single
``{"text": ...}`` document, and writes it *directly* into the data dir at
``<data-dir>/ruler/PaulGrahamEssays.json.gz`` — exactly where the RULER loader
reads it.

The essay text is not redistributable (Apache-2.0 covers NVIDIA's scraper, not
Paul Graham's essays), so it is never bundled in the package or committed. The
``local:`` source scheme treats it as bring-your-own: run this script to produce
the file. The URL list is pinned to a RULER commit SHA (not ``main``) so re-runs
are reproducible against a fixed essay set. This is a regeneration tool; its
html2text/beautifulsoup4/tqdm deps are intentionally NOT part of sieval's runtime
dependency graph (this file lives under ``scripts/`` and is never imported by
``sieval/``).

Usage:
    pdm run python scripts/gen_paul_graham_essays.py --data-dir "$SIEVAL_DATA_DIR"
    # or, with SIEVAL_DATA_DIR exported:
    pdm run python scripts/gen_paul_graham_essays.py

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import argparse
import gzip
import json
import os
import ssl
import time
import urllib.request
from pathlib import Path

import certifi
import html2text
from bs4 import BeautifulSoup
from tqdm import tqdm

# Per-request timeout (seconds) and retry budget — paulgraham.com occasionally
# stalls; without a timeout a single hung socket wedges the whole run.
_TIMEOUT = 30.0
# paulgraham.com intermittently serves an incomplete TLS chain (missing
# intermediate), so a given essay randomly hits CERTIFICATE_VERIFY_FAILED on
# some attempts and succeeds on others. Generous retries make a full 218/218
# fetch reliable; the run aborts rather than write a partial corpus.
_RETRIES = 6

# Verify TLS against certifi's CA bundle explicitly. Some interpreters (system
# python, conda) point ssl at a missing/empty cert store, so the default context
# fails with CERTIFICATE_VERIFY_FAILED — pinning certifi makes the script work
# regardless of which python runs it.
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# Pinned RULER commit that owns the URL list — keeps the essay set fixed across
# regenerations. Same commit the rest of the port pins (``_RULER_DATA_SHA`` in
# sieval/datasets/ruler/_shared.py), so the whole benchmark references one
# upstream revision; the URL list is byte-identical to the earlier 041a952 pin
# this replaced (218 URLs, sha256 9862c06b…).
_RULER_SHA = "ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13"
_URL_LIST = (
    f"https://raw.githubusercontent.com/NVIDIA/RULER/{_RULER_SHA}/"
    "scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt"
)

# The corpus is staged under ``<data-dir>/ruler/`` (the RULER dataset name), the
# same layout the ``local:`` handler reports and the loader reads via `gzip.open`.
_CORPUS_RELPATH = Path("ruler") / "PaulGrahamEssays.json.gz"


def _resolve_out_path(data_dir: str | None) -> Path:
    data_dir = data_dir or os.environ.get("SIEVAL_DATA_DIR")
    if not data_dir:
        raise SystemExit(
            "no data dir: pass --data-dir <path> or export SIEVAL_DATA_DIR; "
            "the corpus is written to <data-dir>/ruler/PaulGrahamEssays.json.gz"
        )
    return Path(data_dir).expanduser() / _CORPUS_RELPATH


def _html_to_text(content: str, converter: html2text.HTML2Text) -> str:
    soup = BeautifulSoup(content, "html.parser")
    specific_tag = soup.find("font")
    return converter.handle(str(specific_tag))


def _fetch(url: str) -> bytes:
    """GET *url* with a per-request timeout and bounded retries."""
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            with urllib.request.urlopen(
                url, timeout=_TIMEOUT, context=_SSL_CTX
            ) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001 — retry any transport error
            last_exc = e
            time.sleep(2**attempt)
    assert last_exc is not None
    raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate RULER's PG corpus.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data dir root (defaults to $SIEVAL_DATA_DIR); the corpus is written "
        "to <data-dir>/ruler/PaulGrahamEssays.json.gz",
    )
    args = parser.parse_args()
    out_path = _resolve_out_path(args.data_dir)

    converter = html2text.HTML2Text()
    converter.ignore_images = True
    converter.ignore_tables = True
    # Upstream sets escape_all, but on html2text >= 2025.4.15 that attribute is a
    # silent no-op (renamed/removed), so upstream's corpus is unescaped markdown.
    # Deliberately not setting escape_snob (the current equivalent) here either,
    # to match upstream's actual (unescaped) output rather than its intent.
    converter.mark_code = False

    urls = [line.strip() for line in _fetch(_URL_LIST).decode("utf-8").splitlines()]
    urls = [u for u in urls if u]

    essays: list[str] = []
    failed: list[str] = []
    for url in tqdm(urls, desc="essays"):
        try:
            raw = _fetch(url)
            if ".html" in url:
                # Mirror RULER's exact per-essay decode: `unicode_escape`
                # here is faithful to the original download_paulgraham_essay.py,
                # not an oversight. (Full-corpus bytes still differ from upstream:
                # this script concatenates essays in the pinned URL-list order,
                # while upstream groups them repo-then-html by folder — glob order
                # within each group. Score-neutral, enumerated in
                # RulerZeroShotGenTask.reference_impl.notes.)
                parsed = _html_to_text(raw.decode("unicode_escape", "utf-8"), converter)
            else:
                parsed = raw.decode("utf-8")
        except Exception as e:  # noqa: BLE001 — best-effort, record and skip
            print(f"Fail download {url} ({e})")
            failed.append(url)
            continue
        essays.append(parsed)

    # A partial corpus would silently change the haystack bytes (and break
    # reproducibility against RULER), so refuse to write an incomplete file.
    if failed:
        raise SystemExit(
            f"{len(failed)}/{len(urls)} essays failed to download; "
            f"refusing to write a partial corpus. Failed: {failed}"
        )

    text = "".join(essays)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 so the gzip header is byte-stable across regenerations — the same
    # essay set always yields identical bytes regardless of when it was run.
    with gzip.GzipFile(out_path, "wb", mtime=0) as gz:
        gz.write(json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"))

    size = out_path.stat().st_size
    print(
        f"Wrote {len(essays)}/{len(urls)} essays "
        f"({size / 1_000_000:.1f} MB) -> {out_path}"
    )


if __name__ == "__main__":
    main()
