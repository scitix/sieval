"""BrowseComp dataset loader (OpenAI, arXiv:2504.12516).

BrowseComp is a 1,266-question live-web browsing benchmark: each question has a
short, verifiable, time-invariant answer that is hard to find (engineered so a
human can't solve it in ~10 minutes and it isn't trivially Googleable).

The official distribution is a single ENCRYPTED CSV in OpenAI simple-evals
(hosted on an Azure blob), columns ``problem, answer, problem_topic, canary``.
``problem`` and ``answer`` are XOR-encrypted; the per-row ``canary`` is the
decryption password. Encryption is anti-crawl/anti-training obfuscation, not a
secret withheld from evaluators — the password ships in the row.

``source`` uses the ``url:`` scheme (like ``gpqa_diamond``): ``sieval dataset
download browsecomp`` fetches the raw ENCRYPTED csv into
``<data>/browsecomp/browse_comp_test_set.csv`` (with a sha256 integrity check).
Decryption happens HERE, in ``load()``, in memory — the plaintext is never
written back to the download cache, so the on-disk copy stays encrypted
(contamination hygiene; keep run outputs out of version control too).

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import base64
import csv
import hashlib
import os
from typing import TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)

CSV_BASENAME = "browse_comp_test_set.csv"
# sha256 of the official encrypted CSV (pinned for integrity/reproducibility).
CSV_SHA256 = "7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf"


class BrowseCompDatasetSample(TypedDict):
    original_index: int
    problem: str
    answer: str
    problem_topic: str


# --- decryption (verbatim from openai/simple-evals browsecomp_eval.py) --------
# https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/browsecomp_eval.py
def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    # derive_key returns exactly len(encrypted) bytes, so strict=True never fires.
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key, strict=True))
    return decrypted.decode()


# ------------------------------------------------------------------------------


@sieval_dataset(
    name="browsecomp",
    display_name="BrowseComp",
    description="BrowseComp — 1,266-question hard live-web browsing benchmark.",
    source=f"url:https://openaipublic.blob.core.windows.net/simple-evals/{CSV_BASENAME}",
    checksums={CSV_BASENAME: f"sha256:{CSV_SHA256}"},
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("english", "browsing", "deep-research", "open-ended"),
    license="MIT",
)
class BrowseCompDataset(Dataset[BrowseCompDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        csv_path = (
            os.path.join(name_or_path, CSV_BASENAME)
            if os.path.isdir(name_or_path)
            else name_or_path
        )
        # Hand-read + from_list (not load_dataset(...).map(decrypt)): .map()
        # caches to ~/.cache/huggingface/datasets, which would persist decrypted
        # plaintext to disk and defeat the contamination hygiene noted above.
        # list[dict] (not the TypedDict) so HFDataset.from_list type-checks.
        rows: list[dict] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                canary = row.get("canary", "")
                rows.append(
                    {
                        "original_index": i,
                        "problem": decrypt(row["problem"], canary),
                        "answer": decrypt(row["answer"], canary),
                        "problem_topic": row.get("problem_topic", ""),
                    }
                )
        if not rows:
            raise ValueError(
                f"BrowseComp produced an empty 'test' split from {csv_path!r}; "
                "check that the dataset has been downloaded via "
                "`sieval dataset download browsecomp`."
            )
        # Built in memory; decrypted plaintext is not persisted to the cache dir.
        return HFDatasetDict({"test": HFDataset.from_list(rows)})
