"""local scheme handler: a bring-your-own corpus staged at ``dest_root/<name>/``.

Some datasets depend on a corpus that cannot be redistributed — e.g. RULER's
Paul Graham essays, where Apache-2.0 covers NVIDIA's scraper but not the essay
text itself. Such a file is produced out-of-band by a generation script (e.g.
``scripts/gen_paul_graham_essays.py``) written *directly* into the data dir at
``{dest_root}/<dataset_name>/``; it is never bundled in the package or committed.

``download`` therefore moves no bytes: if the file is already staged it is a
no-op, otherwise it raises :class:`LocalSourceUnavailable` with instructions.
``is_downloaded`` reports actual presence, so ``sieval dataset download`` can
surface the BYO requirement instead of silently succeeding.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

from pathlib import Path

from sieval.core.datasets.meta import url_path_basename


class LocalSourceUnavailable(RuntimeError):
    """A ``local:`` corpus is required but absent from the data dir (BYO)."""


class LocalHandler:
    scheme = "local"

    def download(
        self,
        source: str,
        dest_root: Path,
        dataset_name: str,
        force: bool,
    ) -> None:
        relpath = self._strip_scheme(source)
        target = dest_root / dataset_name / _basename(relpath)
        # BYO: nothing to fetch. Present → no-op; absent → tell the user how to
        # produce it (force cannot re-fetch a corpus that was never remote).
        if target.exists():
            return
        raise LocalSourceUnavailable(
            f"{source} is a bring-your-own corpus and cannot be fetched "
            f"automatically. Generate or place it at {target} — see the "
            f"dataset's regeneration script under scripts/."
        )

    def is_downloaded(
        self,
        source: str,
        dest_root: Path,
        dataset_name: str,
    ) -> bool:
        relpath = self._strip_scheme(source)
        return (dest_root / dataset_name / _basename(relpath)).exists()

    @staticmethod
    def _strip_scheme(source: str) -> str:
        if not source.startswith("local:"):
            raise ValueError(f"Expected local: scheme, got {source!r}")
        return source[len("local:") :]


def _basename(relpath: str) -> str:
    """Filename the corpus is staged under; shares the url-handler primitive so
    the on-disk name matches the ``url:`` convention."""
    return url_path_basename(relpath) or "download"
