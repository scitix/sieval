"""local scheme handler: stage a package-bundled file into ``dest_root/<name>/``.

For datasets whose corpus is generated once and committed inside the package
(under ``sieval/datasets/_data/``) rather than fetched from a remote. ``download``
copies the bundled file into the same ``{dest_root}/<dataset_name>/`` layout the
url/hf handlers use, so the runtime ``load(name_or_path)`` path is identical.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import shutil
from importlib.resources import files
from pathlib import Path
from posixpath import normpath

from sieval.core.datasets.meta import url_path_basename

# Bundled-data root inside the package; `local:<relpath>` resolves under here.
_DATA_ANCHOR = "sieval.datasets"
_DATA_SUBDIR = "_data"


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
        bundled = self._bundled_path(relpath)
        target_dir = dest_root / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / _basename(relpath)
        if target.exists() and not force:
            return
        tmp = target.with_name(target.name + ".partial")
        try:
            shutil.copyfile(bundled, tmp)
            tmp.replace(target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

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

    @staticmethod
    def _bundled_path(relpath: str) -> Path:
        """Resolve *relpath* under the package data root, rejecting traversal.

        ``local:`` must only ever read files committed inside the package, so an
        absolute path or a ``..`` segment that would escape ``_data/`` is a hard
        error rather than a silently-resolved path.
        """
        if (
            not relpath
            or relpath.startswith("/")
            or ".." in relpath.split("/")
            or normpath(relpath) != relpath
        ):
            raise ValueError(
                f"local: path must be a normalized, package-relative path, "
                f"got {relpath!r}"
            )
        return Path(str(files(_DATA_ANCHOR).joinpath(_DATA_SUBDIR, relpath)))


def _basename(relpath: str) -> str:
    """Filename the bundled file lands under; shares the url-handler primitive
    so the on-disk name matches the ``url:`` convention."""
    return url_path_basename(relpath) or "download"
