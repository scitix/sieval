"""SiEval — Model Delivery Quality Verification System."""

try:
    from sieval._version import (  # type: ignore[unresolved-import]
        __version__ as __version__,
    )
except ImportError:
    # editable install or dev environment without build
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__: str = version("sieval")
    except PackageNotFoundError:
        __version__: str = "0.0.0"

from sieval.meta import load_index as load_index
