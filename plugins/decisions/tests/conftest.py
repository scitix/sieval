"""Fixtures for the standalone decisions plugin.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def anyio_backend():
    return "asyncio"
