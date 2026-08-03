from pathlib import Path

import pytest

ROM_PATH = Path(__file__).resolve().parent.parent / "rom" / "tetris.gb"


@pytest.fixture(scope="session")
def rom_path() -> Path:
    if not ROM_PATH.exists():
        pytest.skip("rom/tetris.gb not present")
    return ROM_PATH
