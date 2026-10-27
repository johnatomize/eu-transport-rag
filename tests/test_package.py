import tomllib
from pathlib import Path

import rag


def test_package_version_matches_pyproject():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        assert rag.__version__ == tomllib.load(f)["project"]["version"]
