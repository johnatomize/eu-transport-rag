from pathlib import Path

import pytest

from rag.chunk import STRATEGIES, Chunk
from rag.corpus import load_manifest
from rag.parse import Tree, parse_corpus

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="session")
def tree() -> Tree:
    """The whole corpus, parsed once for every test that needs it."""
    return parse_corpus(load_manifest(ROOT / "corpus" / "manifest.yaml"), ROOT)


@pytest.fixture(scope="session")
def chunks(tree: Tree) -> dict[str, list[Chunk]]:
    """Every chunking of the whole corpus, cut once for every test that needs it."""
    return {name: strategy(tree) for name, strategy in STRATEGIES.items()}
