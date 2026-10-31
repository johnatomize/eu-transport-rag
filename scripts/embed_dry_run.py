"""Print what embedding the corpus and questions would cost; calls nothing (`make embed-dry-run`).

Every chunking strategy is counted, and the questions of both splits: G embeds the test questions
too, and embedding a question is not scoring it. Fails if any input is over the model's limit.
"""

import datetime as dt
import sys
from pathlib import Path

from rag.chunk import STRATEGIES
from rag.corpus import CorpusError, load_manifest
from rag.embed_estimate import ConfigError, load_config, report
from rag.parse import ParseError, parse_corpus
from rag.questions import validate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
CONFIG = ROOT / "config" / "embedding.yaml"


def main() -> int:
    try:
        config = load_config(CONFIG)
        tree = parse_corpus(load_manifest(MANIFEST), ROOT)
    except (ConfigError, CorpusError, ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    questions, errors = validate(QUESTIONS.read_text(encoding="utf-8"), tree)
    if errors:
        print(f"error: {QUESTIONS.name} is not valid; run `make questions`", file=sys.stderr)
        return 1
    texts = {f"chunks: {name}": [c.text for c in chunk(tree)] for name, chunk in STRATEGIES.items()}
    texts["questions"] = [q.question for q in questions]
    lines, fits = report(texts, config, dt.date.today())
    print("\n".join(lines))
    return 0 if fits else 1


if __name__ == "__main__":
    sys.exit(main())
