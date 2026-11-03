"""Print what embedding everything would take and cost; runs nothing (`make embed-dry-run`).

Every chunking strategy is counted, and the questions of both splits (see `corpus_texts`). Fails
if any input is empty or may be over the model's limit.
"""

import datetime as dt
import sys
from pathlib import Path

from rag.corpus import CorpusError
from rag.embed import EmbedError, corpus_texts
from rag.embed_estimate import ConfigError, load_config, report
from rag.parse import ParseError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "embedding.yaml"


def main() -> int:
    try:
        config = load_config(CONFIG)
        texts = corpus_texts(ROOT, config)
    except (ConfigError, CorpusError, ParseError, EmbedError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    lines, fits = report(texts, config, dt.date.today())
    print("\n".join(lines))
    return 0 if fits else 1


if __name__ == "__main__":
    sys.exit(main())
