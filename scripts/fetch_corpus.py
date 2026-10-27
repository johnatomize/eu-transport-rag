"""Check the raw corpus against corpus/manifest.yaml, downloading any file that is missing.

    fetch_corpus.py              check every pinned file; download missing ones (`make corpus`)
    fetch_corpus.py --refetch    download everything again and compare, writing nothing
    fetch_corpus.py --pin ID...  download documents and record their sha256 and date

Exits with status 1 and the document's ID on any mismatch or download problem.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from rag.corpus import CorpusError, Fetcher, ensure, load_manifest, pin, refetch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refetch", action="store_true", help="download again and compare")
    mode.add_argument("--pin", nargs="+", metavar="ID", help="download and record sha256 and date")
    args = parser.parse_args(argv)

    fetcher = Fetcher()
    try:
        if args.pin:
            for doc_id in args.pin:
                doc = pin(doc_id, MANIFEST, ROOT, fetcher, date.today())
                print(f"pinned   {doc.id}  {doc.sha256}")
            return 0
        docs = load_manifest(MANIFEST)
    except CorpusError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.refetch:
        mismatches, failures = refetch(docs, fetcher)
        for doc, saved in mismatches:
            print(f"CHANGED  {doc.id} ({doc.source}): new download saved to {saved}")
        errors = [str(e) for e in failures]
    else:
        errors = []
        for doc in docs:
            try:
                print(f"{ensure(doc, ROOT, fetcher):8} {doc.id}")
            except CorpusError as e:
                errors.append(str(e))
        mismatches = []
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if mismatches or errors:
        return 1
    print(f"all {len(docs)} documents match the manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
