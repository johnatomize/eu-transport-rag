# Progress

What is built and verified, as of 29 October 2026. The board holds the status of open work; past evidence is in git history and on the closed issues.

Checkpoints A, B and C are built and reviewed by the owner. Checkpoint D, the keyword retrieval baseline, is next (#31).

## Built and verified

- **Repo and tracking:** private GitHub repo, with milestones, tracking issues and sub-issues for checkpoints A–C on the [board](https://github.com/users/johnatomize/projects/2).
- **Secret scanning (#4):** a gitleaks pre-commit hook guards every commit since the first. Before the first commit, a staged fake token was refused.
- **Python tooling (#5):** a `uv` project on Python 3.12 with ruff and pytest, locked in `uv.lock`. `make check` runs lint, tests and a gitleaks scan of the full history, and passes on a fresh clone with only `uv` and gitleaks installed.
- **Local Postgres + pgvector (#7):** `docker-compose.yml` runs `pgvector/pgvector` 0.8.7 on Postgres 18, pinned by tag and digest, with credentials from `.env` and the port bound to localhost only. `make db-up` waits until it is healthy, `make db-check` runs `CREATE EXTENSION vector` and prints `pgvector 0.8.7`, and data survives `make db-down` / `make db-up`. Docker runs in Colima.
- **Corpus manifest and fetch (#11):** `corpus/manifest.yaml` pins the 20 phase 1 documents (three acts, ten guidance notes, seven clarification notes) by sha256, and their 2.0 MB of raw files are committed under `corpus/raw/`. `make corpus` checks them without network. `scripts/fetch_corpus.py --refetch` downloaded all 20 again from CELLAR and the Commission site on 27 October 2026, with identical hashes.
- **Parsing the acts (#12):** `make parse` turns the three acts into 716 nodes, each with a stable ID such as `561/2006:art6(3)`, its heading path, its amendment markers as metadata and its cross-references (ID scheme in `DECISIONS.md`). The counts are 561/2006: 33 articles, 76 paragraphs and 93 points; 165/2014: 48 articles, 148 paragraphs, 212 points and two annexes in 35 parts; 2002/15: 16 articles, rebuilt from flat HTML. Of 318 cross-references, 260 resolve to a node and 55 point to acts outside the corpus. The other 3 are listed: relative references inside 165/2014 Art. 45 that point into 561/2006. The only unparsed content is 9 images, mostly the tachograph pictograms, which stay as `[image]` in the text. A committed snapshot of every ID guards against ID changes.
- **Guidance notes (#13):** `make parse` also turns the 17 guidance and clarification notes into 47 nodes: one per note and one per page (30 pages, such as `guidance-7:p3`), with about 56,000 characters of text (library and cleaning in `DECISIONS.md`). Footnote markers and running headers are removed. 14 notes interpret a single act in the corpus (561/2006 or 165/2014), and their references without an act point into it; guidance-4, guidance-5 and clarification-7 cite either acts outside the corpus or more than one act. The notes' 88 references resolve as follows: 68 to a node, 9 to acts outside the corpus, and 11 left unresolved, each checked by hand and listed in the tests. Images (guidance-4, guidance-7, guidance-10) and pages drawn as a diagram, table or form (guidance-7 pp. 5–6, guidance-9 p. 3, guidance-10 p. 5) are listed as not understood. On guidance-10 p. 5 the pairing of nation codes and countries is lost.
- **Dates of the guidance notes (#26):** every note in the manifest has a publication date with its source (a Commission letter dated in the note, or the PDF's creation date), and `make parse` puts it on the note node. Only guidance-9 and guidance-10 are newer than the 2020 Mobility Package (sources and reasons in `DECISIONS.md`).
- **Question schema and validator (#15):** `data/questions.v1.jsonl` holds the evaluation questions, one JSON object per line, with the fields and rules in `DECISIONS.md`. `make questions`, which `make check` runs, checks every row against the schema and the parsed corpus, and fails if the file is missing. It rejects malformed rows, unknown or repeated fields, key facts missing from the answer, gold node IDs that are not in the corpus, are too broad or sit in an amending article, and rows that break the rule for their question type. Each message names the line, the question ID and the field. It prints a table of question types against split and phrasing.
- **Question set (#16):** 100 questions. They are split 60 dev and 40 test by a seeded shuffle within each type and phrasing, frozen from now on (method in `DECISIONS.md`). The validator also fails if a type has no questions in either split, and a test pins the 40 test questions. Key facts are written with digits, such as `9 hours` and `7.5 tonnes`.

  | Type | Paraphrased | Verbatim | Total | Dev / test |
  | --- | ---: | ---: | ---: | ---: |
  | single fact | 15 | 5 | 20 | 12 / 8 |
  | definition | 5 | 5 | 10 | 6 / 4 |
  | exception | 12 | 4 | 16 | 10 / 6 |
  | multi-article | 11 | 3 | 14 | 8 / 6 |
  | cross-act | 8 | 2 | 10 | 6 / 4 |
  | unanswerable | 13 | 3 | 16 | 10 / 6 |
  | version trap | 11 | 3 | 14 | 8 / 6 |
  | **total** | **75** | **25** | **100** | **60 / 40** |

  - **Gold nodes:** 55 questions have gold nodes in 561/2006, 17 in 165/2014 and 14 in 2002/15. 13 cite a guidance note, and for 8 of them the note is the only evidence.
  - **Version traps:** 11 come from the 2020 Mobility Package, 1 from Reg. 165/2014 and 2 from Reg. 2024/1258. Each old answer was checked against an earlier text of the act. Five traps have a pre-2020 guidance note in the corpus that gives the old answer, including guidance-1 on Art. 12 and guidance-6 on rest on a ferry.
  - **Unanswerable questions:** each was confirmed absent by a keyword search over the parsed text, and its notes record the search terms.
- **Gold answers checked (#17):** the owner checked all 100 gold answers against the text on a review page built by `scripts/gold_review.py`, which shows each question next to its gold nodes, the text that introduces a list item, and the whole article. Four were corrected (q003, q006, q008 and q033). For q062 and q097, whose gold nodes are list items, the fix is to show the introducing text on the page and carry it in chunks (#33). The validator now fails on any question not checked by the owner.
- **Chunking (#33):** `make chunks` cuts the corpus three ways and writes each to `build/chunks-<strategy>.jsonl`: 133 article chunks (mean 284 words, with 165/2014's annexes split into their parts), 412 paragraph chunks (mean 112) and 246 windows of 200 words overlapping by 50 (mean 216 with the prefix). Every chunk starts with its heading path, and a list item carries the text that introduces it, such as "For the purposes of this Regulation the following definitions shall apply:" above each definition in 561/2006 Art. 4. Tests check that every piece of text is in some chunk apart from the amending articles and the acts' titles, that every gold node of the question set can be found in each strategy, and that IDs are the same across runs (choices in `DECISIONS.md`).
- **Retrieval metrics (#35):** `src/rag/retrieval_eval.py` scores any ranked list of chunks against the gold nodes, following the rules in `DECISIONS.md` ("how retrieval evaluation scores a ranking"). It gives recall@1, @3, @5 and @10, fractional recall, MRR@10 and nDCG@10, each with a 95% interval, overall, by phrasing and by question type, and the number of unanswerable questions left out. The Wilson interval matches four published values (Newcombe 1998), the bootstrap gives the same interval for the same seed, and scoring the test split fails unless an explicit flag meant for checkpoint G is set.

## Spend log

Every cost goes here: date, what it paid for, amount and running total. A paid step needs the owner's OK and a dry-run estimate first.

Approved cap: none yet (checkpoint E proposes US$20 for the whole project).

| Date | Item | Cost (US$) | Total (US$) |
| --- | --- | ---: | ---: |
| 2026-10-27 | Project start | 0.00 | 0.00 |
