# Progress

What is built and verified, as of 31 October 2026. The board holds the status of open work; past evidence is in git history and on the closed issues.

Checkpoints A–D are built and reviewed by the owner; D, the keyword retrieval baseline (#31), closed with its first results table. Checkpoint E, embeddings and hybrid search, is next.

## Built and verified

- **Repo and tracking:** private GitHub repo, with milestones, tracking issues and sub-issues for checkpoints A–D on the [board](https://github.com/users/johnatomize/projects/2).
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
- **Keyword search (#34):** BM25 over the chunks of any strategy, with tokens that keep act numbers, article references and decimals whole (choices in `DECISIONS.md`). `make search Q="…"` prints the best five chunks for a typed question. "Can I take my weekly rest in the truck cab?" finds 561/2006 Art. 8(6) first; "How long may I drive in a day?" misses Art. 6(1), because the act says "daily driving time". That vocabulary gap is what E's embeddings have to close.
- **First results table (#36):** `make retrieval-runs` writes one committed run log per configuration to `runs/retrieval/`, and `make tables` rebuilds `results/retrieval.md` from the logs alone (`make check` fails if they disagree). BM25 on the 50 answerable dev questions, with 95% intervals; 10 unanswerable questions are left out:

  | Chunks | Mean words | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | article | 284 | 0.32 (0.21–0.46) | 0.72 (0.58–0.83) | 0.88 (0.76–0.94) | 0.64 (0.54–0.74) | 0.70 (0.61–0.78) |
  | paragraph | 112 | 0.32 (0.21–0.46) | 0.60 (0.46–0.72) | 0.68 (0.54–0.79) | 0.67 (0.55–0.78) | 0.65 (0.55–0.75) |
  | window-200-50 | 216 | 0.30 (0.19–0.44) | 0.60 (0.46–0.72) | 0.76 (0.63–0.86) | 0.61 (0.49–0.72) | 0.63 (0.53–0.72) |

  - **Article chunks find the most**, but they are 2.5 times as long as paragraph chunks, which the lenient matching rule favours. Paragraph chunks put the first right chunk highest (MRR). Every interval overlaps the others: 50 questions can't separate the three yet. G compares them at an equal context budget.
  - **Wording matters:** recall@5 is 0.66, 0.55 and 0.53 on paraphrased questions against 0.92, 0.75 and 0.83 on verbatim ones.
  - **Questions that need two places are the weak spot:** recall@5 is 0.00–0.17 on cross-act questions and 0.25–0.50 on multi-article ones, against 0.83–0.92 on single facts. Version traps are all found by article and paragraph chunks.
  - **Six questions are missed by every run in the top 10**: q023, q048, q050, q051, q062 and q066. Five of them need two or more places. q023, "What is a 'break' under Regulation 561/2006?", is verbatim, but "break" and "561/2006" appear in dozens of chunks, so the short definition in Art. 4(d) never rises.
  - **Baseline for J's quality gate:** recall@5 of 0.72, 0.60 and 0.60 on the dev split, in `results/baseline.json`.
- **Embedding cost estimate (#44):** `make embed-dry-run` estimates, without calling AWS, what embedding the corpus and the questions with Titan Text Embeddings V2 will cost, from the model and price in `config/embedding.yaml`. The three chunkings and the 100 questions come to about 192,000 tokens, counting text shared between chunkings once: US$0.0040, or US$0.0061 with a 50% margin. Bedrock runs in Stockholm, where the price read on 31 October 2026 is a tenth of Frankfurt's (reasons in `DECISIONS.md`). It fails if any input is empty or, with the margin, may be over the model's limit (the longest is about 2,860 tokens of 8,192), and warns when the price is more than a week old.
- **Hybrid search (#48):** `RRF` in `src/rag/retrieve.py` merges the rankings of any retrievers over the same chunks by reciprocal rank fusion (constant 60, each ranking read 50 deep; choices in `DECISIONS.md`). Tests with fixed rankings show that a chunk ranked second by both methods beats one ranked first by only one, that a chunk found by only one still appears, that ties fall in chunk order, and that chunks BM25 pads its top k with, because they hold no query word, get no credit.

## Spend log

Every cost goes here: date, what it paid for, amount and running total. A paid step needs the owner's OK and a dry-run estimate first.

Approved cap: none yet (checkpoint E proposes US$20 for the whole project).

| Date | Item | Cost (US$) | Total (US$) |
| --- | --- | ---: | ---: |
| 2026-10-27 | Project start | 0.00 | 0.00 |
