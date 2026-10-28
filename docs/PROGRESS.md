# Progress

What is built and verified, as of 28 October 2026. The board holds the status of open work; past evidence is in git history and on the closed issues.

Checkpoint A is done. Checkpoint B: every build item is done and the exit gate passes, except the owner's review (#14). Checkpoint C is next.

## Built and verified

- **Repo and tracking:** private GitHub repo, with milestones, tracking issues and sub-issues for checkpoints A–C on the [board](https://github.com/users/johnatomize/projects/2).
- **Secret scanning (#4):** a gitleaks pre-commit hook guards every commit since the first. Before the first commit, a staged fake token was refused.
- **Python tooling (#5):** a `uv` project on Python 3.12 with ruff and pytest, locked in `uv.lock`. `make check` runs lint, tests and a gitleaks scan of the full history, and passes on a fresh clone with only `uv` and gitleaks installed.
- **Local Postgres + pgvector (#7):** `docker-compose.yml` runs `pgvector/pgvector` 0.8.7 on Postgres 18, pinned by tag and digest, with credentials from `.env` and the port bound to localhost only. `make db-up` waits until it is healthy, `make db-check` runs `CREATE EXTENSION vector` and prints `pgvector 0.8.7`, and data survives `make db-down` / `make db-up`. Docker runs in Colima.
- **Corpus manifest and fetch (#11):** `corpus/manifest.yaml` pins the 20 phase 1 documents (three acts, ten guidance notes, seven clarification notes) by sha256, and their 2.0 MB of raw files are committed under `corpus/raw/`. `make corpus` checks them without network. `scripts/fetch_corpus.py --refetch` downloaded all 20 again from CELLAR and the Commission site on 27 October 2026, with identical hashes.
- **Parsing the acts (#12):** `make parse` turns the three acts into 716 nodes, each with a stable ID such as `561/2006:art6(3)`, its heading path, its amendment markers as metadata and its cross-references (ID scheme in `DECISIONS.md`). The counts are 561/2006: 33 articles, 76 paragraphs and 93 points; 165/2014: 48 articles, 148 paragraphs, 212 points and two annexes in 35 parts; 2002/15: 16 articles, rebuilt from flat HTML. Of 318 cross-references, 260 resolve to a node and 55 point to acts outside the corpus. The other 3 are listed: relative references inside 165/2014 Art. 45 that point into 561/2006. The only unparsed content is 9 images, mostly the tachograph pictograms, which stay as `[image]` in the text. A committed snapshot of every ID guards against ID changes.
- **Guidance notes (#13):** `make parse` also turns the 17 guidance and clarification notes into 47 nodes: one per note and one per page (30 pages, such as `guidance-7:p3`), with about 56,000 characters of text (library and cleaning in `DECISIONS.md`). Footnote markers and running headers are removed. 14 notes interpret a single act in the corpus (561/2006 or 165/2014), and their references without an act point into it; guidance-4, guidance-5 and clarification-7 cite either acts outside the corpus or more than one act. The notes' 88 references resolve as follows: 68 to a node, 9 to acts outside the corpus, and 11 left unresolved, each checked by hand and listed in the tests. Images (guidance-4, guidance-7, guidance-10) and pages drawn as a diagram, table or form (guidance-7 pp. 5–6, guidance-9 p. 3, guidance-10 p. 5) are listed as not understood. On guidance-10 p. 5 the pairing of nation codes and countries is lost.

## Spend log

Every cost goes here: date, what it paid for, amount and running total. A paid step needs the owner's OK and a dry-run estimate first.

Approved cap: none yet (checkpoint E proposes US$20 for the whole project).

| Date | Item | Cost (US$) | Total (US$) |
| --- | --- | ---: | ---: |
| 2026-10-27 | Project start | 0.00 | 0.00 |
