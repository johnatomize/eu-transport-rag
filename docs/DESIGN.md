# Design: `eu-transport-rag`

The design the issues refer to. Status lives on the [board](https://github.com/users/johnatomize/projects/2), not here.

## Context

A question-answering service over public EU road-transport and heavy-vehicle legislation, built with retrieval-augmented generation (RAG). You ask *"How many hours may a driver drive in two consecutive weeks?"* and get *"90 hours"* with a citation to Regulation (EC) No 561/2006, Article 6(3), consolidated version of 31.12.2024. If the documents don't contain the answer, the service says so.

It is also a learning project. By the end the owner should understand each piece of a RAG system, why it is there, and how to measure whether it works.

Framing:
- **No deadline**, paced in checkpoints, each reviewed by the owner. Each checkpoint adds a section to `docs/LEARNING.md`.
- **Measured, not claimed.** The deliverable is a results table that compares retrieval and answer quality across variants, with error bars. A working chatbot alone is not the goal.
- **Public data only.** EU legislation and Commission guidance; no private or personal data.
- **Spending is gated.** Checkpoints A–D cost nothing. The first paid call (E), the AWS account setup (I) and any budget increase each need the owner's explicit OK, with a dry-run cost estimate first.

## The corpus

### Where the texts come from

The official source is **EUR-Lex** (`eur-lex.europa.eu`). It is fine for reading in a browser, but it **blocks scripted downloads**: `curl` gets an empty `202` (a bot challenge). Checked 2026-10-27.

Scripts use the **Publications Office CELLAR API** instead. It serves the same documents and works with plain HTTP (checked 2026-10-27):

```sh
# Consolidated text (base act plus all amendments) as XHTML, English
curl -L -H "Accept: application/xhtml+xml" -H "Accept-Language: eng" \
  http://publications.europa.eu/resource/celex/02006R0561-20241231
```

- **CELEX numbers** identify documents. `32006R0561` is the act as originally published. `02006R0561-20241231` is the consolidated version as of 31 Dec 2024.
- The list of consolidated versions comes from the SPARQL endpoint at `https://publications.europa.eu/webapi/rdf/sparql`.
- Swedish texts come from the same URL with `Accept-Language: swe`, which enables an optional bilingual extension (checkpoint L).
- Consolidated texts state that they are "a documentation tool and [have] no legal effect". The demo repeats that disclaimer.
- They contain amendment markers such as `▼M3` (the text below came from the 3rd amendment). The parser removes them from the text but keeps them as metadata.

### What goes in

**Phase 1, the social rules** (small; questions with exact, checkable answers):

| Act | CELEX (base) | Latest consolidated (2026-10-27) |
| --- | --- | --- |
| Reg. 561/2006, driving times, breaks, rest periods | 32006R0561 | 02006R0561-20241231 |
| Reg. 165/2014, tachographs | 32014R0165 | 02014R0165-20241231 |
| Dir. 2002/15/EC, working time of mobile road transport workers | 32002L0015 | none: never amended, so the base act is used (HTML) |
| Commission guidance and clarification notes on 561/2006 (PDFs) | [transport.ec.europa.eu](https://transport.ec.europa.eu/transport-modes/road/social-provisions/driving-time-and-rest-periods/guidance-notes-implementation-community-rules-driving-times-and-rest-periods-professional-drivers_en) | — |

Dir. 2002/15 is the exception to "consolidated versions only". CELLAR lists one consolidated version, `02002L0015-20020323`, dated the act's own publication day, but serves no content for it. The directive has never been amended, so the base act `32002L0015` is the current text. It exists only as old-style HTML 4, not XHTML, so the parser has a small second reader for it. Citations name the original act (OJ L 80, 23.3.2002). Checked 2026-10-27 (#21).

The Commission site rate-limits scripted downloads (HTTP 429 after about six PDFs in a row), so the fetch script waits between requests and checks that each file really is a PDF.

**Phase 2, vehicle rules** (larger and more technical; tests scaling and cross-act retrieval). Phase 2 joins **after checkpoint G** as a scaling step: the corpus, the question set and the G results table are built on phase 1 first; then phase 2 is added and the table re-run to measure what a larger corpus does to retrieval (#10).

| Act | CELEX (base) | Latest consolidated (2026-10-27) |
| --- | --- | --- |
| Dir. 96/53/EC, maximum weights and dimensions | 31996L0053 | 01996L0053-20190814 |
| Reg. 595/2009, Euro VI heavy-duty emissions type-approval | 32009R0595 | 02009R0595-20200901 |
| Reg. 2024/1257, Euro 7 | 32024R1257 | 02024R1257-20260923 |
| Reg. 2019/1242, CO2 standards for new heavy-duty vehicles | 32019R1242 | 02019R1242-20260910 |
| Reg. 2019/2144, General Safety Regulation | 32019R2144 | 02019R2144-20260802 |
| Reg. 2018/858, type-approval framework | 32018R0858 | 02018R0858-20260802 |

`corpus/manifest.yaml` pins the exact consolidated CELEX of every document, with its download date and sha256, so the corpus can be rebuilt byte for byte.

### Four problems found while planning (worth understanding, fixed in the plan)

1. **Amending acts would poison the index.**
   - Reg. 2020/1054 (the Mobility Package) contains text like "Article 6 is replaced by the following: …". Indexed next to the consolidated 561/2006, it puts two versions of each rule in the index, and the older wording can win retrieval.
   - **Fix:** index only consolidated versions. Amending acts are left out, and the amendment markers are kept as metadata.
2. **Legal text is full of cross-references.**
   - Example: "any time spent as described in point (e) of Article 4". The chunk that answers a question often depends on a definition in another article, and plain similarity search won't retrieve it.
   - **Fix:** the parser records references between articles. One variant adds the referenced definitions to the retrieved context, and it is measured against the plain version.
3. **A question set written from the text favours keyword search.**
   - If questions copy the regulation's wording ("daily driving time shall not exceed"), keyword search (BM25) looks better than it would with real users.
   - **Fix:**
     - Write questions the way a driver or fleet manager would ask them ("Can I drive 10 hours today?").
     - Tag each question as *paraphrased* or *verbatim*.
     - Report results for both groups separately.
     - Keep a held-out test split that is used only for the final table.
4. **An LLM judge is a model grading a model.**
   - **Fix:** use deterministic checks wherever possible:
     - gold article IDs for retrieval;
     - exact match on numbers and durations;
     - citation IDs checked against the retrieved set;
     - quoted spans checked as exact substrings.
   - The LLM judge is used only for faithfulness: is every claim supported by the cited text? It is calibrated against about 30 hand-labelled answers, and its agreement rate is reported.

## How the work is organised: checkpoints

### A. Repo skeleton and safety rails (~1–2 h, free)
- `uv` (Python 3.12), ruff, pytest, Makefile, `.gitignore`, `.env.example`, gitleaks pre-commit hook from the first commit.
- `docs/PROGRESS.md` (with a spend log starting at $0), `docs/DECISIONS.md`, `docs/LEARNING.md`, and a `CLAUDE.md` with this project's hard rules.
- A private GitHub repo, created after the owner's OK.
- `docker-compose.yml` skeleton with Postgres + pgvector.

### B. Corpus ingestion (~3–4 h, free)
- `scripts/fetch_corpus.py`: downloads every document in the manifest from CELLAR, checks its sha256, and caches it under `corpus/raw/` (raw files are committed to git; see `DECISIONS.md`).
- `src/rag/parse.py`: turns the XHTML into a tree: act → chapter → article → paragraph → point, plus annexes. Each node gets a stable ID such as `561/2006:art6(3)`, its heading path, its amendment markers, and its outgoing cross-references.
- Guidance-note PDFs are converted to text, with page numbers kept for citations.
- Tests run on saved fixture files: a known article parses into the expected structure, IDs stay the same across runs, and markers are removed from the text but kept in metadata.

### C. Evaluation set, built before any retrieval (~4–5 h, free)
- `data/questions.v1.jsonl`, about 100 questions, each with: gold answer, gold supporting node IDs, question type, *paraphrased/verbatim* tag, and split (dev/test).
- Question types:
  - single fact ("max daily driving time");
  - definition;
  - multi-article (needs two articles);
  - exception or condition ("when may the daily rest be reduced?");
  - cross-act (561/2006 + 2002/15);
  - unanswerable (plausible, but not in the corpus);
  - version trap (the answer changed in 2020).
- Written by hand with LLM help for drafts. Every gold answer is checked by the owner against the text.

### D. Retrieval baseline with keyword search (~3 h, free)
- Chunking strategies as swappable functions:
  - article-level;
  - paragraph-level with the article heading prepended;
  - fixed token windows with overlap.
- BM25 keyword retrieval.
- `src/rag/retrieval_eval.py`: recall@k, MRR and nDCG against the gold node IDs, with Wilson intervals, reusing the stats code from `llm-inference-spike`.
- First results table: BM25 × three chunking strategies, dev split, reported separately for paraphrased and verbatim questions.

### E. Embeddings, vector search, hybrid (~4–5 h, first paid step, gated)
- **Before this step:** a dry-run prints the token count of the whole corpus and the question set, and the estimated cost at that day's Bedrock embedding prices. The owner approves a spending cap (proposed: US$20 for the whole project), and an AWS budget alarm is set first.
- Embeddings via **AWS Bedrock** (Amazon Titan Text Embeddings V2, or Cohere Embed Multilingual if the bilingual extension is planned). Model availability in the chosen EU region is checked and recorded.
  - Zero-cost alternative: a small local embedding model (about 100–500 MB download). That counts as a local model run, so it needs the owner's OK first.
- Embeddings are cached on disk by `sha256(model + text)`, so re-running the evaluation doesn't pay again.
- pgvector in Docker Compose. The corpus is small, so search is **exact** (no approximate index), which makes local and deployed results identical.
- Hybrid search: BM25 + vector results merged with reciprocal rank fusion (RRF).
- Results table extended: BM25 vs vector vs hybrid, × chunking, × with/without cross-reference expansion.

### F. Generation with citations and refusal (~3–4 h, paid, within the cap)
- Claude on Bedrock (the model ID and EU cross-region inference profile are recorded in `DECISIONS.md`).
- The prompt gives the retrieved chunks with their node IDs, tells the model to treat document text as data rather than instructions, and asks for structured output: `{answerable, answer, citations[{node_id, quote}]}`.
- `src/rag/validate.py` checks every answer deterministically:
  - cited IDs must be among the retrieved chunks;
  - quotes must be exact substrings;
  - `answerable: false` must have no citations.
  - An answer that fails is returned as "could not answer reliably", never shown as fact.
- A dry-run mode prints calls and cost, and tests use a mocked client.

### G. Answer evaluation and variant comparison (~4–5 h, paid, within the cap)
- Metrics:
  - answer correctness (exact match for numbers and durations, normalised match otherwise);
  - citation precision and recall against the gold node IDs;
  - refusal accuracy on unanswerable questions, and false refusals on answerable ones;
  - faithfulness (calibrated LLM judge, agreement with hand labels reported);
  - cost, latency and tokens per question.
- Variants:
  - the best 2–3 retrieval setups from E;
  - top-k = 3/5/10;
  - with and without a reranking step (a Bedrock rerank model if available in the region, otherwise LLM reranking).
- Paired comparison between variants on the same questions (sign test), and the final run on the **test split only once**.
- `make tables` regenerates every table from the run logs.

### H. Service and Docker (~3–4 h, free)
- FastAPI app: `POST /ask`, `GET /health`, `GET /ready` (index loaded).
- One JSON log line per request: retrieved IDs, scores, tokens, cost, latency. No secrets.
- Multi-stage `Dockerfile`: small image, non-root user, pinned base image digest.
- `docker compose up` runs the whole stack locally, and a smoke test calls `/ask` against it.

### I. AWS deployment with infrastructure as code (~5–6 h, gated)
- **Before this step:** the owner sets up or approves a personal AWS account, and the budget alarm from E is confirmed.
- **AWS CDK in Python** defines everything:
  - ECR for the image;
  - the service on **Lambda (container image)**, compared in `DECISIONS.md` with ECS Fargate;
  - an IAM role allowed to call only the specific Bedrock models;
  - CloudWatch logs with short retention.
- **Where the index lives is decided in `DECISIONS.md`.** Recommended: an index file built in CI, stored in S3 and loaded into memory, which is cheap and enough for this corpus. The alternatives, RDS with pgvector, or Bedrock Knowledge Bases with OpenSearch Serverless, are realistic at scale but cost money even when idle. Their prices are checked on the day and the trade-off is written down. Retrieval sits behind one interface, and a test checks that the in-memory and pgvector backends return identical rankings.
- `make deploy` and `make destroy`. Leaving nothing running is part of the checklist.

### J. CI/CD with GitHub Actions (~3–4 h)
- On every PR: ruff, pytest, Docker build, and the **retrieval quality gate**. Recall@5 on the dev split must not fall below the recorded baseline. It runs on cached embeddings, so it costs nothing.
- On merge to `main`: build, push to ECR, deploy with CDK, smoke-test `/ready` and one `/ask`.
- AWS access through **GitHub OIDC**: no long-lived AWS keys stored in GitHub.
- Full answer evaluation (paid) is a manual workflow, not run on every PR.

### K. Write-up and demo (~3 h)
- README: what it does, the architecture diagram, the results table, the limitations (small corpus, consolidated texts have no legal effect), and how to run it.
- A 2-minute demo script: a good answer, a refusal, a cross-reference question, and the results table.
- Public release only after a personal-data check, a secret scan, and the owner's approval.

### L. Optional: bilingual (~3–4 h)
- Add the Swedish versions of the same acts, ask questions in Swedish, and cite Swedish text.
- Measure cross-language retrieval: Swedish question → English text, and Swedish → Swedish.

## Hard rules for this project
- No spending without the owner's OK: a dry-run estimate first, then a cap. Every cost goes in the spend log in `PROGRESS.md`.
- No local model download or run without telling the owner what will happen and getting an OK.
- Secrets only in `.env` or AWS, never committed; gitleaks stays enabled.
- Public data only, and the repo stays private until checked and approved.
- `make destroy` after every AWS session unless the owner decides to keep it running.

## Time and cost estimate
- **Time:** about 40–50 hours of working-together time at a learner's pace, so roughly 15–20 sessions of 2–3 h. The biggest learning value is in C (the evaluation set), E (retrieval) and G (evaluation).
- **Money:** embedding the corpus should cost cents. Most of the cost is repeated answer-evaluation runs in F and G, plus AWS resources left running. The cap proposed in E (US$20) is a ceiling, not an expected spend; the real estimate comes from the dry-run with that day's prices.

## Verification ("ready to show" gate)
1. `make check` is green: lint, tests, secret scan, retrieval quality gate.
2. `make corpus` rebuilds the corpus from the manifest with matching hashes.
3. `make tables` regenerates every results table from the raw run logs.
4. The final test-split table shows retrieval and answer metrics with intervals for at least three variants, and the README explains what won and why.
5. Every unanswerable test question is refused or flagged, or the failures are listed and explained.
6. The deployed service answers through its URL, deployed by CI through OIDC. `make destroy` removes everything, and the AWS console shows nothing left.
7. `docs/LEARNING.md` covers every concept used, and the spend log matches the AWS bill.
