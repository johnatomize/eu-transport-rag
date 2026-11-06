# Developer commands. Needs uv and gitleaks on PATH (brew install uv gitleaks).
# The db-* targets also need Docker with Compose and a running Colima VM (see CLAUDE.md).
# Later checkpoints add targets (tables, deploy, destroy) and extend check.

.PHONY: help install lint format test secrets check corpus parse chunks search questions retrieval-runs embed-dry-run embed answers ask judge tables tables-check faithfulness-labels faithfulness-review db-up db-check db-down test-db vector-index gitleaks-present docker-present

help: ## List the targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

install: gitleaks-present ## Create .venv from uv.lock and install the pre-commit hook
	uv sync --locked
	uv run --locked pre-commit install

lint: ## Ruff lint and format check
	uv run --locked ruff check .
	uv run --locked ruff format --check .

format: ## Apply ruff fixes and formatting
	uv run --locked ruff check --fix .
	uv run --locked ruff format .

test: ## Run the test suite, without the tests that need the database
	uv run --locked pytest

secrets: gitleaks-present ## Scan the full git history for secrets
	gitleaks git --redact --no-banner .

check: lint test questions tables-check faithfulness-labels secrets ## Everything that must pass before a pull request

corpus: ## Check corpus/raw against the manifest, downloading missing files
	uv run --locked python scripts/fetch_corpus.py

parse: ## Parse the acts and guidance notes into node trees and report counts, references and anything unparsed
	uv run --locked python scripts/parse_corpus.py --out build/nodes.jsonl

chunks: ## Cut the corpus into chunks with every strategy, report their sizes and write build/chunks-*.jsonl
	uv run --locked python scripts/chunk_corpus.py --out build

search: export Q := $(Q)
search: ## Print the best chunks for a question: make search Q="How long may I drive in a day?"
	@test -n "$$Q" || { echo 'usage: make search Q="your question" [STRATEGY=paragraph] [K=5] [REFS=1]' >&2; exit 1; }
	uv run --locked python scripts/search.py --strategy $(or $(STRATEGY),paragraph) -k $(or $(K),5) $(if $(filter 1,$(REFS)),--refs,) "$$Q"

retrieval-runs: ## Run every retrieval configuration on the dev split and write runs/retrieval/*.json
	uv run --locked python scripts/retrieval_runs.py

embed-dry-run: ## Print what embedding the corpus and the questions would take (and cost, on Bedrock); runs nothing
	uv run --locked python scripts/embed_dry_run.py

embed: ## Report which chunks and questions lack a cached vector; RUN=1 embeds them with the configured model
	uv run --locked $(if $(RUN),--group local-embed,) python scripts/embed.py $(if $(RUN),--run,)

answers: ## Show what answering the dev questions with Claude would send and cost; RUN=1 sends and pays, BATCH=1 at half price
	uv run --locked python scripts/answer.py $(if $(RUN),--run,) $(if $(BATCH),--batch,) $(if $(ROLE),--role $(ROLE),) $(if $(ONLY),--only $(ONLY),)

ask: export Q := $(Q)
ask: ## Answer one typed question: make ask Q="…" shows the passages and cost; RUN=1 sends and pays
	@test -n "$$Q" || { echo 'usage: make ask Q="your question" [RUN=1] [ROLE=scored] [RETRIEVER=hybrid|bm25|vector] [STRATEGY=paragraph] [K=5]' >&2; exit 1; }
	HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error uv run --locked $(if $(filter bm25,$(RETRIEVER)),,--group local-embed) python scripts/ask.py \
		$(if $(RUN),--run,) $(if $(ROLE),--role $(ROLE),) $(if $(RETRIEVER),--retriever $(RETRIEVER),) \
		$(if $(STRATEGY),--strategy $(STRATEGY),) $(if $(K),-k $(K),) "$$Q"

judge: ## Judge faithfulness: make judge LOG=<answer run> [ONLY=q003,q004], or CALIBRATE=1 MODEL=…; RUN=1 sends and pays, BATCH=1 at half price
	@test -n "$(LOG)$(CALIBRATE)" || { echo 'usage: make judge (LOG=<file in runs/generation/> [ONLY=q003,q004] | CALIBRATE=1) [MODEL=claude-haiku-5-5] [BATCH=1] [RUN=1]' >&2; exit 1; }
	uv run --locked python scripts/judge.py $(if $(MODEL),--model $(MODEL),) $(if $(CALIBRATE),--calibrate,) \
		$(if $(LOG),--log $(LOG),) $(if $(ONLY),--only $(ONLY),) $(if $(BATCH),--batch,) $(if $(RUN),--run,)

vector-index: db-up ## Load chunks and their cached vectors into local pgvector: make vector-index [STRATEGY=paragraph]
	uv run --locked python scripts/vector_index.py $(if $(STRATEGY),--strategy $(STRATEGY),)

tables: ## Rebuild results/retrieval.md and results/answers.md from the run logs alone
	uv run --locked python scripts/make_tables.py

tables-check: ## Fail if a results table doesn't match the run logs
	uv run --locked python scripts/make_tables.py --check

faithfulness-labels: ## Check data/faithfulness_labels.v1.jsonl against the answers chosen for labelling (passes before labelling)
	uv run --locked python scripts/faithfulness_labels.py

faithfulness-review: ## Build the page for labelling answers' faithfulness: build/faithfulness-review/
	uv run --locked python scripts/faithfulness_labels.py --page $(if $(ARTIFACT),--artifact,)

questions: ## Validate data/questions.v1.jsonl against its schema and the parsed corpus, and count it
	uv run --locked python scripts/validate_questions.py

db-up: docker-present ## Start local Postgres + pgvector and wait until it is healthy
	docker compose up -d --wait

db-check: db-up ## Enable pgvector in the local database and print its version
	docker compose exec -T db psql -v ON_ERROR_STOP=1 -qtA \
		-c "SET client_min_messages = warning" \
		-c "CREATE EXTENSION IF NOT EXISTS vector" \
		-c "SELECT 'pgvector ' || extversion FROM pg_extension WHERE extname = 'vector'"

test-db: db-up ## Run the tests that need the local database (make test leaves them out)
	uv run --locked pytest -m db

db-down: docker-present ## Stop local Postgres (data is kept in a volume)
	docker compose down

gitleaks-present:
	@command -v gitleaks >/dev/null || { echo "gitleaks not found on PATH: brew install gitleaks" >&2; exit 1; }

docker-present:
	@command -v docker >/dev/null || { echo "docker not found on PATH: brew install colima docker docker-compose" >&2; exit 1; }
	@docker compose version >/dev/null 2>&1 || { echo "docker compose not found: add the Compose plugin directory to ~/.docker/config.json (see CLAUDE.md)" >&2; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "Docker is not running: colima start" >&2; exit 1; }
