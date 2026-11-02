# Developer commands. Needs uv and gitleaks on PATH (brew install uv gitleaks).
# The db-* targets also need Docker with Compose and a running Colima VM (see CLAUDE.md).
# Later checkpoints add targets (tables, deploy, destroy) and extend check.

.PHONY: help install lint format test secrets check corpus parse chunks search questions retrieval-runs embed-dry-run embed tables tables-check db-up db-check db-down test-db vector-index gitleaks-present docker-present

help: ## List the targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

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

check: lint test questions tables-check secrets ## Everything that must pass before a pull request

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

embed-dry-run: ## Print what embedding the corpus and the questions would cost; calls nothing
	uv run --locked python scripts/embed_dry_run.py

embed: ## Report which chunks and questions lack a cached vector and what sending them would cost; SEND=1 pays
	uv run --locked python scripts/embed.py $(if $(SEND),--send,)

vector-index: db-up ## Load chunks and their cached vectors into local pgvector: make vector-index [STRATEGY=paragraph]
	uv run --locked python scripts/vector_index.py $(if $(STRATEGY),--strategy $(STRATEGY),)

tables: ## Rebuild results/retrieval.md from the run logs alone
	uv run --locked python scripts/make_tables.py

tables-check: ## Fail if results/retrieval.md doesn't match the run logs
	uv run --locked python scripts/make_tables.py --check

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
