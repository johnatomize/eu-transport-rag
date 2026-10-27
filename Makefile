# Developer commands. Needs uv and gitleaks on PATH (brew install uv gitleaks).
# The db-* targets also need Docker with Compose and a running Colima VM (see CLAUDE.md).
# Later checkpoints add targets (corpus, tables, deploy, destroy) and extend check.

.PHONY: help install lint format test secrets check db-up db-check db-down gitleaks-present docker-present

help: ## List the targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install: gitleaks-present ## Create .venv from uv.lock and install the pre-commit hook
	uv sync --locked
	uv run --locked pre-commit install

lint: ## Ruff lint and format check
	uv run --locked ruff check .
	uv run --locked ruff format --check .

format: ## Apply ruff fixes and formatting
	uv run --locked ruff check --fix .
	uv run --locked ruff format .

test: ## Run the test suite
	uv run --locked pytest

secrets: gitleaks-present ## Scan the full git history for secrets
	gitleaks git --redact --no-banner .

check: lint test secrets ## Everything that must pass before a pull request

db-up: docker-present ## Start local Postgres + pgvector and wait until it is healthy
	docker compose up -d --wait

db-check: db-up ## Enable pgvector in the local database and print its version
	docker compose exec -T db psql -v ON_ERROR_STOP=1 -qtA \
		-c "SET client_min_messages = warning" \
		-c "CREATE EXTENSION IF NOT EXISTS vector" \
		-c "SELECT 'pgvector ' || extversion FROM pg_extension WHERE extname = 'vector'"

db-down: docker-present ## Stop local Postgres (data is kept in a volume)
	docker compose down

gitleaks-present:
	@command -v gitleaks >/dev/null || { echo "gitleaks not found on PATH: brew install gitleaks" >&2; exit 1; }

docker-present:
	@command -v docker >/dev/null || { echo "docker not found on PATH: brew install colima docker docker-compose" >&2; exit 1; }
	@docker compose version >/dev/null 2>&1 || { echo "docker compose not found: add the Compose plugin directory to ~/.docker/config.json (see CLAUDE.md)" >&2; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "Docker is not running: colima start" >&2; exit 1; }
