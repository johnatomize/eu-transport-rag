"""The connection to the local Postgres + pgvector database from `docker-compose.yml`.

Settings come from the environment, or else from `.env`, the file Docker Compose reads too:
POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB and POSTGRES_PORT (default 5432). The database is
reached on 127.0.0.1 only, as Compose publishes it. The password is never printed.
"""

import os
import re
from pathlib import Path

import psycopg

HOST = "127.0.0.1"


class DatabaseError(Exception):
    pass


def read_env_file(path: Path) -> dict[str, str]:
    """`KEY=VALUE` lines of a `.env` file, read as Docker Compose reads them.

    Blank lines and `#` comments are skipped, as is a leading `export `. A value in quotes is
    taken as it stands; otherwise a `#` after a space starts a comment.
    """
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ("'", '"') and value[0] in value[1:]:
            value = value[1 : value.index(value[0], 1)]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        values[key.strip()] = value
    return values


def settings(env_file: Path) -> dict[str, str]:
    """Connection settings for psycopg, the environment taking precedence over `env_file`."""
    file = read_env_file(env_file)

    def get(key: str, default: str | None = None) -> str:
        value = os.environ.get(key) or file.get(key) or default
        if not value:
            raise DatabaseError(f"{key} is not set: copy .env.example to .env and fill it in")
        return value

    return {
        "host": HOST,
        "port": get("POSTGRES_PORT", "5432"),
        "user": get("POSTGRES_USER"),
        "password": get("POSTGRES_PASSWORD"),
        "dbname": get("POSTGRES_DB"),
    }


def connect(env_file: Path) -> psycopg.Connection:
    """An autocommit connection; work that must be atomic opens `conn.transaction()`."""
    s = settings(env_file)
    try:
        return psycopg.connect(**s, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError as e:
        # The message can name the user and database, never the password.
        first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        raise DatabaseError(
            f"cannot connect to Postgres on {HOST}:{s['port']} ({first}); is it up (make db-up), "
            "and do the settings in .env match the ones it was created with?"
        ) from e
