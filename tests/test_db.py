"""Database settings, read from .env as Docker Compose reads them; no database needed."""

import pytest

from rag.db import DatabaseError, read_env_file, settings


def test_reads_env_lines_as_compose_does(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "POSTGRES_USER=rag\n"
        "POSTGRES_PASSWORD=secret  # local only\n"
        "export POSTGRES_DB=rag_db\n"
        "QUOTED='has # inside'  # and a comment\n"
        'DOUBLE="x y"\n'
        "HASH=a#b\n"
        "EMPTY=\n",
        encoding="utf-8",
    )
    assert read_env_file(env) == {
        "POSTGRES_USER": "rag",
        "POSTGRES_PASSWORD": "secret",
        "POSTGRES_DB": "rag_db",
        "QUOTED": "has # inside",
        "DOUBLE": "x y",
        "HASH": "a#b",
        "EMPTY": "",
    }


def test_the_environment_wins_and_missing_settings_are_named(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("POSTGRES_USER=rag\nPOSTGRES_PASSWORD=x\nPOSTGRES_DB=rag\n", encoding="utf-8")
    for key in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "POSTGRES_PORT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    s = settings(env)
    assert (s["host"], s["port"], s["user"], s["dbname"]) == ("127.0.0.1", "5433", "rag", "rag")
    env.write_text("POSTGRES_USER=rag\nPOSTGRES_DB=rag\n", encoding="utf-8")
    with pytest.raises(DatabaseError, match="POSTGRES_PASSWORD is not set"):
        settings(env)
