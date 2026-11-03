"""`make ask`, the whole chain for one typed question, with stand-ins for Claude and for the
local embedding model: nothing here reaches the network or runs a model."""

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

from rag.embed import EmbeddingCache, EmbedError
from rag.embed_estimate import EmbeddingConfig
from rag.generate import Reply
from rag.prompt import LEGAL_NOTE, NOT_COVERED, UNRELIABLE, Prompt
from rag.spend import Spending, unlogged_runs

ROOT = Path(__file__).parents[1]
WEEKLY = "How long may a driver drive in a week?"
_NODE = re.compile(r'<node id="([^"]+)">\n([^\n]+)')


class Claude:
    """A stand-in for Claude that writes its reply from the prompt it is sent."""

    def __init__(self, write):
        self.write = write
        self.sent: list[Prompt] = []

    def send(self, prompt: Prompt) -> Reply:
        self.sent.append(prompt)
        text = json.dumps(self.write(prompt))
        return Reply(text, "end_turn", None, 2000, 400, 0, 0, "claude-haiku-5-5", "req_1")


def quoting_first_node(prompt: Prompt) -> dict:
    """An answer that quotes the first line of the first node in the prompt, as a model should."""
    node_id, line = _NODE.search(prompt.user).groups()
    return {
        "answerable": True,
        "answer": "At most 56 hours.",
        "citations": [{"node_id": node_id, "quote": line}],
    }


@pytest.fixture
def ask(monkeypatch, tmp_path):
    """scripts/ask.py, with its run logs in tmp_path, and no Claude or local model to be had."""
    spec = importlib.util.spec_from_file_location("ask_script", ROOT / "scripts" / "ask.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.built = []
    module.loaded = []

    def no_client(*args, **kwargs):
        module.built.append(args)
        raise AssertionError("a client was built")

    class NoModel:
        @staticmethod
        def load(config):
            module.loaded.append(config)
            raise AssertionError("the local model was loaded")

    monkeypatch.setattr(module, "AnthropicClient", no_client)
    monkeypatch.setattr(module, "LocalClient", NoModel)
    monkeypatch.setattr(module, "RUNS", tmp_path / "ask")
    monkeypatch.setattr(module, "ENV", tmp_path / ".env")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return module


def pay(ask, monkeypatch, client) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(ask, "AnthropicClient", lambda *a, **k: client)


def test_without_the_flag_it_shows_the_passages_and_the_cost_and_calls_nothing(ask, capsys):
    assert ask.main(["--retriever", "bm25", WEEKLY]) == 0
    out = capsys.readouterr().out
    assert ask.built == [] and ask.loaded == []
    assert "Retrieval: bm25 over paragraph chunks, top 5." in out
    assert len(re.findall(r"^\d\. +[\d.]+  paragraph:\S+$", out, re.MULTILINE)) == 5
    assert re.search(r"Prompt version \w{12}: [\d,]+ characters, about [\d,]+ input tokens", out)
    assert "1 call: about" in out and "Most it can cost" in out
    assert "Nothing was sent" in out
    assert not ask.RUNS.exists()


def test_a_question_from_the_question_set_uses_its_cached_vector(ask, capsys):
    rows = (json.loads(line) for line in (ROOT / "data" / "questions.v1.jsonl").open())
    question = next(row["question"] for row in rows if row["split"] == "dev")
    assert ask.main([question]) == 0
    assert ask.loaded == []
    assert "Retrieval: hybrid over paragraph chunks" in capsys.readouterr().out


def test_a_new_question_is_embedded_by_the_local_model_and_never_saved(ask, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    class FakeModel:
        def embed(self, text):
            assert text.endswith("\nQuery:" + WEEKLY.replace("week", "fortnight"))
            return [1.0] * 1024, 12

    def load(config):
        assert os.environ["HF_HUB_OFFLINE"] == "1"  # so nothing can be downloaded
        return FakeModel()

    monkeypatch.setattr(ask.LocalClient, "load", staticmethod(load))

    def no_save(self):
        raise AssertionError("the embedding cache was saved")

    monkeypatch.setattr(EmbeddingCache, "save", no_save)
    assert ask.main([WEEKLY.replace("week", "fortnight")]) == 0
    captured = capsys.readouterr()
    assert "nothing is downloaded or saved" in captured.err
    assert "Retrieval: hybrid" in captured.out and ask.built == []


def test_a_local_model_that_fails_is_reported_not_raised(ask, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    def load(config):
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(ask.LocalClient, "load", staticmethod(load))
    assert ask.main(["What is a fortnight?"]) == 1
    assert "could not embed the question: MPS backend out of memory" in capsys.readouterr().err


def test_a_question_is_never_embedded_by_a_paid_model(ask):
    paid = EmbeddingConfig(
        provider="bedrock",
        model_id="amazon.titan-embed-text-v2:0",
        dimensions=4,
        normalize=True,
        max_input_tokens=8000,
    )
    with pytest.raises(EmbedError, match="would be paid for"):
        ask.embed_question("anything", EmbeddingCache(Path("unused"), paid))


def test_a_valid_answer_is_shown_with_its_sources_versions_and_quotes(ask, monkeypatch, capsys):
    claude = Claude(quoting_first_node)
    pay(ask, monkeypatch, claude)
    assert ask.main(["--run", "--retriever", "bm25", WEEKLY]) == 0
    out = capsys.readouterr().out
    node_id, line = _NODE.search(claude.sent[0].user).groups()
    assert "At most 56 hours.\n\nSources:\n1. Regulation (EC) No 561/2006, Article " in out
    assert "consolidated version of" in out
    assert f"\N{LEFT DOUBLE QUOTATION MARK}{line.strip()}" in out
    assert LEGAL_NOTE in out

    (path,) = ask.RUNS.glob("*.json")
    assert path.name.startswith("ask-") and path.name.endswith("-claude-haiku-5-5.json")
    log = json.loads(path.read_text(encoding="utf-8"))
    assert log["question"] == WEEKLY
    assert log["config"]["split"] == "typed"
    assert log["config"]["retrieval"] == {"retriever": "bm25", "strategy": "paragraph", "k": 5}
    (entry,) = log["answers"]
    assert entry["verdict"]["ok"] and entry["verdict"]["citations"][0]["node_id"] == node_id
    assert f"names {path.name}" in out
    # Until the spend log names it, the next paid run counts what it cost.
    assert unlogged_runs("# Progress\n\n## Spend log\n", ask.RUNS) == [
        (path.name, log["paid"]["cost_usd"])
    ]


def test_an_answer_citing_a_node_that_was_not_retrieved_is_not_shown(ask, monkeypatch, capsys):
    def citing_elsewhere(prompt):
        reply = quoting_first_node(prompt)
        reply["citations"][0]["node_id"] = "2002/15:art3(a)"
        return reply

    pay(ask, monkeypatch, Claude(citing_elsewhere))
    assert ask.main(["--run", "--retriever", "bm25", WEEKLY]) == 0
    out = capsys.readouterr().out
    assert UNRELIABLE in out
    assert "These passages may be relevant: Regulation (EC) No 561/2006" in out
    assert "2002/15:art3(a)): not a node of the retrieved passages" in out
    assert "At most 56 hours." not in out and "Sources:" not in out


def test_a_refusal_is_shown_as_a_refusal(ask, monkeypatch, capsys):
    refusal = {"answerable": False, "answer": "Nothing on ferry fares.", "citations": []}
    pay(ask, monkeypatch, Claude(lambda prompt: refusal))
    assert ask.main(["--run", "--retriever", "bm25", "What does a ferry ticket cost?"]) == 0
    out = capsys.readouterr().out
    assert NOT_COVERED in out
    assert "Nothing on ferry fares." not in out  # the model's own explanation is never shown


def test_a_paid_question_over_the_cap_or_without_a_key_is_not_sent(ask, monkeypatch, capsys):
    assert ask.main(["--run", "--retriever", "bm25", WEEKLY]) == 1
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(ask, "spending", lambda root: Spending(20.0, 19.9999, ()))
    assert ask.main(["--run", "--retriever", "bm25", WEEKLY]) == 1
    assert "could pass the approved cap" in capsys.readouterr().err
    assert ask.built == [] and not ask.RUNS.exists()


def test_an_empty_question_is_refused(ask, capsys):
    assert ask.main(["  "]) == 1
    assert "the question is empty" in capsys.readouterr().err
