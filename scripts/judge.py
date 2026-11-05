"""Judge answers' faithfulness with Claude, or show what that would send and cost (`make judge`).

    judge.py --model M --calibrate          the owner's 30 labelled answers (#82)
    judge.py --model M --log FILE --only q  a few answers of an answer run, to try the judge out
    judge.py --model M --log FILE           every answer of an answer run that passed the check

FILE is an answer run log's name in runs/generation/. Without `--run` nothing is sent and no API
key is needed: it builds each prompt and prints every call it would make, the estimated cost
with the margin, and whether a paid run would be allowed to start. With `--run` it pays, under
the same checks as `make answers`: prices read in the last 7 days, the most the run can cost
within the approved cap, no prompt over max_prompt_tokens, and ANTHROPIC_API_KEY set.

The judge reads the question, the answer sentence by sentence and the full text of each cited
node (`rag.judge`), never the gold answer or the owner's labels. `--calibrate` judges the
answers the labels file names, after checking it names the chosen answers, and doesn't read
the verdicts. The labelled answers, recognised by their text, are judged only by `--calibrate`
until a calibration run exists, and never by `--only`, so trying the judge out never shows it,
or whoever builds it, how it does on them.

A paid run writes its log to runs/judge/, named `calibration-…`, `try-…` or `run-…` by what it
judged. `make tables` compares each calibration run with the owner's labels in
results/faithfulness_judge.md. Adding the run's row to the spend log, naming the file, stays a
manual step.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from rag.corpus import CorpusError, load_manifest
from rag.faithfulness import LABELLED_RUNS, answers, by_id, select, validate_labels
from rag.generate import (
    AnthropicClient,
    GenerateError,
    RunResult,
    estimate,
    estimate_report,
    input_problems,
    new_run,
    paid_holds,
    read_api_key,
    send_all,
    write_run_log,
)
from rag.generation_config import GenerationConfigError
from rag.generation_config import load_generation_config as load_generation
from rag.judge import (
    BLANK,
    EXPECTED_OUTPUT_TOKENS,
    JUDGE_SCHEMA,
    head,
    judgeable,
    make_call,
    prompt_version,
    read_reply,
    run_log,
)
from rag.parse import ParseError, parse_corpus
from rag.questions import validate
from rag.spend import SpendLogError, spending, spending_report

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"
QUESTIONS = ROOT / "data" / "questions.v1.jsonl"
CONFIG = ROOT / "config" / "generation.yaml"
ANSWER_RUNS = ROOT / "runs" / "generation"
LABELS = ROOT / "data" / "faithfulness_labels.v1.jsonl"
RUNS = ROOT / "runs" / "judge"
ENV = ROOT / ".env"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="the judge: a model priced in the config")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--calibrate", action="store_true", help="the owner's labelled answers")
    which.add_argument("--log", help="an answer run log's file name in runs/generation/")
    parser.add_argument("--only", help="with --log: only these questions' answers, q003,q004")
    parser.add_argument("--run", action="store_true", help="send the calls and pay for them")
    args = parser.parse_args(argv)
    if args.only and not args.log:
        parser.error("--only goes with --log")

    try:
        config = load_generation(CONFIG)
        manifest = load_manifest(MANIFEST)
        tree = parse_corpus(manifest, ROOT)
        docs = {doc.id: doc for doc in manifest}
    except (GenerationConfigError, CorpusError, ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.model not in config.prices:
        print(f"error: {args.model} has no price in {CONFIG.name}", file=sys.stderr)
        return 1
    questions, errors = validate(QUESTIONS.read_text(encoding="utf-8"), tree)
    if errors:
        print(f"error: {QUESTIONS.name} is not valid; run `make questions`", file=sys.stderr)
        return 1
    question_of = {q.id: q.question for q in questions}

    try:
        labelled_logs = {name: read_log(name) for name in LABELLED_RUNS}
        texts = answers(select(questions, labelled_logs), labelled_logs)
        if args.calibrate:
            kind, answer_runs = "calibration", list(LABELLED_RUNS)
            picked = [(run, by_id(labelled_logs[run])[qid]) for qid, run in labelled_keys(texts)]
        else:
            kind, answer_runs = ("try" if args.only else "run"), [args.log]
            labelled = {(qid, text) for (qid, _), text in texts.items()}
            calibrated = any(RUNS.glob("calibration-*.json"))
            picked = pick(args.log, args.only, labelled, calibrated)
        calls = [make_call(question_of[e["id"]], e, run, tree, docs) for run, e in picked]
        if not calls:
            raise ValueError("no answer to judge")
        e = estimate(calls, config, args.model, EXPECTED_OUTPUT_TOKENS)
    except (OSError, ValueError, KeyError, GenerateError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(
        f"Judge {args.model}, effort {config.effort}, at most {config.max_output_tokens:,} "
        f"output tokens a judgement. Instructions version {prompt_version()}."
    )
    print(f"Judging {len(calls)} answers ({kind}) from {', '.join(answer_runs)}.")
    print(f"{'question':9} {'answer run':45} {'sentences':>9} {'est. input':>10}")
    for c in calls:
        print(f"{c.question_id:9} {c.answer_run:45} {len(c.sentences):>9} {c.input_tokens:>10,}")
    print(estimate_report(e, unit="judgement"))

    try:
        spent = spending(ROOT)
    except SpendLogError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(spending_report(spent))
    holds = paid_holds(e, config, spent.spent_usd, spent.cap_usd, dt.date.today())
    errors = input_problems(e, config)
    for problem in errors:
        print(f"error: {problem}", file=sys.stderr)

    if not args.run:
        for problem in holds:
            print(f"warning: a paid run would refuse to start: {problem}", file=sys.stderr)
        print("Nothing was sent. To send and pay, add --run (make judge … RUN=1).")
        return 1 if errors else 0
    key = read_api_key(ENV)
    if not key:
        holds.append("ANTHROPIC_API_KEY is not set: add it to .env (see .env.example)")
    for problem in holds:
        print(f"error: {problem}", file=sys.stderr)
    if errors or holds or not key:
        print("Nothing was sent.", file=sys.stderr)
        return 1

    def save(result: RunResult) -> None:
        write_run_log(path, run_log(run, config, args.model, kind, answer_runs, e, result))

    try:  # before anything is sent, so a run that can't be logged never starts
        run, path = new_run(ROOT, RUNS, args.model, prefix=f"{kind}-")
        save(RunResult())
    except OSError as err:
        print(f"error: cannot write the run log ({err}); nothing was sent", file=sys.stderr)
        return 1
    client = AnthropicClient(config, args.model, api_key=key, schema=JUDGE_SCHEMA)

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    try:
        result = send_all(
            calls, client, config.prices[args.model], head, read_reply, BLANK, progress, save
        )
    except OSError as err:
        print(
            f"error: the run log could not be written ({err}); it is printed above. Save it "
            "to runs/judge/ before anything else is run, so the spend is counted.",
            file=sys.stderr,
        )
        return 1
    usable = [a for a in result.answers if a["verdict"]["ok"]]
    supported = sum(a["verdict"]["supported"] for a in usable)
    print(
        f"Judged {len(result.answers)} of {len(calls)}: {supported} supported, "
        f"{len(usable) - supported} not supported, {len(result.answers) - len(usable)} with "
        "no usable verdict."
    )
    print(
        f"Tokens as counted by the API: {result.input_tokens:,} input, "
        f"{result.output_tokens:,} output; US${result.cost_usd:.4f} at list price."
    )
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(
        f"Run log: {shown}. Add a row to the spend log in docs/PROGRESS.md that names {path.name}."
    )
    if kind == "calibration":
        print("Run `make tables` to compare it with the owner's labels.")
    if result.error:
        print(f"error: stopped early: {result.error}", file=sys.stderr)
        return 1
    return 0


def read_log(name: str) -> dict:
    path = ANSWER_RUNS / name
    if path.parent != ANSWER_RUNS or not path.is_file():
        raise ValueError(f"no answer run log {name} in runs/generation/")
    return json.loads(path.read_text(encoding="utf-8"))


def labelled_keys(texts: dict[tuple[str, str], str]) -> list[tuple[str, str]]:
    """The (question_id, run) of every labelled answer, from the labels file, in its order.

    The file must label exactly the chosen answers (`texts`), checked as `make check` checks
    it; its verdicts are not read here.
    """
    if not LABELS.exists():
        raise ValueError(f"no {LABELS.name}: the owner's labels come first (#82)")
    labels, errors = validate_labels(LABELS.read_text(encoding="utf-8"), texts)
    if errors:
        raise ValueError(f"{LABELS.name} is not valid; run `make faithfulness-labels`")
    return [(label.question_id, label.run) for label in labels]


def pick(
    log_name: str, only: str | None, labelled: set[tuple[str, str]], calibrated: bool
) -> list[tuple[str, dict]]:
    """The answers of an answer run to judge: those that passed the check, or with `only`,
    those questions'.

    An answer the owner labelled, recognised by its question and text, is refused with `only`,
    and in a whole run until a calibration run exists.
    """
    entries = by_id(read_log(log_name))
    if only:
        wanted = [i.strip() for i in only.split(",") if i.strip()]
        if twice := sorted({q for q in wanted if wanted.count(q) > 1}):
            raise ValueError(f"--only names {', '.join(twice)} more than once")
        for qid in wanted:
            if qid not in entries or not judgeable(entries[qid]):
                raise ValueError(f"{qid}: not an answer that passed the check in {log_name}")
        out = [(log_name, entries[qid]) for qid in wanted]
    else:
        out = [(log_name, e) for e in entries.values() if judgeable(e)]
    kept = [e["id"] for _, e in out if (e["id"], e["verdict"]["answer"]) in labelled]
    if kept and (only or not calibrated):
        raise ValueError(
            f"{', '.join(kept)}: answers the owner labelled, judged only with --calibrate"
            + ("" if only else " until a calibration run exists")
        )
    return out


if __name__ == "__main__":
    sys.exit(main())
