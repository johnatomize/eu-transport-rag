import json
from pathlib import Path

import pytest

from rag.keyfacts import missing_facts, states, tokens

QUESTIONS = Path(__file__).parents[1] / "data" / "questions.v1.jsonl"


@pytest.mark.parametrize(
    ("answer", "fact"),
    [
        # Number words, as the acts and the models write them.
        ("within seven calendar days of the card being stolen", "7 calendar days"),
        ("night time is a period of at least four hours", "4 hours"),
        ("After a driving period of four and a half hours, take a break", "4.5 hours"),
        ("a break of forty-five minutes", "45 minutes"),
        ("no more than fifty six hours a week", "56 hours"),
        ("a radius of one hundred kilometres", "100 km"),
        ("a radius of one hundred and fifty km", "150 km"),
        ("at most three reduced daily rest periods", "3 reduced daily rest periods"),
        ("You may not work more than six consecutive hours", "6 consecutive hours"),
        ("over any four consecutive weeks", "4 consecutive weeks"),
        ("interruptions totalling no more than one hour", "1 hour"),
        ("by up to an hour", "1 hour"),
        ("keep them for at least a year after use", "1 year"),
        ("a radius of a hundred km", "100 km"),
        ("an hour and a half", "1.5 hours"),
        ("half an hour", "0.5 hours"),
        # The act's decimal comma, a thousands comma and ½.
        ("the vehicle is 7,5 tonnes or less", "7.5 tonnes"),
        ("over 2,5 tonnes", "2.5 tonnes"),
        ("within 1,000 km", "1000 km"),
        ("after 4½ hours", "4.5 hours"),
        # A number joined to its unit, and singular against plural.
        ("you must take a 45-minute break", "45 minutes"),
        ("after 4.5-hour driving periods", "4.5 hours"),
        ("which covers your one-hour trip", "1 hour"),
        ("an uninterrupted 30-minute break", "30 minutes"),
        ("within 12 consecutive twenty-four-hour periods", "12 consecutive 24-hour periods"),
        ("within 100 kilometres of the base", "100 km"),
        ("max 9h a day", "9 hours"),
        ("a 45 mins break", "45 minutes"),
        ("at least 11 hrs", "11 hours"),
        # Times and dates.
        ("from 00:00 on Monday", "00.00 on Monday"),
        ("until 24:00 on Sunday", "24.00 on Sunday"),
        ("from July 1st, 2026", "1 July 2026"),
        ("since 2nd February 2022", "2 February 2022"),
        ("since the 2nd of February 2022", "2 February 2022"),
        ("within 1,000,000 km", "1000000 km"),
        # Facts already written as the question set writes them.
        ("at most 10 hours, and not more than twice during the week", "twice"),
        ("must not exceed 90 hours", "90 hours"),
    ],
)
def test_states_forgives_other_ways_of_writing_a_number(answer, fact):
    assert states(answer, fact)


@pytest.mark.parametrize(
    ("answer", "fact"),
    [
        # The wrong unit.
        ("a break of 9 minutes", "9 hours"),
        ("a break of nine minutes", "9 hours"),
        ("within seven working days", "7 calendar days"),
        # A number that only contains the fact's digits, or is close to it.
        ("no more than 19 hours", "9 hours"),
        ("after 4.5 hours", "5 hours"),
        ("at least 14 hours", "4 hours"),
        ("at most 11 hours", "1 hour"),
        ("7.5 tonnes", "75 tonnes"),
        ("over 2.5 tonnes", "2 tonnes"),
        # "a" as a rate, not the number 1.
        ("at most twice a week", "1 week"),
        ("up to 9 hours a day", "1 day"),
        ("three times a year", "1 year"),
        # Other words are not forgiven: only numbers and units are normalised.
        ("You may not work more than six hours in a row", "6 consecutive hours"),
        ("at least four weekly rests", "4 weekly rest periods"),
        ("from midnight on Sunday", "24.00 on Sunday"),
        # Half an hour, or an hour and a half, is not one hour.
        ("a break of half an hour", "1 hour"),
        ("after an hour and a half", "1 hour"),
        # An ordinal outside a date is not the number.
        ("in the 4th consecutive week", "4 consecutive weeks"),
        ("during the 2nd hour", "2 hours"),
        # A month before a number that isn't a date.
        ("A driver may 2 hours", "2 May"),
        # An empty answer states nothing.
        ("", "9 hours"),
    ],
)
def test_states_refuses_a_different_fact(answer, fact):
    assert not states(answer, fact)


def test_tokens_spell_numbers_and_units_one_way():
    assert tokens("Four and a half HOURS, then a 45-minute break") == (
        "4.5",
        "hour",
        "then",
        "a",
        "45",
        "minute",
        "break",
    )
    assert tokens("July 1, 2026") == ("1", "july", "2026")
    assert tokens("00.00") == tokens("0") == ("0",)
    assert tokens("4.50 hours") == ("4.5", "hour")


def test_missing_facts_lists_the_facts_not_stated_in_order():
    answer = "Daily driving is limited to nine hours, or ten hours at most twice a week."
    assert missing_facts(answer, ["9 hours", "56 hours", "10 hours", "twice", "90 hours"]) == [
        "56 hours",
        "90 hours",
    ]


def test_every_key_fact_in_the_question_set_states_itself():
    facts = {
        fact
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
        for fact in json.loads(line).get("key_facts") or []
    }
    assert len(facts) > 40
    for fact in facts:
        assert tokens(fact), fact
        assert states(f"In short: {fact}.", fact), fact


@pytest.mark.parametrize(
    ("answer", "facts", "missing"),
    [
        # From F's dev runs (#66), where a plain substring match missed a stated fact.
        (
            "You must apply for a replacement card within seven calendar days.",
            ["7 calendar days"],
            [],
        ),
        (
            "the vehicle or combination is 7,5 tonnes maximum permissible mass or less",
            ["7.5 tonnes"],
            [],
        ),
        (
            "On an occasional passenger service, the 45-minute break due after 4.5 hours of "
            "driving can be replaced by two breaks of at least 15 minutes each.",
            ["45 minutes", "4.5 hours", "15 minutes"],
            [],
        ),
        (
            "You may exceed your daily and weekly driving time by up to one hour to reach your "
            "home. If you first take an uninterrupted 30-minute break immediately before the "
            "extra driving, you may go up to two hours over.",
            ["1 hour", "2 hours", "30 minutes"],
            [],
        ),
        # And where it rightly missed one: the answer says the figure isn't in the passages.
        (
            "56 hours is the absolute weekly driving maximum. The working-time figure itself is "
            "not in these passages.",
            ["56 hours", "48 hours", "60 hours"],
            ["48 hours", "60 hours"],
        ),
    ],
)
def test_real_answers_from_the_first_paid_run(answer, facts, missing):
    assert missing_facts(answer, facts) == missing
