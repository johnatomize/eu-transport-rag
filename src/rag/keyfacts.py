"""Whether an answer states a key fact, matched by code with numbers normalised (#29).

Key facts in the question set are written one way: digits, a decimal point and the unit
(`9 hours`, `4.5 hours`, `7.5 tonnes`). Answers write the same values in other ways, and
`states` forgives exactly those:

- number words: "nine hours", "fifty-six hours", "four and a half hours", "a hundred km",
  "half an hour" (`0.5 hours`) and "an hour and a half" (`1.5 hours`);
- "a" or "an" before a unit: "within an hour" states `1 hour`, but not after a word of
  frequency or another unit, so "twice a week" and "9 hours a day" state no `1 week` or `1 day`;
- the act's decimal comma (`7,5 tonnes`), a thousands comma (`1,000 km`) and `½`;
- a number joined to its unit ("a 45-minute break", "a 4.5-hour period"), and a unit in the
  singular or plural, spelled out ("kilometres" for `km`) or short ("9h", "45 mins");
- times written with a colon (`00:00` for `00.00`), dates written month first or with an
  ordinal ("July 1st, 2026" for `1 July 2026`; an ordinal elsewhere, "the 4th week", is not a
  count), and leading or trailing zeros in a number.

Nothing else is forgiven. The fact's words must appear in order, as whole words, so "9 minutes"
doesn't state `9 hours`, "19 hours" or "4.5 hours" doesn't state `9 hours` or `5 hours`, and
"six hours in a row" doesn't state `6 consecutive hours`. Both the answer and the fact go
through the same `tokens`, after `validate.normalise` has dealt with dashes, quote marks and
invisible characters. Choices and reasons: "matching key facts" in docs/DECISIONS.md.
"""

import re
from collections.abc import Iterable

from rag.validate import normalise

_ONES = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {
    word: value * 10
    for value, word in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)
}
_UNITS = {
    **dict.fromkeys(("hour", "hours", "h", "hr", "hrs"), "hour"),
    **dict.fromkeys(("minute", "minutes", "min", "mins"), "minute"),
    **dict.fromkeys(("day", "days"), "day"),
    **dict.fromkeys(("week", "weeks"), "week"),
    **dict.fromkeys(("month", "months"), "month"),
    **dict.fromkeys(("year", "years"), "year"),
    **dict.fromkeys(("tonne", "tonnes"), "tonne"),
    **dict.fromkeys(("km", "kilometre", "kilometres", "kilometer", "kilometers"), "km"),
}
_MONTHS = frozenset(
    "january february march april may june july august september october november december".split()
)
_MONTH = rf"\b(?:{'|'.join(_MONTHS)})\b"
# Before these, "a" or "an" is a rate ("twice a week", "9 hours a day"), not the number 1.
_FREQUENCY = frozenset({"once", "twice", "times", *_UNITS})

_TOKEN = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+")


def _number(token: str) -> str:
    """One spelling for a number: no leading zeros, no trailing zeros after the point."""
    whole, _, fraction = token.partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def _spelled(words: list[str], i: int) -> tuple[float, int] | None:
    """The number spelled out from `words[i]`, and how many words it takes, or None."""
    j = i
    if words[j] in _TENS:
        value = _TENS[words[j]]
        j += 1
        if j < len(words) and 0 < _ONES.get(words[j], 0) < 10:
            value += _ONES[words[j]]
            j += 1
    elif words[j] in _ONES:
        value = _ONES[words[j]]
        j += 1
    else:
        return None
    if j < len(words) and words[j] == "hundred":
        value *= 100
        j += 1
        rest = _spelled(words, j + 1) if words[j : j + 1] == ["and"] else None
        if rest and rest[0] < 100 and rest[0] == int(rest[0]):
            value += rest[0]
            j += 1 + rest[1]
    if words[j : j + 3] == ["and", "a", "half"]:
        return value + 0.5, j + 3 - i
    return value, j - i


def tokens(text: str) -> tuple[str, ...]:
    """The words and numbers of `text`, lower-cased, with numbers and units in one spelling."""
    text = normalise(text)[0].lower()
    text = re.sub(r"(\d)\s*½", r"\1.5", text)
    text = re.sub(r"(\d{1,2}):(\d{2})(?!\d)", r"\1.\2", text)  # 00:00 is 00.00
    text = re.sub(r"(\d),(?=\d{3}(?!\d))", r"\1", text)  # 1,000 km and 1,000,000 km
    text = re.sub(r"(\d),(\d{1,2})(?!\d)", r"\1.\2", text)  # 7,5 tonnes
    # An ordinal only in a date, "1st July" or "July 1st", so "the 4th week" isn't "4 week".
    text = re.sub(rf"(\d)(?:st|nd|rd|th)(?:\s+of)?(?=\s+{_MONTH})", r"\1", text)
    text = re.sub(rf"({_MONTH}\s+\d{{1,2}})(?:st|nd|rd|th)\b", r"\1", text)
    words = _TOKEN.findall(text)

    out: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        if word in ("a", "an") and words[i + 1 : i + 2] == ["hundred"]:
            words[i] = word = "one"  # "a hundred km"
        if (
            word == "half"
            and i + 2 < len(words)
            and words[i + 1] in ("a", "an")
            and words[i + 2] in _UNITS
        ):
            out.append("0.5")  # "half an hour" is not `1 hour`
            i += 2
            continue
        spelled = _spelled(words, i)
        if spelled:
            value, length = spelled
            out.append(_number(f"{value:g}"))
            i += length
            continue
        if (
            word in ("a", "an")
            and i + 1 < len(words)
            and words[i + 1] in _UNITS
            and not (out and out[-1] in _FREQUENCY)
        ):
            out.append("1")
        elif word[0].isdigit():
            out.append(_number(word))
        elif (
            word in _UNITS
            and words[i + 1 : i + 4] == ["and", "a", "half"]
            and out
            and re.fullmatch(r"\d+(?:\.\d+)?", out[-1])
        ):
            # "an hour and a half" is `1.5 hours`, not `1 hour`.
            out[-1] = _number(f"{float(out[-1]) + 0.5:g}")
            out.append(_UNITS[word])
            i += 4
            continue
        else:
            out.append(_UNITS.get(word, word))
        i += 1

    # A date written month first, "July 1 2026", becomes "1 July 2026". The year must follow,
    # so "may" before a number is left alone.
    for k in range(len(out) - 2):
        day, year = out[k + 1], out[k + 2]
        if (
            out[k] in _MONTHS
            and day.isdigit()
            and 1 <= int(day) <= 31
            and re.fullmatch(r"\d{4}", year)
        ):
            out[k], out[k + 1] = day, out[k]
    return tuple(out)


def _contains(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    n = len(needle)
    return n > 0 and any(haystack[k : k + n] == needle for k in range(len(haystack) - n + 1))


def states(answer: str, fact: str) -> bool:
    """Whether `answer` states the key fact `fact`, with numbers and units normalised."""
    return _contains(tokens(answer), tokens(fact))


def missing_facts(answer: str, facts: Iterable[str]) -> list[str]:
    """The key facts `answer` doesn't state, in the order given."""
    found = tokens(answer)
    return [fact for fact in facts if not _contains(found, tokens(fact))]
