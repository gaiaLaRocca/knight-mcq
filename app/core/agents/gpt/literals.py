"""Literal values: dates and measurements as KG nodes.

Literal values take their chunks from the head entity, verified (docs/phase1_kb_quality_plan.md,
"Work plan for test_8"). A date or a measurement has no Wikipedia page of its own, yet it is
the answer of the most factual questions (when did the bridge open, how tall is it). Looked
up like an entity it lands on a page chosen by chance: on test_7 `may 27, 1937` got the page
*1937*, scored 0.18 at the gate and was dropped; on test_6 `746 feet` got the topic's page,
which the topic-page exclusion now forbids. So a literal is never looked up: it is checked
against the text of the head of its triplet, and kept only if its value appears there.

This module holds the pure part: recognising a literal, its canonical name, and finding its
value in a text. Where the check runs is described in `process_triplet` (chatbot.py) and in
the thesis runner, which holds the recorded chunks.
"""
import re
from nltk.tokenize import sent_tokenize

_MONTHS = (
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
)
_MONTH = "(?:" + "|".join(_MONTHS) + ")"
_DAY = r"\d{1,2}"
_YEAR = r"(?:1\d{3}|20\d{2})"
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_SCALE = r"(?:thousand|million|billion)"

# Canonical unit -> spellings. Canonical forms are used only for matching, never written to a
# node name: "746 feet" must match "746 ft (227 m)" and "746-foot".
_UNITS = {
    "ft": ("feet", "foot", "ft"),
    "mi": ("miles", "mile", "mi"),
    "m": ("meters", "meter", "metres", "metre", "m"),
    "km": ("kilometers", "kilometer", "kilometres", "kilometre", "km"),
    "inch": ("inches", "inch"),  # never "in": it would match the preposition
    "yd": ("yards", "yard", "yd"),
    "acre": ("acres", "acre"),
    "ha": ("hectares", "hectare", "ha"),
    "ton": ("tons", "ton"),
    "tonne": ("tonnes", "tonne"),
    "lb": ("pounds", "pound", "lbs", "lb"),
    "kg": ("kilograms", "kilogram", "kg"),
    "square": ("square", "sq"),
    "cubic": ("cubic", "cu"),
    "%": ("percent", "%"),
}
_CANONICAL_UNIT = {spelling: canon for canon, spellings in _UNITS.items() for spelling in spellings}
_UNIT = "(?:" + "|".join(
    sorted((s for c, ss in _UNITS.items() if c not in ("square", "cubic") for s in ss), key=len, reverse=True)
) + ")"

_FULL_DATE = re.compile(rf"^{_MONTH} {_DAY},? {_YEAR}$|^{_DAY} {_MONTH},? {_YEAR}$")
_MONTH_YEAR = re.compile(rf"^{_MONTH},? {_YEAR}$")
_BARE_YEAR = re.compile(rf"^{_YEAR}$")
_MEASUREMENT = re.compile(rf"^{_NUMBER}(?: {_SCALE})?(?: (?:square|sq|cubic|cu))? ?{_UNIT}$")


def is_literal(name):
    """True if the whole (normalised) name is a date, a month and year, a year, or a number
    with a unit.

    A pattern on the whole name, never "contains a digit": routes such as `us route 101` and
    `california state route 1` carry digits and are entities. Vague periods
    (`late 19th century`) and expressions (`1.7 * 5280 feet`) stay ordinary terms and follow
    page selection. Checked on the 27 digit-bearing node names of test_1-test_7.
    """
    return bool(
        _FULL_DATE.match(name) or _MONTH_YEAR.match(name)
        or _BARE_YEAR.match(name) or _MEASUREMENT.match(name)
    )


def canonical_literal(name):
    """Drop the commas of a literal, so that one value is one node.

    test_7 held `may 27, 1937` beside `may 27 1937`, earlier runs `82,116 acres` beside
    `82116 acres`. Called by `normalize_node_name` on names that are literals, so it only ever
    sees one: a comma elsewhere is part of the name (`washington, dc`).
    """
    return re.sub(r"\s+", " ", name.replace(",", " ")).strip() if _FULL_DATE.match(name) \
        else name.replace(",", "")


def _tokens(text):
    """Lowercase word and number tokens, units canonicalised, thousands separators removed.

    Punctuation is dropped, except a dot inside a number (`1.7`); a hyphen separates
    (`746-foot` -> `746 ft`).
    """
    text = text.lower().replace("-", " ")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    return [_CANONICAL_UNIT.get(t, t) for t in re.findall(r"\d+(?:\.\d+)?|[a-z]+|%", text)]


def _value_pattern(name):
    """Regex over a token string (`_tokens` joined by spaces) that finds the literal's value.

    A number with a unit and a year must appear as the same contiguous tokens. A full date is
    accepted in either order (`may 27 1937`, `27 may 1937`); a month and year also inside a
    full date (`may 1937` is found in "May 27, 1937").
    """
    toks = _tokens(name)
    if _FULL_DATE.match(name):
        month = next(t for t in toks if t in _MONTHS)
        day, year = [t for t in toks if t != month]
        body = rf"{month} {day} {year}|{day} {month} {year}"
    elif _MONTH_YEAR.match(name):
        month, year = toks
        body = rf"{month} (?:\d{{1,2}} )?{year}"
    else:
        body = " ".join(re.escape(t) for t in toks)
    return re.compile(rf"(?<!\S)(?:{body})(?!\S)")


def find_literal_evidence(name, chunks):
    """The sentence of `chunks` that states the literal's value, or None if no chunk does.

    The value must appear within one chunk. The sentence is returned verbatim because it
    becomes the literal's node description: informative in the 150 characters the QA
    validator reads, and containing nothing that is not in the KB. When the value spans a
    sentence boundary (an abbreviation such as "ft." misread as a full stop), the chunk is
    returned instead.
    """
    pattern = _value_pattern(name)
    for chunk in chunks:
        if not pattern.search(" ".join(_tokens(chunk))):
            continue
        for sentence in sent_tokenize(chunk):
            if pattern.search(" ".join(_tokens(sentence))):
                return sentence.strip()
        return chunk.strip()
    return None
