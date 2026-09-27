import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app.core.utils.wikipedia_lookup as wl

TOPIC = "Golden Gate Bridge"

# Scores logged on test_7 for `golden gate strait`: the topic's page was in the entity band
# and won the topic tie-break.
TITLE_SCORES = {"Golden Gate": 0.817, "Golden Gate Bridge": 0.763, "Golden Gate (film)": 0.50}
TOPIC_SCORES = {"Golden Gate": 0.8182, "Golden Gate Bridge": 0.8521, "Golden Gate (film)": 0.10}


def _lookup(term):
    """Run the ranked selection on the three candidates above, without the network."""
    with patch.object(wl.wikipedia, "search", return_value=list(TITLE_SCORES)), \
         patch.object(wl.wikipedia, "page", side_effect=lambda t, auto_suggest: SimpleNamespace(title=t)), \
         patch.object(wl, "_fetch_and_select", side_effect=lambda t, *a, **k: ([f"chunk of {t}"], t)), \
         patch.object(wl, "entity_relevance_scorer", lambda term, title: TITLE_SCORES[title]), \
         patch.object(wl, "topic_relevance_scorer", lambda lead: TOPIC_SCORES[lead]):
        return wl.get_wikipedia_chunks(llm=None, term=term, topic=TOPIC)


class TestTopicPageExclusion(unittest.TestCase):
    def test_non_topic_term_no_longer_gets_the_topic_page(self):
        chunks, ambiguous, title = _lookup("golden gate strait")
        self.assertEqual(title, "Golden Gate")
        self.assertEqual(chunks, ["chunk of Golden Gate"])

    def test_the_topic_itself_still_gets_its_page(self):
        _, _, title = _lookup("golden gate bridge")
        self.assertEqual(title, "Golden Gate Bridge")

    def test_comparison_runs_on_normalised_names(self):
        self.assertTrue(wl._is_topic_page_for_other_term("golden gate strait", "Golden Gate Bridge", TOPIC))
        self.assertFalse(wl._is_topic_page_for_other_term("the golden gate bridge", "Golden Gate Bridge", TOPIC))
        self.assertFalse(wl._is_topic_page_for_other_term(
            "golden gate bridge design", "Golden Gate Bridge, Highway and Transportation District", TOPIC))

    def test_inactive_without_topic(self):
        self.assertFalse(wl._is_topic_page_for_other_term("golden gate strait", "Golden Gate Bridge", None))


# `charles ellis`, measured live (2026-09): the top search result is a disambiguation page
# with 14 options; title scores against the term, topic scores of each page's opening window.
ELLIS_SEARCH = ["Charles Ellis", "Chuck Schumer", "Charles D. Ellis", "Charles Drummond Ellis",
                "Charles Alton Ellis"]
ELLIS_OPTIONS = ["Charles Alton Ellis", "Charles Ellis (soccer)", "Charles Ellis, 1st Baron Seaford",
                 "Charlie Ellis", "Charles Ellis Johnson", "Charles D. Ellis"]
ELLIS_TITLE = {"Charles D. Ellis": 0.935, "Charles Alton Ellis": 0.888, "Charles Ellis (soccer)": 0.841,
               "Charles Ellis Johnson": 0.836, "Charlie Ellis": 0.834, "Charles Drummond Ellis": 0.783,
               "Charles Ellis, 1st Baron Seaford": 0.715, "Chuck Schumer": 0.318}
ELLIS_TOPIC = {"Charles D. Ellis": -0.0391, "Charles Alton Ellis": 0.3855,
               "Charles Ellis (soccer)": 0.0119, "Charles Drummond Ellis": 0.0}


def _ellis_page(title, auto_suggest):
    if title in ("Charles Ellis", "Charlie Ellis"):  # the second: an option that is itself one
        raise wl.wikipedia.exceptions.DisambiguationError(title, ELLIS_OPTIONS)
    if title == "Charles Ellis Johnson":
        raise wl.wikipedia.exceptions.PageError(title)
    return SimpleNamespace(title=title)


def _ellis_lookup(topic_scores=ELLIS_TOPIC):
    fetch = patch.object(wl, "_fetch_and_select", side_effect=lambda t, *a, **k: ([f"chunk of {t}"], t))
    with patch.object(wl.wikipedia, "search", return_value=ELLIS_SEARCH), \
         patch.object(wl.wikipedia, "page", side_effect=_ellis_page), \
         fetch as fetched, \
         patch.object(wl, "entity_relevance_scorer", lambda term, title: ELLIS_TITLE[title]), \
         patch.object(wl, "topic_relevance_scorer", lambda lead: topic_scores[lead]):
        result = wl.get_wikipedia_chunks(llm=None, term="charles ellis", topic=TOPIC)
        return result, [c.args[0] for c in fetched.call_args_list]


class TestDisambiguationThroughPageSelection(unittest.TestCase):
    def test_options_join_the_pool_and_the_right_sense_wins(self):
        (chunks, ambiguous, title), _ = _ellis_lookup()
        self.assertEqual(title, "Charles Alton Ellis")
        self.assertEqual(chunks, ["chunk of Charles Alton Ellis"])
        self.assertFalse(ambiguous)  # else the recorder would discard the page

    def test_only_valid_in_band_pages_are_fetched(self):
        _, fetched = _ellis_lookup()
        # In band (>= 0.785): D. Ellis, Alton Ellis, soccer, Johnson (no page), Charlie (itself
        # a disambiguation page, not followed). Each page fetched once.
        self.assertEqual(sorted(fetched), ["Charles Alton Ellis", "Charles D. Ellis", "Charles Ellis (soccer)"])

    def test_ambiguous_only_when_no_page_is_found(self):
        # No in-band page yields chunks: the flag the legacy path raised comes back.
        with patch.object(wl, "_fetch_and_select", return_value=([], "")), \
             patch.object(wl.wikipedia, "search", return_value=ELLIS_SEARCH), \
             patch.object(wl.wikipedia, "page", side_effect=_ellis_page), \
             patch.object(wl, "entity_relevance_scorer", lambda term, title: ELLIS_TITLE[title]), \
             patch.object(wl, "topic_relevance_scorer", lambda lead: 0.0):
            self.assertEqual(wl.get_wikipedia_chunks(llm=None, term="charles ellis", topic=TOPIC),
                             ([], True, None))


if __name__ == "__main__":
    unittest.main()
