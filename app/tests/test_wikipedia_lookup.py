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


if __name__ == "__main__":
    unittest.main()
