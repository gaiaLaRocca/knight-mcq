import unittest
from app.core.agents.gpt.text_processing import (
    preprocess_text, normalize_node_name, clean_triplet, _drop_self_loops,
)

class TestTextProcessing(unittest.TestCase):
    def test_preprocess_text(self):
        text = "This is a sentence. And another one!"
        sentences = preprocess_text(text)
        self.assertEqual(len(sentences), 2)

class TestNormalizeNodeName(unittest.TestCase):
    # Duplicate pairs observed on test_7: each must collapse onto one key.
    def test_test7_duplicates_merge(self):
        pairs = [
            ("the golden gate bridge", "golden gate bridge"),
            ("the san francisco peninsula", "san francisco peninsula"),
            ("u.s. route 101", "us route 101"),
        ]
        for variant, canonical in pairs:
            self.assertEqual(normalize_node_name(variant), canonical)
            self.assertEqual(normalize_node_name(canonical), canonical)

    def test_decimal_dots_kept(self):
        self.assertEqual(normalize_node_name("1.7 miles"), "1.7 miles")
        self.assertEqual(normalize_node_name("2,743.2 meters"), "2743.2 meters")  # literal: comma dropped
        self.assertEqual(normalize_node_name("1.7 * 5280 feet"), "1.7 * 5280 feet")

    def test_abbreviation_dots_removed(self):
        self.assertEqual(normalize_node_name("u.s. national park service"), "us national park service")
        self.assertEqual(normalize_node_name("washington, d.c."), "washington, dc")

    def test_only_leading_the_is_stripped(self):
        self.assertEqual(normalize_node_name("theater district"), "theater district")
        self.assertEqual(normalize_node_name("bridge of the gods"), "bridge of the gods")
        self.assertEqual(normalize_node_name("the"), "the")

    def test_case_underscores_whitespace(self):
        self.assertEqual(normalize_node_name("  The_Golden  Gate\nBridge "), "golden gate bridge")

    def test_literal_spellings_merge(self):
        # Pairs observed on test_7 and earlier runs.
        self.assertEqual(normalize_node_name("may 27, 1937"), normalize_node_name("may 27 1937"))
        self.assertEqual(normalize_node_name("82,116 acres"), normalize_node_name("82116 acres"))
        # A comma outside a literal is part of the name.
        self.assertEqual(normalize_node_name("golden gate bridge, highway and transportation district"),
                         "golden gate bridge, highway and transportation district")

    def test_idempotent(self):
        for name in ["the golden gate bridge", "u.s. route 101", "1.7 miles", "The Hague",
                     "May 27, 1937", "4,200 feet"]:
            once = normalize_node_name(name)
            self.assertEqual(normalize_node_name(once), once)

class TestCleanTriplet(unittest.TestCase):
    def test_head_and_tail_normalised(self):
        cleaned = clean_triplet(
            {"head": "The Golden Gate Bridge", "relation": "carries", "tail": "U.S. Route 101"}
        )
        self.assertEqual(
            cleaned, {"head": "golden gate bridge", "relation": "carries", "tail": "us route 101"}
        )

    def test_self_loops_created_by_normalisation_are_dropped(self):
        triplets = [
            clean_triplet({"head": "the golden gate bridge", "relation": "is", "tail": "golden gate bridge"}),
            clean_triplet({"head": "golden gate bridge", "relation": "spans", "tail": "golden gate strait"}),
        ]
        self.assertEqual(
            _drop_self_loops(triplets),
            [{"head": "golden gate bridge", "relation": "spans", "tail": "golden gate strait"}],
        )

if __name__ == "__main__":
    unittest.main()
