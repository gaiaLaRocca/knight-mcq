import unittest
from app.core.agents.gpt.literals import is_literal, find_literal_evidence
from app.core.agents.gpt.text_processing import normalize_node_name

# The 27 digit-bearing node names of test_1-test_7, split by hand.
LITERALS = [
    "1.7 miles", "1933", "1937", "2,743.2 meters", "2.7 kilometers", "63 million square miles",
    "656 miles", "746 feet", "82,116 acres", "82116 acres", "8,981 feet", "90 feet",
    "december 31, 1900", "december 31, 1914", "january 1, 1881", "january 1, 1901",
    "may 27 1937", "may 27, 1937",
]
NOT_LITERALS = [
    "1.7 * 5280 feet", "1.7 times the length of one mile", "20th century",
    "california state route 1", "early 20th century", "late 19th century",
    "period spanning from january 1, 1901, to december 31, 1919", "u.s. route 101", "us route 101",
]

# Opening of the Golden Gate Bridge lead (oldid 1359105856).
GGB_LEAD = (
    "The bridge opened to the public on May 27, 1937, and has undergone various retrofits "
    "and other improvement projects in the decades since. Its main span is 4,200 feet "
    "(1,280 m) and its total height is 746 feet (227 m)."
)


class TestIsLiteral(unittest.TestCase):
    def test_logged_names(self):
        for name in LITERALS:
            self.assertTrue(is_literal(normalize_node_name(name)), name)
        for name in NOT_LITERALS:
            self.assertFalse(is_literal(normalize_node_name(name)), name)

    def test_other_shapes(self):
        for name in ["may 1937", "27 may 1937", "746 ft", "50%", "4,200 feet"]:
            self.assertTrue(is_literal(normalize_node_name(name)), name)
        for name in ["5280", "route 66", "feet"]:
            self.assertFalse(is_literal(normalize_node_name(name)), name)


class TestFindLiteralEvidence(unittest.TestCase):
    def evidence(self, name, chunks):
        return find_literal_evidence(normalize_node_name(name), chunks)

    def test_value_found_returns_its_sentence(self):
        self.assertEqual(
            self.evidence("may 27, 1937", [GGB_LEAD]),
            "The bridge opened to the public on May 27, 1937, and has undergone various "
            "retrofits and other improvement projects in the decades since.",
        )
        self.assertIn("4,200 feet", self.evidence("4200 feet", [GGB_LEAD]))

    def test_units_and_hyphens_are_normalised(self):
        self.assertIsNotNone(self.evidence("746 feet", ["Its height is 746 ft (227 m)."]))
        self.assertIsNotNone(self.evidence("746 feet", ["A 746-foot tower."]))
        self.assertIsNotNone(self.evidence("1280 meters", [GGB_LEAD]))

    def test_month_and_year_inside_a_full_date(self):
        self.assertIsNotNone(self.evidence("may 1937", [GGB_LEAD]))

    def test_absent_value_is_not_verified(self):
        self.assertIsNone(self.evidence("656 miles", [GGB_LEAD]))
        self.assertIsNone(self.evidence("may 28, 1937", [GGB_LEAD]))
        # The number alone, or with another unit, is not the value.
        self.assertIsNone(self.evidence("746 miles", [GGB_LEAD]))
        self.assertIsNone(self.evidence("1,937 feet", [GGB_LEAD]))

    def test_decimals_are_not_split(self):
        self.assertIsNone(self.evidence("7 miles", ["It is 1.7 miles long."]))
        self.assertIsNotNone(self.evidence("1.7 miles", ["It is 1.7 miles long."]))

    def test_no_chunks(self):
        self.assertIsNone(self.evidence("746 feet", []))


if __name__ == "__main__":
    unittest.main()
