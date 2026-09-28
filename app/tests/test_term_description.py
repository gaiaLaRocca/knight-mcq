import unittest
from types import SimpleNamespace

from app.core.agents.gpt import term_description as td

WIKI = ("Leon Solomon Moisseiff was a leading American engineer in suspension bridge design. "
        "He was a consulting engineer on the Golden Gate Bridge.")
PREAMBLE_REPLY = ("Okay, here's a detailed explanation of \"Leon Moisseiff\", drawing on the context:\n\n"
                  "Leon Moisseiff was an American bridge engineer. He consulted on the Golden Gate Bridge.")


class _FakeLLM:
    """Records the messages it receives and answers with a fixed reply."""
    def __init__(self, reply):
        self.reply, self.messages = reply, None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.reply)


class _FakeLookup:
    def __init__(self, summary, ambiguous=False):
        self.result = (summary, ambiguous)

    def lookup(self, term, context_hint=None, llm=None):
        return self.result


def _describe(reply, summary=WIKI, parent="golden gate bridge"):
    llm = _FakeLLM(reply)
    description, used = td.generate_term_description.__wrapped__(
        llm, "leon moisseiff", parent_term=parent, external_lookup=_FakeLookup(summary))
    return description, used, llm.messages


class TestDescriptionPrompt(unittest.TestCase):
    def test_prose_prompt_replaces_the_template(self):
        _, _, messages = _describe("Leon Moisseiff was an engineer.")
        system, human = messages[0].content, messages[1].content
        for section in ("Domains of Use", "Subfields", "Case Studies", "Current Research", "cite notable research"):
            self.assertNotIn(section, system)
        for clause in ("continuous prose", "no preamble", "one-sentence definition", "about 300 words",
                       "stay close to what it says"):
            self.assertIn(clause, system)
        self.assertIn("Explain the term: 'leon moisseiff'.", human)
        self.assertIn(WIKI, human)
        self.assertIn("Also explain how it relates to 'golden gate bridge'.", human)

    def test_same_prompt_without_wikipedia(self):
        _, used, messages = _describe("A note.", summary=None)
        self.assertFalse(used)
        self.assertEqual(messages[0].content, td.DESCRIPTION_SYSTEM_PROMPT)


class TestPreamble(unittest.TestCase):
    def test_preamble_paragraph_is_dropped(self):
        description, used, _ = _describe(PREAMBLE_REPLY)
        self.assertTrue(used)
        self.assertTrue(description.startswith("Leon Moisseiff was an American bridge engineer."))

    def test_other_announcing_forms_are_dropped(self):
        # test_7 also had "Okay, let's delve into...".
        for preamble in ("Okay, let's delve into a comprehensive explanation of the term:",
                         "Here is a short note on Leon Moisseiff:",
                         "Sure! I'll explain the term below."):
            self.assertEqual(td._strip_preamble(f"{preamble}\n\nLeon Moisseiff was an engineer."),
                             "Leon Moisseiff was an engineer.")

    def test_preamble_ending_in_a_colon_on_the_same_line_as_content(self):
        # A test_7 opening: "...adhering to the requested structure: **Term:** US Route 101".
        reply = ("Okay, here's a detailed explanation of US Route 101, adhering to the requested "
                 "structure: US Route 101 is a highway along the Pacific coast.\n\nIt crosses the bridge.")
        self.assertEqual(td._strip_preamble(reply),
                         "US Route 101 is a highway along the Pacific coast.\n\nIt crosses the bridge.")

    def test_only_the_announcement_is_removed_never_the_rest(self):
        reply = ("Okay, here's a detailed explanation of Marin County based on the provided context.\n"
                 "Marin County lies north of the Golden Gate.\n\nIts county seat is San Rafael.")
        self.assertEqual(td._strip_preamble(reply),
                         "Marin County lies north of the Golden Gate.\n\nIts county seat is San Rafael.")

    def test_content_is_never_dropped(self):
        for reply in ("Okay, Leon Moisseiff was an American engineer.\n\nHe consulted on the bridge.",
                      "Sure, the Pacific Ocean is the largest ocean.\n\nIt covers a third of the Earth.",
                      "Okayama is a city in Japan.\n\nIt lies on the Seto Inland Sea.",
                      "Hereford is a city in England.\n\nIt lies on the River Wye.",
                      # A fact after the announcement, on the same line: the whole line stays.
                      "Okay, here's a note on Leon Moisseiff. He was an American engineer.\n\nMore.",
                      # "U.S." looks like a sentence end: in doubt, nothing is removed.
                      "Okay, here's a detailed explanation of U.S. Route 101 from the context.\n\nIt is a highway.",
                      "Okay, here's the whole answer on one line with no paragraph break."):
            self.assertEqual(td._strip_preamble(reply), reply)


if __name__ == "__main__":
    unittest.main()
