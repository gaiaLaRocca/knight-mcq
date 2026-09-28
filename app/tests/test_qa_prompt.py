import re
import unittest

from app.generation.qa_generation import _format_multihop_qa_prompt

PATH = {
    "nodes": [
        {"name": "golden gate bridge", "description": "A suspension bridge."},
        {"name": "joseph strauss", "description": "Chief engineer of the Golden Gate Bridge."},
    ],
    "relationships": ["ASSOCIATED_WITH"],
}


def _examples(prompt):
    """The few-shot part of the prompt: from the first example to the topic instruction,
    which legitimately names the topic."""
    return prompt[prompt.index("Example 1:"):prompt.index("IMPORTANT: The generated Question")]


class TestQaPromptExamples(unittest.TestCase):
    def setUp(self):
        self.prompt = _format_multihop_qa_prompt(PATH, topic="Golden Gate Bridge")

    def test_examples_carry_no_topic_content(self):
        # test_7: q_003 and q_005 answered with example 3's caissons, absent from the KB.
        examples = _examples(self.prompt).lower()
        for leaked in ("golden gate", "caisson", "strauss", "san francisco"):
            self.assertNotIn(leaked, examples)

    def test_two_association_examples_with_different_openers(self):
        examples = _examples(self.prompt)
        self.assertEqual(examples.count("[:ASSOCIATED_WITH]"), 2)
        openers = re.findall(r"^Question: (\w+)", examples, flags=re.M)
        self.assertEqual(openers, ["Which", "What", "Which", "What"])
        self.assertNotIn("How", openers)

    def test_association_rule_present(self):
        self.assertIn("An ASSOCIATED_WITH step means", self.prompt)


if __name__ == "__main__":
    unittest.main()
