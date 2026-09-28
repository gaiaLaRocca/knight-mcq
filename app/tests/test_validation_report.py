import unittest
from unittest.mock import patch

import app.generation.qa_generation as qg

PASS = "Grammar_Fluency: YES\nAnswerable_From_Source: YES\nTopic_Relevant: YES\nEthics_Privacy_Safe: YES"
FAIL_ANSWERABLE = "Grammar_Fluency: YES\nAnswerable_From_Source: NO\nTopic_Relevant: YES\nEthics_Privacy_Safe: YES"
GARBLED = "Okay, here is my evaluation: the question looks fine."

SOURCE = {"nodes": [{"name": "golden gate bridge"}, {"name": "joseph strauss"}],
          "relationships": ["ASSOCIATED_WITH"]}


def _pair(question):
    return {"question": question, "answer": "An answer.", "source_details": SOURCE}


class TestValidationReport(unittest.TestCase):
    def _validate(self, pairs, verdicts):
        """Run validate_qa_pairs with the validator's replies fixed per question."""
        report = {}
        with patch.object(qg, "safe_generate",
                          side_effect=lambda llm, messages, **k: verdicts[messages[1].content.split("Question: ")[1].split("\n")[0]]):
            accepted = qg.validate_qa_pairs(pairs, llm_client=object(), topic="Golden Gate Bridge", report=report)
        return accepted, report

    def test_each_rejection_is_recorded_with_stage_and_checks(self):
        pairs = [_pair("Who was the chief engineer of the bridge?"),
                 _pair("What role did Strauss play in the bridge?"),
                 _pair("Short?"),  # under MIN_QUESTION_LEN: structural
                 _pair("Which garbled verdict does this one receive?")]
        verdicts = {pairs[0]["question"]: PASS, pairs[1]["question"]: FAIL_ANSWERABLE,
                    pairs[3]["question"]: GARBLED}
        accepted, report = self._validate(pairs, verdicts)
        self.assertEqual([p["question"] for p in accepted], [pairs[0]["question"]])
        self.assertEqual(
            [(r["index"], r["stage"], r["failed_checks"]) for r in report["rejected"]],
            [(1, "llm", ["answerable_from_source"]),
             (2, "structural", ["Invalid/missing/length question"]),
             (3, "llm", ["unparseable_verdict"])],
        )
        self.assertEqual(report["rejected"][0]["raw_verdict"], FAIL_ANSWERABLE)
        self.assertEqual(report["rejected"][0]["source_details"], SOURCE)

    def test_timed_out_validation_is_counted_not_silent(self):
        pairs = [_pair("Who was the chief engineer of the bridge?")]
        accepted, report = self._validate(pairs, {pairs[0]["question"]: None})
        self.assertEqual(len(accepted), 1)  # still accepted, as before
        self.assertEqual(report["accepted_without_validation"], [pairs[0]["question"]])

    def test_without_report_behaviour_is_unchanged(self):
        pairs = [_pair("Who was the chief engineer of the bridge?")]
        with patch.object(qg, "safe_generate", return_value=PASS):
            self.assertEqual(qg.validate_qa_pairs(pairs, llm_client=object()), pairs)


if __name__ == "__main__":
    unittest.main()
