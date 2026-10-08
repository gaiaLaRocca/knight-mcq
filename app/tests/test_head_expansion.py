import unittest
from unittest.mock import patch

import app.core.agents.gpt.chatbot as chatbot

TOPIC = "golden gate bridge"

# What each node's description yields when mined for sub-triplets (one each, under the cap).
SUB_TRIPLETS = {
    "joseph strauss": [{"head": "joseph strauss", "relation": "born_in", "tail": "cincinnati"}],
    "chief engineer": [{"head": "chief engineer", "relation": "oversees", "tail": "vessel machinery"}],
    "golden gate strait": [{"head": "golden gate strait", "relation": "connects_to", "tail": "pacific ocean"}],
    "suspension bridge": [{"head": "suspension bridge", "relation": "is_a", "tail": "bridge"}],
}


def _build(triplets, depth=1, max_depth=2):
    """Process seed triplets with Neo4j and the LLM mocked; return (edges, expanded terms)."""
    edges, expanded = [], []

    def mine(description):
        term = description.removeprefix("description of ")
        expanded.append(term)
        return SUB_TRIPLETS.get(term, [])

    chatbot.expanded_terms.reset()
    with patch.object(chatbot, "save_term_as_node"), \
         patch.object(chatbot, "_get_llm"), \
         patch.object(chatbot, "record_literal_head"), \
         patch.object(chatbot, "_is_wiki_fact_checked", return_value=True), \
         patch.object(chatbot, "query_term_description", side_effect=lambda conn, term: f"description of {term}"), \
         patch.object(chatbot, "extract_clean_special_terms", side_effect=mine), \
         patch.object(chatbot, "create_relationship",
                      side_effect=lambda conn, src, dst, rel="HAS_TERM": edges.append((src, rel, dst))):
        processed = {TOPIC}
        for triplet in triplets:
            chatbot.process_triplet(None, None, triplet, TOPIC, depth=depth, max_depth=max_depth,
                                    current_query_processed_terms=processed)
    return edges, expanded


class TestAssociationHeadExpansion(unittest.TestCase):
    def test_head_attached_by_association_is_expanded(self):
        # test_8: Strauss hung from the topic by an association edge and got no sub-triplet.
        edges, expanded = _build([{"head": "joseph strauss", "relation": "served_as", "tail": "chief engineer"}])
        self.assertIn("joseph strauss", expanded)
        self.assertIn(("joseph strauss", "BORN_IN", "cincinnati"), edges)

    def test_tail_is_still_expanded(self):
        edges, expanded = _build([{"head": "joseph strauss", "relation": "served_as", "tail": "chief engineer"}])
        self.assertIn("chief engineer", expanded)

    def test_the_parent_itself_is_not_expanded_again(self):
        _, expanded = _build([{"head": TOPIC, "relation": "is_a", "tail": "suspension bridge"}])
        self.assertEqual(expanded, ["suspension bridge"])

    def test_no_expansion_at_the_maximum_depth(self):
        _, expanded = _build([{"head": "joseph strauss", "relation": "served_as", "tail": "chief engineer"}],
                             depth=2, max_depth=2)
        self.assertEqual(expanded, [])


class TestBranchSelectionInTheBuild(unittest.TestCase):
    def test_only_the_selected_branches_reach_the_graph(self):
        strauss = [
            {"head": "joseph strauss", "relation": "designed", "tail": TOPIC},
            {"head": "charles alton ellis", "relation": "was", "tail": "primary designer"},
            {"head": "joseph strauss", "relation": "born_in", "tail": "cincinnati"},
        ]
        with patch.object(chatbot.branch_selection, "relevance_scorer", lambda tail: 0.0), \
             patch.object(chatbot, "seed_nodes", frozenset({TOPIC, "joseph strauss", "chief engineer"})), \
             patch.dict(SUB_TRIPLETS, {"joseph strauss": strauss}):
            edges, _ = _build([{"head": "joseph strauss", "relation": "served_as", "tail": "chief engineer"}])
        self.assertIn(("joseph strauss", "BORN_IN", "cincinnati"), edges)
        self.assertNotIn(("joseph strauss", "DESIGNED", TOPIC), edges)
        self.assertNotIn(("charles alton ellis", "WAS", "primary designer"), edges)


class TestExpandedOnce(unittest.TestCase):
    def test_a_node_reached_as_tail_and_as_head_is_expanded_once(self):
        # test_8: the strait is the tail of `spans` and the head of both `connects` triplets.
        _, expanded = _build([
            {"head": TOPIC, "relation": "spans", "tail": "golden gate strait"},
            {"head": "golden gate strait", "relation": "connects", "tail": "san francisco bay"},
            {"head": "golden gate strait", "relation": "connects", "tail": "pacific ocean"},
        ])
        self.assertEqual(expanded.count("golden gate strait"), 1)

    def test_a_tail_reached_by_several_triplets_is_expanded_once(self):
        # test_8: `suspension bridge` closed three seed triplets and was expanded three times.
        _, expanded = _build([
            {"head": TOPIC, "relation": relation, "tail": "suspension bridge"}
            for relation in ("is_a", "was_longest", "was_tallest")
        ])
        self.assertEqual(expanded.count("suspension bridge"), 1)


if __name__ == "__main__":
    unittest.main()
