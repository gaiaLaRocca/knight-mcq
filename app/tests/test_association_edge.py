import unittest
from unittest.mock import patch

import app.core.agents.gpt.chatbot as chatbot
from app.core.common.neo4j_connection import Neo4jConnection, ASSOCIATION_RELATION


def _edges_written(triplet, parent_term):
    """Run process_triplet with Neo4j and the LLM mocked; return the edges it creates."""
    edges = []
    with patch.object(chatbot, "save_term_as_node"), \
         patch.object(chatbot, "query_term_description", return_value="a description"), \
         patch.object(chatbot, "record_literal_head"), \
         patch.object(chatbot, "create_relationship",
                      side_effect=lambda conn, src, dst, rel="HAS_TERM": edges.append((src, rel, dst))):
        # depth == max_depth: no recursion, only the edges of this triplet.
        chatbot.process_triplet(None, None, triplet, parent_term, depth=1, max_depth=1,
                                current_query_processed_terms=set())
    return edges


class TestAssociationEdge(unittest.TestCase):
    def test_head_other_than_parent_gets_an_association_edge(self):
        # test_7: from the seed essay, whose parent is the topic.
        edges = _edges_written(
            {"head": "leon moisseiff", "relation": "contributed_to", "tail": "golden gate bridge design"},
            "golden gate bridge",
        )
        self.assertEqual(edges, [
            ("golden gate bridge", ASSOCIATION_RELATION, "leon moisseiff"),
            ("leon moisseiff", "CONTRIBUTED_TO", "golden gate bridge design"),
        ])

    def test_no_borrowed_relation_from_parent_to_tail(self):
        # test_7 wrote `golden gate bridge -CONNECTS-> pacific ocean`: the strait connects them.
        edges = _edges_written(
            {"head": "golden gate strait", "relation": "connects", "tail": "pacific ocean"},
            "golden gate bridge",
        )
        self.assertNotIn(("golden gate bridge", "CONNECTS", "pacific ocean"), edges)
        self.assertIn(("golden gate strait", "CONNECTS", "pacific ocean"), edges)

    def test_head_equal_to_parent_writes_only_the_real_edge(self):
        edges = _edges_written(
            {"head": "golden gate bridge", "relation": "is_a", "tail": "suspension bridge"},
            "golden gate bridge",
        )
        self.assertEqual(edges, [("golden gate bridge", "IS_A", "suspension bridge")])

    def test_role_in_tail_no_longer_orphans_the_person(self):
        # test_7: Strauss hung only from `chief engineer`, which the entity filter drops.
        edges = _edges_written(
            {"head": "joseph strauss", "relation": "served_as", "tail": "chief engineer"},
            "golden gate bridge",
        )
        self.assertIn(("golden gate bridge", ASSOCIATION_RELATION, "joseph strauss"), edges)


class TestFindPathsShape(unittest.TestCase):
    def _query_for(self, **kwargs):
        """(cypher, parameters) that find_paths sends; no driver, only the query is inspected."""
        conn = object.__new__(Neo4jConnection)
        captured = []
        conn.query = lambda cypher, parameters=None, **k: captured.append((cypher, parameters)) or []
        conn.find_paths(**kwargs)
        return captured[0]

    def test_at_most_one_association_hop(self):
        expected = f"size([rel IN relationships(p) WHERE type(rel) = '{ASSOCIATION_RELATION}']) <= 1"
        self.assertIn(expected, self._query_for(max_length=2)[0])
        self.assertIn(expected, self._query_for(exact_length=2)[0])

    def test_start_bound_to_the_named_node_by_parameter(self):
        for kwargs in ({"max_length": 2}, {"exact_length": 2}):
            cypher, params = self._query_for(start_name="golden gate bridge", **kwargs)
            self.assertIn("MATCH p=(start_node:Term {name: $start_name})-", cypher)
            self.assertEqual(params, {"start_name": "golden gate bridge"})

    def test_without_start_any_node_can_start(self):
        cypher, params = self._query_for(max_length=2)
        self.assertIn("MATCH p=(start_node:Term)-", cypher)
        self.assertIsNone(params)


class TestPathsFromTopic(unittest.TestCase):
    def test_qa_generation_starts_paths_at_the_topic_node(self):
        from unittest.mock import MagicMock
        from app.generation.qa_generation import generate_qa_from_paths
        conn = MagicMock()
        conn.find_paths.return_value = []
        generate_qa_from_paths(conn, llm_client=None, max_complexity=2, topic="Golden Gate Bridge")
        conn.find_paths.assert_called_once_with(
            max_length=2, exact_length=None, start_name="golden gate bridge")


if __name__ == "__main__":
    unittest.main()
