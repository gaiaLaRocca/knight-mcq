import unittest
from unittest.mock import MagicMock, patch

from app.generation.qa_generation import _one_path_per_node_sequence, generate_qa_from_paths


def _path(*steps):
    """_path("a", "R", "b", "S", "c") -> the dict find_paths returns for (a)-[:R]->(b)-[:S]->(c)."""
    return {"nodes": [{"name": name} for name in steps[::2]], "relationships": list(steps[1::2])}


# test_8: the seed states `connects`, the strait's own description `connects_to`.
CONNECTS = _path("golden gate bridge", "SPANS", "golden gate strait", "CONNECTS", "pacific ocean")
CONNECTS_TO = _path("golden gate bridge", "SPANS", "golden gate strait", "CONNECTS_TO", "pacific ocean")
STRAIT = _path("golden gate bridge", "SPANS", "golden gate strait")


class TestOnePathPerNodeSequence(unittest.TestCase):
    def test_parallel_edges_leave_one_path(self):
        kept, dropped = _one_path_per_node_sequence([STRAIT, CONNECTS, CONNECTS_TO])
        self.assertEqual(kept, [STRAIT, CONNECTS])
        self.assertEqual(dropped, [CONNECTS_TO])

    def test_the_copy_kept_does_not_depend_on_order(self):
        for order in ([CONNECTS, CONNECTS_TO], [CONNECTS_TO, CONNECTS]):
            kept, _ = _one_path_per_node_sequence(order)
            self.assertEqual(kept, [CONNECTS])

    def test_different_node_sequences_are_all_kept(self):
        # A path contained in a longer one is a different sequence, not a duplicate.
        paths = [STRAIT, CONNECTS, _path("golden gate bridge", "LINKS", "marin county")]
        kept, dropped = _one_path_per_node_sequence(paths)
        self.assertEqual(kept, paths)
        self.assertEqual(dropped, [])


class TestGenerationUsesOnePathPerSequence(unittest.TestCase):
    def test_only_the_kept_copy_is_generated(self):
        conn = MagicMock()
        conn.find_paths.return_value = [CONNECTS_TO, CONNECTS]
        with patch("app.generation.qa_generation._process_single_path", return_value=[]) as process:
            generate_qa_from_paths(conn, llm_client=None, topic="Golden Gate Bridge")
        self.assertEqual([call.args[0] for call in process.call_args_list], [CONNECTS])


if __name__ == "__main__":
    unittest.main()
