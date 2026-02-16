import os
import unittest
from app.core.common.config import MAX_DEPTH
from app.core.common.neo4j_connection import Neo4jConnection
from app.core.agents.gpt.chatbot import generate_response

# Use env for integration test; skip when Neo4j or API is not available
NEO4J_URI = os.getenv("GPT_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("GPT_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("GPT_NEO4J_PASSWORD", "password")


class TestChatbot(unittest.TestCase):
    def setUp(self):
        self.conn = None
        try:
            self.conn = Neo4jConnection(uri=NEO4J_URI, user=NEO4J_USER, pwd=NEO4J_PASSWORD)
            if hasattr(self.conn._driver, "verify_connectivity"):
                self.conn._driver.verify_connectivity()
        except Exception:
            self.skipTest("Neo4j not available")

    def test_generate_response(self):
        response = generate_response("Tell me about AI", self.conn, MAX_DEPTH)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def tearDown(self):
        if self.conn is not None:
            self.conn.close()

if __name__ == "__main__":
    unittest.main()
