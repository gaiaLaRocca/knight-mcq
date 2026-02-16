import unittest
from app.core.agents.gpt.text_processing import preprocess_text

class TestTextProcessing(unittest.TestCase):
    def test_preprocess_text(self):
        text = "This is a sentence. And another one!"
        sentences = preprocess_text(text)
        self.assertEqual(len(sentences), 2)

if __name__ == "__main__":
    unittest.main()
