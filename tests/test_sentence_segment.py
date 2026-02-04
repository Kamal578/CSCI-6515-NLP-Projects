import unittest

from src.sentence_segment import sentence_segment


class TestSentenceSegment(unittest.TestCase):
    def test_kv_verst_continuation(self):
        text = "(Novobayazid) — ərazisinə (4239 kv. verst) görə quberniyanın ən böyük qəzası idi."
        sents = sentence_segment(text)
        self.assertEqual(len(sents), 1)
        self.assertIn("4239 kv. verst", sents[0])

    def test_abbreviation_not_split(self):
        text = "prof. Samir gəlir. Salam."
        sents = sentence_segment(text)
        self.assertEqual(len(sents), 2)
        self.assertTrue(sents[0].startswith("prof. Samir"))

if __name__ == "__main__":
    unittest.main()
