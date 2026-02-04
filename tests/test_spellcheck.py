import unittest
from collections import Counter

from src.spellcheck import suggest


class TestSpellcheck(unittest.TestCase):
    def test_suggest_ranks_by_distance_then_freq(self):
        vocab = Counter({"azərbaycan": 10, "azerbaycan": 5, "kitab": 3})
        cands = suggest("azrbaycan", vocab, max_dist=2, top_k=3)
        self.assertTrue(len(cands) >= 1)
        top_word, _ = cands[0]
        self.assertEqual(top_word, "azərbaycan")


if __name__ == "__main__":
    unittest.main()
