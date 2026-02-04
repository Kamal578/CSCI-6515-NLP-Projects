import unittest

from src.tokenize import tokenize


class TestTokenize(unittest.TestCase):
    def test_azerbaijani_letters_and_lowercase(self):
        text = "Salam, Azərbaycan!"
        toks = tokenize(text, lowercase=True)
        self.assertEqual(toks, ["salam", "azərbaycan"])

    def test_decimals_and_numbers(self):
        text = "Qiymət 3,14 manatdır, 2024-cü il."
        toks = tokenize(text, lowercase=True)
        self.assertIn("3,14", toks)
        self.assertIn("2024", toks)


if __name__ == "__main__":
    unittest.main()
