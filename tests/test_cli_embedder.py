import unittest

import cli


class TestCliEmbedder(unittest.TestCase):
    def test_gemini_embedder_fails_cleanly_without_name_error(self):
        with self.assertRaises(RuntimeError):
            cli._build_embedder("gemini", 384)


if __name__ == "__main__":
    unittest.main()


