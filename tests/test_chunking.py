import unittest

from rag_indexer.chunking import chunk_markdown, chunk_text


class TestChunking(unittest.TestCase):
    def test_text_offsets(self):
        text = "a\nb\nc\n"
        chunks = chunk_text(text, max_chars=2, overlap=0, min_chunk_chars=1)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].line_start, 1)
        self.assertEqual(chunks[0].line_end, 1)
        self.assertEqual(chunks[1].line_start, 2)
        self.assertEqual(chunks[1].line_end, 2)

    def test_markdown_section_path(self):
        text = "# Title\nLine 1\n## Sub\nLine 2\n"
        chunks = chunk_markdown(text, max_chars=200, overlap=0, min_chunk_chars=1)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].section_path, "Title")
        self.assertEqual(chunks[1].section_path, "Title > Sub")


if __name__ == "__main__":
    unittest.main()
