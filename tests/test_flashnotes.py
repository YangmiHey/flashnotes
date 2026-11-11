"""Tests for FlashNotes pure functions (no API key needed).

Run: python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flashnotes import (  # noqa: E402
    DEFAULT_MODEL,
    MERGE_PROMPTS,
    SYSTEM_PROMPTS,
    chunk_text,
    estimate_tokens,
    extract_content,
)


class TestChunking(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n  "), [])

    def test_short_text_is_single_chunk(self):
        text = "今天开会讨论了上线计划。"
        self.assertEqual(chunk_text(text), [text])

    def test_long_text_splits_into_multiple_chunks(self):
        para = "这是一段很长的会议记录。" * 400
        text = "\n\n".join([para] * 5)
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].startswith("这是一段很长的会议记录"))
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 6000 + 400 + 10)

    def test_overlap_keeps_context(self):
        para = "abcdefghijklmnopqrstuvwxyz" * 400
        text = "\n\n".join([para] * 3)
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 1)
        # Each chunk after the first starts with the tail of the previous one.
        self.assertTrue(chunks[1].startswith(chunks[0][-400:]))


class TestHelpers(unittest.TestCase):
    def test_estimate_tokens_positive(self):
        self.assertGreater(estimate_tokens("hello world"), 0)
        self.assertEqual(estimate_tokens(""), 1)

    def test_extract_content_ok(self):
        data = {"choices": [{"message": {"content": "hello"}}]}
        self.assertEqual(extract_content(data), "hello")

    def test_extract_content_raises_on_bad_payload(self):
        with self.assertRaises(ValueError):
            extract_content({"choices": []})
        with self.assertRaises(ValueError):
            extract_content({})

    def test_prompts_cover_both_languages(self):
        for lang in ("zh", "en"):
            self.assertIn(lang, SYSTEM_PROMPTS)
            self.assertIn(lang, MERGE_PROMPTS)
            self.assertIn("TL;DR", SYSTEM_PROMPTS[lang])

    def test_default_model_is_deepseek_flash_1017(self):
        self.assertEqual(DEFAULT_MODEL, "deepseek/deepseek-v4.1-flash-1017")


if __name__ == "__main__":
    unittest.main()
