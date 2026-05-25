from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from edge_trader_bot.config import BotConfig
from edge_trader_bot.rag_scanner import (
    OPENROUTER_CHAT_COMPLETIONS_ENDPOINT,
    build_default_llm_summarizer,
    build_llm_headers,
    build_llm_provider_config_from_env,
)


class LlmConfigTests(unittest.TestCase):
    def test_bot_config_defaults_to_openrouter_deepseek(self) -> None:
        config = BotConfig()

        self.assertEqual(config.llm_provider, "openrouter")
        self.assertEqual(config.llm_model, "deepseek/deepseek-chat")
        self.assertEqual(config.max_llm_evidence_items, 5)

    def test_env_builds_openrouter_provider_config(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EDGE_TRADER_LLM_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "test-key",
                "EDGE_TRADER_LLM_MODEL": "deepseek/deepseek-chat",
            },
            clear=False,
        ):
            provider = build_llm_provider_config_from_env()

        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider.provider, "openrouter")
        self.assertEqual(provider.model, "deepseek/deepseek-chat")
        self.assertEqual(provider.endpoint, OPENROUTER_CHAT_COMPLETIONS_ENDPOINT)

    def test_openrouter_headers_include_metadata(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EDGE_TRADER_LLM_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_APP_TITLE": "test-bot",
            },
            clear=False,
        ):
            provider = build_llm_provider_config_from_env()
            assert provider is not None
            headers = build_llm_headers(provider)

        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["X-Title"], "test-bot")
        self.assertIn("HTTP-Referer", headers)

    def test_missing_openrouter_key_disables_default_summarizer(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EDGE_TRADER_LLM_PROVIDER": "openrouter",
            },
            clear=True,
        ):
            self.assertIsNone(build_llm_provider_config_from_env())
            self.assertIsNone(build_default_llm_summarizer())

    def test_config_from_env_reads_provider_and_model(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EDGE_TRADER_LLM_PROVIDER": "openrouter",
                "EDGE_TRADER_LLM_MODEL": "deepseek/deepseek-r1",
                "EDGE_TRADER_ENABLE_LLM_RAG": "1",
                "EDGE_TRADER_MAX_LLM_EVIDENCE_ITEMS": "5",
            },
            clear=True,
        ):
            config = BotConfig.from_env()

        self.assertEqual(config.llm_provider, "openrouter")
        self.assertEqual(config.llm_model, "deepseek/deepseek-r1")
        self.assertTrue(config.enable_llm_rag_summary)
        self.assertEqual(config.max_llm_evidence_items, 5)


if __name__ == "__main__":
    unittest.main()
