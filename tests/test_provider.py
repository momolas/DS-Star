import os
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).parent.parent))

from provider import GeminiProvider, OpenAIProvider, OllamaProvider

class TestProviderPrecedence(unittest.TestCase):

    @patch('provider.genai')
    def test_gemini_precedence(self, mock_genai):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
            p = GeminiProvider("config-key", "gemini-pro")
            self.assertEqual(p.api_key, "env-key")

        with patch.dict(os.environ, {}, clear=True):
            if "GEMINI_API_KEY" in os.environ: del os.environ["GEMINI_API_KEY"]
            p = GeminiProvider("config-key", "gemini-pro")
            self.assertEqual(p.api_key, "config-key")

    @patch('provider.openai')
    def test_openai_precedence(self, mock_openai):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            p = OpenAIProvider("config-key", "gpt-4")
            self.assertEqual(p.api_key, "env-key")

        with patch.dict(os.environ, {}, clear=True):
            if "OPENAI_API_KEY" in os.environ: del os.environ["OPENAI_API_KEY"]
            p = OpenAIProvider("config-key", "gpt-4")
            self.assertEqual(p.api_key, "config-key")

if __name__ == '__main__':
    unittest.main()
