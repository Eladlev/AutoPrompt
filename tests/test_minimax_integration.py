"""Integration tests for MiniMax provider (require MINIMAX_API_KEY)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

MINIMAX_API_KEY = os.environ.get('MINIMAX_API_KEY', '')


@unittest.skipUnless(MINIMAX_API_KEY, 'MINIMAX_API_KEY not set')
class TestMiniMaxIntegration(unittest.TestCase):
    """End-to-end tests that call the real MiniMax API."""

    def _get_llm(self, model='MiniMax-M2.7', temperature=0.5):
        from unittest.mock import patch
        env = {
            'openai': {'OPENAI_API_KEY': '', 'OPENAI_API_BASE': '', 'OPENAI_ORGANIZATION': ''},
            'minimax': {'MINIMAX_API_KEY': MINIMAX_API_KEY, 'MINIMAX_API_BASE': 'https://api.minimax.io/v1'},
        }
        with patch('utils.config.LLM_ENV', env):
            from utils.config import get_llm
            return get_llm({'type': 'MiniMax', 'name': model, 'temperature': temperature})

    def test_simple_completion(self):
        """MiniMax M2.7 can complete a simple prompt."""
        llm = self._get_llm()
        result = llm.invoke('Say hello in one word.')
        self.assertTrue(len(result.content) > 0)

    def test_highspeed_model(self):
        """MiniMax M2.7-highspeed responds correctly."""
        llm = self._get_llm(model='MiniMax-M2.7-highspeed')
        result = llm.invoke('What is 2+2? Reply with just the number.')
        self.assertIn('4', result.content)

    def test_temperature_clamping_zero(self):
        """Temperature 0 is clamped and model still works."""
        llm = self._get_llm(temperature=0)
        result = llm.invoke('Reply with OK.')
        self.assertTrue(len(result.content) > 0)


if __name__ == '__main__':
    unittest.main()
