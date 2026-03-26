"""Unit tests for MiniMax LLM provider integration."""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestGetLlmMiniMax(unittest.TestCase):
    """Tests for the MiniMax branch in get_llm()."""

    def setUp(self):
        """Patch LLM_ENV so config.py doesn't need a real llm_env.yml."""
        self.env_patcher = patch('utils.config.LLM_ENV', {
            'openai': {'OPENAI_API_KEY': '', 'OPENAI_API_BASE': '', 'OPENAI_ORGANIZATION': ''},
            'azure': {'AZURE_OPENAI_API_KEY': '', 'AZURE_OPENAI_ENDPOINT': '', 'OPENAI_API_VERSION': ''},
            'google': {'GOOGLE_API_KEY': ''},
            'anthropic': {'ANTHROPIC_API_KEY': ''},
            'minimax': {'MINIMAX_API_KEY': 'test-mm-key', 'MINIMAX_API_BASE': 'https://api.minimax.io/v1'},
        })
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch('utils.config.ChatOpenAI')
    def test_minimax_basic(self, mock_chat):
        """MiniMax type routes to ChatOpenAI with correct base URL and model."""
        from utils.config import get_llm
        config = {'type': 'MiniMax', 'name': 'MiniMax-M2.7', 'temperature': 0.8}
        get_llm(config)
        mock_chat.assert_called_once()
        kwargs = mock_chat.call_args
        self.assertEqual(kwargs[1]['model_name'], 'MiniMax-M2.7')
        self.assertEqual(kwargs[1]['openai_api_key'], 'test-mm-key')
        self.assertEqual(kwargs[1]['openai_api_base'], 'https://api.minimax.io/v1')
        self.assertAlmostEqual(kwargs[1]['temperature'], 0.8)

    @patch('utils.config.ChatOpenAI')
    def test_minimax_case_insensitive(self, mock_chat):
        """Provider type matching is case-insensitive."""
        from utils.config import get_llm
        config = {'type': 'minimax', 'name': 'MiniMax-M2.5', 'temperature': 0.5}
        get_llm(config)
        mock_chat.assert_called_once()
        self.assertEqual(mock_chat.call_args[1]['model_name'], 'MiniMax-M2.5')

    @patch('utils.config.ChatOpenAI')
    def test_minimax_temperature_clamping_zero(self, mock_chat):
        """Temperature 0 is clamped to 0.01 for MiniMax."""
        from utils.config import get_llm
        config = {'type': 'MiniMax', 'name': 'MiniMax-M2.7', 'temperature': 0}
        get_llm(config)
        self.assertAlmostEqual(mock_chat.call_args[1]['temperature'], 0.01)

    @patch('utils.config.ChatOpenAI')
    def test_minimax_temperature_default_zero(self, mock_chat):
        """When temperature not set, defaults to 0 which is clamped to 0.01."""
        from utils.config import get_llm
        config = {'type': 'MiniMax', 'name': 'MiniMax-M2.7'}
        get_llm(config)
        self.assertAlmostEqual(mock_chat.call_args[1]['temperature'], 0.01)

    @patch('utils.config.ChatOpenAI')
    def test_minimax_temperature_valid(self, mock_chat):
        """Temperature within (0, 1] is passed through without clamping."""
        from utils.config import get_llm
        config = {'type': 'MiniMax', 'name': 'MiniMax-M2.7', 'temperature': 0.7}
        get_llm(config)
        self.assertAlmostEqual(mock_chat.call_args[1]['temperature'], 0.7)

    @patch('utils.config.ChatOpenAI')
    def test_minimax_custom_api_key(self, mock_chat):
        """API key from config overrides LLM_ENV."""
        from utils.config import get_llm
        config = {'type': 'MiniMax', 'name': 'MiniMax-M2.7', 'minimax_api_key': 'custom-key'}
        get_llm(config)
        self.assertEqual(mock_chat.call_args[1]['openai_api_key'], 'custom-key')

    @patch('utils.config.ChatOpenAI')
    def test_minimax_custom_api_base(self, mock_chat):
        """Custom API base URL from config overrides LLM_ENV default."""
        from utils.config import get_llm
        config = {'type': 'MiniMax', 'name': 'MiniMax-M2.7', 'minimax_api_base': 'https://custom.minimax.io/v1'}
        get_llm(config)
        self.assertEqual(mock_chat.call_args[1]['openai_api_base'], 'https://custom.minimax.io/v1')

    @patch('utils.config.ChatOpenAI')
    def test_minimax_model_kwargs(self, mock_chat):
        """model_kwargs are forwarded to ChatOpenAI."""
        from utils.config import get_llm
        config = {'type': 'MiniMax', 'name': 'MiniMax-M2.7', 'model_kwargs': {'seed': 42}}
        get_llm(config)
        self.assertEqual(mock_chat.call_args[1]['model_kwargs'], {'seed': 42})

    @patch('utils.config.ChatOpenAI')
    def test_minimax_no_env_section(self, mock_chat):
        """Works gracefully when minimax section missing from LLM_ENV."""
        from utils.config import get_llm
        with patch('utils.config.LLM_ENV', {
            'openai': {'OPENAI_API_KEY': '', 'OPENAI_API_BASE': '', 'OPENAI_ORGANIZATION': ''},
        }):
            config = {'type': 'MiniMax', 'name': 'MiniMax-M2.7', 'minimax_api_key': 'k'}
            get_llm(config)
            mock_chat.assert_called_once()
            self.assertEqual(mock_chat.call_args[1]['openai_api_base'], 'https://api.minimax.io/v1')

    @patch('utils.config.ChatOpenAI')
    def test_minimax_m27_highspeed(self, mock_chat):
        """M2.7-highspeed model name is correctly passed."""
        from utils.config import get_llm
        config = {'type': 'MiniMax', 'name': 'MiniMax-M2.7-highspeed', 'temperature': 0.5}
        get_llm(config)
        self.assertEqual(mock_chat.call_args[1]['model_name'], 'MiniMax-M2.7-highspeed')


class TestChainWrapperMiniMax(unittest.TestCase):
    """Tests for MiniMax support in ChainWrapper."""

    @patch('utils.config.LLM_ENV', {
        'openai': {'OPENAI_API_KEY': '', 'OPENAI_API_BASE': '', 'OPENAI_ORGANIZATION': ''},
        'minimax': {'MINIMAX_API_KEY': 'test-key', 'MINIMAX_API_BASE': 'https://api.minimax.io/v1'},
    })
    @patch('utils.config.ChatOpenAI')
    @patch('utils.llm_chain.load_prompt')
    @patch('utils.llm_chain.LLMChain')
    def test_minimax_uses_openai_callback(self, mock_llm_chain, mock_load_prompt, mock_chat):
        """MiniMax provider uses get_openai_callback for cost tracking."""
        from easydict import EasyDict
        mock_load_prompt.return_value = MagicMock()
        mock_llm_instance = MagicMock()
        mock_chat.return_value = mock_llm_instance

        from utils.llm_chain import ChainWrapper, get_openai_callback
        llm_config = EasyDict({'type': 'MiniMax', 'name': 'MiniMax-M2.7', 'temperature': 0.5})
        wrapper = ChainWrapper(llm_config, 'dummy.prompt')
        self.assertEqual(wrapper.callback, get_openai_callback)

    @patch('utils.config.LLM_ENV', {
        'openai': {'OPENAI_API_KEY': '', 'OPENAI_API_BASE': '', 'OPENAI_ORGANIZATION': ''},
        'minimax': {'MINIMAX_API_KEY': 'test-key', 'MINIMAX_API_BASE': 'https://api.minimax.io/v1'},
    })
    @patch('utils.config.ChatOpenAI')
    @patch('utils.llm_chain.load_prompt')
    def test_minimax_structured_output(self, mock_load_prompt, mock_chat):
        """MiniMax provider supports structured output via with_structured_output."""
        from easydict import EasyDict
        mock_load_prompt.return_value = MagicMock()
        mock_llm_instance = MagicMock()
        mock_chat.return_value = mock_llm_instance

        schema = {'type': 'object', 'properties': {'label': {'type': 'string'}}}
        from utils.llm_chain import ChainWrapper
        llm_config = EasyDict({'type': 'MiniMax', 'name': 'MiniMax-M2.7', 'temperature': 0.5})
        wrapper = ChainWrapper(llm_config, 'dummy.prompt', json_schema=schema)
        # Structured output should have called with_structured_output
        mock_llm_instance.with_structured_output.assert_called_once_with(schema)

    @patch('utils.config.LLM_ENV', {
        'openai': {'OPENAI_API_KEY': '', 'OPENAI_API_BASE': '', 'OPENAI_ORGANIZATION': ''},
        'minimax': {'MINIMAX_API_KEY': 'test-key', 'MINIMAX_API_BASE': 'https://api.minimax.io/v1'},
    })
    @patch('utils.config.ChatOpenAI')
    @patch('utils.llm_chain.load_prompt')
    @patch('utils.llm_chain.LLMChain')
    def test_minimax_no_schema_uses_llmchain(self, mock_llm_chain, mock_load_prompt, mock_chat):
        """Without json_schema, MiniMax falls back to LLMChain."""
        from easydict import EasyDict
        mock_load_prompt.return_value = MagicMock()
        mock_llm_instance = MagicMock()
        mock_chat.return_value = mock_llm_instance

        from utils.llm_chain import ChainWrapper
        llm_config = EasyDict({'type': 'MiniMax', 'name': 'MiniMax-M2.7', 'temperature': 0.5})
        wrapper = ChainWrapper(llm_config, 'dummy.prompt')
        mock_llm_instance.with_structured_output.assert_not_called()
        mock_llm_chain.assert_called_once()


class TestLlmEnvConfig(unittest.TestCase):
    """Tests for MiniMax section in llm_env.yml."""

    def test_minimax_section_exists(self):
        """llm_env.yml has a minimax section with required keys."""
        import yaml
        env_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_env.yml')
        with open(env_path) as f:
            env = yaml.safe_load(f)
        self.assertIn('minimax', env)
        self.assertIn('MINIMAX_API_KEY', env['minimax'])
        self.assertIn('MINIMAX_API_BASE', env['minimax'])
        self.assertEqual(env['minimax']['MINIMAX_API_BASE'], 'https://api.minimax.io/v1')

    def test_config_default_mentions_minimax(self):
        """config_default.yml comment mentions MiniMax as a supported provider."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_default.yml')
        with open(config_path) as f:
            content = f.read()
        self.assertIn('MiniMax', content)


class TestNotImplementedPreserved(unittest.TestCase):
    """Ensure unknown providers still raise NotImplementedError."""

    @patch('utils.config.LLM_ENV', {
        'openai': {'OPENAI_API_KEY': '', 'OPENAI_API_BASE': '', 'OPENAI_ORGANIZATION': ''},
    })
    def test_unknown_type_raises(self):
        from utils.config import get_llm
        with self.assertRaises(NotImplementedError):
            get_llm({'type': 'UnknownProvider', 'name': 'test'})


if __name__ == '__main__':
    unittest.main()
