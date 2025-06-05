"""
   Unit tests for OpenAI provider implementation.
   """

from llm_contracts.contracts.base import PromptLengthContract, JSONFormatContract
from llm_contracts.validators.basic_validators import InputValidator, OutputValidator
from llm_contracts.core.exceptions import ProviderError, ContractViolationError
from llm_contracts.providers.openai_provider import OpenAIProvider
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestOpenAIProvider(unittest.TestCase):
    """Test the OpenAI provider implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the OpenAI package availability
        self.openai_available_patcher = patch(
            'llm_contracts.providers.openai_provider._has_openai', True)
        self.openai_available_patcher.start()

        # Mock OpenAI classes
        self.mock_openai = patch(
            'llm_contracts.providers.openai_provider.OpenAI')
        self.mock_async_openai = patch(
            'llm_contracts.providers.openai_provider.AsyncOpenAI')
        self.mock_chat_completion = patch(
            'llm_contracts.providers.openai_provider.ChatCompletion')

        self.mock_openai_client = self.mock_openai.start()
        self.mock_async_openai_client = self.mock_async_openai.start()
        self.mock_chat_completion_class = self.mock_chat_completion.start()

        # Set up mock instances
        self.mock_client = Mock()
        self.mock_async_client = Mock()
        self.mock_openai_client.return_value = self.mock_client
        self.mock_async_openai_client.return_value = self.mock_async_client

    def tearDown(self):
        """Clean up test fixtures."""
        self.openai_available_patcher.stop()
        self.mock_openai.stop()
        self.mock_async_openai.stop()
        self.mock_chat_completion.stop()

    def test_provider_initialization(self):
        """Test OpenAI provider initialization."""
        provider = OpenAIProvider(model="gpt-4")

        self.assertEqual(provider.provider_name, "openai")
        self.assertEqual(provider.model, "gpt-4")
        self.assertIsNotNone(provider.client)
        self.assertIsNotNone(provider.async_client)

    def test_provider_initialization_without_openai(self):
        """Test that provider raises error when OpenAI is not available."""
        with patch('llm_contracts.providers.openai_provider._has_openai', False):
            with self.assertRaises(ProviderError) as context:
                OpenAIProvider()

            self.assertIn("OpenAI Python package not installed",
                          str(context.exception))

    def test_prepare_input_string_prompt(self):
        """Test input preparation with string prompt."""
        provider = OpenAIProvider()

        params = provider._prepare_input("Hello world")

        self.assertEqual(params["model"], "gpt-3.5-turbo")
        self.assertEqual(params["messages"], [
                         {"role": "user", "content": "Hello world"}])

    def test_prepare_input_messages_format(self):
        """Test input preparation with messages format."""
        provider = OpenAIProvider()
        messages = [{"role": "user", "content": "Hello"}]

        params = provider._prepare_input(messages)

        self.assertEqual(params["messages"], messages)

    def test_prepare_input_with_validation(self):
        """Test input preparation with contract validation."""
        provider = OpenAIProvider()

        # Add input validator with length contract
        input_validator = InputValidator()
        input_validator.add_contract(PromptLengthContract(max_tokens=10))
        provider.set_input_validator(input_validator)

        # Test with valid input
        params = provider._prepare_input("Hello")
        self.assertIsNotNone(params)

        # Test with invalid input (too long)
        long_prompt = "This is a very long prompt that exceeds the token limit. " * 20
        with self.assertRaises(ContractViolationError):
            provider._prepare_input(long_prompt)

    def test_extract_content_chat_completion(self):
        """Test content extraction from ChatCompletion response."""
        provider = OpenAIProvider()

        # Mock ChatCompletion response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"

        # Mock the hasattr checks
        with patch('llm_contracts.providers.openai_provider.ChatCompletion') as mock_chat_completion:
            mock_chat_completion.__name__ = 'ChatCompletion'
            content = provider._extract_content(mock_response)
            self.assertEqual(content, "Test response")

    def test_extract_content_string(self):
        """Test content extraction from string response."""
        provider = OpenAIProvider()

        content = provider._extract_content("Test string")
        self.assertEqual(content, "Test string")

    def test_validate_output_with_contracts(self):
        """Test output validation with contracts."""
        provider = OpenAIProvider()

        # Add output validator with JSON contract
        output_validator = OutputValidator()
        output_validator.add_contract(JSONFormatContract())
        provider.set_output_validator(output_validator)

        # Test with valid JSON response object that extracts to JSON string
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '{"test": "value"}'

        # This should not raise an exception
        provider._validate_output(mock_response)

        # Test with invalid JSON response object
        mock_bad_response = Mock()
        mock_bad_response.choices = [Mock()]
        mock_bad_response.choices[0].message = Mock()
        mock_bad_response.choices[0].message.content = "invalid json"

        with self.assertRaises(ContractViolationError):
            provider._validate_output(mock_bad_response)

    def test_call_method_success(self):
        """Test successful API call."""
        provider = OpenAIProvider()

        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"

        provider.client.chat.completions.create.return_value = mock_response

        result = provider.call("Test prompt")

        self.assertEqual(result, mock_response)
        provider.client.chat.completions.create.assert_called_once()

    def test_call_method_with_validation_failure(self):
        """Test API call with validation failure."""
        provider = OpenAIProvider()

        # Add input validator with strict length limit
        input_validator = InputValidator()
        input_validator.add_contract(PromptLengthContract(max_tokens=1))
        provider.set_input_validator(input_validator)

        # Test with prompt that's too long
        with self.assertRaises(ContractViolationError):
            provider.call("This prompt is too long for the contract")

    @patch('llm_contracts.providers.openai_provider.asyncio')
    async def test_acall_method(self):
        """Test async API call."""
        provider = OpenAIProvider()

        # Mock async response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Async test response"

        provider.async_client.chat.completions.create.return_value = mock_response

        result = await provider.acall("Test prompt")

        self.assertEqual(result, mock_response)

    def test_estimate_tokens(self):
        """Test token estimation functionality."""
        provider = OpenAIProvider()

        # Test with short text
        tokens = provider.estimate_tokens("Hello world")
        self.assertGreater(tokens, 0)
        self.assertIsInstance(tokens, int)

        # Test with longer text
        long_text = "This is a longer text that should have more tokens. " * 10
        long_tokens = provider.estimate_tokens(long_text)
        self.assertGreater(long_tokens, tokens)

    def test_str_representation(self):
        """Test string representation of provider."""
        provider = OpenAIProvider(model="gpt-4")

        str_repr = str(provider)
        self.assertIn("OpenAIProvider", str_repr)
        self.assertIn("gpt-4", str_repr)

    def test_get_available_models_success(self):
        """Test getting available models successfully."""
        provider = OpenAIProvider()

        # Mock models response
        mock_model1 = Mock()
        mock_model1.id = "gpt-3.5-turbo"
        mock_model2 = Mock()
        mock_model2.id = "gpt-4"

        mock_models_response = Mock()
        mock_models_response.data = [mock_model1, mock_model2]

        provider.client.models.list.return_value = mock_models_response

        models = provider.get_available_models()

        self.assertEqual(models, ["gpt-3.5-turbo", "gpt-4"])

    def test_get_available_models_failure(self):
        """Test handling of models list failure."""
        provider = OpenAIProvider()

        # Mock API failure
        provider.client.models.list.side_effect = Exception("API Error")

        with self.assertRaises(ProviderError):
            provider.get_available_models()


class TestOpenAIProviderIntegration(unittest.TestCase):
    """Integration tests for OpenAI provider without actual API calls."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock OpenAI availability
        self.openai_patcher = patch(
            'llm_contracts.providers.openai_provider._has_openai', True)
        self.openai_patcher.start()

        # Mock the OpenAI classes
        self.mock_openai = patch(
            'llm_contracts.providers.openai_provider.OpenAI')
        self.mock_async_openai = patch(
            'llm_contracts.providers.openai_provider.AsyncOpenAI')

        self.mock_openai_class = self.mock_openai.start()
        self.mock_async_openai_class = self.mock_async_openai.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.openai_patcher.stop()
        self.mock_openai.stop()
        self.mock_async_openai.stop()

    def test_full_workflow_with_contracts(self):
        """Test complete workflow with input and output contracts."""
        provider = OpenAIProvider(model="gpt-3.5-turbo")

        # Set up input validation
        input_validator = InputValidator("test_input")
        input_validator.add_contract(PromptLengthContract(max_tokens=100))
        provider.set_input_validator(input_validator)

        # Set up output validation
        output_validator = OutputValidator("test_output")
        output_validator.add_contract(JSONFormatContract())
        provider.set_output_validator(output_validator)

        # Mock a successful API response with valid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"response": "Hello!"}'

        provider.client.chat.completions.create.return_value = mock_response

        # Test successful call
        result = provider.call("Generate a JSON response")

        self.assertEqual(result, mock_response)

        # Verify the API was called with correct parameters
        call_args = provider.client.chat.completions.create.call_args
        self.assertIn("messages", call_args[1])
        self.assertEqual(call_args[1]["model"], "gpt-3.5-turbo")


if __name__ == '__main__':
    unittest.main()
