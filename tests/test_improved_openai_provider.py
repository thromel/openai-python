"""
Tests for the improved OpenAI provider with performance optimizations.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Any, Dict

# Import the improved provider
from llm_contracts.providers.improved_openai_provider import (
    ImprovedOpenAIProvider,
    ContractMetrics,
    ContractCircuitBreaker,
    CircuitState,
    StreamingValidator,
    StreamValidationResult
)
from llm_contracts.contracts.base import (
    PromptLengthContract,
    JSONFormatContract,
    ContentPolicyContract
)
from llm_contracts.core.exceptions import ContractViolationError, ProviderError


class TestContractMetrics:
    """Test the metrics collection system."""

    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        metrics = ContractMetrics()

        assert len(metrics.validation_times) == 0
        assert len(metrics.violation_counts) == 0
        assert len(metrics.auto_fix_success_rate) == 0
        assert len(metrics.contract_performance) == 0

    def test_record_validation_time(self):
        """Test recording validation times."""
        metrics = ContractMetrics()

        # Record successful validation
        metrics.record_validation_time("test_contract", 0.05, violated=False)

        assert "test_contract" in metrics.validation_times
        assert len(metrics.validation_times["test_contract"]) == 1
        assert metrics.validation_times["test_contract"][0] == 0.05
        assert metrics.violation_counts["test_contract"] == 0

        # Record violated validation
        metrics.record_validation_time("test_contract", 0.03, violated=True)

        assert len(metrics.validation_times["test_contract"]) == 2
        assert metrics.violation_counts["test_contract"] == 1
        assert metrics.contract_performance["test_contract"]["call_count"] == 2

    def test_auto_fix_attempt_recording(self):
        """Test recording auto-fix attempts."""
        metrics = ContractMetrics()

        # Record successful auto-fix
        metrics.record_auto_fix_attempt("test_contract", True)

        stats = metrics.auto_fix_success_rate["test_contract"]
        assert stats["total"] == 1
        assert stats["success"] == 1

        # Record failed auto-fix
        metrics.record_auto_fix_attempt("test_contract", False)

        assert stats["total"] == 2
        assert stats["success"] == 1

    def test_health_report(self):
        """Test health report generation."""
        metrics = ContractMetrics()

        # Add some data
        metrics.record_validation_time("fast_contract", 0.01)
        metrics.record_validation_time("slow_contract", 0.1)
        metrics.record_validation_time(
            "violated_contract", 0.05, violated=True)

        report = metrics.get_health_report()

        assert "total_validations" in report
        assert "violation_rate" in report
        assert "slowest_contracts" in report
        assert "most_violated_contracts" in report
        assert report["total_validations"] == 3
        assert report["violation_rate"] == 1 / \
            3  # 1 violation out of 3 validations


class TestContractCircuitBreaker:
    """Test the circuit breaker functionality."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initial state."""
        cb = ContractCircuitBreaker()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.last_failure is None
        assert not cb.should_skip()

    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opens after threshold failures."""
        cb = ContractCircuitBreaker(failure_threshold=3)

        # Record failures below threshold
        cb.record_failure("test_contract")
        cb.record_failure("test_contract")
        assert cb.state == CircuitState.CLOSED
        assert not cb.should_skip()

        # Hit threshold
        cb.record_failure("test_contract")
        assert cb.state == CircuitState.OPEN
        assert cb.should_skip()

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = ContractCircuitBreaker(failure_threshold=1, timeout=0.1)

        # Trigger circuit breaker
        cb.record_failure("test_contract")
        assert cb.state == CircuitState.OPEN
        assert cb.should_skip()

        # Wait for timeout
        time.sleep(0.2)
        assert not cb.should_skip()  # Should transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

        # Record success to close circuit
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert not cb.should_skip()


class TestStreamingValidator:
    """Test the streaming validation system."""

    def test_streaming_validator_initialization(self):
        """Test streaming validator setup."""
        # Mock contracts
        streaming_contract = Mock()
        streaming_contract.supports_streaming = True

        regular_contract = Mock()
        regular_contract.supports_streaming = False

        validator = StreamingValidator([streaming_contract, regular_contract])

        assert len(validator.chunk_validators) == 1
        assert len(validator.final_validators) == 1
        assert validator.buffer == ""

    @pytest.mark.asyncio
    async def test_chunk_validation(self):
        """Test chunk-by-chunk validation."""
        # Mock streaming contract
        streaming_contract = Mock()
        streaming_contract.supports_streaming = True
        streaming_contract.should_validate_at_length.return_value = True

        # Mock validation result
        mock_result = Mock()
        mock_result.is_violation = False
        streaming_contract.validate_partial = AsyncMock(
            return_value=mock_result)

        validator = StreamingValidator([streaming_contract])

        # Validate chunk
        result = await validator.validate_chunk("test chunk")

        assert not result.should_terminate
        assert len(result.partial_results) == 1
        assert validator.buffer == "test chunk"

    @pytest.mark.asyncio
    async def test_critical_violation_termination(self):
        """Test that critical violations terminate streaming."""
        # Mock streaming contract with critical violation
        streaming_contract = Mock()
        streaming_contract.supports_streaming = True
        streaming_contract.should_validate_at_length.return_value = True

        # Mock critical violation
        mock_result = Mock()
        mock_result.is_violation = True
        mock_result.severity = Mock()
        mock_result.severity.name = "CRITICAL"
        streaming_contract.validate_partial = AsyncMock(
            return_value=mock_result)

        validator = StreamingValidator([streaming_contract])

        # Validate chunk
        result = await validator.validate_chunk("bad content")

        assert result.should_terminate
        assert result.violation == mock_result


class TestImprovedOpenAIProvider:
    """Test the main ImprovedOpenAIProvider class."""

    @patch('llm_contracts.providers.improved_openai_provider._has_openai', True)
    @patch('llm_contracts.providers.improved_openai_provider.OpenAI')
    @patch('llm_contracts.providers.improved_openai_provider.AsyncOpenAI')
    def test_provider_initialization(self, mock_async_openai, mock_openai):
        """Test provider initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_async_client = Mock()
        mock_async_openai.return_value = mock_async_client

        provider = ImprovedOpenAIProvider(api_key="test-key")

        assert provider._client == mock_client
        assert provider._async_client == mock_async_client
        assert len(provider.input_contracts) == 0
        assert len(provider.output_contracts) == 0
        assert provider.max_retries == 3
        assert provider.auto_remediation is True

    def test_provider_initialization_without_openai(self):
        """Test provider initialization fails without OpenAI package."""
        with patch('llm_contracts.providers.improved_openai_provider._has_openai', False):
            with pytest.raises(ProviderError, match="OpenAI Python package not installed"):
                ImprovedOpenAIProvider()

    @patch('llm_contracts.providers.improved_openai_provider._has_openai', True)
    @patch('llm_contracts.providers.improved_openai_provider.OpenAI')
    @patch('llm_contracts.providers.improved_openai_provider.AsyncOpenAI')
    def test_contract_management(self, mock_async_openai, mock_openai):
        """Test adding and managing contracts."""
        mock_openai.return_value = Mock()
        mock_async_openai.return_value = Mock()

        provider = ImprovedOpenAIProvider()

        # Add input contract
        input_contract = PromptLengthContract(max_tokens=1000)
        provider.add_input_contract(input_contract)

        assert len(provider.input_contracts) == 1
        assert provider.input_contracts[0] == input_contract

        # Add output contract
        output_contract = JSONFormatContract()
        provider.add_output_contract(output_contract)

        assert len(provider.output_contracts) == 1
        assert provider.output_contracts[0] == output_contract

        # Test auto-categorization
        auto_contract = ContentPolicyContract(banned_patterns=["test"])
        provider.add_contract(auto_contract)

        # Should be added as output contract by default
        assert len(provider.output_contracts) == 2

    @patch('llm_contracts.providers.improved_openai_provider._has_openai', True)
    @patch('llm_contracts.providers.improved_openai_provider.OpenAI')
    @patch('llm_contracts.providers.improved_openai_provider.AsyncOpenAI')
    def test_direct_attribute_passthrough(self, mock_async_openai, mock_openai):
        """Test that non-wrapped attributes pass through directly."""
        mock_client = Mock()
        mock_client.models = "models_attr"
        mock_client.files = "files_attr"
        mock_openai.return_value = mock_client
        mock_async_openai.return_value = Mock()

        provider = ImprovedOpenAIProvider()

        # These should be direct passthroughs with zero overhead
        assert provider.models == "models_attr"
        assert provider.files == "files_attr"

        # Test fallback __getattr__ for other attributes
        mock_client.some_other_attr = "other_value"
        assert provider.some_other_attr == "other_value"

    @patch('llm_contracts.providers.improved_openai_provider._has_openai', True)
    @patch('llm_contracts.providers.improved_openai_provider.OpenAI')
    @patch('llm_contracts.providers.improved_openai_provider.AsyncOpenAI')
    def test_metrics_integration(self, mock_async_openai, mock_openai):
        """Test metrics collection integration."""
        mock_openai.return_value = Mock()
        mock_async_openai.return_value = Mock()

        provider = ImprovedOpenAIProvider()

        # Test metrics access
        metrics = provider.get_metrics()
        assert "total_validations" in metrics
        assert "violation_rate" in metrics

        # Test circuit breaker reset
        provider._circuit_breaker.state = CircuitState.OPEN
        provider.reset_circuit_breaker()
        assert provider._circuit_breaker.state == CircuitState.CLOSED

    @patch('llm_contracts.providers.improved_openai_provider._has_openai', True)
    @patch('llm_contracts.providers.improved_openai_provider.OpenAI')
    @patch('llm_contracts.providers.improved_openai_provider.AsyncOpenAI')
    def test_input_validation_sync(self, mock_async_openai, mock_openai):
        """Test synchronous input validation."""
        mock_openai.return_value = Mock()
        mock_async_openai.return_value = Mock()

        provider = ImprovedOpenAIProvider()

        # Add a contract that will fail
        failing_contract = Mock()
        failing_contract.name = "test_contract"
        failing_contract.validate.return_value = Mock(
            is_valid=False, message="Test failure")
        provider.add_input_contract(failing_contract)

        # Test validation failure
        with pytest.raises(ContractViolationError, match="Input validation failed"):
            provider._validate_input_sync(
                messages=[{"role": "user", "content": "test"}])

    @patch('llm_contracts.providers.improved_openai_provider._has_openai', True)
    @patch('llm_contracts.providers.improved_openai_provider.OpenAI')
    @patch('llm_contracts.providers.improved_openai_provider.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_input_validation_async(self, mock_async_openai, mock_openai):
        """Test asynchronous input validation."""
        mock_openai.return_value = Mock()
        mock_async_openai.return_value = Mock()

        provider = ImprovedOpenAIProvider()

        # Add a contract that will pass
        passing_contract = Mock()
        passing_contract.name = "test_contract"
        passing_contract.validate = AsyncMock(return_value=Mock(is_valid=True))
        provider.add_input_contract(passing_contract)

        # Test validation success (should not raise)
        await provider._validate_input_async(messages=[{"role": "user", "content": "test"}])

        # Verify contract was called
        passing_contract.validate.assert_called_once()

    @patch('llm_contracts.providers.improved_openai_provider._has_openai', True)
    @patch('llm_contracts.providers.improved_openai_provider.OpenAI')
    @patch('llm_contracts.providers.improved_openai_provider.AsyncOpenAI')
    def test_output_validation_with_auto_fix(self, mock_async_openai, mock_openai):
        """Test output validation with auto-remediation."""
        mock_openai.return_value = Mock()
        mock_async_openai.return_value = Mock()

        provider = ImprovedOpenAIProvider()

        # Add a contract with auto-fix suggestion
        contract_with_fix = Mock()
        contract_with_fix.name = "test_contract"
        validation_result = Mock()
        validation_result.is_valid = False
        validation_result.auto_fix_suggestion = "corrected content"
        validation_result.message = "Test failure"
        contract_with_fix.validate.return_value = validation_result
        provider.add_output_contract(contract_with_fix)

        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "original content"

        # Test auto-fix
        corrected_response = provider._validate_output_sync(mock_response)

        # Should have been corrected
        assert corrected_response.choices[0].message.content == "corrected content"

    def test_content_extraction(self):
        """Test content extraction from OpenAI responses."""
        with patch('llm_contracts.providers.improved_openai_provider._has_openai', True):
            with patch('llm_contracts.providers.improved_openai_provider.OpenAI'):
                with patch('llm_contracts.providers.improved_openai_provider.AsyncOpenAI'):
                    provider = ImprovedOpenAIProvider()

                    # Test chat completion response
                    chat_response = Mock()
                    chat_response.choices = [Mock()]
                    chat_response.choices[0].message = Mock()
                    chat_response.choices[0].message.content = "test content"

                    content = provider._extract_content(chat_response)
                    assert content == "test content"

                    # Test completion response
                    completion_response = Mock()
                    completion_response.choices = [Mock()]
                    completion_response.choices[0].text = "completion text"

                    content = provider._extract_content(completion_response)
                    assert content == "completion text"

                    # Test string response
                    content = provider._extract_content("direct string")
                    assert content == "direct string"

    def test_string_representation(self):
        """Test string representation of provider."""
        with patch('llm_contracts.providers.improved_openai_provider._has_openai', True):
            with patch('llm_contracts.providers.improved_openai_provider.OpenAI'):
                with patch('llm_contracts.providers.improved_openai_provider.AsyncOpenAI'):
                    provider = ImprovedOpenAIProvider()

                    str_repr = str(provider)
                    assert "ImprovedOpenAIProvider" in str_repr
                    assert "contracts=0" in str_repr
                    assert "metrics_enabled=True" in str_repr

                    # Add contracts and test again
                    provider.add_input_contract(Mock())
                    provider.add_output_contract(Mock())

                    str_repr = str(provider)
                    assert "contracts=2" in str_repr


if __name__ == "__main__":
    pytest.main([__file__])
