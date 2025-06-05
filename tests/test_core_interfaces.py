"""
Tests for core interfaces of the LLM Design by Contract framework.
"""

import pytest
from typing import Any, Dict, List, Optional
from llm_contracts.core.interfaces import (
    ContractBase,
    ValidatorBase,
    ProviderAdapter,
    ContractType,
    ValidationResult,
    ConversationContext
)


class MockContract(ContractBase):
    """Mock contract for testing."""

    def _get_contract_type(self) -> ContractType:
        return ContractType.INPUT

    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        # Simple validation: data should be a string
        if isinstance(data, str):
            return ValidationResult(True, "Valid string input")
        return ValidationResult(False, "Input must be a string")


class MockValidator(ValidatorBase):
    """Mock validator for testing."""

    def validate_all(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        results = []
        for contract in self.contracts:
            results.append(contract.validate(data, context))
        return results


class MockProvider(ProviderAdapter):
    """Mock provider for testing."""

    def call(self, prompt, **kwargs: Any) -> str:
        return f"Mock response to: {prompt}"

    async def acall(self, prompt, **kwargs: Any) -> str:
        return f"Mock async response to: {prompt}"

    def stream(self, prompt, **kwargs: Any):
        for word in f"Mock streaming response to: {prompt}".split():
            yield word

    async def astream(self, prompt, **kwargs: Any):
        for word in f"Mock async streaming response to: {prompt}".split():
            yield word


def test_contract_base():
    """Test ContractBase functionality."""
    contract = MockContract("test_contract", "A test contract")

    assert contract.name == "test_contract"
    assert contract.description == "A test contract"
    assert contract.is_required is True
    assert contract.contract_type == ContractType.INPUT


def test_validator_base():
    """Test ValidatorBase functionality."""
    validator = MockValidator("test_validator")
    contract = MockContract("test_contract")

    # Test adding contracts
    validator.add_contract(contract)
    assert len(validator.contracts) == 1


def test_provider_adapter():
    """Test ProviderAdapter functionality."""
    provider = MockProvider("test_provider")
    validator = MockValidator("test_validator")

    # Test setting validators
    provider.set_input_validator(validator)
    provider.set_output_validator(validator)

    assert provider.input_validator is not None
    assert provider.output_validator is not None


def test_conversation_context():
    """Test ConversationContext functionality."""
    context = ConversationContext("test_conversation")

    assert context.conversation_id == "test_conversation"
    assert len(context.messages) == 0


if __name__ == "__main__":
    pytest.main([__file__])
