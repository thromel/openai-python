"""
Exception classes for the LLM Design by Contract framework.
"""

from typing import Any, Dict, Optional


class LLMContractError(Exception):
    """Base exception for all LLM contract-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ContractViolationError(LLMContractError):
    """Raised when a contract is violated during validation."""

    def __init__(
        self,
        message: str,
        contract_type: str,
        contract_name: str,
        violation_details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, violation_details)
        self.contract_type = contract_type
        self.contract_name = contract_name


class ValidationError(LLMContractError):
    """Raised when validation fails for input or output."""

    def __init__(
        self,
        message: str,
        validation_stage: str,  # 'input' or 'output'
        validator_name: str,
        error_details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_details)
        self.validation_stage = validation_stage
        self.validator_name = validator_name


class ProviderError(LLMContractError):
    """Raised when there's an error with the LLM provider adapter."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        provider_details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, provider_details)
        self.provider_name = provider_name


class ContractParsingError(LLMContractError):
    """Raised when contract specification cannot be parsed."""
    pass


class TemporalViolationError(ContractViolationError):
    """Raised when temporal/sequence contracts are violated."""

    def __init__(
        self,
        message: str,
        expected_sequence: str,
        actual_sequence: str,
        violation_details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "temporal", "sequence", violation_details)
        self.expected_sequence = expected_sequence
        self.actual_sequence = actual_sequence
