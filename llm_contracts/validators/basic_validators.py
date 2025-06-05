"""
Basic validator implementations for input and output validation.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from ..core.interfaces import ValidatorBase, ValidationResult, ContractBase


class InputValidator(ValidatorBase):
    """Basic input validator that enforces input contracts."""

    def __init__(self, name: str = "input_validator"):
        super().__init__(name)

    def validate_all(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate input data against all contracts."""
        results: List[ValidationResult] = []
        for contract in self.contracts:
            try:
                result = contract.validate(data, context)
                results.append(result)
            except Exception as e:
                # Handle validation errors gracefully
                results.append(ValidationResult(
                    False,
                    f"Contract '{contract.name}' validation failed: {str(e)}",
                    details={"contract_name": contract.name, "error": str(e)}
                ))
        return results


class OutputValidator(ValidatorBase):
    """Basic output validator that enforces output contracts."""

    def __init__(self, name: str = "output_validator"):
        super().__init__(name)

    def validate_all(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate output data against all contracts."""
        results: List[ValidationResult] = []
        for contract in self.contracts:
            try:
                result = contract.validate(data, context)
                results.append(result)
            except Exception as e:
                # Handle validation errors gracefully
                results.append(ValidationResult(
                    False,
                    f"Contract '{contract.name}' validation failed: {str(e)}",
                    details={"contract_name": contract.name, "error": str(e)}
                ))
        return results


class CompositeValidator(ValidatorBase):
    """Validator that can combine multiple validator types."""

    def __init__(self, name: str = "composite_validator"):
        super().__init__(name)
        self.validators: List[ValidatorBase] = []

    def add_validator(self, validator: ValidatorBase) -> None:
        """Add a sub-validator to this composite."""
        self.validators.append(validator)

    def validate_all(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate data against all contracts in all sub-validators."""
        all_results: List[ValidationResult] = []

        # First validate with our own contracts
        for contract in self.contracts:
            try:
                result = contract.validate(data, context)
                all_results.append(result)
            except Exception as e:
                all_results.append(ValidationResult(
                    False,
                    f"Contract '{contract.name}' validation failed: {str(e)}",
                    details={"contract_name": contract.name, "error": str(e)}
                ))

        # Then validate with sub-validators
        for validator in self.validators:
            try:
                results = validator.validate_all(data, context)
                all_results.extend(results)
            except Exception as e:
                all_results.append(ValidationResult(
                    False,
                    f"Sub-validator '{validator.name}' failed: {str(e)}",
                    details={"validator_name": validator.name, "error": str(e)}
                ))

        return all_results


class ConditionalValidator(ValidatorBase):
    """Validator that applies contracts conditionally based on context."""

    def __init__(self, name: str = "conditional_validator"):
        super().__init__(name)
        self.conditional_contracts: List[Tuple[Callable[[
            Any, Optional[Dict[str, Any]]], bool], ContractBase]] = []

    def add_conditional_contract(self, condition: Callable[[Any, Optional[Dict[str, Any]]], bool], contract: ContractBase) -> None:
        """Add a contract that is only applied if condition returns True."""
        self.conditional_contracts.append((condition, contract))

    def validate_all(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate data against applicable contracts."""
        results: List[ValidationResult] = []

        # Always validate unconditional contracts
        for contract in self.contracts:
            try:
                result = contract.validate(data, context)
                results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    False,
                    f"Contract '{contract.name}' validation failed: {str(e)}",
                    details={"contract_name": contract.name, "error": str(e)}
                ))

        # Validate conditional contracts
        for condition, contract in self.conditional_contracts:
            try:
                if condition(data, context):
                    result = contract.validate(data, context)
                    results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    False,
                    f"Conditional contract '{contract.name}' validation failed: {str(e)}",
                    details={"contract_name": contract.name, "error": str(e)}
                ))

        return results
