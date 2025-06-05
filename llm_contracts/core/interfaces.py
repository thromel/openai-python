"""
Core abstract interfaces for the LLM Design by Contract framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Iterator
from enum import Enum
import time


class ContractType(Enum):
    """Types of contracts supported by the framework."""
    INPUT = "input"
    OUTPUT = "output"
    TEMPORAL = "temporal"
    SEMANTIC_CONSISTENCY = "semantic_consistency"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOMAIN_SPECIFIC = "domain_specific"


class ValidationResult:
    """Result of a contract validation."""

    def __init__(
        self,
        is_valid: bool,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        auto_fix_suggestion: Optional[str] = None
    ):
        self.is_valid = is_valid
        self.message = message
        self.details = details or {}
        self.auto_fix_suggestion = auto_fix_suggestion
        self.timestamp = time.time()


class ContractBase(ABC):
    """Abstract base class for all contract types."""

    def __init__(self, name: str, description: str = "", is_required: bool = True):
        self.name = name
        self.description = description
        self.is_required = is_required
        self.contract_type = self._get_contract_type()

    @abstractmethod
    def _get_contract_type(self) -> ContractType:
        """Return the type of this contract."""
        pass

    @abstractmethod
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate the given data against this contract."""
        pass

    def __str__(self) -> str:  # type: ignore
        return f"{self.__class__.__name__}(name='{self.name}', type={self.contract_type.value})"


class ValidatorBase(ABC):
    """Abstract base class for validators that enforce contracts."""

    def __init__(self, name: str):
        self.name = name
        self.contracts: List[ContractBase] = []

    def add_contract(self, contract: ContractBase) -> None:
        """Add a contract to this validator."""
        self.contracts.append(contract)

    def remove_contract(self, contract_name: str) -> bool:
        """Remove a contract by name. Returns True if found and removed."""
        for i, contract in enumerate(self.contracts):
            if contract.name == contract_name:
                del self.contracts[i]
                return True
        return False

    @abstractmethod
    def validate_all(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate data against all contracts and return results."""
        pass

    def get_failed_validations(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Get only the failed validation results."""
        return [result for result in self.validate_all(data, context) if not result.is_valid]


class ProviderAdapter(ABC):
    """Abstract base class for LLM provider adapters."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.input_validator: Optional[ValidatorBase] = None
        self.output_validator: Optional[ValidatorBase] = None

    def set_input_validator(self, validator: ValidatorBase) -> None:
        """Set the input validator for this provider."""
        self.input_validator = validator

    def set_output_validator(self, validator: ValidatorBase) -> None:
        """Set the output validator for this provider."""
        self.output_validator = validator

    @abstractmethod
    def call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        **kwargs: Any
    ) -> Any:
        """Make a synchronous call to the LLM provider."""
        pass

    @abstractmethod
    async def acall(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        **kwargs: Any
    ) -> Any:
        """Make an asynchronous call to the LLM provider."""
        pass

    @abstractmethod
    def stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        **kwargs: Any
    ) -> Iterator[Any]:
        """Make a streaming call to the LLM provider."""
        pass

    @abstractmethod
    async def astream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Make an asynchronous streaming call to the LLM provider."""
        pass

    def validate_input(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate input data if input validator is set."""
        if self.input_validator:
            return self.input_validator.validate_all(data, context)
        return []

    def validate_output(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate output data if output validator is set."""
        if self.output_validator:
            return self.output_validator.validate_all(data, context)
        return []


class ConversationContext:
    """Context manager for multi-turn conversations."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.created_at = time.time()

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation context."""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.messages.append(message)

    def get_last_n_messages(self, n: int) -> List[Dict[str, Any]]:
        """Get the last n messages from the conversation."""
        return self.messages[-n:] if n > 0 else []

    def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Get all messages from a specific role."""
        return [msg for msg in self.messages if msg["role"] == role]


class StreamingValidatorInterface(ABC):
    """Interface for validators that can handle streaming data."""

    @abstractmethod
    def validate_chunk(self, chunk: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a single chunk of streaming data."""
        pass

    @abstractmethod
    def validate_complete_stream(self, complete_data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate the complete streamed data once streaming is finished."""
        pass
