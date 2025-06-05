"""
Base contract classes for the LLM Design by Contract framework taxonomy.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
import re
import json
import time
from ..core.interfaces import ContractBase, ContractType, ValidationResult


class InputContract(ContractBase):
    """Base class for input contracts (preconditions)."""

    def _get_contract_type(self) -> ContractType:
        return ContractType.INPUT


class OutputContract(ContractBase):
    """Base class for output contracts (postconditions)."""

    def _get_contract_type(self) -> ContractType:
        return ContractType.OUTPUT


class TemporalContract(ContractBase):
    """Base class for temporal/sequence contracts."""

    def _get_contract_type(self) -> ContractType:
        return ContractType.TEMPORAL


class SemanticConsistencyContract(ContractBase):
    """Base class for semantic consistency contracts."""

    def _get_contract_type(self) -> ContractType:
        return ContractType.SEMANTIC_CONSISTENCY


class PerformanceContract(ContractBase):
    """Base class for performance and resource contracts."""

    def _get_contract_type(self) -> ContractType:
        return ContractType.PERFORMANCE


class SecurityContract(ContractBase):
    """Base class for security and safety contracts."""

    def _get_contract_type(self) -> ContractType:
        return ContractType.SECURITY


class DomainSpecificContract(ContractBase):
    """Base class for domain-specific contracts."""

    def _get_contract_type(self) -> ContractType:
        return ContractType.DOMAIN_SPECIFIC


# Concrete implementations of common contract patterns

class PromptLengthContract(InputContract):
    """Contract to validate prompt length constraints."""

    def __init__(self, max_tokens: int, name: str = "prompt_length", description: str = ""):
        super().__init__(
            name, description or f"Prompt must not exceed {max_tokens} tokens")
        self.max_tokens = max_tokens

    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate that prompt length is within limits."""
        if isinstance(data, str):
            # Simple token estimation (rough approximation)
            estimated_tokens = len(data.split()) * 1.3  # Rough estimate
            if estimated_tokens <= self.max_tokens:
                return ValidationResult(True, f"Prompt length OK ({estimated_tokens:.0f} tokens)")
            else:
                return ValidationResult(
                    False,
                    f"Prompt too long: {estimated_tokens:.0f} tokens > {self.max_tokens}",
                    auto_fix_suggestion=f"Truncate prompt to approximately {self.max_tokens} tokens"
                )
        elif isinstance(data, list):
            # Handle message format
            total_tokens = 0
            for msg in data:
                if isinstance(msg, dict) and "content" in msg:
                    total_tokens += len(str(msg["content"]).split()) * 1.3

            if total_tokens <= self.max_tokens:
                return ValidationResult(True, f"Messages length OK ({total_tokens:.0f} tokens)")
            else:
                return ValidationResult(
                    False,
                    f"Messages too long: {total_tokens:.0f} tokens > {self.max_tokens}",
                    auto_fix_suggestion="Remove older messages or truncate content"
                )

        return ValidationResult(False, "Invalid input format for prompt length validation")


class JSONFormatContract(OutputContract):
    """Contract to validate JSON output format."""

    def __init__(self, schema: Optional[Dict[str, Any]] = None, name: str = "json_format", description: str = ""):
        super().__init__(name, description or "Output must be valid JSON")
        self.schema = schema

    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate that output is valid JSON."""
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data

            # Basic schema validation if provided
            if self.schema:
                if not self._validate_schema(parsed, self.schema):
                    return ValidationResult(
                        False,
                        "JSON does not match required schema",
                        auto_fix_suggestion="Ensure JSON contains required fields"
                    )

            return ValidationResult(True, "Valid JSON format")

        except json.JSONDecodeError as e:
            return ValidationResult(
                False,
                f"Invalid JSON: {str(e)}",
                auto_fix_suggestion="Fix JSON syntax errors"
            )

    def _validate_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Basic schema validation (simplified)."""
        if not isinstance(data, dict):
            return False

        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                return False

        return True


class ContentPolicyContract(SecurityContract):
    """Contract to validate content against policy rules."""

    def __init__(self, banned_patterns: List[str], name: str = "content_policy", description: str = ""):
        super().__init__(name, description or "Content must comply with policy")
        self.banned_patterns = [re.compile(
            pattern, re.IGNORECASE) for pattern in banned_patterns]

    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate content against policy rules."""
        content = str(data)

        for pattern in self.banned_patterns:
            if pattern.search(content):
                return ValidationResult(
                    False,
                    f"Content violates policy: matches pattern '{pattern.pattern}'",
                    auto_fix_suggestion="Remove or rephrase problematic content"
                )

        return ValidationResult(True, "Content complies with policy")


class PromptInjectionContract(SecurityContract):
    """Contract to detect potential prompt injection attacks."""

    def __init__(self, name: str = "prompt_injection", description: str = ""):
        super().__init__(name, description or "Input must not contain prompt injection")
        # Common prompt injection patterns
        self.injection_patterns = [
            re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
            re.compile(r"forget\s+everything", re.IGNORECASE),
            re.compile(r"new\s+instructions", re.IGNORECASE),
            re.compile(r"system\s*:\s*", re.IGNORECASE),
            re.compile(r"<\s*system\s*>", re.IGNORECASE),
        ]

    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate input for prompt injection patterns."""
        content = str(data)

        for pattern in self.injection_patterns:
            if pattern.search(content):
                return ValidationResult(
                    False,
                    f"Potential prompt injection detected: '{pattern.pattern}'",
                    auto_fix_suggestion="Remove or sanitize suspicious instructions"
                )

        return ValidationResult(True, "No prompt injection detected")


class ResponseTimeContract(PerformanceContract):
    """Contract to validate response time constraints."""

    def __init__(self, max_seconds: float, name: str = "response_time", description: str = ""):
        super().__init__(
            name, description or f"Response must complete within {max_seconds} seconds")
        self.max_seconds = max_seconds
        self.start_time: Optional[float] = None

    def start_timing(self) -> None:
        """Start timing the operation."""
        self.start_time = time.time()

    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate response time."""
        if self.start_time is None:
            return ValidationResult(False, "Timing not started")

        elapsed = time.time() - self.start_time

        if elapsed <= self.max_seconds:
            return ValidationResult(True, f"Response time OK ({elapsed:.2f}s)")
        else:
            return ValidationResult(
                False,
                f"Response too slow: {elapsed:.2f}s > {self.max_seconds}s",
                auto_fix_suggestion="Consider using a faster model or optimizing the prompt"
            )


class ConversationConsistencyContract(SemanticConsistencyContract):
    """Contract to validate consistency across conversation turns."""

    def __init__(self, name: str = "conversation_consistency", description: str = ""):
        super().__init__(name, description or "Responses must be consistent with conversation history")

    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate conversation consistency."""
        if not context or "conversation_history" not in context:
            return ValidationResult(True, "No conversation history to check")

        current_response = str(data)
        history = context["conversation_history"]

        # Simple consistency check - look for direct contradictions
        # This is a simplified implementation; real semantic consistency would need NLP models
        contradiction_keywords = ["no", "not",
                                  "never", "incorrect", "wrong", "false"]

        for msg in history:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                prev_content = str(msg.get("content", ""))

                # Very basic contradiction detection
                if any(keyword in current_response.lower() for keyword in contradiction_keywords):
                    if self._potential_contradiction(prev_content, current_response):
                        return ValidationResult(
                            False,
                            "Potential contradiction with previous response detected",
                            auto_fix_suggestion="Ensure response is consistent with conversation history"
                        )

        return ValidationResult(True, "Response appears consistent with conversation history")

    def _potential_contradiction(self, prev_content: str, current_content: str) -> bool:
        """Simple heuristic to detect potential contradictions."""
        # This is a very basic implementation
        # Real implementation would use semantic similarity models
        prev_words = set(prev_content.lower().split())
        current_words = set(current_content.lower().split())

        # If there's significant overlap but contradiction keywords, flag it
        overlap = len(prev_words.intersection(current_words))
        return overlap > 3  # Arbitrary threshold


class MedicalDisclaimerContract(DomainSpecificContract):
    """Contract to ensure medical advice includes appropriate disclaimers."""

    def __init__(self, name: str = "medical_disclaimer", description: str = ""):
        super().__init__(name, description or "Medical advice must include disclaimer")
        self.medical_keywords = ["diagnose", "treatment",
                                 "medication", "symptoms", "disease", "condition"]
        self.disclaimer_keywords = [
            "doctor", "physician", "medical professional", "consult", "disclaimer"]

    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate medical disclaimer presence."""
        content = str(data).lower()

        # Check if content contains medical advice
        has_medical_content = any(
            keyword in content for keyword in self.medical_keywords)

        if has_medical_content:
            # Check if disclaimer is present
            has_disclaimer = any(
                keyword in content for keyword in self.disclaimer_keywords)

            if not has_disclaimer:
                return ValidationResult(
                    False,
                    "Medical advice detected without appropriate disclaimer",
                    auto_fix_suggestion="Add disclaimer to consult medical professional"
                )

        return ValidationResult(True, "Medical disclaimer requirements met")
