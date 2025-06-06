"""
Improved OpenAI provider with performance-optimized selective proxying and contract enforcement.

This implementation addresses the critical performance and compatibility issues by:
1. Using selective proxying to avoid __getattr__ overhead
2. Pre-wrapping only critical methods (chat.completions.create)
3. Direct passthrough for other attributes (zero overhead)
4. Circuit breaker pattern for degraded operation
5. Comprehensive metrics collection and observability
6. Async-first design with concurrent validation
7. Maintaining 100% OpenAI SDK API compatibility
"""

import asyncio
import time
import logging
from typing import Any, List, Optional, Dict, Iterator, AsyncIterator
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import statistics

# Handle optional OpenAI dependency
try:
    from openai import OpenAI, AsyncOpenAI
    _has_openai = True
except ImportError:
    _has_openai = False
    OpenAI = None
    AsyncOpenAI = None

from ..core.exceptions import ProviderError, ContractViolationError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker activated
    HALF_OPEN = "half_open"  # Testing if issue is resolved


@dataclass
class ContractMetrics:
    """Metrics collection for contract validation performance."""
    validation_times: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list))
    violation_counts: "Counter[str]" = field(default_factory=Counter)
    auto_fix_success_rate: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"success": 0, "total": 0})
    )
    contract_performance: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {
            "avg_latency": 0.0,
            "p95_latency": 0.0,
            "error_rate": 0.0,
            "call_count": 0
        })
    )
    total_calls: int = 0

    def record_validation_time(self, contract_name: str, duration: float, violated: bool = False):
        """Record validation timing and outcome."""
        self.validation_times[contract_name].append(duration)
        if violated:
            self.violation_counts[contract_name] += 1

        # Update performance metrics
        perf = self.contract_performance[contract_name]
        perf["call_count"] += 1

        # Update running average
        perf["avg_latency"] = (
            perf["avg_latency"] * (perf["call_count"] - 1) + duration
        ) / perf["call_count"]

        # Calculate p95 latency
        times = self.validation_times[contract_name]
        if len(times) >= 20:  # Calculate p95 with sufficient data
            perf["p95_latency"] = statistics.quantiles(
                times[-100:], n=20)[18] if len(times[-100:]) > 1 else times[-1]

    def record_auto_fix_attempt(self, contract_name: str, success: bool):
        """Record auto-remediation attempt."""
        stats = self.auto_fix_success_rate[contract_name]
        stats["total"] += 1
        if success:
            stats["success"] += 1

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        total_validations = sum(len(times)
                                for times in self.validation_times.values())
        total_violations = sum(self.violation_counts.values())

        return {
            "total_validations": total_validations,
            "violation_rate": total_violations / max(total_validations, 1),
            "slowest_contracts": self._get_slowest_contracts(),
            "most_violated_contracts": self._get_most_violated_contracts(),
            "auto_fix_success_rates": dict(self.auto_fix_success_rate),
            "performance_summary": dict(self.contract_performance)
        }

    def _get_slowest_contracts(self) -> List[Dict[str, Any]]:
        """Get contracts with highest average latency."""
        return sorted([
            {"name": name, "avg_latency": perf["avg_latency"]}
            for name, perf in self.contract_performance.items()
        ], key=lambda x: x["avg_latency"], reverse=True)[:5]

    def _get_most_violated_contracts(self) -> List[Dict[str, Any]]:
        """Get contracts with most violations."""
        return [
            {"name": name, "violations": count}
            for name, count in self.violation_counts.most_common(5)
        ]


@dataclass
class ContractCircuitBreaker:
    """Circuit breaker for contract validation to prevent cascade failures."""
    failure_threshold: int = 5
    timeout: int = 60
    failure_count: int = 0
    last_failure: Optional[float] = None
    state: CircuitState = CircuitState.CLOSED

    def record_success(self):
        """Reset circuit breaker on successful validation."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self, contract_name: str):
        """Record contract validation failure."""
        self.failure_count += 1
        self.last_failure = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened for contract {contract_name}")

    def should_skip(self) -> bool:
        """Check if contract validation should be skipped."""
        if self.state == CircuitState.OPEN:
            if self.last_failure and time.time() - self.last_failure > self.timeout:
                self.state = CircuitState.HALF_OPEN
                return False
            return True
        return False


# Old streaming classes removed - now using the new streaming module


class ImprovedOpenAIProvider:
    """
    Performance-optimized OpenAI provider with selective proxying and contract enforcement.

    This provider maintains 100% compatibility with the OpenAI SDK while adding
    contract enforcement through selective method interception, avoiding __getattr__ overhead.

    Usage:
        # Drop-in replacement for openai.OpenAI()
        client = ImprovedOpenAIProvider(api_key="...")
        client.add_input_contract(PromptLengthContract())
        client.add_output_contract(JSONFormatContract())

        # Use exact same API as OpenAI SDK
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(self, client: Optional[Any] = None, **kwargs: Any):
        """Initialize with same parameters as openai.OpenAI() or with existing client."""
        if not _has_openai:
            raise ProviderError(
                "OpenAI Python package not installed. Install with: pip install openai",
                "openai"
            )

        # Create or use provided OpenAI client
        if client is not None:
            self._client = client
        elif OpenAI is not None:
            self._client = OpenAI(**kwargs)
        else:
            raise ProviderError("OpenAI client not available", "openai")

        # Create async client with same parameters
        if AsyncOpenAI is not None:
            # Use the same kwargs for async client - simpler and safer approach
            self._async_client = AsyncOpenAI(**kwargs)
        else:
            raise ProviderError("AsyncOpenAI client not available", "openai")

        # Contract storage
        self.input_contracts: List[Any] = []
        self.output_contracts: List[Any] = []
        self.streaming_contracts: List[Any] = []

        # Performance optimization and reliability components
        self._metrics = ContractMetrics()
        self._circuit_breaker = ContractCircuitBreaker()

        # Configuration
        self.max_retries = 3
        self.auto_remediation = True

        # Pre-wrap only critical methods to avoid __getattr__ overhead
        self.chat = self._wrap_chat_namespace(self._client.chat)
        self.completions = self._wrap_completions_namespace(
            self._client.completions)

        # Direct passthrough for other attributes (zero overhead)
        self.models = self._client.models
        self.files = self._client.files
        self.fine_tuning = self._client.fine_tuning
        self.images = self._client.images
        self.audio = self._client.audio
        self.embeddings = self._client.embeddings
        self.moderations = self._client.moderations
        self.beta = self._client.beta

        # Pass through client properties
        if hasattr(self._client, 'api_key'):
            self.api_key = self._client.api_key
        if hasattr(self._client, 'organization'):
            self.organization = self._client.organization
        if hasattr(self._client, 'base_url'):
            self.base_url = self._client.base_url

    def add_input_contract(self, contract: Any) -> None:
        """Add an input contract for validation."""
        self.input_contracts.append(contract)

    def add_output_contract(self, contract: Any) -> None:
        """Add an output contract for validation."""
        self.output_contracts.append(contract)

    def add_contract(self, contract: Any) -> None:
        """Add a contract (automatically categorizes as input or output)."""
        # Simple categorization - could be more sophisticated
        if hasattr(contract, 'validate_input') or 'Input' in contract.__class__.__name__:
            self.add_input_contract(contract)
        elif hasattr(contract, 'validate_output') or 'Output' in contract.__class__.__name__:
            self.add_output_contract(contract)
        else:
            # Default to output contract
            self.add_output_contract(contract)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics and health report."""
        return self._metrics.get_health_report()

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        self._circuit_breaker.record_success()

    def _wrap_chat_namespace(self, chat_attr: Any) -> Any:
        """Wrap only the chat namespace to intercept completions.create."""

        class WrappedChat:
            def __init__(self, original_chat: Any, provider: 'ImprovedOpenAIProvider'):
                self._original_chat = original_chat
                self._provider = provider

                # Wrap only completions
                self.completions = self._provider._wrap_completions_create(
                    original_chat.completions)

            def __getattr__(self, name: str) -> Any:
                """Forward all other chat methods unchanged."""
                return getattr(self._original_chat, name)

        return WrappedChat(chat_attr, self)

    def _wrap_completions_namespace(self, completions_attr: Any) -> Any:
        """Wrap the completions namespace if accessed directly."""
        return self._wrap_completions_create(completions_attr)

    def _wrap_completions_create(self, completions_attr: Any) -> Any:
        """High-performance method wrapping with validation."""

        class WrappedCompletions:
            def __init__(self, original_completions: Any, provider: 'ImprovedOpenAIProvider'):
                self._original_completions = original_completions
                self._provider = provider
                # Store the original create method
                self._original_create = original_completions.create

            def create(self, **kwargs: Any) -> Any:
                """Sync create with contract enforcement."""
                # Circuit breaker check
                if self._provider._circuit_breaker.should_skip():
                    logger.warning(
                        "Contract validation skipped due to circuit breaker")
                    return self._original_create(**kwargs)

                # Sync input validation
                validation_start = time.time()
                try:
                    self._provider._validate_input_sync(**kwargs)
                    self._provider._metrics.record_validation_time(
                        'input', time.time() - validation_start)
                    self._provider._circuit_breaker.record_success()
                except Exception as e:
                    self._provider._metrics.record_validation_time(
                        'input', time.time() - validation_start, violated=True)
                    self._provider._circuit_breaker.record_failure(
                        'input_validation')
                    if isinstance(e, ContractViolationError):
                        raise
                    logger.warning(f"Input validation error: {e}")

                # Handle streaming
                if kwargs.get('stream', False):
                    stream = self._original_create(**kwargs)
                    if self._provider.output_contracts:
                        # Use the new streaming validation system
                        from ..streaming import StreamingValidator, StreamWrapper
                        validator = StreamingValidator(
                            self._provider.output_contracts,
                            performance_monitoring=True,
                            early_termination=True
                        )
                        return StreamWrapper(stream, validator)
                    return stream

                # Call original OpenAI method
                response = self._original_create(**kwargs)

                # Sync output validation with auto-remediation
                validation_start = time.time()
                try:
                    validated_response = self._provider._validate_output_sync(
                        response, **kwargs)
                    self._provider._metrics.record_validation_time(
                        'output', time.time() - validation_start)
                    self._provider._circuit_breaker.record_success()
                    return validated_response
                except Exception as e:
                    self._provider._metrics.record_validation_time(
                        'output', time.time() - validation_start, violated=True)
                    self._provider._circuit_breaker.record_failure(
                        'output_validation')
                    if isinstance(e, ContractViolationError):
                        raise
                    logger.warning(f"Output validation error: {e}")
                    return response

            async def acreate(self, **kwargs: Any) -> Any:
                """Async create with contract enforcement."""
                # Circuit breaker check
                if self._provider._circuit_breaker.should_skip():
                    logger.warning(
                        "Contract validation skipped due to circuit breaker")
                    return await self._original_completions.acreate(**kwargs)

                # Async input validation
                validation_start = time.time()
                try:
                    await self._provider._validate_input_async(**kwargs)
                    self._provider._metrics.record_validation_time(
                        'input', time.time() - validation_start)
                    self._provider._circuit_breaker.record_success()
                except Exception as e:
                    self._provider._metrics.record_validation_time(
                        'input', time.time() - validation_start, violated=True)
                    self._provider._circuit_breaker.record_failure(
                        'input_validation')
                    if isinstance(e, ContractViolationError):
                        raise
                    logger.warning(f"Input validation error: {e}")

                # Handle streaming
                if kwargs.get('stream', False):
                    stream = await self._original_completions.acreate(**kwargs)
                    if self._provider.output_contracts:
                        # Use the new streaming validation system
                        from ..streaming import StreamingValidator, AsyncStreamWrapper
                        validator = StreamingValidator(
                            self._provider.output_contracts,
                            performance_monitoring=True,
                            early_termination=True
                        )
                        return AsyncStreamWrapper(stream, validator)
                    return stream

                # Call original OpenAI method
                response = await self._original_completions.acreate(**kwargs)

                # Async output validation with auto-remediation
                validation_start = time.time()
                try:
                    validated_response = await self._provider._validate_output_async(response, **kwargs)
                    self._provider._metrics.record_validation_time(
                        'output', time.time() - validation_start)
                    self._provider._circuit_breaker.record_success()
                    return validated_response
                except Exception as e:
                    self._provider._metrics.record_validation_time(
                        'output', time.time() - validation_start, violated=True)
                    self._provider._circuit_breaker.record_failure(
                        'output_validation')
                    if isinstance(e, ContractViolationError):
                        raise
                    logger.warning(f"Output validation error: {e}")
                    return response

            def __getattr__(self, name: str) -> Any:
                """Forward all other completion methods unchanged."""
                return getattr(self._original_completions, name)

        return WrappedCompletions(completions_attr, self)

    async def _validate_input_async(self, **kwargs: Any) -> None:
        """Async input validation with concurrent execution."""
        if not self.input_contracts:
            return

        # Create validation tasks for concurrent execution
        validation_tasks: List[Any] = []
        for contract in self.input_contracts:
            if hasattr(contract, 'validate'):
                if asyncio.iscoroutinefunction(contract.validate):
                    validation_tasks.append(contract.validate(kwargs))
                else:
                    # Run sync validation in thread pool
                    validation_tasks.append(asyncio.get_event_loop().run_in_executor(
                        None, contract.validate, kwargs
                    ))
            elif hasattr(contract, 'validate_input'):
                if asyncio.iscoroutinefunction(contract.validate_input):
                    validation_tasks.append(contract.validate_input(kwargs))
                else:
                    validation_tasks.append(asyncio.get_event_loop().run_in_executor(
                        None, contract.validate_input, kwargs
                    ))

        if validation_tasks:
            # Execute all validations concurrently
            results: List[Any] = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                contract = self.input_contracts[i]
                if isinstance(result, Exception):
                    if isinstance(result, ContractViolationError):
                        raise result
                    logger.warning(
                        f"Input contract validation failed for {contract.name if hasattr(contract, 'name') else 'unknown'}: {result}")
                    continue

                if hasattr(result, 'is_valid') and not result.is_valid:
                    if hasattr(result, 'auto_fix_suggestion') and result.auto_fix_suggestion and self.auto_remediation:
                        # Apply auto-fix
                        if isinstance(result.auto_fix_suggestion, dict):
                            kwargs.update(result.auto_fix_suggestion)
                        self._metrics.record_auto_fix_attempt(
                            contract.name if hasattr(contract, 'name') else 'unknown', True)
                    else:
                        self._metrics.record_auto_fix_attempt(
                            contract.name if hasattr(contract, 'name') else 'unknown', False)
                        raise ContractViolationError(
                            f"Input validation failed: {result.message if hasattr(result, 'message') else str(result)}",
                            contract_type="input",
                            contract_name=contract.name if hasattr(
                                contract, 'name') else 'unknown'
                        )

    def _validate_input_sync(self, **kwargs: Any) -> None:
        """Synchronous input validation."""
        if not self.input_contracts:
            return

        # Extract content from kwargs for validation
        content_to_validate = kwargs  # For input contracts that validate the full request
        message_dict = kwargs  # Default to kwargs

        # For prompt-specific contracts, extract the messages content
        if 'messages' in kwargs and kwargs['messages']:
            # Also provide a dict format for contracts that expect it
            message_dict = {'messages': kwargs['messages']}

        for contract in self.input_contracts:
            try:
                result = None
                # Different contracts may have different validation methods
                if hasattr(contract, 'validate'):
                    # Try validating with message dict first (for prompt length contracts)
                    if 'messages' in kwargs:
                        result = contract.validate(message_dict)
                    else:
                        result = contract.validate(content_to_validate)
                elif hasattr(contract, 'validate_input'):
                    result = contract.validate_input(content_to_validate)
                else:
                    continue

                if hasattr(result, 'is_valid') and not result.is_valid:
                    if hasattr(result, 'auto_fix_suggestion') and result.auto_fix_suggestion and self.auto_remediation:
                        # Apply auto-fix
                        if isinstance(result.auto_fix_suggestion, dict):
                            kwargs.update(result.auto_fix_suggestion)
                        self._metrics.record_auto_fix_attempt(
                            contract.name if hasattr(contract, 'name') else 'unknown', True)
                    else:
                        self._metrics.record_auto_fix_attempt(
                            contract.name if hasattr(contract, 'name') else 'unknown', False)
                        raise ContractViolationError(
                            f"Input validation failed: {result.message if hasattr(result, 'message') else str(result)}",
                            contract_type="input",
                            contract_name=contract.name if hasattr(
                                contract, 'name') else 'unknown'
                        )
            except Exception as e:
                if isinstance(e, ContractViolationError):
                    raise
                logger.warning(f"Input contract validation failed: {e}")

    async def _validate_output_async(self, response: Any, **kwargs: Any) -> Any:
        """Async output validation with auto-remediation."""
        if not self.output_contracts:
            return response

        content = self._extract_content(response)

        # Create validation tasks for concurrent execution
        validation_tasks: List[Any] = []
        for contract in self.output_contracts:
            if hasattr(contract, 'validate'):
                if asyncio.iscoroutinefunction(contract.validate):
                    validation_tasks.append(contract.validate(content))
                else:
                    validation_tasks.append(asyncio.get_event_loop().run_in_executor(
                        None, contract.validate, content
                    ))
            elif hasattr(contract, 'validate_output'):
                if asyncio.iscoroutinefunction(contract.validate_output):
                    validation_tasks.append(contract.validate_output(content))
                else:
                    validation_tasks.append(asyncio.get_event_loop().run_in_executor(
                        None, contract.validate_output, content
                    ))

        if validation_tasks:
            # Execute all validations concurrently
            results: List[Any] = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                contract = self.output_contracts[i]
                if isinstance(result, Exception):
                    if isinstance(result, ContractViolationError):
                        raise result
                    logger.warning(
                        f"Output contract validation failed for {contract.name if hasattr(contract, 'name') else 'unknown'}: {result}")
                    continue

                if hasattr(result, 'is_valid') and not result.is_valid:
                    if hasattr(result, 'auto_fix_suggestion') and result.auto_fix_suggestion and self.auto_remediation:
                        # Create corrected response
                        corrected_response = self._create_corrected_response(
                            response, result.auto_fix_suggestion)
                        self._metrics.record_auto_fix_attempt(
                            contract.name if hasattr(contract, 'name') else 'unknown', True)
                        return corrected_response
                    elif self.max_retries > 0 and self.auto_remediation:
                        # Retry with correction instruction
                        retry_response = await self._retry_with_correction_async(response, result.message if hasattr(result, 'message') else str(result), kwargs)
                        self._metrics.record_auto_fix_attempt(
                            contract.name if hasattr(contract, 'name') else 'unknown', True)
                        return retry_response
                    else:
                        self._metrics.record_auto_fix_attempt(
                            contract.name if hasattr(contract, 'name') else 'unknown', False)
                        raise ContractViolationError(
                            f"Output validation failed: {result.message if hasattr(result, 'message') else str(result)}",
                            contract_type="output",
                            contract_name=contract.name if hasattr(
                                contract, 'name') else 'unknown'
                        )

        return response

    def _validate_output_sync(self, response: Any, **kwargs: Any) -> Any:
        """Synchronous output validation with auto-remediation."""
        if not self.output_contracts:
            return response

        content = self._extract_content(response)

        for contract in self.output_contracts:
            try:
                result = None
                # Different contracts may have different validation methods
                if hasattr(contract, 'validate'):
                    result = contract.validate(content)
                elif hasattr(contract, 'validate_output'):
                    result = contract.validate_output(content)
                else:
                    continue

                if hasattr(result, 'is_valid') and not result.is_valid:
                    if hasattr(result, 'auto_fix_suggestion') and result.auto_fix_suggestion and self.auto_remediation:
                        # Create corrected response
                        corrected_response = self._create_corrected_response(
                            response, result.auto_fix_suggestion)
                        self._metrics.record_auto_fix_attempt(
                            contract.name if hasattr(contract, 'name') else 'unknown', True)
                        return corrected_response
                    elif self.max_retries > 0 and self.auto_remediation:
                        # Retry with correction instruction
                        retry_response = self._retry_with_correction_sync(
                            response, result.message if hasattr(result, 'message') else str(result), kwargs)
                        self._metrics.record_auto_fix_attempt(
                            contract.name if hasattr(contract, 'name') else 'unknown', True)
                        return retry_response
                    else:
                        self._metrics.record_auto_fix_attempt(
                            contract.name if hasattr(contract, 'name') else 'unknown', False)
                        raise ContractViolationError(
                            f"Output validation failed: {result.message if hasattr(result, 'message') else str(result)}",
                            contract_type="output",
                            contract_name=contract.name if hasattr(
                                contract, 'name') else 'unknown'
                        )
            except Exception as e:
                if isinstance(e, ContractViolationError):
                    raise
                logger.warning(f"Output contract validation failed: {e}")

        return response

    def _extract_content(self, response: Any) -> str:
        """Extract content from OpenAI response."""
        try:
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    return choice.message.content or ""
                elif hasattr(choice, 'text'):
                    return choice.text or ""
        except (AttributeError, IndexError):
            pass

        if isinstance(response, str):
            return response

        return str(response)

    def _create_corrected_response(self, original_response: Any, correction: str) -> Any:
        """Create a corrected response maintaining OpenAI response structure."""
        try:
            if hasattr(original_response, 'choices') and original_response.choices:
                # Create a copy and modify the content
                import copy
                corrected_response = copy.deepcopy(original_response)
                if hasattr(corrected_response.choices[0], 'message'):
                    corrected_response.choices[0].message.content = correction
                elif hasattr(corrected_response.choices[0], 'text'):
                    corrected_response.choices[0].text = correction
                return corrected_response
        except (AttributeError, IndexError):
            pass

        return original_response

    async def _retry_with_correction_async(self, original_response: Any, error_message: str, original_kwargs: Any) -> Any:
        """Async retry the API call with correction instructions."""
        try:
            # Extract original messages
            messages = original_kwargs.get('messages', [])
            if not messages:
                return original_response

            # Add correction instruction
            correction_messages = messages + [{
                "role": "system",
                "content": f"Previous response violated contract: {error_message}. Please correct and retry."
            }]

            # Create new parameters for retry
            retry_kwargs = original_kwargs.copy()
            retry_kwargs['messages'] = correction_messages

            # Temporarily disable auto-remediation to avoid infinite loops
            old_retries = self.max_retries
            old_auto_remediation = self.auto_remediation
            self.max_retries = 0
            self.auto_remediation = False

            try:
                # Call original OpenAI method for retry
                retry_response = await self._async_client.chat.completions.create(**retry_kwargs)
                return retry_response
            finally:
                # Restore settings
                self.max_retries = old_retries
                self.auto_remediation = old_auto_remediation

        except Exception as e:
            logger.warning(f"Retry with correction failed: {e}")
            return original_response

    def _retry_with_correction_sync(self, original_response: Any, violation_message: str, kwargs: Dict[str, Any]) -> Any:
        """Retry API call with correction instruction for sync operations."""
        correction_prompt = f"The previous response violated a contract: {violation_message}. Please provide a corrected response."

        # Modify the last message to include correction instruction
        messages = kwargs.get('messages', [])
        if messages:
            corrected_messages = messages.copy()
            corrected_messages.append({
                "role": "user",
                "content": correction_prompt
            })

            # Update kwargs for retry
            retry_kwargs = kwargs.copy()
            retry_kwargs['messages'] = corrected_messages

            # Call original OpenAI API
            return self._client.chat.completions.create(**retry_kwargs)

        return original_response

    def __getattr__(self, name: str) -> Any:
        """Fallback to original client for any attributes not explicitly wrapped."""
        return getattr(self._client, name)

    def __str__(self) -> str:  # type: ignore[override]
        return f"ImprovedOpenAIProvider(contracts={len(self.input_contracts + self.output_contracts)}, metrics_enabled=True)"

    def __repr__(self) -> str:  # type: ignore[override]
        return self.__str__()


# For backwards compatibility, keep the old class name as an alias
OpenAIProvider = ImprovedOpenAIProvider


# Usage Example:
if __name__ == "__main__":
    # This maintains exact OpenAI SDK compatibility:

    client = ImprovedOpenAIProvider(api_key="your-key")

    # Add contracts
    # client.add_input_contract(PromptLengthContract(max_length=1000))
    # client.add_output_contract(JSONFormatContract())

    # Use exactly like OpenAI SDK - all parameters work:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100,
        stream=False
        # Any OpenAI parameter works unchanged!
    )

    # Contracts are enforced automatically
    print("Response received with contract enforcement!")
