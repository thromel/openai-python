"""
Performance-optimized input validation with async support, circuit breaker integration,
metrics collection, and OpenTelemetry tracing integration.

This module implements the enhanced InputValidator required for Task 4 that integrates
with the ImprovedOpenAIProvider architecture and provides comprehensive input validation
capabilities with performance optimization.
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Set
from collections import defaultdict
from dataclasses import dataclass, field
import re
import json

from ..core.interfaces import ValidatorBase, ValidationResult, ContractBase
from ..core.exceptions import ContractViolationError
from ..providers.openai_provider import ContractCircuitBreaker, ContractMetrics
from ..utils.telemetry import (
    ContractTracer, TraceContext, OperationType, 
    get_tracer, trace_contract_validation
)


logger = logging.getLogger(__name__)


@dataclass
class InputValidationContext:
    """Context information for input validation."""
    request_id: str
    model: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    prompt_type: str = "chat"  # chat, completion, embedding, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class TokenCounter:
    """Efficient token counting utility."""

    def __init__(self):
        self._cache: Dict[str, int] = {}
        self._max_cache_size = 1000

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens with caching for performance."""
        cache_key = f"{model}:{hash(text)}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Simple estimation - replace with tiktoken for accuracy
        # For now, use rough estimation: ~4 chars per token
        estimated_tokens = len(text) // 4

        # Cache the result
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = estimated_tokens
        return estimated_tokens


class PromptInjectionDetector:
    """Advanced prompt injection detection."""

    def __init__(self):
        # Common prompt injection patterns
        self.injection_patterns = [
            r"ignore previous instructions",
            r"ignore all previous",
            r"disregard.*instructions",
            r"you are now",
            r"new instructions",
            r"system prompt",
            r"prompt injection",
            r"jailbreak",
            r"override.*settings",
            r"forget.*context",
        ]

        # Compile patterns for performance
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.injection_patterns
        ]

    def detect_injection(self, text: str) -> "tuple[bool, List[str]]":
        """Detect potential prompt injection attempts."""
        violations: List[str] = []

        for pattern in self.compiled_patterns:
            if pattern.search(text):
                violations.append(pattern.pattern)

        return len(violations) > 0, violations


class PerformanceOptimizedInputValidator(ValidatorBase):
    """
    High-performance input validator with async support, circuit breaker integration,
    metrics collection, and comprehensive validation capabilities.
    """

    def __init__(self,
                 name: str = "performance_input_validator",
                 enable_circuit_breaker: bool = True,
                 enable_metrics: bool = True,
                 enable_tracing: bool = True,
                 max_concurrent_validations: int = 10):
        super().__init__(name)

        # Performance optimization components
        self.circuit_breaker = ContractCircuitBreaker() if enable_circuit_breaker else None
        self.metrics = ContractMetrics() if enable_metrics else None
        self.enable_tracing = enable_tracing
        self.tracer = get_tracer("llm_contracts.input_validator") if enable_tracing else None

        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent_validations)

        # Validation utilities
        self.token_counter = TokenCounter()
        self.injection_detector = PromptInjectionDetector()

        # Lazy contract loading registry
        self.lazy_contracts: Dict[str, Callable[[], ContractBase]] = {}
        self.loaded_contracts: Set[str] = set()

        # Validation cache for performance
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.cache_max_size = 100

        logger.info(f"Initialized {self.__class__.__name__} with "
                    f"circuit_breaker={enable_circuit_breaker}, "
                    f"metrics={enable_metrics}, "
                    f"tracing={enable_tracing}")

    def register_lazy_contract(self, name: str, loader: Callable[[], ContractBase]) -> None:
        """Register a contract that will be loaded on first use."""
        self.lazy_contracts[name] = loader
        logger.debug(f"Registered lazy contract: {name}")

    def _load_lazy_contracts(self) -> None:
        """Load any lazy contracts that haven't been loaded yet."""
        for name, loader in self.lazy_contracts.items():
            if name not in self.loaded_contracts:
                try:
                    contract = loader()
                    self.contracts.append(contract)
                    self.loaded_contracts.add(name)
                    logger.debug(f"Loaded lazy contract: {name}")
                except Exception as e:
                    logger.error(f"Failed to load lazy contract {name}: {e}")

    async def validate_async(self,
                             data: Any,
                             context: Optional[InputValidationContext] = None) -> List[ValidationResult]:
        """
        Asynchronous validation with performance optimization and observability.
        """
        if context is None:
            context = InputValidationContext(
                request_id=f"req_{int(time.time())}")

        # Circuit breaker check
        if self.circuit_breaker and self.circuit_breaker.should_skip():
            logger.warning("Validation skipped due to circuit breaker")
            return [ValidationResult(
                is_valid=True,
                message="Validation skipped due to circuit breaker"
            )]

        # Enhanced OpenTelemetry tracing
        if self.tracer:
            trace_context = TraceContext(
                operation_type=OperationType.INPUT_VALIDATION,
                contract_name=self.name,
                request_id=context.request_id,
                model=context.model,
                metadata={
                    "prompt_type": context.prompt_type,
                    "user_id": context.user_id or "unknown",
                    "conversation_id": context.conversation_id or "none"
                }
            )
            
            with self.tracer.trace_operation(trace_context, "input_validation") as span:
                return await self._perform_validation(data, context, span)
        else:
            return await self._perform_validation(data, context)

    async def _perform_validation(self,
                                  data: Any,
                                  context: InputValidationContext,
                                  span: Optional[Any] = None) -> List[ValidationResult]:
        """Internal validation logic with performance optimization."""

        # Load lazy contracts
        self._load_lazy_contracts()

        # Check cache
        cache_key = self._generate_cache_key(data, context)
        if cache_key in self.validation_cache:
            cached_result = self.validation_cache[cache_key]
            if self.metrics:
                self.metrics.record_validation_time("cache_hit", 0.0)
            return [cached_result]

        validation_start = time.time()
        results: List[ValidationResult] = []

        try:
            # Concurrent validation with semaphore control
            async with self.semaphore:
                # Built-in validation types
                builtin_results = await self._perform_builtin_validations(data, context)
                results.extend(builtin_results)

                # Contract-based validations
                if self.contracts:
                    contract_tasks = [
                        self._validate_with_contract(contract, data, context)
                        for contract in self.contracts
                    ]
                    contract_results = await asyncio.gather(*contract_tasks, return_exceptions=True)

                    for result in contract_results:
                        if isinstance(result, Exception):
                            results.append(ValidationResult(
                                is_valid=False,
                                message=f"Contract validation error: {str(result)}"
                            ))
                        else:
                            results.append(result)

            # Update metrics
            validation_duration = time.time() - validation_start
            violations = [r for r in results if not r.is_valid]

            if self.metrics:
                self.metrics.record_validation_time(
                    "input_validation",
                    validation_duration,
                    violated=len(violations) > 0
                )

                if violations and self.circuit_breaker:
                    self.circuit_breaker.record_failure("input_validation")
                elif self.circuit_breaker:
                    self.circuit_breaker.record_success()

            # Cache successful validations
            if not violations and len(results) == 1:
                self._cache_result(cache_key, results[0])

            # Enhanced tracing and metrics
            if span and self.tracer:
                # Add detailed span attributes
                self.tracer.add_event(span, "validation_completed", {
                    "duration_ms": validation_duration * 1000,
                    "contract_count": len(self.contracts),
                    "violation_count": len(violations),
                    "cache_hit": cache_key in self.validation_cache
                })
                
                # Record violations with tracer
                for violation in violations:
                    self.tracer.record_contract_violation(
                        contract_name=self.name,
                        violation_type="input_validation",
                        severity="medium",
                        metadata={"message": violation.message}
                    )

            return results

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            
            # Record error in tracing
            if span and self.tracer:
                self.tracer.add_event(span, "validation_error", {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })

            if self.circuit_breaker:
                self.circuit_breaker.record_failure("input_validation")

            return [ValidationResult(
                is_valid=False,
                message=f"Validation system error: {str(e)}"
            )]

    async def _perform_builtin_validations(self,
                                           data: Any,
                                           context: InputValidationContext) -> List[ValidationResult]:
        """Perform built-in validation types efficiently."""
        results: List[ValidationResult] = []

        # Token length validation
        token_result = await self._validate_token_length(data, context)
        if token_result:
            results.append(token_result)

        # Parameter validation
        param_result = await self._validate_parameters(data, context)
        if param_result:
            results.append(param_result)

        # Prompt injection detection
        injection_result = await self._validate_prompt_injection(data, context)
        if injection_result:
            results.append(injection_result)

        return results

    async def _validate_token_length(self,
                                     data: Any,
                                     context: InputValidationContext) -> Optional[ValidationResult]:
        """Validate token length limits efficiently."""
        try:
            if isinstance(data, dict) and 'messages' in data:
                total_tokens = 0
                for message in data['messages']:
                    if isinstance(message, dict) and 'content' in message:
                        content = str(message['content'])
                        tokens = self.token_counter.count_tokens(
                            content, context.model or "gpt-4")
                        total_tokens += tokens

                # Model-specific limits
                limits = {
                    "gpt-4": 8192,
                    "gpt-4-32k": 32768,
                    "gpt-3.5-turbo": 4096,
                    "gpt-3.5-turbo-16k": 16384,
                }

                limit = limits.get(context.model or "gpt-4", 4096)

                # Record token metrics
                if self.tracer:
                    self.tracer.record_token_metrics(
                        token_count=total_tokens,
                        model=context.model or "unknown",
                        request_type=context.prompt_type
                    )

                if total_tokens > limit:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Token count {total_tokens} exceeds limit {limit} for model {context.model}",
                        auto_fix_suggestion=f"Reduce token count to {limit // 2} or less"
                    )

            return None  # No validation needed/successful

        except Exception as e:
            logger.error(f"Token length validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                message=f"Token length validation error: {str(e)}"
            )

    async def _validate_parameters(self,
                                   data: Any,
                                   context: InputValidationContext) -> Optional[ValidationResult]:
        """Validate parameter types and ranges."""
        try:
            if not isinstance(data, dict):
                return ValidationResult(
                    is_valid=False,
                    message="Input data must be a dictionary"
                )

            # Check required parameters
            required_fields = [
                'messages'] if context.prompt_type == 'chat' else ['prompt']
            missing_fields = [
                field for field in required_fields if field not in data]

            if missing_fields:
                return ValidationResult(
                    is_valid=False,
                    message=f"Missing required fields: {missing_fields}"
                )

            # Validate parameter ranges
            if 'temperature' in data:
                temp = data['temperature']
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    fixed_temp = max(0, min(2, float(temp))) if isinstance(temp, (int, float)) else 1.0
                    return ValidationResult(
                        is_valid=False,
                        message=f"Temperature {temp} must be between 0 and 2",
                        auto_fix_suggestion=f"Use temperature={fixed_temp} instead"
                    )

            if 'max_tokens' in data:
                max_tokens = data['max_tokens']
                if not isinstance(max_tokens, int) or max_tokens < 1:
                    fixed_tokens = max(1, abs(int(max_tokens))) if isinstance(max_tokens, (int, float)) else 100
                    return ValidationResult(
                        is_valid=False,
                        message=f"max_tokens {max_tokens} must be a positive integer",
                        auto_fix_suggestion=f"Use max_tokens={fixed_tokens} instead"
                    )

            return None  # Validation successful

        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                message=f"Parameter validation error: {str(e)}"
            )

    async def _validate_prompt_injection(self,
                                         data: Any,
                                         context: InputValidationContext) -> Optional[ValidationResult]:
        """Detect potential prompt injection attempts."""
        try:
            content_parts = []

            if isinstance(data, dict):
                if 'messages' in data:
                    for message in data['messages']:
                        if isinstance(message, dict) and 'content' in message:
                            content_parts.append(str(message['content']))
                elif 'prompt' in data:
                    content_parts.append(str(data['prompt']))

            for content in content_parts:
                is_injection, patterns = self.injection_detector.detect_injection(
                    content)
                if is_injection:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Potential prompt injection detected. Patterns: {patterns}",
                        details={"detected_patterns": patterns,
                                 "content_length": len(content)}
                    )

            return None  # No injection detected

        except Exception as e:
            logger.error(f"Prompt injection validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                message=f"Prompt injection validation error: {str(e)}"
            )

    async def _validate_with_contract(self,
                                      contract: ContractBase,
                                      data: Any,
                                      context: InputValidationContext) -> ValidationResult:
        """Validate data with a specific contract."""
        try:
            validation_start = time.time()

            # Convert context to dict for contract validation
            context_dict = {
                "request_id": context.request_id,
                "model": context.model,
                "user_id": context.user_id,
                "conversation_id": context.conversation_id,
                "prompt_type": context.prompt_type,
                "metadata": context.metadata,
                "timestamp": context.timestamp
            }

            result = contract.validate(data, context_dict)

            # Record contract-specific metrics
            if self.metrics:
                duration = time.time() - validation_start
                self.metrics.record_validation_time(
                    contract.name,
                    duration,
                    violated=not result.is_valid
                )

            return result

        except Exception as e:
            logger.error(f"Contract {contract.name} validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                message=f"Contract '{contract.name}' validation error: {str(e)}"
            )

    def _generate_cache_key(self, data: Any, context: InputValidationContext) -> str:
        """Generate cache key for validation results."""
        try:
            data_hash = hash(str(data)) if data else 0
            context_hash = hash(f"{context.model}:{context.prompt_type}")
            return f"{data_hash}:{context_hash}"
        except:
            return f"uncacheable_{int(time.time())}"

    def _cache_result(self, cache_key: str, result: ValidationResult) -> None:
        """Cache validation result with size limit."""
        if len(self.validation_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.validation_cache))
            del self.validation_cache[oldest_key]

        self.validation_cache[cache_key] = result

    def validate_all(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Synchronous validation wrapper for compatibility."""
        input_context = InputValidationContext(
            request_id=f"sync_{int(time.time())}"
        )

        if context:
            input_context.model = context.get("model")
            input_context.user_id = context.get("user_id")
            input_context.conversation_id = context.get("conversation_id")
            input_context.prompt_type = context.get("prompt_type", "chat")
            input_context.metadata = context.get("metadata", {})

        # Run async validation in event loop
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.validate_async(data, input_context))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.validate_async(data, input_context))
            finally:
                loop.close()

    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        if not self.metrics:
            return {"metrics_disabled": True}

        report = self.metrics.get_health_report()
        report.update({
            "validator_name": self.name,
            "circuit_breaker_state": self.circuit_breaker.state.value if self.circuit_breaker else "disabled",
            "cached_validations": len(self.validation_cache),
            "loaded_contracts": len(self.loaded_contracts),
            "lazy_contracts": len(self.lazy_contracts)
        })

        return report

    def reset_metrics(self) -> None:
        """Reset all metrics and caches."""
        if self.metrics:
            self.metrics = ContractMetrics()

        if self.circuit_breaker:
            self.circuit_breaker = ContractCircuitBreaker()

        self.validation_cache.clear()
        logger.info("Metrics and caches reset")


# Backward compatibility alias
InputValidator = PerformanceOptimizedInputValidator
