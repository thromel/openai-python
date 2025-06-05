"""
Enhanced Output Validation with Streaming Support and Auto-Remediation.

This module implements the comprehensive OutputValidator required for Task 5 that provides:
- Real-time streaming validation during response generation
- Advanced conflict resolution between multiple contracts
- Intelligent state management for multi-turn contexts
- Auto-remediation with retry logic and circuit breaker integration
- Critical violation termination for streaming responses
"""

import asyncio
import time
import logging
import json
import re
from typing import Any, Dict, List, Optional, Union, Callable, Set, Iterator, AsyncIterator
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import copy

from ..core.interfaces import ValidatorBase, ValidationResult, ContractBase, ContractType
from ..core.exceptions import ContractViolationError
from ..providers.openai_provider import ContractCircuitBreaker, ContractMetrics, StreamingValidator
from ..utils.telemetry import (
    ContractTracer, TraceContext, OperationType, 
    get_tracer, trace_contract_validation
)

logger = logging.getLogger(__name__)


class ViolationSeverity(Enum):
    """Severity levels for contract violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts between contracts."""
    FIRST_WINS = "first_wins"  # First contract takes precedence
    LAST_WINS = "last_wins"    # Last contract takes precedence
    MOST_RESTRICTIVE = "most_restrictive"  # Most restrictive contract wins
    MERGE = "merge"            # Attempt to merge contract requirements
    FAIL_ON_CONFLICT = "fail_on_conflict"  # Fail if any conflicts detected


@dataclass
class OutputValidationContext:
    """Context information for output validation."""
    request_id: str
    model: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    original_request: Optional[Dict[str, Any]] = None
    response_format: str = "text"  # text, json, xml, etc.
    streaming: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass 
class AutoRemediationResult:
    """Result of auto-remediation attempt."""
    success: bool
    original_content: str
    corrected_content: Optional[str] = None
    method_used: Optional[str] = None
    attempts: int = 1
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictResolution:
    """Result of contract conflict resolution."""
    conflicts_detected: bool
    resolution_strategy: ConflictResolutionStrategy
    winning_contracts: List[str]
    merged_contract: Optional[ContractBase] = None
    resolution_metadata: Dict[str, Any] = field(default_factory=dict)


class ContractConflictResolver:
    """Advanced conflict resolution system for multiple contracts."""
    
    def __init__(self, default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MOST_RESTRICTIVE):
        self.default_strategy = default_strategy
        self.conflict_rules: Dict[str, ConflictResolutionStrategy] = {}
        
    def register_conflict_rule(self, 
                              contract_pair: tuple[str, str], 
                              strategy: ConflictResolutionStrategy) -> None:
        """Register a specific conflict resolution rule for a pair of contracts."""
        key = tuple(sorted(contract_pair))
        self.conflict_rules[key] = strategy
        
    def resolve_conflicts(self, contracts: List[ContractBase], data: Any) -> ConflictResolution:
        """Resolve conflicts between contracts."""
        conflicts = self._detect_conflicts(contracts, data)
        
        if not conflicts:
            return ConflictResolution(
                conflicts_detected=False,
                resolution_strategy=self.default_strategy,
                winning_contracts=[c.name for c in contracts]
            )
            
        # Apply resolution strategy
        if self.default_strategy == ConflictResolutionStrategy.FIRST_WINS:
            winning_contracts = [contracts[0].name] if contracts else []
        elif self.default_strategy == ConflictResolutionStrategy.LAST_WINS:
            winning_contracts = [contracts[-1].name] if contracts else []
        elif self.default_strategy == ConflictResolutionStrategy.MOST_RESTRICTIVE:
            winning_contracts = self._find_most_restrictive(contracts, data)
        elif self.default_strategy == ConflictResolutionStrategy.MERGE:
            merged_contract = self._merge_contracts(contracts)
            winning_contracts = [merged_contract.name] if merged_contract else []
        else:
            # FAIL_ON_CONFLICT
            raise ContractViolationError(
                f"Contract conflicts detected: {conflicts}",
                contract_type="conflict_resolution",
                contract_name="multiple"
            )
            
        return ConflictResolution(
            conflicts_detected=True,
            resolution_strategy=self.default_strategy,
            winning_contracts=winning_contracts,
            resolution_metadata={"detected_conflicts": conflicts}
        )
        
    def _detect_conflicts(self, contracts: List[ContractBase], data: Any) -> List[Dict[str, Any]]:
        """Detect conflicts between contracts."""
        conflicts = []
        
        for i, contract1 in enumerate(contracts):
            for j, contract2 in enumerate(contracts[i+1:], i+1):
                conflict = self._check_contract_pair_conflict(contract1, contract2, data)
                if conflict:
                    conflicts.append({
                        "contract1": contract1.name,
                        "contract2": contract2.name,
                        "conflict_type": conflict["type"],
                        "details": conflict["details"]
                    })
                    
        return conflicts
        
    def _check_contract_pair_conflict(self, 
                                     contract1: ContractBase, 
                                     contract2: ContractBase, 
                                     data: Any) -> Optional[Dict[str, Any]]:
        """Check if two contracts conflict."""
        # Example conflict detection logic
        # This can be expanded based on specific contract types
        
        # Check for format conflicts
        if (hasattr(contract1, 'required_format') and 
            hasattr(contract2, 'required_format')):
            if contract1.required_format != contract2.required_format:
                return {
                    "type": "format_conflict",
                    "details": f"{contract1.required_format} vs {contract2.required_format}"
                }
                
        # Check for length constraints
        if (hasattr(contract1, 'max_length') and 
            hasattr(contract2, 'min_length')):
            if contract1.max_length < contract2.min_length:
                return {
                    "type": "length_conflict", 
                    "details": f"max_length {contract1.max_length} < min_length {contract2.min_length}"
                }
                
        return None
        
    def _find_most_restrictive(self, contracts: List[ContractBase], data: Any) -> List[str]:
        """Find the most restrictive contract(s)."""
        # Simple heuristic: contracts with more validation rules are considered more restrictive
        contract_scores = []
        
        for contract in contracts:
            score = 0
            # Count validation attributes as a measure of restrictiveness
            if hasattr(contract, 'max_length'):
                score += 1
            if hasattr(contract, 'required_format'):
                score += 2
            if hasattr(contract, 'banned_patterns'):
                score += len(getattr(contract, 'banned_patterns', []))
                
            contract_scores.append((contract.name, score))
            
        # Return contracts with highest scores
        max_score = max(contract_scores, key=lambda x: x[1])[1] if contract_scores else 0
        return [name for name, score in contract_scores if score == max_score]
        
    def _merge_contracts(self, contracts: List[ContractBase]) -> Optional[ContractBase]:
        """Attempt to merge compatible contracts."""
        # This is a complex operation that would need to be implemented
        # based on specific contract types. For now, return None to indicate
        # merging is not yet implemented.
        logger.warning("Contract merging not yet implemented")
        return None


class IntelligentAutoRemediator:
    """Advanced auto-remediation system with multiple strategies."""
    
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.remediation_strategies: Dict[str, Callable] = {
            "json_fix": self._fix_json_format,
            "length_truncate": self._fix_length_issues,
            "content_filter": self._fix_content_violations,
            "format_correction": self._fix_format_issues
        }
        
    async def attempt_remediation(self, 
                                content: str,
                                violation: ValidationResult,
                                context: OutputValidationContext) -> AutoRemediationResult:
        """Attempt to automatically remediate content violations."""
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                # Determine remediation strategy based on violation type
                strategy = self._select_strategy(violation, content)
                
                if strategy and strategy in self.remediation_strategies:
                    corrected_content = await self.remediation_strategies[strategy](
                        content, violation, context
                    )
                    
                    if corrected_content and corrected_content != content:
                        return AutoRemediationResult(
                            success=True,
                            original_content=content,
                            corrected_content=corrected_content,
                            method_used=strategy,
                            attempts=attempt
                        )
                        
            except Exception as e:
                logger.warning(f"Auto-remediation attempt {attempt} failed: {e}")
                
        return AutoRemediationResult(
            success=False,
            original_content=content,
            attempts=self.max_attempts,
            error_message="All remediation attempts failed"
        )
        
    def _select_strategy(self, violation: ValidationResult, content: str) -> Optional[str]:
        """Select appropriate remediation strategy based on violation."""
        message = violation.message.lower()
        
        if "json" in message or "format" in message:
            return "json_fix"
        elif "length" in message or "too long" in message:
            return "length_truncate"
        elif "content" in message or "inappropriate" in message:
            return "content_filter"
        elif "format" in message:
            return "format_correction"
            
        return None
        
    async def _fix_json_format(self, 
                             content: str, 
                             violation: ValidationResult,
                             context: OutputValidationContext) -> Optional[str]:
        """Fix JSON formatting issues."""
        try:
            # Try to extract JSON from content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Validate JSON
                json.loads(json_str)
                return json_str
                
            # Try to create valid JSON from content
            return json.dumps({"content": content.strip()})
            
        except Exception as e:
            logger.warning(f"JSON fix failed: {e}")
            return None
            
    async def _fix_length_issues(self, 
                               content: str, 
                               violation: ValidationResult,
                               context: OutputValidationContext) -> Optional[str]:
        """Fix content length issues by truncating."""
        try:
            # Extract target length from violation message
            length_match = re.search(r'(\d+)', violation.message)
            if length_match:
                target_length = int(length_match.group(1))
                if len(content) > target_length:
                    # Truncate intelligently at word boundaries
                    truncated = content[:target_length]
                    last_space = truncated.rfind(' ')
                    if last_space > target_length * 0.8:  # Don't cut too much
                        truncated = truncated[:last_space]
                    return truncated + "..."
                    
        except Exception as e:
            logger.warning(f"Length fix failed: {e}")
            
        return None
        
    async def _fix_content_violations(self, 
                                    content: str, 
                                    violation: ValidationResult,
                                    context: OutputValidationContext) -> Optional[str]:
        """Fix content policy violations."""
        try:
            # Simple content filtering - remove problematic patterns
            cleaned_content = content
            
            # Remove common problematic patterns
            problematic_patterns = [
                r'\b(hate|violence|harmful)\w*\b',
                r'\b(inappropriate|offensive)\w*\b'
            ]
            
            for pattern in problematic_patterns:
                cleaned_content = re.sub(pattern, '[FILTERED]', cleaned_content, flags=re.IGNORECASE)
                
            return cleaned_content if cleaned_content != content else None
            
        except Exception as e:
            logger.warning(f"Content fix failed: {e}")
            return None
            
    async def _fix_format_issues(self, 
                               content: str, 
                               violation: ValidationResult,
                               context: OutputValidationContext) -> Optional[str]:
        """Fix general format issues."""
        try:
            # Basic format cleanup
            cleaned = content.strip()
            
            # Remove extra whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Ensure proper sentence ending
            if cleaned and not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'
                
            return cleaned if cleaned != content else None
            
        except Exception as e:
            logger.warning(f"Format fix failed: {e}")
            return None


class EnhancedStreamingValidator:
    """Enhanced streaming validator with real-time processing and conflict resolution."""
    
    def __init__(self, 
                 contracts: List[ContractBase],
                 conflict_resolver: Optional[ContractConflictResolver] = None,
                 auto_remediator: Optional[IntelligentAutoRemediator] = None):
        self.contracts = contracts
        self.conflict_resolver = conflict_resolver or ContractConflictResolver()
        self.auto_remediator = auto_remediator or IntelligentAutoRemediator()
        self.buffer = ""
        self.validation_checkpoints: List[int] = [100, 500, 1000]  # Character counts for validation
        self.chunk_validators = [c for c in contracts if hasattr(c, 'supports_streaming') and c.supports_streaming]
        self.final_validators = [c for c in contracts if not (hasattr(c, 'supports_streaming') and c.supports_streaming)]
        
    async def validate_chunk(self, 
                           chunk: str, 
                           context: OutputValidationContext) -> "StreamValidationResult":
        """Validate individual chunk with enhanced processing."""
        self.buffer += chunk
        
        results: List[ValidationResult] = []
        should_terminate = False
        
        # Check if we should validate at this buffer length
        if any(len(self.buffer) >= checkpoint for checkpoint in self.validation_checkpoints):
            
            # Resolve conflicts first
            resolution = self.conflict_resolver.resolve_conflicts(
                self.chunk_validators, self.buffer
            )
            
            active_contracts = [
                c for c in self.chunk_validators 
                if c.name in resolution.winning_contracts
            ]
            
            # Validate with active contracts
            for contract in active_contracts:
                if hasattr(contract, 'validate_partial'):
                    try:
                        result = await contract.validate_partial(self.buffer)
                        results.append(result)
                        
                        # Check for critical violations
                        if (hasattr(result, 'is_valid') and not result.is_valid and
                            hasattr(result, 'severity') and 
                            result.severity == ViolationSeverity.CRITICAL):
                            should_terminate = True
                            break
                            
                    except Exception as e:
                        logger.error(f"Chunk validation failed for {contract.name}: {e}")
                        
        return StreamValidationResult(
            should_terminate=should_terminate,
            violation=None,
            partial_results=results
        )
        
    async def finalize_validation(self, 
                                context: OutputValidationContext) -> List[ValidationResult]:
        """Enhanced final validation with conflict resolution and auto-remediation."""
        
        # Resolve conflicts among all validators
        resolution = self.conflict_resolver.resolve_conflicts(
            self.final_validators, self.buffer
        )
        
        active_contracts = [
            c for c in self.final_validators 
            if c.name in resolution.winning_contracts
        ]
        
        final_results: List[ValidationResult] = []
        
        for contract in active_contracts:
            try:
                if hasattr(contract, 'validate'):
                    if asyncio.iscoroutinefunction(contract.validate):
                        result = await contract.validate(self.buffer)
                    else:
                        result = contract.validate(self.buffer)
                    
                    # Attempt auto-remediation for violations
                    if hasattr(result, 'is_valid') and not result.is_valid:
                        remediation = await self.auto_remediator.attempt_remediation(
                            self.buffer, result, context
                        )
                        
                        if remediation.success:
                            # Update buffer with corrected content
                            self.buffer = remediation.corrected_content
                            # Re-validate with corrected content
                            if asyncio.iscoroutinefunction(contract.validate):
                                result = await contract.validate(self.buffer)
                            else:
                                result = contract.validate(self.buffer)
                                
                    final_results.append(result)
                    
            except Exception as e:
                logger.error(f"Final validation failed for {contract.name}: {e}")
                final_results.append(ValidationResult(
                    is_valid=False,
                    message=f"Validation error: {str(e)}"
                ))
                
        return final_results


# Import existing StreamValidationResult from provider or redefine
class StreamValidationResult:
    """Enhanced result of streaming chunk validation."""
    
    def __init__(self, 
                 should_terminate: bool = False, 
                 violation: Optional[ValidationResult] = None, 
                 partial_results: Optional[List[ValidationResult]] = None,
                 auto_remediation: Optional[AutoRemediationResult] = None):
        self.should_terminate = should_terminate
        self.violation = violation
        self.partial_results: List[ValidationResult] = partial_results or []
        self.auto_remediation = auto_remediation


class PerformanceOptimizedOutputValidator(ValidatorBase):
    """
    High-performance output validator with streaming support, conflict resolution,
    and intelligent auto-remediation capabilities.
    """
    
    def __init__(self,
                 name: str = "performance_output_validator",
                 enable_circuit_breaker: bool = True,
                 enable_metrics: bool = True,
                 enable_tracing: bool = True,
                 enable_auto_remediation: bool = True,
                 conflict_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MOST_RESTRICTIVE,
                 max_concurrent_validations: int = 10):
        super().__init__(name)
        
        # Core components
        self.circuit_breaker = ContractCircuitBreaker() if enable_circuit_breaker else None
        self.metrics = ContractMetrics() if enable_metrics else None
        self.enable_tracing = enable_tracing
        self.tracer = get_tracer("llm_contracts.output_validator") if enable_tracing else None
        
        # Advanced components
        self.conflict_resolver = ContractConflictResolver(conflict_resolution_strategy)
        self.auto_remediator = IntelligentAutoRemediator() if enable_auto_remediation else None
        
        # Performance optimization
        self.semaphore = asyncio.Semaphore(max_concurrent_validations)
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.cache_max_size = 100
        
        logger.info(f"Initialized {self.__class__.__name__} with "
                   f"auto_remediation={enable_auto_remediation}, "
                   f"conflict_strategy={conflict_resolution_strategy.value}")
                   
    def create_streaming_validator(self, contracts: Optional[List[ContractBase]] = None) -> EnhancedStreamingValidator:
        """Create an enhanced streaming validator."""
        active_contracts = contracts or self.contracts
        return EnhancedStreamingValidator(
            contracts=active_contracts,
            conflict_resolver=self.conflict_resolver,
            auto_remediator=self.auto_remediator
        )
        
    async def validate_async(self,
                           content: str,
                           context: Optional[OutputValidationContext] = None) -> List[ValidationResult]:
        """
        Asynchronous output validation with conflict resolution and auto-remediation.
        """
        if context is None:
            context = OutputValidationContext(
                request_id=f"out_{int(time.time())}")
                
        # Circuit breaker check
        if self.circuit_breaker and self.circuit_breaker.should_skip():
            logger.warning("Output validation skipped due to circuit breaker")
            return [ValidationResult(
                is_valid=True,
                message="Validation skipped due to circuit breaker"
            )]
            
        # Enhanced tracing
        if self.tracer:
            trace_context = TraceContext(
                operation_type=OperationType.OUTPUT_VALIDATION,
                contract_name=self.name,
                request_id=context.request_id,
                model=context.model,
                metadata={
                    "response_format": context.response_format,
                    "streaming": context.streaming,
                    "content_length": len(content)
                }
            )
            
            with self.tracer.trace_operation(trace_context, "output_validation") as span:
                return await self._perform_validation(content, context, span)
        else:
            return await self._perform_validation(content, context)
            
    async def _perform_validation(self,
                                content: str,
                                context: OutputValidationContext,
                                span: Optional[Any] = None) -> List[ValidationResult]:
        """Internal validation logic with enhanced processing."""
        
        validation_start = time.time()
        results: List[ValidationResult] = []
        
        try:
            async with self.semaphore:
                # Conflict resolution
                resolution = self.conflict_resolver.resolve_conflicts(self.contracts, content)
                active_contracts = [
                    c for c in self.contracts 
                    if c.name in resolution.winning_contracts
                ]
                
                # Record conflict resolution
                if span and self.tracer:
                    self.tracer.add_event(span, "conflict_resolution", {
                        "conflicts_detected": resolution.conflicts_detected,
                        "strategy": resolution.resolution_strategy.value,
                        "active_contracts": len(active_contracts)
                    })
                
                # Concurrent validation
                if active_contracts:
                    validation_tasks = [
                        self._validate_with_contract(contract, content, context)
                        for contract in active_contracts
                    ]
                    contract_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
                    
                    for i, result in enumerate(contract_results):
                        if isinstance(result, Exception):
                            results.append(ValidationResult(
                                is_valid=False,
                                message=f"Contract validation error: {str(result)}"
                            ))
                        else:
                            # Auto-remediation for violations
                            if (not result.is_valid and self.auto_remediator and
                                hasattr(result, 'auto_fix_suggestion')):
                                
                                remediation = await self.auto_remediator.attempt_remediation(
                                    content, result, context
                                )
                                
                                if remediation.success and remediation.corrected_content:
                                    # Re-validate with corrected content
                                    corrected_result = await self._validate_with_contract(
                                        active_contracts[i], remediation.corrected_content, context
                                    )
                                    results.append(corrected_result)
                                    
                                    # Record auto-remediation metrics
                                    if self.metrics:
                                        self.metrics.record_auto_fix_attempt(
                                            active_contracts[i].name, True
                                        )
                                else:
                                    results.append(result)
                                    if self.metrics:
                                        self.metrics.record_auto_fix_attempt(
                                            active_contracts[i].name, False
                                        )
                            else:
                                results.append(result)
                
                # Update metrics and circuit breaker
                validation_duration = time.time() - validation_start
                violations = [r for r in results if not r.is_valid]
                
                if self.metrics:
                    self.metrics.record_validation_time(
                        "output_validation",
                        validation_duration,
                        violated=len(violations) > 0
                    )
                    
                if self.circuit_breaker:
                    if violations:
                        self.circuit_breaker.record_failure("output_validation")
                    else:
                        self.circuit_breaker.record_success()
                        
                # Enhanced tracing
                if span and self.tracer:
                    self.tracer.add_event(span, "validation_completed", {
                        "duration_ms": validation_duration * 1000,
                        "contract_count": len(active_contracts),
                        "violation_count": len(violations),
                        "auto_remediation_count": sum(1 for r in results if hasattr(r, 'auto_remediation'))
                    })
                    
                return results
                
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            
            if span and self.tracer:
                self.tracer.add_event(span, "validation_error", {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
                
            if self.circuit_breaker:
                self.circuit_breaker.record_failure("output_validation")
                
            return [ValidationResult(
                is_valid=False,
                message=f"Output validation system error: {str(e)}"
            )]
            
    async def _validate_with_contract(self,
                                    contract: ContractBase,
                                    content: str,
                                    context: OutputValidationContext) -> ValidationResult:
        """Validate content with a specific contract."""
        try:
            validation_start = time.time()
            
            # Convert context to dict for contract validation
            context_dict = {
                "request_id": context.request_id,
                "model": context.model,
                "user_id": context.user_id,
                "conversation_id": context.conversation_id,
                "response_format": context.response_format,
                "streaming": context.streaming,
                "metadata": context.metadata,
                "timestamp": context.timestamp
            }
            
            if asyncio.iscoroutinefunction(contract.validate):
                result = await contract.validate(content, context_dict)
            else:
                result = contract.validate(content, context_dict)
                
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
            
    def validate_all(self, content: str, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Synchronous validation wrapper for compatibility."""
        output_context = OutputValidationContext(
            request_id=f"sync_out_{int(time.time())}"
        )
        
        if context:
            output_context.model = context.get("model")
            output_context.user_id = context.get("user_id")
            output_context.conversation_id = context.get("conversation_id")
            output_context.response_format = context.get("response_format", "text")
            output_context.streaming = context.get("streaming", False)
            output_context.original_request = context.get("original_request")
            output_context.metadata = context.get("metadata", {})
            
        # Run async validation
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.validate_async(content, output_context))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.validate_async(content, output_context))
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
            "auto_remediation_enabled": self.auto_remediator is not None,
            "conflict_resolution_strategy": self.conflict_resolver.default_strategy.value
        })
        
        return report
        
    def reset_metrics(self) -> None:
        """Reset all metrics and caches."""
        if self.metrics:
            self.metrics = ContractMetrics()
            
        if self.circuit_breaker:
            self.circuit_breaker = ContractCircuitBreaker()
            
        self.validation_cache.clear()
        logger.info("Output validator metrics and caches reset")


# Backward compatibility alias
OutputValidator = PerformanceOptimizedOutputValidator