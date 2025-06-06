"""Real-time streaming validation system with critical violation termination."""

import asyncio
import time
import json
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator, Iterator, Set, Tuple
import logging

from ..contracts.base import ContractBase, ValidationResult
from ..core.exceptions import ContractViolationError

logger = logging.getLogger(__name__)


class ChunkValidationMode(Enum):
    """Modes for chunk validation."""
    IMMEDIATE = auto()      # Validate every chunk immediately
    BUFFERED = auto()       # Buffer chunks and validate periodically
    THRESHOLD = auto()      # Validate when buffer reaches certain thresholds
    ADAPTIVE = auto()       # Adapt validation frequency based on content


class ViolationSeverity(Enum):
    """Severity levels for streaming violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StreamChunk:
    """Represents a chunk in the streaming response."""
    content: str
    index: int
    timestamp: float
    cumulative_content: str
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationCheckpoint:
    """Checkpoint for validation during streaming."""
    position: int
    content: str
    timestamp: float
    validation_results: List[ValidationResult] = field(default_factory=list)
    should_continue: bool = True
    auto_fix_applied: Optional[str] = None


@dataclass
class StreamValidationResult:
    """Result of streaming validation."""
    is_valid: bool
    should_terminate: bool = False
    violation_severity: Optional[ViolationSeverity] = None
    violation_message: Optional[str] = None
    checkpoint: Optional[ValidationCheckpoint] = None
    auto_fix_suggestion: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class StreamingMetrics:
    """Metrics for streaming validation performance."""
    total_chunks: int = 0
    validated_chunks: int = 0
    violations_detected: int = 0
    critical_violations: int = 0
    average_validation_latency_ms: float = 0.0
    peak_validation_latency_ms: float = 0.0
    total_validation_time_ms: float = 0.0
    stream_start_time: Optional[float] = None
    stream_end_time: Optional[float] = None
    early_terminations: int = 0
    auto_fixes_applied: int = 0
    
    def add_validation_latency(self, latency_ms: float):
        """Add a validation latency measurement."""
        self.total_validation_time_ms += latency_ms
        self.validated_chunks += 1
        self.average_validation_latency_ms = self.total_validation_time_ms / self.validated_chunks
        self.peak_validation_latency_ms = max(self.peak_validation_latency_ms, latency_ms)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        duration = (self.stream_end_time or time.time()) - (self.stream_start_time or 0)
        return {
            "total_chunks": self.total_chunks,
            "validated_chunks": self.validated_chunks,
            "validation_coverage": self.validated_chunks / max(self.total_chunks, 1),
            "violations_detected": self.violations_detected,
            "critical_violations": self.critical_violations,
            "early_terminations": self.early_terminations,
            "auto_fixes_applied": self.auto_fixes_applied,
            "average_validation_latency_ms": self.average_validation_latency_ms,
            "peak_validation_latency_ms": self.peak_validation_latency_ms,
            "total_validation_time_ms": self.total_validation_time_ms,
            "stream_duration_seconds": duration,
            "validation_overhead_percentage": (self.total_validation_time_ms / (duration * 1000)) * 100 if duration > 0 else 0,
        }


class StreamingContract(ContractBase):
    """Base class for contracts that support streaming validation."""
    
    def __init__(
        self,
        name: str,
        description: str,
        validation_mode: ChunkValidationMode = ChunkValidationMode.ADAPTIVE,
        min_chunk_size: int = 1,
        max_buffer_size: int = 1000,
        validation_interval_ms: int = 100,
        critical_patterns: Optional[List[str]] = None,
    ):
        super().__init__(name, description)
        self.validation_mode = validation_mode
        self.min_chunk_size = min_chunk_size
        self.max_buffer_size = max_buffer_size
        self.validation_interval_ms = validation_interval_ms
        self.critical_patterns = critical_patterns or []
        self.supports_streaming = True
    
    async def validate_chunk(
        self,
        chunk: StreamChunk,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Validate a single chunk. Override in subclasses."""
        return StreamValidationResult(is_valid=True)
    
    async def validate_incremental(
        self,
        cumulative_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Validate cumulative content incrementally. Override in subclasses."""
        return StreamValidationResult(is_valid=True)
    
    def should_validate_at_position(self, position: int, content: str) -> bool:
        """Determine if validation should run at this position."""
        if self.validation_mode == ChunkValidationMode.IMMEDIATE:
            return True
        elif self.validation_mode == ChunkValidationMode.THRESHOLD:
            return len(content) >= self.min_chunk_size
        elif self.validation_mode == ChunkValidationMode.ADAPTIVE:
            # Adaptive logic: validate more frequently if critical patterns detected
            for pattern in self.critical_patterns:
                if pattern.lower() in content.lower():
                    return True
            return position % 10 == 0  # Otherwise validate every 10th chunk
        return False
    
    def detect_critical_violations(self, content: str) -> List[str]:
        """Detect critical violations that require immediate termination."""
        violations = []
        for pattern in self.critical_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(f"Critical pattern detected: {pattern}")
        return violations


class StreamingValidator:
    """Enhanced real-time streaming validator with critical violation termination."""
    
    def __init__(
        self,
        contracts: List[Union[ContractBase, StreamingContract]],
        validation_mode: ChunkValidationMode = ChunkValidationMode.ADAPTIVE,
        max_buffer_size: int = 10000,
        enable_auto_fix: bool = True,
        performance_monitoring: bool = True,
        early_termination: bool = True,
    ):
        self.contracts = contracts
        self.validation_mode = validation_mode
        self.max_buffer_size = max_buffer_size
        self.enable_auto_fix = enable_auto_fix
        self.performance_monitoring = performance_monitoring
        self.early_termination = early_termination
        
        # State management
        self.buffer = ""
        self.chunk_history = deque(maxlen=1000)
        self.validation_checkpoints: List[ValidationCheckpoint] = []
        self.metrics = StreamingMetrics()
        self.is_terminated = False
        self.last_validation_time = 0.0
        
        # Performance optimization
        self.validation_cache: Dict[str, StreamValidationResult] = {}
        self.async_validation_tasks: Set[asyncio.Task] = set()
        
        # Categorize contracts
        self.streaming_contracts = [
            c for c in contracts if isinstance(c, StreamingContract) or 
            (hasattr(c, 'supports_streaming') and c.supports_streaming)
        ]
        self.batch_contracts = [
            c for c in contracts if c not in self.streaming_contracts
        ]
        
        logger.info(f"StreamingValidator initialized with {len(self.streaming_contracts)} streaming contracts and {len(self.batch_contracts)} batch contracts")
    
    async def process_chunk(
        self,
        chunk_content: str,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Process a single chunk with real-time validation."""
        if self.is_terminated:
            return StreamValidationResult(
                is_valid=False,
                should_terminate=True,
                violation_message="Stream already terminated due to critical violation"
            )
        
        start_time = time.time()
        
        # Create chunk object
        self.buffer += chunk_content
        chunk = StreamChunk(
            content=chunk_content,
            index=chunk_index,
            timestamp=start_time,
            cumulative_content=self.buffer,
            token_count=len(chunk_content.split()),
            metadata=metadata or {}
        )
        
        self.chunk_history.append(chunk)
        self.metrics.total_chunks += 1
        
        if self.metrics.stream_start_time is None:
            self.metrics.stream_start_time = start_time
        
        # Check buffer size limits
        if len(self.buffer) > self.max_buffer_size:
            # Truncate buffer while preserving recent content
            truncate_point = len(self.buffer) - (self.max_buffer_size // 2)
            self.buffer = self.buffer[truncate_point:]
            logger.warning(f"Buffer truncated at position {truncate_point}")
        
        # Immediate critical violation detection
        critical_violations = []
        for contract in self.streaming_contracts:
            if isinstance(contract, StreamingContract):
                violations = contract.detect_critical_violations(chunk.cumulative_content)
                critical_violations.extend(violations)
        
        if critical_violations and self.early_termination:
            self.is_terminated = True
            self.metrics.critical_violations += len(critical_violations)
            self.metrics.early_terminations += 1
            
            return StreamValidationResult(
                is_valid=False,
                should_terminate=True,
                violation_severity=ViolationSeverity.CRITICAL,
                violation_message=f"Critical violations detected: {'; '.join(critical_violations)}",
                performance_metrics={"validation_latency_ms": (time.time() - start_time) * 1000}
            )
        
        # Determine if we should validate at this position
        should_validate = self._should_validate_now(chunk)
        
        if not should_validate:
            return StreamValidationResult(
                is_valid=True,
                performance_metrics={"validation_latency_ms": (time.time() - start_time) * 1000}
            )
        
        # Perform streaming validation
        validation_result = await self._validate_streaming_chunk(chunk)
        
        # Record metrics
        validation_latency = (time.time() - start_time) * 1000
        if self.performance_monitoring:
            self.metrics.add_validation_latency(validation_latency)
            validation_result.performance_metrics["validation_latency_ms"] = validation_latency
        
        # Handle violations
        if not validation_result.is_valid:
            self.metrics.violations_detected += 1
            
            if validation_result.violation_severity == ViolationSeverity.CRITICAL:
                self.metrics.critical_violations += 1
                if self.early_termination:
                    self.is_terminated = True
                    self.metrics.early_terminations += 1
                    validation_result.should_terminate = True
        
        # Apply auto-fix if available and enabled
        if (not validation_result.is_valid and 
            self.enable_auto_fix and 
            validation_result.auto_fix_suggestion):
            
            # In a real implementation, you might apply the fix to the buffer
            # For now, we just record that auto-fix was suggested
            self.metrics.auto_fixes_applied += 1
            logger.info(f"Auto-fix suggested: {validation_result.auto_fix_suggestion}")
        
        self.last_validation_time = time.time()
        return validation_result
    
    async def _validate_streaming_chunk(self, chunk: StreamChunk) -> StreamValidationResult:
        """Validate a chunk using streaming contracts."""
        results = []
        auto_fix_suggestions = []
        max_severity = ViolationSeverity.LOW
        
        # Validate with streaming contracts
        for contract in self.streaming_contracts:
            try:
                if isinstance(contract, StreamingContract):
                    if contract.should_validate_at_position(chunk.index, chunk.cumulative_content):
                        result = await contract.validate_chunk(chunk)
                        results.append(result)
                        
                        if not result.is_valid:
                            if result.violation_severity:
                                if (result.violation_severity.value == ViolationSeverity.CRITICAL.value or
                                    result.violation_severity == ViolationSeverity.CRITICAL):
                                    max_severity = ViolationSeverity.CRITICAL
                                elif max_severity != ViolationSeverity.CRITICAL:
                                    max_severity = max(max_severity, result.violation_severity, key=lambda x: ['low', 'medium', 'high', 'critical'].index(x.value if hasattr(x, 'value') else str(x)))
                            
                            if result.auto_fix_suggestion:
                                auto_fix_suggestions.append(result.auto_fix_suggestion)
                
                elif hasattr(contract, 'validate_partial'):
                    # Legacy streaming contract
                    result = await contract.validate_partial(chunk.cumulative_content)
                    results.append(result)
            
            except Exception as e:
                logger.error(f"Error validating chunk with contract {contract}: {e}")
                results.append(StreamValidationResult(
                    is_valid=False,
                    violation_severity=ViolationSeverity.HIGH,
                    violation_message=f"Validation error: {str(e)}"
                ))
        
        # Aggregate results
        is_valid = all(r.is_valid for r in results if hasattr(r, 'is_valid'))
        should_terminate = any(r.should_terminate for r in results if hasattr(r, 'should_terminate'))
        
        # Create checkpoint
        checkpoint = ValidationCheckpoint(
            position=chunk.index,
            content=chunk.cumulative_content,
            timestamp=chunk.timestamp,
            validation_results=[r for r in results if hasattr(r, 'is_valid')],
            should_continue=not should_terminate,
            auto_fix_applied=auto_fix_suggestions[0] if auto_fix_suggestions else None
        )
        
        self.validation_checkpoints.append(checkpoint)
        
        return StreamValidationResult(
            is_valid=is_valid,
            should_terminate=should_terminate,
            violation_severity=max_severity if not is_valid else None,
            violation_message="; ".join([r.violation_message for r in results if hasattr(r, 'violation_message') and r.violation_message]) or None,
            checkpoint=checkpoint,
            auto_fix_suggestion=auto_fix_suggestions[0] if auto_fix_suggestions else None
        )
    
    def _should_validate_now(self, chunk: StreamChunk) -> bool:
        """Determine if validation should run for this chunk."""
        # Always validate if we haven't validated recently
        time_since_last = time.time() - self.last_validation_time
        if time_since_last > (self.validation_mode.value if hasattr(self.validation_mode, 'value') else 100) / 1000:
            return True
        
        # Check contract-specific validation requirements
        for contract in self.streaming_contracts:
            if isinstance(contract, StreamingContract):
                if contract.should_validate_at_position(chunk.index, chunk.cumulative_content):
                    return True
        
        # Adaptive validation based on content patterns
        if self.validation_mode == ChunkValidationMode.ADAPTIVE:
            # Validate more frequently for certain patterns
            risk_patterns = ['error', 'fail', 'wrong', 'bad', 'harmful', 'dangerous']
            content_lower = chunk.content.lower()
            if any(pattern in content_lower for pattern in risk_patterns):
                return True
        
        return False
    
    async def finalize_validation(self) -> StreamValidationResult:
        """Perform final validation on complete content."""
        if self.metrics.stream_start_time:
            self.metrics.stream_end_time = time.time()
        
        if not self.batch_contracts:
            return StreamValidationResult(is_valid=True)
        
        start_time = time.time()
        final_results = []
        
        # Validate complete content with batch contracts
        for contract in self.batch_contracts:
            try:
                if hasattr(contract, 'validate'):
                    if asyncio.iscoroutinefunction(contract.validate):
                        result = await contract.validate(self.buffer)
                    else:
                        result = contract.validate(self.buffer)
                    final_results.append(result)
            except Exception as e:
                logger.error(f"Error in final validation with contract {contract}: {e}")
                final_results.append(ValidationResult(
                    is_valid=False,
                    message=f"Final validation error: {str(e)}"
                ))
        
        # Aggregate final results
        is_valid = all(r.is_valid for r in final_results)
        violation_messages = [r.message for r in final_results if not r.is_valid and hasattr(r, 'message')]
        
        # Record final validation latency
        if self.performance_monitoring:
            final_latency = (time.time() - start_time) * 1000
            self.metrics.add_validation_latency(final_latency)
        
        return StreamValidationResult(
            is_valid=is_valid,
            should_terminate=False,  # Stream is already complete
            violation_message="; ".join(violation_messages) if violation_messages else None,
            performance_metrics=self.metrics.get_summary()
        )
    
    def get_buffer(self) -> str:
        """Get current buffer content."""
        return self.buffer
    
    def get_metrics(self) -> StreamingMetrics:
        """Get current metrics."""
        return self.metrics
    
    def reset(self):
        """Reset validator state for new stream."""
        self.buffer = ""
        self.chunk_history.clear()
        self.validation_checkpoints.clear()
        self.metrics = StreamingMetrics()
        self.is_terminated = False
        self.last_validation_time = 0.0
        self.validation_cache.clear()
    
    async def cleanup(self):
        """Clean up async resources."""
        # Cancel any running validation tasks
        for task in self.async_validation_tasks:
            if not task.done():
                task.cancel()
        
        if self.async_validation_tasks:
            await asyncio.gather(*self.async_validation_tasks, return_exceptions=True)
        
        self.async_validation_tasks.clear()