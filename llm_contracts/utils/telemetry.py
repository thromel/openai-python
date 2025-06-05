"""
OpenTelemetry integration utilities for the LLM Contracts framework.

This module provides comprehensive observability features including:
- Distributed tracing for contract validation
- Metrics collection and aggregation
- Span annotations for debugging
- Performance monitoring
- Error tracking and alerting
"""

import time
import logging
from typing import Any, Dict, List, Optional, Callable, ContextManager
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

# Optional OpenTelemetry integration
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode, Span
    from opentelemetry.metrics import Meter
    from opentelemetry.semconv.trace import SpanAttributes
    _has_otel = True
except ImportError:
    _has_otel = False
    trace = None
    metrics = None
    Status = None
    StatusCode = None
    Span = None
    Meter = None
    SpanAttributes = None

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be traced."""
    CONTRACT_VALIDATION = "contract_validation"
    INPUT_VALIDATION = "input_validation"
    OUTPUT_VALIDATION = "output_validation"
    STREAMING_VALIDATION = "streaming_validation"
    AUTO_REMEDIATION = "auto_remediation"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class TraceContext:
    """Context information for tracing operations."""
    operation_type: OperationType
    contract_name: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)


class ContractTracer:
    """
    High-performance tracing utilities for contract validation operations.
    
    Provides comprehensive observability with minimal performance overhead.
    """

    def __init__(self, service_name: str = "llm_contracts"):
        self.service_name = service_name
        self.enabled = _has_otel
        
        if self.enabled:
            self.tracer = trace.get_tracer(service_name)
            self.meter = metrics.get_meter(service_name)
            
            # Create metrics instruments
            self._setup_metrics()
        else:
            self.tracer = None
            self.meter = None
            
        logger.info(f"ContractTracer initialized (enabled={self.enabled})")

    def _setup_metrics(self) -> None:
        """Set up OpenTelemetry metrics instruments."""
        if not self.enabled or not self.meter:
            return
            
        try:
            # Counters
            self.validation_counter = self.meter.create_counter(
                name="contract_validations_total",
                description="Total number of contract validations",
                unit="1"
            )
            
            self.violation_counter = self.meter.create_counter(
                name="contract_violations_total", 
                description="Total number of contract violations",
                unit="1"
            )
            
            self.auto_fix_counter = self.meter.create_counter(
                name="auto_remediation_attempts_total",
                description="Total number of auto-remediation attempts",
                unit="1"
            )
            
            # Histograms
            self.validation_duration = self.meter.create_histogram(
                name="contract_validation_duration_seconds",
                description="Duration of contract validation operations",
                unit="s"
            )
            
            self.token_count_histogram = self.meter.create_histogram(
                name="token_count_distribution",
                description="Distribution of token counts in requests",
                unit="tokens"
            )
            
            # Gauges (using observable gauge)
            self.active_validations = self.meter.create_observable_gauge(
                name="active_validations",
                description="Number of currently active validations",
                unit="1"
            )
            
        except Exception as e:
            logger.warning(f"Failed to setup OpenTelemetry metrics: {e}")
            self.enabled = False

    @contextmanager
    def trace_operation(self, 
                       context: TraceContext,
                       span_name: Optional[str] = None) -> ContextManager[Optional[Span]]:
        """
        Context manager for tracing contract operations.
        
        Usage:
            with tracer.trace_operation(TraceContext(...)) as span:
                # Perform operation
                if span:
                    span.set_attribute("custom.attr", "value")
        """
        if not self.enabled or not self.tracer:
            yield None
            return
            
        operation_name = span_name or f"{context.operation_type.value}"
        
        with self.tracer.start_as_current_span(operation_name) as span:
            try:
                # Set standard attributes
                self._set_span_attributes(span, context)
                
                # Record operation start
                start_time = time.time()
                context.start_time = start_time
                
                yield span
                
                # Record successful completion
                duration = time.time() - start_time
                span.set_attribute("operation.duration_ms", duration * 1000)
                span.set_status(Status(StatusCode.OK))
                
                # Record metrics
                self._record_operation_metrics(context, duration, success=True)
                
            except Exception as e:
                # Record error
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.type", type(e).__name__)
                
                # Record error metrics
                duration = time.time() - context.start_time
                self._record_operation_metrics(context, duration, success=False)
                
                raise

    def _set_span_attributes(self, span: Span, context: TraceContext) -> None:
        """Set standard span attributes."""
        try:
            # Operation metadata
            span.set_attribute("operation.type", context.operation_type.value)
            span.set_attribute("service.name", self.service_name)
            
            if context.contract_name:
                span.set_attribute("contract.name", context.contract_name)
                
            if context.request_id:
                span.set_attribute("request.id", context.request_id)
                
            if context.user_id:
                span.set_attribute("user.id", context.user_id)
                
            if context.model:
                span.set_attribute("llm.model", context.model)
            
            # Custom metadata
            for key, value in context.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"custom.{key}", value)
                    
        except Exception as e:
            logger.warning(f"Failed to set span attributes: {e}")

    def _record_operation_metrics(self, 
                                 context: TraceContext,
                                 duration: float,
                                 success: bool) -> None:
        """Record metrics for the operation."""
        if not self.enabled:
            return
            
        try:
            # Common attributes
            attributes = {
                "operation_type": context.operation_type.value,
                "success": str(success).lower()
            }
            
            if context.contract_name:
                attributes["contract_name"] = context.contract_name
                
            if context.model:
                attributes["model"] = context.model
            
            # Record duration
            if hasattr(self, 'validation_duration'):
                self.validation_duration.record(duration, attributes)
            
            # Record counter
            if hasattr(self, 'validation_counter'):
                self.validation_counter.add(1, attributes)
                
        except Exception as e:
            logger.warning(f"Failed to record operation metrics: {e}")

    def record_contract_violation(self,
                                 contract_name: str,
                                 violation_type: str,
                                 severity: str = "medium",
                                 auto_fixed: bool = False,
                                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a contract violation with detailed metadata."""
        if not self.enabled:
            return
            
        try:
            attributes = {
                "contract_name": contract_name,
                "violation_type": violation_type,
                "severity": severity,
                "auto_fixed": str(auto_fixed).lower()
            }
            
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        attributes[f"metadata_{key}"] = str(value)
            
            # Record violation counter
            if hasattr(self, 'violation_counter'):
                self.violation_counter.add(1, attributes)
                
            # Record auto-fix attempt if applicable
            if auto_fixed and hasattr(self, 'auto_fix_counter'):
                self.auto_fix_counter.add(1, {
                    "contract_name": contract_name,
                    "success": "true"
                })
                
        except Exception as e:
            logger.warning(f"Failed to record contract violation: {e}")

    def record_token_metrics(self,
                           token_count: int,
                           model: str,
                           request_type: str = "unknown") -> None:
        """Record token count metrics."""
        if not self.enabled or not hasattr(self, 'token_count_histogram'):
            return
            
        try:
            attributes = {
                "model": model,
                "request_type": request_type
            }
            
            self.token_count_histogram.record(token_count, attributes)
            
        except Exception as e:
            logger.warning(f"Failed to record token metrics: {e}")

    def create_child_span(self,
                         name: str,
                         parent_span: Optional[Span] = None,
                         attributes: Optional[Dict[str, Any]] = None) -> Optional[Span]:
        """Create a child span for detailed operation tracking."""
        if not self.enabled or not self.tracer:
            return None
            
        try:
            span = self.tracer.start_span(name)
            
            if attributes:
                for key, value in attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(key, value)
                        
            return span
            
        except Exception as e:
            logger.warning(f"Failed to create child span: {e}")
            return None

    def add_event(self,
                 span: Optional[Span],
                 name: str,
                 attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to a span."""
        if not self.enabled or not span:
            return
            
        try:
            span.add_event(name, attributes or {})
        except Exception as e:
            logger.warning(f"Failed to add span event: {e}")

    def get_trace_id(self, span: Optional[Span] = None) -> Optional[str]:
        """Get the current trace ID for correlation."""
        if not self.enabled:
            return None
            
        try:
            if span:
                return format(span.get_span_context().trace_id, '032x')
            
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                return format(current_span.get_span_context().trace_id, '032x')
                
        except Exception as e:
            logger.warning(f"Failed to get trace ID: {e}")
            
        return None


# Global tracer instance
_global_tracer: Optional[ContractTracer] = None


def get_tracer(service_name: str = "llm_contracts") -> ContractTracer:
    """Get or create the global tracer instance."""
    global _global_tracer
    
    if _global_tracer is None:
        _global_tracer = ContractTracer(service_name)
        
    return _global_tracer


def trace_contract_validation(contract_name: str,
                            operation_type: OperationType = OperationType.CONTRACT_VALIDATION):
    """
    Decorator for tracing contract validation methods.
    
    Usage:
        @trace_contract_validation("my_contract")
        def validate(self, data, context):
            # validation logic
            return result
    """
    def decorator(func: Callable) -> Callable:
        if not _has_otel:
            return func  # Return original function if tracing not available
            
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            context = TraceContext(
                operation_type=operation_type,
                contract_name=contract_name
            )
            
            with tracer.trace_operation(context) as span:
                if span:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


# Convenience functions for common operations
def start_validation_trace(contract_name: str, 
                          request_id: Optional[str] = None,
                          model: Optional[str] = None) -> TraceContext:
    """Start a validation trace and return context."""
    return TraceContext(
        operation_type=OperationType.CONTRACT_VALIDATION,
        contract_name=contract_name,
        request_id=request_id,
        model=model
    )


def record_validation_success(tracer: ContractTracer,
                            contract_name: str,
                            duration: float,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
    """Record a successful validation."""
    context = TraceContext(
        operation_type=OperationType.CONTRACT_VALIDATION,
        contract_name=contract_name,
        metadata=metadata or {}
    )
    
    tracer._record_operation_metrics(context, duration, success=True)


def record_validation_failure(tracer: ContractTracer,
                            contract_name: str,
                            duration: float,
                            error_message: str,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
    """Record a failed validation."""
    context = TraceContext(
        operation_type=OperationType.CONTRACT_VALIDATION,
        contract_name=contract_name,
        metadata=metadata or {}
    )
    
    tracer._record_operation_metrics(context, duration, success=False)
    tracer.record_contract_violation(
        contract_name=contract_name,
        violation_type="validation_failure",
        metadata={"error_message": error_message}
    )


# Export main classes and functions
__all__ = [
    "ContractTracer",
    "TraceContext", 
    "OperationType",
    "get_tracer",
    "trace_contract_validation",
    "start_validation_trace",
    "record_validation_success",
    "record_validation_failure"
]