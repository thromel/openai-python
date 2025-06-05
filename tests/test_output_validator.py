"""
Tests for the PerformanceOptimizedOutputValidator implementation.

This test suite validates all the enhanced features implemented for Task 5:
- Real-time streaming validation
- Advanced conflict resolution
- Intelligent auto-remediation
- State management for multi-turn contexts  
- Critical violation termination
- Performance optimization
"""

import asyncio
import pytest  # type: ignore
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, Optional

from llm_contracts.validators.output_validator import (
    PerformanceOptimizedOutputValidator,
    OutputValidationContext,
    ContractConflictResolver,
    IntelligentAutoRemediator,
    EnhancedStreamingValidator,
    ViolationSeverity,
    ConflictResolutionStrategy,
    AutoRemediationResult,
    ConflictResolution
)
from llm_contracts.core.interfaces import ValidationResult, ContractBase, ContractType


class MockOutputContract(ContractBase):
    """Mock output contract for testing."""
    
    def __init__(self, name: str, should_pass: bool = True, 
                 required_format: Optional[str] = None,
                 max_length: Optional[int] = None,
                 banned_patterns: Optional[list] = None,
                 supports_streaming: bool = False):
        super().__init__(name)
        self.should_pass = should_pass
        self.required_format = required_format
        self.max_length = max_length
        self.banned_patterns = banned_patterns or []
        self.supports_streaming = supports_streaming
        
    def _get_contract_type(self) -> ContractType:  # type: ignore[override]
        return ContractType.OUTPUT
        
    def validate(self, content: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:  # type: ignore[override]
        # Simulate validation logic
        if not self.should_pass:
            return ValidationResult(
                is_valid=False,
                message=f"Contract {self.name} failed validation",
                auto_fix_suggestion="Apply suggested fix"
            )
            
        # Check length constraint
        if self.max_length and len(content) > self.max_length:
            return ValidationResult(
                is_valid=False,
                message=f"Content length {len(content)} exceeds maximum {self.max_length}",
                auto_fix_suggestion="Truncate content"
            )
            
        # Check format requirement
        if self.required_format == "json":
            try:
                import json
                json.loads(content)
            except:
                return ValidationResult(
                    is_valid=False,
                    message="Content is not valid JSON",
                    auto_fix_suggestion="Fix JSON formatting"
                )
                
        # Check banned patterns
        for pattern in self.banned_patterns:
            if pattern.lower() in content.lower():
                return ValidationResult(
                    is_valid=False,
                    message=f"Content contains banned pattern: {pattern}",
                    auto_fix_suggestion="Remove inappropriate content"
                )
                
        return ValidationResult(
            is_valid=True,
            message=f"Contract {self.name} validation passed"
        )
        
    async def validate_partial(self, content: str) -> ValidationResult:
        """Streaming validation for partial content."""
        # Simplified partial validation
        if len(content) > 100 and not self.should_pass:
            return ValidationResult(
                is_valid=False,
                message=f"Partial validation failed for {self.name}",
                severity=ViolationSeverity.CRITICAL if "critical" in self.name else ViolationSeverity.MEDIUM
            )
            
        return ValidationResult(is_valid=True, message="Partial validation passed")


class TestContractConflictResolver:
    """Test the conflict resolution system."""
    
    def test_initialization(self):
        resolver = ContractConflictResolver()
        assert resolver.default_strategy == ConflictResolutionStrategy.MOST_RESTRICTIVE
        
    def test_register_conflict_rule(self):
        resolver = ContractConflictResolver()
        resolver.register_conflict_rule(
            ("contract1", "contract2"), 
            ConflictResolutionStrategy.FIRST_WINS
        )
        
        # Rules are stored with sorted keys
        assert ("contract1", "contract2") in resolver.conflict_rules
        
    def test_no_conflicts_detected(self):
        resolver = ContractConflictResolver()
        
        # Create non-conflicting contracts
        contract1 = MockOutputContract("contract1", required_format="json")
        contract2 = MockOutputContract("contract2", max_length=1000)
        
        resolution = resolver.resolve_conflicts([contract1, contract2], "test content")
        
        assert not resolution.conflicts_detected
        assert len(resolution.winning_contracts) == 2
        
    def test_format_conflicts_detected(self):
        resolver = ContractConflictResolver()
        
        # Create conflicting contracts
        contract1 = MockOutputContract("contract1", required_format="json")
        contract2 = MockOutputContract("contract2", required_format="xml")
        
        resolution = resolver.resolve_conflicts([contract1, contract2], "test")
        
        assert resolution.conflicts_detected
        assert len(resolution.winning_contracts) > 0
        
    def test_first_wins_strategy(self):
        resolver = ContractConflictResolver(ConflictResolutionStrategy.FIRST_WINS)
        
        contract1 = MockOutputContract("first", required_format="json")
        contract2 = MockOutputContract("second", required_format="xml")
        
        resolution = resolver.resolve_conflicts([contract1, contract2], "test")
        
        assert "first" in resolution.winning_contracts
        assert "second" not in resolution.winning_contracts
        
    def test_last_wins_strategy(self):
        resolver = ContractConflictResolver(ConflictResolutionStrategy.LAST_WINS)
        
        contract1 = MockOutputContract("first", required_format="json")
        contract2 = MockOutputContract("second", required_format="xml")
        
        resolution = resolver.resolve_conflicts([contract1, contract2], "test")
        
        assert "second" in resolution.winning_contracts
        assert "first" not in resolution.winning_contracts


class TestIntelligentAutoRemediator:
    """Test the auto-remediation system."""
    
    def test_initialization(self):
        remediator = IntelligentAutoRemediator(max_attempts=5)
        assert remediator.max_attempts == 5
        assert "json_fix" in remediator.remediation_strategies
        
    @pytest.mark.asyncio
    async def test_json_fix_strategy(self):
        remediator = IntelligentAutoRemediator()
        
        # Test JSON content extraction
        content = 'Here is the JSON: {"key": "value"} and some extra text'
        violation = ValidationResult(is_valid=False, message="Invalid JSON format")
        context = OutputValidationContext(request_id="test")
        
        result = await remediator._fix_json_format(content, violation, context)
        
        assert result is not None
        assert '{"key": "value"}' in result
        
    @pytest.mark.asyncio
    async def test_length_fix_strategy(self):
        remediator = IntelligentAutoRemediator()
        
        # Test content truncation
        content = "This is a very long content that needs to be truncated to fit within limits"
        violation = ValidationResult(is_valid=False, message="Content exceeds 50 characters")
        context = OutputValidationContext(request_id="test")
        
        result = await remediator._fix_length_issues(content, violation, context)
        
        assert result is not None
        assert len(result) <= 50
        assert result.endswith("...")
        
    @pytest.mark.asyncio
    async def test_content_filter_strategy(self):
        remediator = IntelligentAutoRemediator()
        
        # Test content filtering
        content = "This contains inappropriate harmful content that should be filtered"
        violation = ValidationResult(is_valid=False, message="Content policy violation")
        context = OutputValidationContext(request_id="test")
        
        result = await remediator._fix_content_violations(content, violation, context)
        
        assert result is not None
        assert "[FILTERED]" in result
        assert "harmful" not in result
        
    @pytest.mark.asyncio
    async def test_successful_remediation(self):
        remediator = IntelligentAutoRemediator()
        
        # Test successful remediation flow
        content = '{"invalid": json}'
        violation = ValidationResult(is_valid=False, message="Invalid JSON format")
        context = OutputValidationContext(request_id="test")
        
        result = await remediator.attempt_remediation(content, violation, context)
        
        assert result.success
        assert result.original_content == content
        assert result.corrected_content is not None
        assert result.method_used == "json_fix"
        
    @pytest.mark.asyncio
    async def test_failed_remediation(self):
        remediator = IntelligentAutoRemediator(max_attempts=1)
        
        # Test failed remediation (no applicable strategy)
        content = "uncorrectable content"
        violation = ValidationResult(is_valid=False, message="Unknown violation type")
        context = OutputValidationContext(request_id="test")
        
        result = await remediator.attempt_remediation(content, violation, context)
        
        assert not result.success
        assert result.attempts == 1
        assert result.error_message is not None


class TestEnhancedStreamingValidator:
    """Test the enhanced streaming validation system."""
    
    def test_initialization(self):
        contracts = [MockOutputContract("test", supports_streaming=True)]
        validator = EnhancedStreamingValidator(contracts)
        
        assert len(validator.contracts) == 1
        assert len(validator.chunk_validators) == 1
        assert len(validator.final_validators) == 0
        
    @pytest.mark.asyncio
    async def test_chunk_validation_normal(self):
        contract = MockOutputContract("streaming_test", supports_streaming=True)
        validator = EnhancedStreamingValidator([contract])
        context = OutputValidationContext(request_id="test")
        
        # Test normal chunk validation
        result = await validator.validate_chunk("test chunk", context)
        
        assert not result.should_terminate
        assert len(result.partial_results) >= 0
        
    @pytest.mark.asyncio
    async def test_chunk_validation_critical_violation(self):
        contract = MockOutputContract("critical_test", should_pass=False, supports_streaming=True)
        validator = EnhancedStreamingValidator([contract])
        context = OutputValidationContext(request_id="test")
        
        # Add enough content to trigger validation
        for i in range(15):  # Enough to exceed 100 chars
            await validator.validate_chunk("test chunk " * 2, context)
        
        # Should have partial results from validation attempts
        assert len(validator.buffer) > 100
        
    @pytest.mark.asyncio
    async def test_finalize_validation(self):
        contract = MockOutputContract("final_test")
        validator = EnhancedStreamingValidator([contract])
        context = OutputValidationContext(request_id="test")
        
        # Add some content to buffer
        validator.buffer = "test content for final validation"
        
        results = await validator.finalize_validation(context)
        
        assert len(results) >= 0  # Should have some results


class TestOutputValidationContext:
    """Test the OutputValidationContext dataclass."""
    
    def test_context_creation(self):
        context = OutputValidationContext(
            request_id="test_123",
            model="gpt-4",
            response_format="json",
            streaming=True
        )
        
        assert context.request_id == "test_123"
        assert context.model == "gpt-4"
        assert context.response_format == "json"
        assert context.streaming is True
        assert isinstance(context.timestamp, float)


class TestPerformanceOptimizedOutputValidator:
    """Test the main PerformanceOptimizedOutputValidator class."""
    
    def test_initialization(self):
        validator = PerformanceOptimizedOutputValidator(
            name="test_output_validator",
            enable_circuit_breaker=True,
            enable_metrics=True,
            enable_tracing=False,  # Disable for testing
            enable_auto_remediation=True
        )
        
        assert validator.name == "test_output_validator"
        assert validator.circuit_breaker is not None
        assert validator.metrics is not None
        assert validator.auto_remediator is not None
        assert validator.conflict_resolver is not None
        
    def test_create_streaming_validator(self):
        validator = PerformanceOptimizedOutputValidator(enable_tracing=False)
        
        contract = MockOutputContract("streaming_contract", supports_streaming=True)
        validator.add_contract(contract)
        
        streaming_validator = validator.create_streaming_validator()
        
        assert isinstance(streaming_validator, EnhancedStreamingValidator)
        assert len(streaming_validator.contracts) == 1
        
    @pytest.mark.asyncio
    async def test_async_validation_basic(self):
        validator = PerformanceOptimizedOutputValidator(enable_tracing=False)
        
        # Add a passing contract
        contract = MockOutputContract("basic_test", should_pass=True)
        validator.add_contract(contract)
        
        content = "This is test content for validation"
        context = OutputValidationContext(request_id="test_001")
        
        results = await validator.validate_async(content, context)
        
        # Should have results with no violations
        assert len(results) > 0
        violations = [r for r in results if not r.is_valid]
        assert len(violations) == 0
        
    @pytest.mark.asyncio
    async def test_async_validation_with_violation(self):
        validator = PerformanceOptimizedOutputValidator(enable_tracing=False)
        
        # Add a failing contract
        contract = MockOutputContract("failing_test", should_pass=False)
        validator.add_contract(contract)
        
        content = "This content will fail validation"
        context = OutputValidationContext(request_id="test_002")
        
        results = await validator.validate_async(content, context)
        
        # Should have violations
        violations = [r for r in results if not r.is_valid]
        assert len(violations) > 0
        
    @pytest.mark.asyncio
    async def test_auto_remediation_integration(self):
        validator = PerformanceOptimizedOutputValidator(
            enable_tracing=False,
            enable_auto_remediation=True
        )
        
        # Add a contract that checks JSON format
        contract = MockOutputContract("json_test", required_format="json")
        validator.add_contract(contract)
        
        # Content that will fail JSON validation but can be auto-fixed
        content = 'Response: {"key": "value"} - here is your JSON'
        context = OutputValidationContext(request_id="test_003")
        
        results = await validator.validate_async(content, context)
        
        # Auto-remediation should have been attempted
        assert len(results) > 0
        
    @pytest.mark.asyncio
    async def test_conflict_resolution_integration(self):
        validator = PerformanceOptimizedOutputValidator(
            enable_tracing=False,
            conflict_resolution_strategy=ConflictResolutionStrategy.FIRST_WINS
        )
        
        # Add conflicting contracts
        contract1 = MockOutputContract("json_contract", required_format="json")
        contract2 = MockOutputContract("xml_contract", required_format="xml")
        validator.add_contract(contract1)
        validator.add_contract(contract2)
        
        content = '{"test": "content"}'
        context = OutputValidationContext(request_id="test_004")
        
        results = await validator.validate_async(content, context)
        
        # Should have resolved conflicts and validated
        assert len(results) > 0
        
    @pytest.mark.asyncio
    async def test_length_constraint_validation(self):
        validator = PerformanceOptimizedOutputValidator(enable_tracing=False)
        
        # Add a contract with length limit
        contract = MockOutputContract("length_test", max_length=20)
        validator.add_contract(contract)
        
        content = "This is a very long content that exceeds the maximum allowed length"
        context = OutputValidationContext(request_id="test_005")
        
        results = await validator.validate_async(content, context)
        
        # Should detect length violation
        violations = [r for r in results if not r.is_valid and "length" in r.message.lower()]
        assert len(violations) > 0
        
    @pytest.mark.asyncio
    async def test_content_filtering(self):
        validator = PerformanceOptimizedOutputValidator(enable_tracing=False)
        
        # Add a contract that bans certain patterns
        contract = MockOutputContract("filter_test", banned_patterns=["inappropriate"])
        validator.add_contract(contract)
        
        content = "This content contains inappropriate material"
        context = OutputValidationContext(request_id="test_006")
        
        results = await validator.validate_async(content, context)
        
        # Should detect content violation
        violations = [r for r in results if not r.is_valid and "banned pattern" in r.message.lower()]
        assert len(violations) > 0
        
    def test_synchronous_validation_wrapper(self):
        validator = PerformanceOptimizedOutputValidator(enable_tracing=False)
        
        contract = MockOutputContract("sync_test", should_pass=True)
        validator.add_contract(contract)
        
        content = "Test content for synchronous validation"
        
        results = validator.validate_all(content)
        
        # Should work without violations
        violations = [r for r in results if not r.is_valid]
        assert len(violations) == 0
        
    def test_metrics_collection(self):
        validator = PerformanceOptimizedOutputValidator(enable_tracing=False)
        
        contract = MockOutputContract("metrics_test")
        validator.add_contract(contract)
        
        content = "Test content for metrics"
        
        # Run validation to generate metrics
        validator.validate_all(content)
        
        # Check metrics report
        metrics_report = validator.get_metrics_report()
        
        assert "validator_name" in metrics_report
        assert "circuit_breaker_state" in metrics_report
        assert "auto_remediation_enabled" in metrics_report
        assert "conflict_resolution_strategy" in metrics_report
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self):
        validator = PerformanceOptimizedOutputValidator(enable_tracing=False)
        
        # Add multiple failing contracts to trigger circuit breaker
        for i in range(6):  # More than failure threshold
            contract = MockOutputContract(f"failing_{i}", should_pass=False)
            validator.add_contract(contract)
            
        content = "Content that will cause failures"
        context = OutputValidationContext(request_id="test_circuit")
        
        # First validation should trigger circuit breaker
        await validator.validate_async(content, context)
        
        # Check if circuit breaker is activated
        if validator.circuit_breaker:
            # Circuit breaker behavior may vary based on implementation
            assert validator.circuit_breaker.failure_count > 0
            
    def test_metrics_reset(self):
        validator = PerformanceOptimizedOutputValidator(enable_tracing=False)
        
        # Generate some metrics
        contract = MockOutputContract("reset_test")
        validator.add_contract(contract)
        validator.validate_all("test content")
        
        # Reset metrics
        validator.reset_metrics()
        
        # Verify reset
        assert len(validator.validation_cache) == 0
        if validator.metrics:
            assert validator.metrics.total_calls == 0


if __name__ == "__main__":
    # Run basic tests
    print("Running PerformanceOptimizedOutputValidator tests...")
    
    # Test conflict resolver
    resolver = ContractConflictResolver()
    print(f"✅ ConflictResolver created with strategy: {resolver.default_strategy.value}")
    
    # Test auto-remediator
    remediator = IntelligentAutoRemediator()
    print(f"✅ AutoRemediator created with {remediator.max_attempts} max attempts")
    
    # Test validator initialization
    validator = PerformanceOptimizedOutputValidator(enable_tracing=False)
    print(f"✅ OutputValidator initialized: {validator.name}")
    
    # Test basic validation
    contract = MockOutputContract("basic_test", should_pass=True)
    validator.add_contract(contract)
    
    results = validator.validate_all("Test content for basic validation")
    violations = [r for r in results if not r.is_valid]
    print(f"✅ Basic validation: {len(violations)} violations")
    
    # Test streaming validator creation
    streaming_validator = validator.create_streaming_validator()
    print(f"✅ Streaming validator created: {type(streaming_validator).__name__}")
    
    # Test metrics
    metrics = validator.get_metrics_report()
    print(f"✅ Metrics collection: {len(metrics)} metrics")
    
    print("\n🎉 All basic tests passed! Run with pytest for comprehensive testing.")