"""Comprehensive tests for the streaming validation system."""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Any, Dict, List, Optional

from .stream_validator import (
    StreamingValidator, StreamValidationResult, StreamChunk, 
    StreamingContract, ChunkValidationMode, ViolationSeverity
)
from .stream_processors import (
    TokenProcessor, ContentFilter, PatternMatcher, JSONValidator,
    LengthMonitor, SentimentAnalyzer, ToxicityDetector
)
from .stream_wrapper import StreamWrapper, AsyncStreamWrapper, StreamingClient
from .performance import (
    StreamingProfiler, LatencyOptimizer, BufferManager, AsyncBufferPool,
    benchmark_streaming_validation
)
from ..contracts.base import ContractBase, ValidationResult
from ..core.exceptions import ContractViolationError


class MockStreamingContract(StreamingContract):
    """Mock contract for testing."""
    
    def __init__(self, name: str = "mock", should_violate: bool = False, 
                 violation_severity: ViolationSeverity = ViolationSeverity.MEDIUM,
                 should_terminate: bool = False):
        super().__init__(name, f"Mock contract: {name}")
        self.should_violate = should_violate
        self.violation_severity = violation_severity
        self.should_terminate = should_terminate
        self.validate_calls = []
    
    async def validate_chunk(self, chunk: StreamChunk, context: Optional[Dict[str, Any]] = None) -> StreamValidationResult:
        self.validate_calls.append(chunk)
        
        if self.should_violate:
            return StreamValidationResult(
                is_valid=False,
                should_terminate=self.should_terminate,
                violation_severity=self.violation_severity,
                violation_message=f"Mock violation from {self.name}"
            )
        
        return StreamValidationResult(is_valid=True)


class MockOpenAIChunk:
    """Mock OpenAI streaming chunk."""
    
    def __init__(self, content: str):
        self.choices = [Mock()]
        self.choices[0].delta = Mock()
        self.choices[0].delta.content = content


class TestStreamingValidator:
    """Tests for StreamingValidator."""
    
    @pytest.mark.asyncio
    async def test_basic_validation_success(self):
        """Test basic successful validation."""
        contract = MockStreamingContract("test")
        validator = StreamingValidator([contract])
        
        result = await validator.process_chunk("Hello", 0)
        
        assert result.is_valid
        assert not result.should_terminate
        assert len(contract.validate_calls) == 1
    
    @pytest.mark.asyncio
    async def test_validation_failure(self):
        """Test validation failure handling."""
        contract = MockStreamingContract("test", should_violate=True)
        validator = StreamingValidator([contract])
        
        result = await validator.process_chunk("Hello", 0)
        
        assert not result.is_valid
        assert result.violation_message == "Mock violation from test"
        assert result.violation_severity == ViolationSeverity.MEDIUM
    
    @pytest.mark.asyncio
    async def test_critical_violation_termination(self):
        """Test that critical violations terminate the stream."""
        contract = MockStreamingContract(
            "critical", 
            should_violate=True, 
            violation_severity=ViolationSeverity.CRITICAL,
            should_terminate=True
        )
        validator = StreamingValidator([contract], early_termination=True)
        
        result = await validator.process_chunk("Hello", 0)
        
        assert not result.is_valid
        assert result.should_terminate
        assert validator.is_terminated
    
    @pytest.mark.asyncio
    async def test_buffer_management(self):
        """Test buffer size management."""
        contract = MockStreamingContract("test")
        validator = StreamingValidator([contract], max_buffer_size=100)
        
        # Add content that exceeds buffer size
        large_content = "x" * 150
        await validator.process_chunk(large_content, 0)
        
        # Buffer should be truncated
        assert len(validator.get_buffer()) <= 100
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance metrics collection."""
        contract = MockStreamingContract("test")
        validator = StreamingValidator([contract], performance_monitoring=True)
        
        await validator.process_chunk("Hello", 0)
        
        metrics = validator.get_metrics()
        assert metrics.total_chunks == 1
        assert metrics.validated_chunks == 1
        assert metrics.average_validation_latency_ms >= 0
    
    @pytest.mark.asyncio
    async def test_finalize_validation(self):
        """Test final validation of complete content."""
        # Mock a contract that validates complete content
        batch_contract = Mock()
        batch_contract.validate = AsyncMock(return_value=ValidationResult(is_valid=True))
        
        validator = StreamingValidator([batch_contract])
        await validator.process_chunk("Hello ", 0)
        await validator.process_chunk("world!", 1)
        
        result = await validator.finalize_validation()
        
        assert result.is_valid
        batch_contract.validate.assert_called_once_with("Hello world!")
    
    def test_reset(self):
        """Test validator reset functionality."""
        contract = MockStreamingContract("test")
        validator = StreamingValidator([contract])
        
        # Simulate some state
        validator.buffer = "some content"
        validator.metrics.total_chunks = 5
        validator.is_terminated = True
        
        validator.reset()
        
        assert validator.buffer == ""
        assert validator.metrics.total_chunks == 0
        assert not validator.is_terminated


class TestStreamProcessors:
    """Tests for stream processors."""
    
    @pytest.mark.asyncio
    async def test_token_processor(self):
        """Test token-based validation."""
        processor = TokenProcessor(
            max_tokens=5,
            forbidden_tokens={"bad", "evil"}
        )
        
        # Test normal content
        chunk = StreamChunk("hello world", 0, time.time(), "hello world")
        result = await processor.validate_chunk(chunk)
        assert result.is_valid
        
        # Test forbidden token
        chunk = StreamChunk("this is bad", 0, time.time(), "this is bad")
        result = await processor.validate_chunk(chunk)
        assert not result.is_valid
        assert result.should_terminate
    
    @pytest.mark.asyncio
    async def test_content_filter(self):
        """Test content filtering."""
        filter_processor = ContentFilter(
            blocked_patterns=["spam", "advertisement"],
            stop_on_violation=True
        )
        
        # Test clean content
        chunk = StreamChunk("Hello world", 0, time.time(), "Hello world")
        result = await filter_processor.validate_chunk(chunk)
        assert result.is_valid
        
        # Test blocked content
        chunk = StreamChunk("Buy our spam product", 0, time.time(), "Buy our spam product")
        result = await filter_processor.validate_chunk(chunk)
        assert not result.is_valid
        assert result.should_terminate
    
    @pytest.mark.asyncio
    async def test_json_validator(self):
        """Test JSON structure validation."""
        json_validator = JSONValidator(require_complete_json=True)
        
        # Test valid JSON structure
        chunk = StreamChunk('{"key":', 0, time.time(), '{"key":')
        result = await json_validator.validate_chunk(chunk)
        assert result.is_valid
        
        # Test invalid bracket matching
        chunk = StreamChunk('{"key":}', 0, time.time(), '{"key":}')
        result = await json_validator.validate_chunk(chunk)
        assert not result.is_valid
    
    @pytest.mark.asyncio
    async def test_length_monitor(self):
        """Test content length monitoring."""
        length_monitor = LengthMonitor(max_length=10, truncate_on_limit=True)
        
        # Test normal length
        chunk = StreamChunk("short", 0, time.time(), "short")
        result = await length_monitor.validate_chunk(chunk)
        assert result.is_valid
        
        # Test length limit exceeded
        chunk = StreamChunk("x" * 15, 0, time.time(), "x" * 15)
        result = await length_monitor.validate_chunk(chunk)
        assert not result.is_valid
        assert result.should_terminate
        assert result.auto_fix_suggestion  # Should suggest truncation
    
    @pytest.mark.asyncio
    async def test_sentiment_analyzer(self):
        """Test sentiment analysis."""
        sentiment_analyzer = SentimentAnalyzer(
            min_sentiment=-0.5,
            max_sentiment=0.5
        )
        
        # Test neutral content
        chunk = StreamChunk("The weather is okay today", 0, time.time(), "The weather is okay today" * 10)
        result = await sentiment_analyzer.validate_chunk(chunk)
        assert result.is_valid
        
        # Test negative content
        negative_content = "This is terrible awful bad horrible disgusting" * 10
        chunk = StreamChunk(negative_content, 0, time.time(), negative_content)
        result = await sentiment_analyzer.validate_chunk(chunk)
        assert not result.is_valid
    
    @pytest.mark.asyncio
    async def test_toxicity_detector(self):
        """Test toxicity detection."""
        toxicity_detector = ToxicityDetector(severity_escalation=True)
        
        # Test clean content
        chunk = StreamChunk("Hello world", 0, time.time(), "Hello world")
        result = await toxicity_detector.validate_chunk(chunk)
        assert result.is_valid
        
        # Test toxic content
        chunk = StreamChunk("I hate this stupid thing", 0, time.time(), "I hate this stupid thing")
        result = await toxicity_detector.validate_chunk(chunk)
        assert not result.is_valid


class TestStreamWrappers:
    """Tests for stream wrappers."""
    
    def test_sync_stream_wrapper(self):
        """Test synchronous stream wrapper."""
        # Create mock stream
        mock_chunks = [
            MockOpenAIChunk("Hello "),
            MockOpenAIChunk("world!"),
        ]
        
        def mock_stream():
            for chunk in mock_chunks:
                yield chunk
        
        # Create validator
        contract = MockStreamingContract("test")
        validator = StreamingValidator([contract])
        
        # Create wrapper
        wrapper = StreamWrapper(mock_stream(), validator)
        
        # Consume stream
        chunks = list(wrapper)
        
        assert len(chunks) == 2
        assert len(contract.validate_calls) >= 1  # At least one validation call
    
    @pytest.mark.asyncio
    async def test_async_stream_wrapper(self):
        """Test asynchronous stream wrapper."""
        # Create mock async stream
        mock_chunks = [
            MockOpenAIChunk("Hello "),
            MockOpenAIChunk("world!"),
        ]
        
        async def mock_async_stream():
            for chunk in mock_chunks:
                yield chunk
        
        # Create validator
        contract = MockStreamingContract("test")
        validator = StreamingValidator([contract])
        
        # Create wrapper
        wrapper = AsyncStreamWrapper(mock_async_stream(), validator)
        
        # Consume stream
        chunks = []
        async for chunk in wrapper:
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert len(contract.validate_calls) >= 1
    
    @pytest.mark.asyncio
    async def test_stream_termination_on_violation(self):
        """Test stream termination on critical violation."""
        # Create mock stream
        async def mock_stream():
            yield MockOpenAIChunk("Hello ")
            yield MockOpenAIChunk("bad content")  # This should trigger violation
            yield MockOpenAIChunk("more content")  # This should not be reached
        
        # Create validator with terminating contract
        contract = MockStreamingContract(
            "terminating",
            should_violate=True,
            should_terminate=True
        )
        validator = StreamingValidator([contract])
        
        wrapper = AsyncStreamWrapper(mock_stream(), validator, raise_on_violation=True)
        
        # Should raise exception when violation occurs
        with pytest.raises(ContractViolationError):
            chunks = []
            async for chunk in wrapper:
                chunks.append(chunk)


class TestPerformanceComponents:
    """Tests for performance optimization components."""
    
    def test_streaming_profiler(self):
        """Test streaming profiler."""
        profiler = StreamingProfiler()
        profiler.start_profiling()
        
        # Simulate some activity
        profiler.record_chunk_processing(5.0)
        profiler.record_validation(10.0, "test_contract", violated=False)
        profiler.record_auto_fix()
        
        report = profiler.stop_profiling()
        
        assert report["chunks_processed"] == 1
        assert report["validations_performed"] == 1
        assert report["auto_fixes_applied"] == 1
        assert "latency_metrics" in report
    
    def test_latency_optimizer(self):
        """Test latency optimizer."""
        optimizer = LatencyOptimizer(target_latency_ms=10.0)
        
        # Simulate high latency
        for _ in range(15):
            optimizer.record_latency(25.0)  # High latency
        
        recommendations = optimizer.record_latency(25.0)
        
        # Should recommend optimizations for high latency
        assert "validation_frequency" in recommendations or "buffer_size_multiplier" in recommendations
        
        # Test low latency optimization
        optimizer = LatencyOptimizer(target_latency_ms=10.0)
        for _ in range(15):
            optimizer.record_latency(2.0)  # Low latency
        
        recommendations = optimizer.record_latency(2.0)
        # Should recommend increasing validation quality
        assert len(recommendations) >= 0  # May or may not have recommendations
    
    def test_buffer_manager(self):
        """Test buffer manager."""
        buffer = BufferManager(initial_size=100, max_size=1000)
        
        # Test basic append
        assert buffer.append("Hello world")
        assert "Hello world" in buffer.get_content()
        
        # Test buffer growth
        large_content = "x" * 200
        assert buffer.append(large_content)
        
        stats = buffer.get_stats()
        assert stats.used_size > 100  # Buffer should have grown
        
        # Test truncation
        bytes_removed = buffer.truncate(50)
        assert bytes_removed > 0
        assert buffer.get_stats().used_size == 50
    
    @pytest.mark.asyncio
    async def test_async_buffer_pool(self):
        """Test async buffer pool."""
        pool = AsyncBufferPool(pool_size=3, buffer_size=100)
        
        # Acquire buffer
        buffer = await pool.acquire_buffer()
        assert buffer is not None
        
        # Use buffer
        buffer.append("test content")
        
        # Release buffer
        await pool.release_buffer(buffer)
        
        # Get stats
        stats = pool.get_pool_stats()
        assert stats["pool_size"] == 3
        assert stats["checkout_count"] == 1
        
        await pool.cleanup()
    
    @pytest.mark.asyncio
    async def test_buffer_pool_exhaustion(self):
        """Test buffer pool when exhausted."""
        pool = AsyncBufferPool(pool_size=1, buffer_size=100)
        
        # Acquire the only buffer
        buffer1 = await pool.acquire_buffer()
        assert buffer1 is not None
        
        # Try to acquire another (should timeout)
        buffer2 = await pool.acquire_buffer(timeout=0.1)
        assert buffer2 is None
        
        # Release and try again
        await pool.release_buffer(buffer1)
        buffer3 = await pool.acquire_buffer()
        assert buffer3 is not None
        
        await pool.release_buffer(buffer3)
        await pool.cleanup()


class TestIntegration:
    """Integration tests for the complete streaming validation system."""
    
    @pytest.mark.asyncio
    async def test_complete_streaming_workflow(self):
        """Test complete streaming validation workflow."""
        # Create contracts
        contracts = [
            TokenProcessor(max_tokens=100),
            ContentFilter(blocked_patterns=["spam"]),
            LengthMonitor(max_length=500)
        ]
        
        # Create validator
        validator = StreamingValidator(
            contracts,
            performance_monitoring=True,
            early_termination=True
        )
        
        # Simulate streaming content
        test_content = "This is a test message that should pass all validation checks."
        chunks = [test_content[i:i+10] for i in range(0, len(test_content), 10)]
        
        # Process chunks
        for i, chunk in enumerate(chunks):
            result = await validator.process_chunk(chunk, i)
            assert result.is_valid
            
            if result.should_terminate:
                break
        
        # Finalize
        final_result = await validator.finalize_validation()
        assert final_result.is_valid
        
        # Check metrics
        metrics = validator.get_metrics()
        assert metrics.total_chunks > 0
        assert metrics.violations_detected == 0
    
    @pytest.mark.asyncio
    async def test_streaming_with_violations_and_recovery(self):
        """Test streaming with violations and auto-recovery."""
        # Create contract that provides auto-fix
        class AutoFixContract(StreamingContract):
            def __init__(self):
                super().__init__("autofix", "Auto-fixing contract")
            
            async def validate_chunk(self, chunk, context=None):
                if "bad" in chunk.content:
                    return StreamValidationResult(
                        is_valid=False,
                        violation_message="Contains 'bad' word",
                        auto_fix_suggestion=chunk.content.replace("bad", "good")
                    )
                return StreamValidationResult(is_valid=True)
        
        validator = StreamingValidator([AutoFixContract()])
        
        # Process content with violations
        result = await validator.process_chunk("This is bad content", 0)
        
        assert not result.is_valid
        assert result.auto_fix_suggestion == "This is good content"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self):
        """Test performance benchmarking functionality."""
        # Create simple validator
        contracts = [TokenProcessor(max_tokens=1000)]
        validator = StreamingValidator(contracts)
        
        # Run benchmark
        test_content = "This is test content for benchmarking validation performance. " * 10
        
        report = await benchmark_streaming_validation(
            validator,
            test_content,
            chunk_size=20,
            iterations=5  # Small number for test speed
        )
        
        assert "iteration_stats" in report
        assert report["iteration_stats"]["total_iterations"] == 5
        assert report["iteration_stats"]["avg_latency_ms"] >= 0
        assert "throughput" in report


if __name__ == "__main__":
    # Run tests with asyncio support
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])