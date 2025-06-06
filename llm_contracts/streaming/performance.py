"""Performance optimization utilities for streaming validation."""

import asyncio
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, Iterator, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency tracking for performance optimization."""
    measurements: deque = field(default_factory=lambda: deque(maxlen=1000))
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    average: float = 0.0
    peak: float = 0.0
    
    def add_measurement(self, latency_ms: float):
        """Add latency measurement and update statistics."""
        self.measurements.append(latency_ms)
        self.peak = max(self.peak, latency_ms)
        
        if len(self.measurements) >= 10:  # Minimum samples for statistics
            sorted_measurements = sorted(self.measurements)
            n = len(sorted_measurements)
            
            self.average = sum(sorted_measurements) / n
            self.p50 = sorted_measurements[int(n * 0.5)]
            self.p95 = sorted_measurements[int(n * 0.95)]
            self.p99 = sorted_measurements[int(n * 0.99)]


@dataclass 
class BufferStats:
    """Buffer utilization statistics."""
    total_size: int = 0
    used_size: int = 0
    peak_usage: int = 0
    allocations: int = 0
    reallocations: int = 0
    
    @property
    def utilization_percentage(self) -> float:
        """Current buffer utilization as percentage."""
        return (self.used_size / max(self.total_size, 1)) * 100
    
    @property
    def peak_utilization_percentage(self) -> float:
        """Peak buffer utilization as percentage."""
        return (self.peak_usage / max(self.total_size, 1)) * 100


class StreamingProfiler:
    """Comprehensive profiler for streaming validation performance."""
    
    def __init__(self, enable_detailed_tracking: bool = True):
        self.enable_detailed_tracking = enable_detailed_tracking
        self.validation_latency = LatencyMetrics()
        self.chunk_processing_latency = LatencyMetrics()
        self.buffer_stats = BufferStats()
        
        # Performance counters
        self.chunks_processed = 0
        self.validations_performed = 0
        self.violations_detected = 0
        self.auto_fixes_applied = 0
        self.streams_terminated = 0
        
        # Timing data
        self.start_time: Optional[float] = None
        self.contract_timings: Dict[str, LatencyMetrics] = defaultdict(LatencyMetrics)
        
        # Threading for concurrent profiling
        self._profile_lock = threading.Lock()
    
    def start_profiling(self):
        """Start profiling session."""
        self.start_time = time.time()
        logger.info("Streaming validation profiling started")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return comprehensive report."""
        end_time = time.time()
        duration = end_time - (self.start_time or end_time)
        
        report = {
            "session_duration_seconds": duration,
            "chunks_processed": self.chunks_processed,
            "validations_performed": self.validations_performed,
            "violations_detected": self.violations_detected,
            "auto_fixes_applied": self.auto_fixes_applied,
            "streams_terminated": self.streams_terminated,
            "throughput": {
                "chunks_per_second": self.chunks_processed / max(duration, 0.001),
                "validations_per_second": self.validations_performed / max(duration, 0.001),
            },
            "latency_metrics": {
                "validation": {
                    "average_ms": self.validation_latency.average,
                    "p50_ms": self.validation_latency.p50,
                    "p95_ms": self.validation_latency.p95,
                    "p99_ms": self.validation_latency.p99,
                    "peak_ms": self.validation_latency.peak,
                },
                "chunk_processing": {
                    "average_ms": self.chunk_processing_latency.average,
                    "p50_ms": self.chunk_processing_latency.p50,
                    "p95_ms": self.chunk_processing_latency.p95,
                    "p99_ms": self.chunk_processing_latency.p99,
                    "peak_ms": self.chunk_processing_latency.peak,
                }
            },
            "buffer_performance": {
                "total_size": self.buffer_stats.total_size,
                "peak_usage": self.buffer_stats.peak_usage,
                "peak_utilization_percent": self.buffer_stats.peak_utilization_percentage,
                "allocations": self.buffer_stats.allocations,
                "reallocations": self.buffer_stats.reallocations,
            },
            "contract_performance": {
                name: {
                    "average_ms": metrics.average,
                    "p95_ms": metrics.p95,
                    "peak_ms": metrics.peak,
                    "measurements": len(metrics.measurements)
                }
                for name, metrics in self.contract_timings.items()
            }
        }
        
        logger.info(f"Streaming validation profiling completed: {self.chunks_processed} chunks, {self.validations_performed} validations")
        return report
    
    def record_chunk_processing(self, latency_ms: float):
        """Record chunk processing latency."""
        with self._profile_lock:
            self.chunks_processed += 1
            self.chunk_processing_latency.add_measurement(latency_ms)
    
    def record_validation(self, latency_ms: float, contract_name: Optional[str] = None, violated: bool = False):
        """Record validation latency and outcome."""
        with self._profile_lock:
            self.validations_performed += 1
            self.validation_latency.add_measurement(latency_ms)
            
            if violated:
                self.violations_detected += 1
            
            if contract_name and self.enable_detailed_tracking:
                self.contract_timings[contract_name].add_measurement(latency_ms)
    
    def record_auto_fix(self):
        """Record auto-fix application."""
        with self._profile_lock:
            self.auto_fixes_applied += 1
    
    def record_stream_termination(self):
        """Record stream termination due to violation."""
        with self._profile_lock:
            self.streams_terminated += 1
    
    def update_buffer_stats(self, total_size: int, used_size: int):
        """Update buffer utilization statistics."""
        with self._profile_lock:
            self.buffer_stats.total_size = total_size
            self.buffer_stats.used_size = used_size
            self.buffer_stats.peak_usage = max(self.buffer_stats.peak_usage, used_size)


class LatencyOptimizer:
    """Adaptive latency optimizer for streaming validation."""
    
    def __init__(self, target_latency_ms: float = 10.0):
        self.target_latency_ms = target_latency_ms
        self.recent_latencies = deque(maxlen=50)
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Adaptive parameters
        self.validation_frequency = 1.0  # Start at 100% validation
        self.buffer_size_multiplier = 1.0
        self.concurrent_validation_limit = 5
        
        # Performance thresholds
        self.high_latency_threshold = target_latency_ms * 2
        self.low_latency_threshold = target_latency_ms * 0.5
    
    def record_latency(self, latency_ms: float) -> Dict[str, Any]:
        """Record latency and return optimization recommendations."""
        self.recent_latencies.append(latency_ms)
        
        if len(self.recent_latencies) < 10:
            return {}  # Need more data
        
        avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)
        recommendations = {}
        
        # High latency optimization
        if avg_latency > self.high_latency_threshold:
            recommendations.update(self._optimize_for_high_latency(avg_latency))
        
        # Low latency optimization (increase quality)
        elif avg_latency < self.low_latency_threshold:
            recommendations.update(self._optimize_for_low_latency(avg_latency))
        
        if recommendations:
            self.optimization_history.append({
                "timestamp": time.time(),
                "avg_latency": avg_latency,
                "recommendations": recommendations
            })
        
        return recommendations
    
    def _optimize_for_high_latency(self, avg_latency: float) -> Dict[str, Any]:
        """Generate recommendations for reducing high latency."""
        recommendations = {}
        
        # Reduce validation frequency
        if self.validation_frequency > 0.3:
            new_frequency = max(0.3, self.validation_frequency * 0.8)
            recommendations["validation_frequency"] = new_frequency
            self.validation_frequency = new_frequency
        
        # Increase buffer size to reduce validation frequency
        if self.buffer_size_multiplier < 3.0:
            new_multiplier = min(3.0, self.buffer_size_multiplier * 1.2)
            recommendations["buffer_size_multiplier"] = new_multiplier
            self.buffer_size_multiplier = new_multiplier
        
        # Reduce concurrent validations
        if self.concurrent_validation_limit > 2:
            new_limit = max(2, self.concurrent_validation_limit - 1)
            recommendations["concurrent_validation_limit"] = new_limit
            self.concurrent_validation_limit = new_limit
        
        logger.info(f"High latency detected ({avg_latency:.1f}ms), optimizing: {recommendations}")
        return recommendations
    
    def _optimize_for_low_latency(self, avg_latency: float) -> Dict[str, Any]:
        """Generate recommendations for increasing validation quality when latency is low."""
        recommendations = {}
        
        # Increase validation frequency
        if self.validation_frequency < 1.0:
            new_frequency = min(1.0, self.validation_frequency * 1.1)
            recommendations["validation_frequency"] = new_frequency
            self.validation_frequency = new_frequency
        
        # Decrease buffer size for more frequent validation
        if self.buffer_size_multiplier > 0.5:
            new_multiplier = max(0.5, self.buffer_size_multiplier * 0.9)
            recommendations["buffer_size_multiplier"] = new_multiplier
            self.buffer_size_multiplier = new_multiplier
        
        # Increase concurrent validations
        if self.concurrent_validation_limit < 10:
            new_limit = min(10, self.concurrent_validation_limit + 1)
            recommendations["concurrent_validation_limit"] = new_limit
            self.concurrent_validation_limit = new_limit
        
        logger.debug(f"Low latency detected ({avg_latency:.1f}ms), increasing validation quality: {recommendations}")
        return recommendations
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current optimization settings."""
        return {
            "validation_frequency": self.validation_frequency,
            "buffer_size_multiplier": self.buffer_size_multiplier,
            "concurrent_validation_limit": self.concurrent_validation_limit,
            "target_latency_ms": self.target_latency_ms,
            "recent_avg_latency": sum(self.recent_latencies) / len(self.recent_latencies) if self.recent_latencies else 0
        }


class BufferManager:
    """High-performance buffer manager for streaming content."""
    
    def __init__(self, 
                 initial_size: int = 8192,
                 max_size: int = 1024 * 1024,  # 1MB max
                 growth_factor: float = 1.5):
        self.initial_size = initial_size
        self.max_size = max_size
        self.growth_factor = growth_factor
        
        self._buffer = bytearray(initial_size)
        self._position = 0
        self._stats = BufferStats(total_size=initial_size)
        
        # Thread safety
        self._lock = threading.Lock()
    
    def append(self, content: str) -> bool:
        """Append content to buffer. Returns True if successful, False if buffer full."""
        content_bytes = content.encode('utf-8')
        content_length = len(content_bytes)
        
        with self._lock:
            # Check if we need to grow the buffer
            if self._position + content_length > len(self._buffer):
                if not self._grow_buffer(content_length):
                    return False  # Buffer at max size, cannot grow
            
            # Append content
            self._buffer[self._position:self._position + content_length] = content_bytes
            self._position += content_length
            
            # Update stats
            self._stats.used_size = self._position
            self._stats.peak_usage = max(self._stats.peak_usage, self._position)
            
            return True
    
    def get_content(self, start: int = 0, end: Optional[int] = None) -> str:
        """Get content from buffer."""
        with self._lock:
            end = end or self._position
            return self._buffer[start:end].decode('utf-8', errors='ignore')
    
    def get_recent_content(self, max_length: int) -> str:
        """Get recent content up to max_length characters."""
        with self._lock:
            start = max(0, self._position - max_length)
            return self.get_content(start)
    
    def truncate(self, keep_recent: int) -> int:
        """Truncate buffer keeping only recent content. Returns bytes removed."""
        with self._lock:
            if self._position <= keep_recent:
                return 0
            
            bytes_to_remove = self._position - keep_recent
            
            # Move recent content to beginning
            self._buffer[:keep_recent] = self._buffer[bytes_to_remove:self._position]
            self._position = keep_recent
            
            self._stats.used_size = self._position
            return bytes_to_remove
    
    def clear(self):
        """Clear buffer content."""
        with self._lock:
            self._position = 0
            self._stats.used_size = 0
    
    def _grow_buffer(self, required_additional: int) -> bool:
        """Grow buffer to accommodate additional content."""
        current_size = len(self._buffer)
        required_size = self._position + required_additional
        
        if required_size > self.max_size:
            return False
        
        # Calculate new size
        new_size = current_size
        while new_size < required_size:
            new_size = int(new_size * self.growth_factor)
        
        new_size = min(new_size, self.max_size)
        
        # Grow buffer
        new_buffer = bytearray(new_size)
        new_buffer[:self._position] = self._buffer[:self._position]
        self._buffer = new_buffer
        
        # Update stats
        self._stats.total_size = new_size
        self._stats.reallocations += 1
        
        logger.debug(f"Buffer grown from {current_size} to {new_size} bytes")
        return True
    
    def get_stats(self) -> BufferStats:
        """Get buffer statistics."""
        with self._lock:
            return BufferStats(
                total_size=self._stats.total_size,
                used_size=self._stats.used_size,
                peak_usage=self._stats.peak_usage,
                allocations=self._stats.allocations,
                reallocations=self._stats.reallocations
            )


class AsyncBufferPool:
    """Pool of buffers for concurrent streaming validation."""
    
    def __init__(self, 
                 pool_size: int = 10,
                 buffer_size: int = 8192):
        self.pool_size = pool_size
        self.buffer_size = buffer_size
        
        # Create buffer pool
        self._available_buffers: asyncio.Queue = asyncio.Queue()
        self._all_buffers: List[BufferManager] = []
        
        # Initialize pool
        for _ in range(pool_size):
            buffer_manager = BufferManager(initial_size=buffer_size)
            self._all_buffers.append(buffer_manager)
            self._available_buffers.put_nowait(buffer_manager)
        
        # Statistics
        self._checkout_count = 0
        self._wait_count = 0
        self._lock = asyncio.Lock()
    
    async def acquire_buffer(self, timeout: float = 1.0) -> Optional[BufferManager]:
        """Acquire a buffer from the pool."""
        self._checkout_count += 1
        
        try:
            buffer = await asyncio.wait_for(
                self._available_buffers.get(), 
                timeout=timeout
            )
            buffer.clear()  # Reset buffer state
            return buffer
        except asyncio.TimeoutError:
            self._wait_count += 1
            logger.warning("Buffer pool exhausted, validation may be delayed")
            return None
    
    async def release_buffer(self, buffer: BufferManager):
        """Return a buffer to the pool."""
        buffer.clear()
        await self._available_buffers.put(buffer)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get buffer pool statistics."""
        return {
            "pool_size": self.pool_size,
            "available_buffers": self._available_buffers.qsize(),
            "checkout_count": self._checkout_count,
            "wait_count": self._wait_count,
            "wait_rate": self._wait_count / max(self._checkout_count, 1),
            "buffer_stats": [buf.get_stats().__dict__ for buf in self._all_buffers]
        }
    
    async def cleanup(self):
        """Clean up the buffer pool."""
        # Clear all buffers
        while not self._available_buffers.empty():
            buffer = await self._available_buffers.get()
            buffer.clear()


# Utility functions for performance optimization

def create_optimized_validator(
    contracts: List[Any],
    target_latency_ms: float = 10.0,
    enable_profiling: bool = True
) -> Tuple[Any, StreamingProfiler, LatencyOptimizer]:
    """Create a performance-optimized streaming validator."""
    from .stream_validator import StreamingValidator, ChunkValidationMode
    
    # Create profiler and optimizer
    profiler = StreamingProfiler(enable_detailed_tracking=enable_profiling)
    optimizer = LatencyOptimizer(target_latency_ms=target_latency_ms)
    
    # Create validator with adaptive settings
    validator = StreamingValidator(
        contracts=contracts,
        validation_mode=ChunkValidationMode.ADAPTIVE,
        performance_monitoring=enable_profiling,
        early_termination=True
    )
    
    if enable_profiling:
        profiler.start_profiling()
    
    return validator, profiler, optimizer


async def benchmark_streaming_validation(
    validator: Any,
    test_content: str,
    chunk_size: int = 50,
    iterations: int = 100
) -> Dict[str, Any]:
    """Benchmark streaming validation performance."""
    profiler = StreamingProfiler()
    profiler.start_profiling()
    
    latencies = []
    
    for i in range(iterations):
        # Reset validator for each iteration
        validator.reset()
        
        # Process content in chunks
        start_time = time.time()
        
        for j in range(0, len(test_content), chunk_size):
            chunk = test_content[j:j + chunk_size]
            chunk_start = time.time()
            
            try:
                result = await validator.process_chunk(chunk, j // chunk_size)
                
                chunk_latency = (time.time() - chunk_start) * 1000
                profiler.record_chunk_processing(chunk_latency)
                
                if result and hasattr(result, 'should_terminate') and result.should_terminate:
                    break
                    
            except Exception as e:
                logger.warning(f"Validation error during benchmark: {e}")
        
        # Finalize validation
        try:
            await validator.finalize_validation()
        except Exception as e:
            logger.warning(f"Finalization error during benchmark: {e}")
        
        iteration_latency = (time.time() - start_time) * 1000
        latencies.append(iteration_latency)
    
    # Generate benchmark report
    report = profiler.stop_profiling()
    
    # Add iteration statistics
    latencies.sort()
    report["iteration_stats"] = {
        "total_iterations": iterations,
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": latencies[int(len(latencies) * 0.5)],
        "p95_latency_ms": latencies[int(len(latencies) * 0.95)],
        "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
    }
    
    return report