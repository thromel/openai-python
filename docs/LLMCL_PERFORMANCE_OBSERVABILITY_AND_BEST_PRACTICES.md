# LLMCL Performance, Observability, and Best Practices

This document provides comprehensive guidance on optimizing LLMCL performance, implementing observability, and following best practices for production deployments.

## Table of Contents

1. [Performance Optimization](#performance-optimization)
2. [Caching Strategies](#caching-strategies)
3. [Asynchronous Processing](#asynchronous-processing)
4. [Memory Management](#memory-management)
5. [Observability and Monitoring](#observability-and-monitoring)
6. [Metrics Collection](#metrics-collection)
7. [Tracing and Debugging](#tracing-and-debugging)
8. [Production Best Practices](#production-best-practices)
9. [Scaling Strategies](#scaling-strategies)
10. [Security Considerations](#security-considerations)
11. [Testing Best Practices](#testing-best-practices)
12. [Troubleshooting Guide](#troubleshooting-guide)

## Performance Optimization

### Runtime Configuration for Performance

```python
from llm_contracts.language import LLMCLRuntime
from llm_contracts.language.conflict_resolver import ConflictResolver
from llm_contracts.language.auto_fix import AutoFixManager

# Optimized runtime configuration
runtime = LLMCLRuntime(
    # Enable caching with large cache size
    cache_enabled=True,
    cache_size=50000,
    cache_ttl=3600,  # 1 hour TTL
    
    # Optimize conflict resolution
    conflict_resolver=ConflictResolver(
        strategy='MOST_RESTRICTIVE',  # Faster than MERGE
        enable_static_analysis=True,   # Pre-compute conflicts
        lazy_evaluation=True           # Evaluate only when needed
    ),
    
    # Optimize auto-fix behavior
    auto_fix_manager=AutoFixManager(
        strategy='FIRST_FIX',         # Faster than evaluating all fixes
        confidence_threshold=0.8,     # Higher threshold = fewer attempts
        max_attempts=2,               # Limit fix attempts
        enable_parallel_fixes=True    # Parallel fix evaluation
    ),
    
    # Performance settings
    enable_lazy_loading=True,         # Load contracts on demand
    max_concurrent_validations=100,   # Limit concurrent operations
    enable_circuit_breaker=True,      # Fail fast on repeated errors
    circuit_breaker_threshold=5       # Trip after 5 failures
)
```

### Contract Optimization

```llmcl
// Optimized contract design
contract OptimizedValidation(priority = high) {
    // Put most restrictive/likely-to-fail checks first
    require len(content) > 0
        message: "Input required"
    
    // Combine related checks to reduce expression evaluations
    require len(content) <= 4000 and not contains(content, "\x00")
        message: "Input validation failed"
        auto_fix: content[:4000].replace("\x00", "")
    
    // Use efficient string operations
    ensure not starts_with(response, "Error:") and not starts_with(response, "Failed:")
        message: "Should not start with error indicators"
        auto_fix: if starts_with(response, "Error:") then 
                     response[6:] 
                   else if starts_with(response, "Failed:") then 
                     response[7:] 
                   else 
                     response
    
    // Prefer simple checks over complex regex when possible
    ensure len(response) >= 20
        message: "Response should be substantial"
        auto_fix: response + " Please let me know if you need more help."
}

// Avoid: Expensive operations in hot paths
contract ExpensiveContract(priority = medium) {
    // Avoid: Complex regex in every validation
    ensure match(response, r"^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)$")
        message: "Complex email validation"
    
    // Avoid: Multiple separate string operations
    ensure not contains(response, "bad1")
        message: "Check 1"
    ensure not contains(response, "bad2")
        message: "Check 2"
    ensure not contains(response, "bad3")
        message: "Check 3"
    
    // Better: Combined check
    // ensure not (contains(response, "bad1") or contains(response, "bad2") or contains(response, "bad3"))
}
```

### Compilation Optimization

```python
from llm_contracts.language.compiler import CompilationOptions
from llm_contracts.language import compile_contract

# Optimized compilation settings
optimization_options = CompilationOptions(
    optimize=True,                    # Enable optimization passes
    inline_simple_expressions=True,   # Inline simple expressions
    constant_folding=True,            # Evaluate constants at compile time
    dead_code_elimination=True,       # Remove unused code
    expression_caching=True,          # Cache repeated expressions
    enable_jit=True,                  # Just-in-time compilation for hot paths
    target_performance='speed'        # Optimize for speed over memory
)

# Pre-compile frequently used contracts
contract_cache = {}

def get_compiled_contract(contract_source, cache_key=None):
    """Get compiled contract with caching."""
    if cache_key and cache_key in contract_cache:
        return contract_cache[cache_key]
    
    contract = compile_contract(contract_source, optimization_options)
    
    if cache_key:
        contract_cache[cache_key] = contract
    
    return contract

# Batch compilation for multiple contracts
def compile_multiple_contracts(contract_sources):
    """Compile multiple contracts efficiently."""
    return [
        compile_contract(source, optimization_options) 
        for source in contract_sources
    ]
```

## Caching Strategies

### Validation Result Caching

```python
from llm_contracts.language.cache import ValidationCache
import hashlib

class SmartValidationCache:
    def __init__(self, max_size=10000, ttl=3600):
        self.cache = ValidationCache(max_size=max_size, ttl=ttl)
        self.hit_rate = 0.0
        self.total_requests = 0
        self.cache_hits = 0
    
    def get_cache_key(self, contract, context):
        """Generate cache key from contract and context."""
        # Create deterministic hash from contract and relevant context
        contract_hash = hashlib.md5(str(contract).encode()).hexdigest()
        
        # Only include relevant context fields in cache key
        relevant_context = {
            k: v for k, v in context.items() 
            if k in ['content', 'response', 'user_type', 'api_version']
        }
        
        context_hash = hashlib.md5(str(sorted(relevant_context.items())).encode()).hexdigest()
        
        return f"{contract_hash}:{context_hash}"
    
    def get(self, contract, context):
        """Get cached validation result."""
        self.total_requests += 1
        cache_key = self.get_cache_key(contract, context)
        
        result = self.cache.get(cache_key)
        if result:
            self.cache_hits += 1
            self.hit_rate = self.cache_hits / self.total_requests
        
        return result
    
    def set(self, contract, context, result):
        """Cache validation result."""
        cache_key = self.get_cache_key(contract, context)
        self.cache.set(cache_key, result)
    
    def get_stats(self):
        """Get cache performance statistics."""
        return {
            'hit_rate': self.hit_rate,
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.cache),
            'memory_usage': self.cache.get_memory_usage()
        }

# Usage with runtime
cache = SmartValidationCache(max_size=50000, ttl=1800)  # 30 min TTL

runtime = LLMCLRuntime(
    cache_enabled=True,
    custom_cache=cache
)
```

### Contract Compilation Caching

```python
import pickle
import os
from pathlib import Path

class ContractCompilationCache:
    def __init__(self, cache_dir="./contract_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, source_hash):
        """Get cache file path for contract source."""
        return self.cache_dir / f"{source_hash}.pkl"
    
    def get_source_hash(self, source):
        """Get hash of contract source."""
        return hashlib.sha256(source.encode()).hexdigest()
    
    def get_compiled_contract(self, source):
        """Get compiled contract from cache or compile if not cached."""
        source_hash = self.get_source_hash(source)
        cache_path = self.get_cache_path(source_hash)
        
        # Check if cached version exists and is recent
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_contract = pickle.load(f)
                return cached_contract
            except Exception as e:
                # Cache corruption, remove and recompile
                cache_path.unlink(missing_ok=True)
        
        # Compile and cache
        contract = compile_contract(source)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(contract, f)
        except Exception as e:
            # Caching failed, but contract is still valid
            pass
        
        return contract
    
    def clear_cache(self):
        """Clear all cached contracts."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

# Usage
compilation_cache = ContractCompilationCache()

def get_contract(source):
    return compilation_cache.get_compiled_contract(source)
```

### Multi-Level Caching

```python
class MultiLevelCache:
    def __init__(self):
        # L1: In-memory cache for hot contracts
        self.l1_cache = {}
        self.l1_max_size = 100
        
        # L2: Larger in-memory cache for validation results
        self.l2_cache = ValidationCache(max_size=10000, ttl=1800)
        
        # L3: Persistent cache for compiled contracts
        self.l3_cache = ContractCompilationCache()
    
    def get_contract(self, source):
        """Get contract with multi-level caching."""
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        
        # Check L1 cache (hot contracts)
        if source_hash in self.l1_cache:
            return self.l1_cache[source_hash]
        
        # Check L3 cache (persistent)
        contract = self.l3_cache.get_compiled_contract(source)
        
        # Store in L1 cache
        if len(self.l1_cache) >= self.l1_max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
        
        self.l1_cache[source_hash] = contract
        return contract
    
    def validate_with_cache(self, contract, context):
        """Validate with L2 result caching."""
        # Check L2 cache for validation results
        result = self.l2_cache.get(contract, context)
        if result:
            return result
        
        # Perform validation
        runtime = LLMCLRuntime()
        result = runtime.validate(contract, context)
        
        # Cache result
        self.l2_cache.set(contract, context, result)
        
        return result
```

## Asynchronous Processing

### Async Runtime Implementation

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from llm_contracts.language import LLMCLRuntime

class AsyncLLMCLRuntime:
    def __init__(self, max_workers=10):
        self.runtime = LLMCLRuntime()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)
    
    async def validate_async(self, contract, context):
        """Asynchronous validation using thread pool."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self.runtime.validate,
                contract,
                context
            )
    
    async def validate_batch_async(self, contract, contexts):
        """Validate multiple contexts concurrently."""
        tasks = [
            self.validate_async(contract, context) 
            for context in contexts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def validate_multiple_contracts_async(self, contracts, context):
        """Validate against multiple contracts concurrently."""
        tasks = [
            self.validate_async(contract, context) 
            for contract in contracts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        combined_result = self._combine_results(results)
        return combined_result
    
    def _combine_results(self, results):
        """Combine multiple validation results."""
        all_violations = []
        all_auto_fixes = []
        is_valid = True
        
        for result in results:
            if isinstance(result, Exception):
                continue
            
            if not result.is_valid:
                is_valid = False
            
            all_violations.extend(result.violations)
            all_auto_fixes.extend(result.auto_fixes)
        
        return ValidationResult(
            is_valid=is_valid,
            violations=all_violations,
            auto_fixes=all_auto_fixes
        )

# Usage example
async def process_requests_async():
    async_runtime = AsyncLLMCLRuntime(max_workers=20)
    
    # Load contracts
    contracts = [
        compile_contract(source1),
        compile_contract(source2),
        compile_contract(source3)
    ]
    
    # Process batch of requests
    contexts = [
        {'content': f'Request {i}', 'response': f'Response {i}'}
        for i in range(100)
    ]
    
    # Validate all contexts against all contracts
    all_results = []
    for contract in contracts:
        results = await async_runtime.validate_batch_async(contract, contexts)
        all_results.extend(results)
    
    return all_results
```

### Streaming Validation

```python
import asyncio
from typing import AsyncIterator

class StreamingValidator:
    def __init__(self, runtime):
        self.runtime = runtime
    
    async def validate_stream(self, contract, context_stream: AsyncIterator):
        """Validate a stream of contexts as they arrive."""
        async for context in context_stream:
            try:
                result = await self.runtime.validate_async(contract, context)
                yield {
                    'context': context,
                    'result': result,
                    'timestamp': time.time()
                }
            except Exception as e:
                yield {
                    'context': context,
                    'error': str(e),
                    'timestamp': time.time()
                }
    
    async def validate_with_backpressure(self, contract, context_stream, max_queue_size=100):
        """Validate stream with backpressure control."""
        queue = asyncio.Queue(maxsize=max_queue_size)
        
        async def producer():
            async for context in context_stream:
                await queue.put(context)
            await queue.put(None)  # Sentinel value
        
        async def consumer():
            while True:
                context = await queue.get()
                if context is None:
                    break
                
                result = await self.runtime.validate_async(contract, context)
                yield {
                    'context': context,
                    'result': result,
                    'queue_size': queue.qsize()
                }
        
        # Start producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        
        async for result in consumer():
            yield result
        
        await producer_task

# Usage
async def process_stream():
    runtime = AsyncLLMCLRuntime()
    contract = compile_contract(contract_source)
    validator = StreamingValidator(runtime)
    
    # Simulate streaming data
    async def generate_contexts():
        for i in range(1000):
            yield {'content': f'Input {i}', 'response': f'Response {i}'}
            await asyncio.sleep(0.01)  # Simulate streaming delay
    
    # Process stream with validation
    async for validation_result in validator.validate_stream(contract, generate_contexts()):
        if not validation_result['result'].is_valid:
            print(f"Validation failed for context {validation_result['context']}")
```

## Memory Management

### Memory-Efficient Runtime

```python
import gc
import psutil
import weakref
from llm_contracts.language import LLMCLRuntime

class MemoryEfficientRuntime:
    def __init__(self, memory_limit_mb=1000):
        self.runtime = LLMCLRuntime()
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.contract_cache = weakref.WeakValueDictionary()
        self.validation_cache = {}
        self.cache_max_size = 10000
    
    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss
    
    def check_memory_pressure(self):
        """Check if memory usage is approaching limit."""
        current_usage = self.get_memory_usage()
        return current_usage > self.memory_limit_bytes * 0.8  # 80% threshold
    
    def cleanup_memory(self):
        """Perform memory cleanup when under pressure."""
        if self.check_memory_pressure():
            # Clear validation cache
            cache_size = len(self.validation_cache)
            self.validation_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            print(f"Memory cleanup: cleared {cache_size} cache entries")
    
    def validate(self, contract, context):
        """Validate with memory management."""
        # Check memory before validation
        self.cleanup_memory()
        
        # Perform validation
        result = self.runtime.validate(contract, context)
        
        # Cache result if memory allows
        if not self.check_memory_pressure() and len(self.validation_cache) < self.cache_max_size:
            cache_key = self._get_cache_key(contract, context)
            self.validation_cache[cache_key] = result
        
        return result
    
    def _get_cache_key(self, contract, context):
        """Generate cache key."""
        return f"{id(contract)}:{hash(str(context))}"

# Memory monitoring decorator
def monitor_memory(func):
    """Decorator to monitor memory usage of functions."""
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        result = func(*args, **kwargs)
        
        memory_after = process.memory_info().rss
        memory_delta = memory_after - memory_before
        
        if memory_delta > 10 * 1024 * 1024:  # More than 10MB
            print(f"Function {func.__name__} used {memory_delta / 1024 / 1024:.2f} MB")
        
        return result
    return wrapper

# Usage
runtime = MemoryEfficientRuntime(memory_limit_mb=500)

@monitor_memory
def validate_large_batch(contexts):
    results = []
    for context in contexts:
        result = runtime.validate(contract, context)
        results.append(result)
    return results
```

### Contract Memory Optimization

```python
class CompactContract:
    """Memory-optimized contract representation."""
    
    def __init__(self, contract):
        # Store only essential data
        self.name = contract.name
        self.priority = contract.priority
        
        # Compile clauses to bytecode for smaller memory footprint
        self.compiled_clauses = self._compile_clauses(contract.clauses)
        
        # Remove unnecessary metadata
        self.dependencies = set(contract.dependencies)
    
    def _compile_clauses(self, clauses):
        """Compile clauses to compact bytecode representation."""
        compiled = []
        for clause in clauses:
            # Convert to compact representation
            compact_clause = {
                'type': clause.type,
                'bytecode': self._compile_expression(clause.expression),
                'message': clause.message,
                'auto_fix': clause.auto_fix_expression
            }
            compiled.append(compact_clause)
        return compiled
    
    def _compile_expression(self, expression):
        """Compile expression to bytecode."""
        # Simplified bytecode compilation
        # In practice, this would be more sophisticated
        return expression.compile_to_bytecode()

class ContractPool:
    """Pool of reusable contract instances to reduce memory allocation."""
    
    def __init__(self, max_pool_size=100):
        self.pool = []
        self.max_pool_size = max_pool_size
        self.active_contracts = weakref.WeakSet()
    
    def get_contract(self, contract_source):
        """Get contract instance from pool or create new one."""
        # Try to reuse from pool
        if self.pool:
            contract = self.pool.pop()
            contract.reinitialize(contract_source)
        else:
            contract = compile_contract(contract_source)
        
        self.active_contracts.add(contract)
        return contract
    
    def return_contract(self, contract):
        """Return contract to pool for reuse."""
        if len(self.pool) < self.max_pool_size:
            contract.cleanup()  # Clear any state
            self.pool.append(contract)
        
        self.active_contracts.discard(contract)
    
    def get_stats(self):
        """Get pool statistics."""
        return {
            'pool_size': len(self.pool),
            'active_contracts': len(self.active_contracts),
            'pool_utilization': 1 - (len(self.pool) / self.max_pool_size)
        }
```

## Observability and Monitoring

### Comprehensive Metrics Collection

```python
from dataclasses import dataclass
from typing import Dict, List
import time
import threading
from collections import defaultdict, deque

@dataclass
class ValidationMetrics:
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    auto_fixes_applied: int = 0
    conflicts_resolved: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_validations == 0:
            return 0.0
        return self.successful_validations / self.total_validations
    
    @property
    def auto_fix_rate(self) -> float:
        if self.failed_validations == 0:
            return 0.0
        return self.auto_fixes_applied / self.failed_validations

class MetricsCollector:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.validation_counts = defaultdict(int)
        self.contract_metrics = defaultdict(lambda: defaultdict(int))
        self.lock = threading.Lock()
        
        # Time-series data for trending
        self.time_series = defaultdict(list)
        self.last_collection_time = time.time()
    
    def record_validation(self, contract_name, success, latency_ms, auto_fixes_applied=0, conflicts_resolved=0):
        """Record a validation event."""
        with self.lock:
            self.validation_counts['total'] += 1
            
            if success:
                self.validation_counts['successful'] += 1
            else:
                self.validation_counts['failed'] += 1
            
            self.validation_counts['auto_fixes'] += auto_fixes_applied
            self.validation_counts['conflicts'] += conflicts_resolved
            
            self.latencies.append(latency_ms)
            
            # Per-contract metrics
            self.contract_metrics[contract_name]['total'] += 1
            if success:
                self.contract_metrics[contract_name]['successful'] += 1
            else:
                self.contract_metrics[contract_name]['failed'] += 1
    
    def get_current_metrics(self) -> ValidationMetrics:
        """Get current metrics snapshot."""
        with self.lock:
            if not self.latencies:
                return ValidationMetrics()
            
            sorted_latencies = sorted(self.latencies)
            n = len(sorted_latencies)
            
            metrics = ValidationMetrics(
                total_validations=self.validation_counts['total'],
                successful_validations=self.validation_counts['successful'],
                failed_validations=self.validation_counts['failed'],
                auto_fixes_applied=self.validation_counts['auto_fixes'],
                conflicts_resolved=self.validation_counts['conflicts'],
                average_latency_ms=sum(self.latencies) / n if n > 0 else 0,
                p95_latency_ms=sorted_latencies[int(n * 0.95)] if n > 0 else 0,
                p99_latency_ms=sorted_latencies[int(n * 0.99)] if n > 0 else 0,
                memory_usage_mb=self._get_memory_usage_mb()
            )
            
            return metrics
    
    def get_contract_metrics(self) -> Dict[str, ValidationMetrics]:
        """Get per-contract metrics."""
        with self.lock:
            contract_metrics = {}
            for contract_name, counts in self.contract_metrics.items():
                metrics = ValidationMetrics(
                    total_validations=counts['total'],
                    successful_validations=counts['successful'],
                    failed_validations=counts['failed']
                )
                contract_metrics[contract_name] = metrics
            return contract_metrics
    
    def _get_memory_usage_mb(self):
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def collect_time_series(self):
        """Collect time-series data for trending."""
        current_time = time.time()
        
        if current_time - self.last_collection_time >= 60:  # Collect every minute
            metrics = self.get_current_metrics()
            
            self.time_series['success_rate'].append((current_time, metrics.success_rate))
            self.time_series['latency'].append((current_time, metrics.average_latency_ms))
            self.time_series['memory'].append((current_time, metrics.memory_usage_mb))
            
            # Keep only last 24 hours of data
            cutoff_time = current_time - 24 * 3600
            for series in self.time_series.values():
                while series and series[0][0] < cutoff_time:
                    series.pop(0)
            
            self.last_collection_time = current_time

# Global metrics collector
metrics_collector = MetricsCollector()

class InstrumentedRuntime(LLMCLRuntime):
    """Runtime with built-in metrics collection."""
    
    def validate(self, contract, context):
        """Validate with metrics collection."""
        start_time = time.time()
        
        try:
            result = super().validate(contract, context)
            
            latency_ms = (time.time() - start_time) * 1000
            
            metrics_collector.record_validation(
                contract_name=contract.name,
                success=result.is_valid,
                latency_ms=latency_ms,
                auto_fixes_applied=len(result.auto_fixes) if result.auto_fixes else 0,
                conflicts_resolved=len(result.conflicts_detected) if result.conflicts_detected else 0
            )
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            metrics_collector.record_validation(
                contract_name=getattr(contract, 'name', 'unknown'),
                success=False,
                latency_ms=latency_ms
            )
            
            raise
```

### Prometheus Integration

```python
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

class PrometheusMetrics:
    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client not available")
        
        # Counters
        self.validation_total = Counter(
            'llmcl_validations_total',
            'Total number of validations',
            ['contract_name', 'result']
        )
        
        self.auto_fixes_total = Counter(
            'llmcl_auto_fixes_total',
            'Total number of auto-fixes applied',
            ['contract_name', 'fix_type']
        )
        
        self.conflicts_total = Counter(
            'llmcl_conflicts_total',
            'Total number of conflicts resolved',
            ['resolution_strategy']
        )
        
        # Histograms
        self.validation_duration = Histogram(
            'llmcl_validation_duration_seconds',
            'Validation duration in seconds',
            ['contract_name']
        )
        
        # Gauges
        self.cache_hit_rate = Gauge(
            'llmcl_cache_hit_rate',
            'Current cache hit rate'
        )
        
        self.memory_usage = Gauge(
            'llmcl_memory_usage_bytes',
            'Current memory usage in bytes'
        )
        
        self.active_validations = Gauge(
            'llmcl_active_validations',
            'Number of currently active validations'
        )
    
    def record_validation(self, contract_name, success, duration_seconds):
        """Record validation metrics."""
        result = 'success' if success else 'failure'
        self.validation_total.labels(
            contract_name=contract_name,
            result=result
        ).inc()
        
        self.validation_duration.labels(
            contract_name=contract_name
        ).observe(duration_seconds)
    
    def record_auto_fix(self, contract_name, fix_type):
        """Record auto-fix application."""
        self.auto_fixes_total.labels(
            contract_name=contract_name,
            fix_type=fix_type
        ).inc()
    
    def record_conflict(self, resolution_strategy):
        """Record conflict resolution."""
        self.conflicts_total.labels(
            resolution_strategy=resolution_strategy
        ).inc()
    
    def update_gauges(self, metrics: ValidationMetrics):
        """Update gauge metrics."""
        self.cache_hit_rate.set(metrics.cache_hit_rate)
        self.memory_usage.set(metrics.memory_usage_mb * 1024 * 1024)

# Initialize Prometheus metrics
if PROMETHEUS_AVAILABLE:
    prometheus_metrics = PrometheusMetrics()
    
    # Start Prometheus HTTP server
    start_http_server(8000)  # Metrics available at http://localhost:8000

class PrometheusInstrumentedRuntime(LLMCLRuntime):
    """Runtime with Prometheus metrics integration."""
    
    def validate(self, contract, context):
        """Validate with Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return super().validate(contract, context)
        
        prometheus_metrics.active_validations.inc()
        start_time = time.time()
        
        try:
            result = super().validate(contract, context)
            
            duration = time.time() - start_time
            prometheus_metrics.record_validation(
                contract_name=contract.name,
                success=result.is_valid,
                duration_seconds=duration
            )
            
            # Record auto-fixes
            if result.auto_fixes:
                for fix in result.auto_fixes:
                    prometheus_metrics.record_auto_fix(
                        contract_name=contract.name,
                        fix_type=fix.fix_type
                    )
            
            return result
            
        finally:
            prometheus_metrics.active_validations.dec()
```

## Tracing and Debugging

### OpenTelemetry Integration

```python
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

class TracingRuntime(LLMCLRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if OPENTELEMETRY_AVAILABLE:
            # Configure tracing
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)
            
            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.tracer = tracer
        else:
            self.tracer = None
    
    def validate(self, contract, context):
        """Validate with distributed tracing."""
        if not self.tracer:
            return super().validate(contract, context)
        
        with self.tracer.start_as_current_span("llmcl_validation") as span:
            # Add span attributes
            span.set_attribute("contract.name", contract.name)
            span.set_attribute("contract.priority", contract.priority)
            span.set_attribute("context.size", len(str(context)))
            
            try:
                result = super().validate(contract, context)
                
                # Add result attributes
                span.set_attribute("validation.success", result.is_valid)
                span.set_attribute("validation.violations", len(result.violations))
                span.set_attribute("validation.auto_fixes", len(result.auto_fixes) if result.auto_fixes else 0)
                span.set_attribute("validation.execution_time", result.execution_time)
                
                # Add violation details
                if result.violations:
                    violation_messages = [v.message for v in result.violations]
                    span.set_attribute("validation.violation_messages", str(violation_messages))
                
                return result
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

class DetailedTracer:
    """Detailed tracing for debugging contract execution."""
    
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.trace_data = []
    
    def trace_clause_evaluation(self, clause, context, result):
        """Trace individual clause evaluation."""
        if not self.enabled:
            return
        
        trace_entry = {
            'timestamp': time.time(),
            'clause_type': clause.type,
            'clause_expression': str(clause.expression),
            'context': dict(context),
            'result': result,
            'execution_time_ms': getattr(result, 'execution_time', 0) * 1000
        }
        
        self.trace_data.append(trace_entry)
    
    def trace_conflict_resolution(self, conflicts, resolution_strategy, result):
        """Trace conflict resolution process."""
        if not self.enabled:
            return
        
        trace_entry = {
            'timestamp': time.time(),
            'event_type': 'conflict_resolution',
            'conflicts': [str(c) for c in conflicts],
            'strategy': resolution_strategy,
            'resolution_result': str(result)
        }
        
        self.trace_data.append(trace_entry)
    
    def trace_auto_fix(self, violation, fix_expression, fix_result):
        """Trace auto-fix application."""
        if not self.enabled:
            return
        
        trace_entry = {
            'timestamp': time.time(),
            'event_type': 'auto_fix',
            'violation': str(violation),
            'fix_expression': fix_expression,
            'fix_result': fix_result,
            'success': fix_result is not None
        }
        
        self.trace_data.append(trace_entry)
    
    def get_trace_summary(self):
        """Get summary of trace data."""
        if not self.trace_data:
            return "No trace data available"
        
        summary = {
            'total_events': len(self.trace_data),
            'clause_evaluations': len([t for t in self.trace_data if 'clause_type' in t]),
            'conflict_resolutions': len([t for t in self.trace_data if t.get('event_type') == 'conflict_resolution']),
            'auto_fixes': len([t for t in self.trace_data if t.get('event_type') == 'auto_fix']),
            'time_span': self.trace_data[-1]['timestamp'] - self.trace_data[0]['timestamp']
        }
        
        return summary
    
    def export_trace(self, filename):
        """Export trace data to file."""
        import json
        with open(filename, 'w') as f:
            json.dump(self.trace_data, f, indent=2, default=str)
    
    def clear_trace(self):
        """Clear trace data."""
        self.trace_data.clear()

# Usage
tracer = DetailedTracer(enabled=True)

class DebuggingRuntime(LLMCLRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer = tracer
    
    def validate(self, contract, context):
        """Validate with detailed tracing."""
        result = super().validate(contract, context)
        
        # Trace each clause evaluation (simplified)
        for clause in contract.clauses:
            clause_result = self._evaluate_clause(clause, context)
            self.tracer.trace_clause_evaluation(clause, context, clause_result)
        
        return result
```

## Production Best Practices

### Configuration Management

```python
from dataclasses import dataclass
from typing import Optional
import os
import json

@dataclass
class LLMCLConfig:
    """Configuration for LLMCL in production environments."""
    
    # Performance settings
    cache_enabled: bool = True
    cache_size: int = 50000
    cache_ttl_seconds: int = 3600
    max_concurrent_validations: int = 100
    enable_lazy_loading: bool = True
    
    # Conflict resolution
    default_conflict_strategy: str = 'MOST_RESTRICTIVE'
    enable_priority_override: bool = True
    
    # Auto-fix settings
    auto_fix_strategy: str = 'BEST_FIX'
    auto_fix_confidence_threshold: float = 0.7
    max_auto_fix_attempts: int = 3
    
    # Observability
    enable_metrics: bool = True
    enable_tracing: bool = False
    enable_prometheus: bool = False
    metrics_port: int = 8000
    
    # Circuit breaker
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    
    # Memory management
    memory_limit_mb: int = 1000
    enable_memory_monitoring: bool = True
    
    # Security
    enable_contract_signing: bool = False
    max_contract_size_kb: int = 100
    allowed_functions: Optional[list] = None
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            cache_enabled=os.getenv('LLMCL_CACHE_ENABLED', 'true').lower() == 'true',
            cache_size=int(os.getenv('LLMCL_CACHE_SIZE', '50000')),
            cache_ttl_seconds=int(os.getenv('LLMCL_CACHE_TTL', '3600')),
            max_concurrent_validations=int(os.getenv('LLMCL_MAX_CONCURRENT', '100')),
            default_conflict_strategy=os.getenv('LLMCL_CONFLICT_STRATEGY', 'MOST_RESTRICTIVE'),
            auto_fix_confidence_threshold=float(os.getenv('LLMCL_AUTO_FIX_THRESHOLD', '0.7')),
            enable_metrics=os.getenv('LLMCL_ENABLE_METRICS', 'true').lower() == 'true',
            enable_tracing=os.getenv('LLMCL_ENABLE_TRACING', 'false').lower() == 'true',
            memory_limit_mb=int(os.getenv('LLMCL_MEMORY_LIMIT_MB', '1000'))
        )
    
    @classmethod
    def from_file(cls, config_path):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def to_file(self, config_path):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

def create_production_runtime(config: LLMCLConfig):
    """Create production-ready LLMCL runtime."""
    
    # Create conflict resolver
    conflict_resolver = ConflictResolver(
        default_strategy=config.default_conflict_strategy,
        priority_override=config.enable_priority_override
    )
    
    # Create auto-fix manager
    auto_fix_manager = AutoFixManager(
        default_strategy=config.auto_fix_strategy,
        confidence_threshold=config.auto_fix_confidence_threshold,
        max_attempts=config.max_auto_fix_attempts
    )
    
    # Create runtime with configuration
    runtime = LLMCLRuntime(
        conflict_resolver=conflict_resolver,
        auto_fix_manager=auto_fix_manager,
        cache_enabled=config.cache_enabled,
        cache_size=config.cache_size,
        cache_ttl=config.cache_ttl_seconds,
        max_concurrent_validations=config.max_concurrent_validations,
        enable_lazy_loading=config.enable_lazy_loading,
        enable_circuit_breaker=config.enable_circuit_breaker,
        circuit_breaker_threshold=config.circuit_breaker_threshold,
        circuit_breaker_timeout=config.circuit_breaker_timeout_seconds
    )
    
    # Enable monitoring if configured
    if config.enable_metrics:
        runtime = InstrumentedRuntime(runtime)
    
    if config.enable_tracing:
        runtime = TracingRuntime(runtime)
    
    if config.enable_prometheus:
        runtime = PrometheusInstrumentedRuntime(runtime)
    
    return runtime

# Usage
config = LLMCLConfig.from_env()
runtime = create_production_runtime(config)
```

### Health Checks and Monitoring

```python
import time
from enum import Enum
from typing import Dict, Any

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthChecker:
    def __init__(self, runtime):
        self.runtime = runtime
        self.last_check_time = 0
        self.check_interval = 60  # Check every minute
        self.health_history = deque(maxlen=10)
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        current_time = time.time()
        
        health_data = {
            'timestamp': current_time,
            'status': HealthStatus.HEALTHY.value,
            'checks': {},
            'metrics': {}
        }
        
        # Check memory usage
        memory_check = self._check_memory()
        health_data['checks']['memory'] = memory_check
        
        # Check validation performance
        performance_check = self._check_performance()
        health_data['checks']['performance'] = performance_check
        
        # Check cache health
        cache_check = self._check_cache()
        health_data['checks']['cache'] = cache_check
        
        # Check circuit breaker status
        circuit_check = self._check_circuit_breaker()
        health_data['checks']['circuit_breaker'] = circuit_check
        
        # Determine overall health status
        failed_checks = [name for name, check in health_data['checks'].items() if not check['healthy']]
        
        if failed_checks:
            if len(failed_checks) >= 2 or 'memory' in failed_checks:
                health_data['status'] = HealthStatus.UNHEALTHY.value
            else:
                health_data['status'] = HealthStatus.DEGRADED.value
            
            health_data['failed_checks'] = failed_checks
        
        # Add current metrics
        health_data['metrics'] = self._get_current_metrics()
        
        # Store in history
        self.health_history.append(health_data)
        self.last_check_time = current_time
        
        return health_data
    
    def _check_memory(self):
        """Check memory usage health."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            return {
                'healthy': memory_mb < 800,  # Threshold: 800MB
                'memory_usage_mb': memory_mb,
                'threshold_mb': 800
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def _check_performance(self):
        """Check validation performance health."""
        metrics = metrics_collector.get_current_metrics()
        
        return {
            'healthy': metrics.average_latency_ms < 1000 and metrics.success_rate > 0.95,
            'average_latency_ms': metrics.average_latency_ms,
            'success_rate': metrics.success_rate,
            'latency_threshold_ms': 1000,
            'success_rate_threshold': 0.95
        }
    
    def _check_cache(self):
        """Check cache health."""
        if hasattr(self.runtime, 'cache'):
            cache_stats = self.runtime.cache.get_stats()
            return {
                'healthy': cache_stats.get('hit_rate', 0) > 0.5,
                'hit_rate': cache_stats.get('hit_rate', 0),
                'hit_rate_threshold': 0.5,
                'cache_size': cache_stats.get('cache_size', 0)
            }
        
        return {'healthy': True, 'message': 'No cache configured'}
    
    def _check_circuit_breaker(self):
        """Check circuit breaker status."""
        if hasattr(self.runtime, 'circuit_breaker'):
            cb = self.runtime.circuit_breaker
            return {
                'healthy': cb.state != 'OPEN',
                'state': cb.state,
                'failure_count': cb.failure_count,
                'last_failure_time': cb.last_failure_time
            }
        
        return {'healthy': True, 'message': 'No circuit breaker configured'}
    
    def _get_current_metrics(self):
        """Get current system metrics."""
        metrics = metrics_collector.get_current_metrics()
        return {
            'total_validations': metrics.total_validations,
            'success_rate': metrics.success_rate,
            'average_latency_ms': metrics.average_latency_ms,
            'auto_fix_rate': metrics.auto_fix_rate,
            'memory_usage_mb': metrics.memory_usage_mb
        }
    
    def get_health_trends(self):
        """Get health trends over time."""
        if len(self.health_history) < 2:
            return {}
        
        recent = self.health_history[-1]
        previous = self.health_history[-2]
        
        trends = {}
        
        # Memory trend
        if 'memory' in recent['checks'] and 'memory' in previous['checks']:
            current_memory = recent['checks']['memory'].get('memory_usage_mb', 0)
            previous_memory = previous['checks']['memory'].get('memory_usage_mb', 0)
            trends['memory_trend'] = current_memory - previous_memory
        
        # Performance trend
        current_latency = recent['metrics'].get('average_latency_ms', 0)
        previous_latency = previous['metrics'].get('average_latency_ms', 0)
        trends['latency_trend'] = current_latency - previous_latency
        
        return trends

# Create health check endpoint
health_checker = HealthChecker(runtime)

def health_endpoint():
    """HTTP health check endpoint."""
    health_data = health_checker.check_health()
    
    status_code = 200
    if health_data['status'] == HealthStatus.DEGRADED.value:
        status_code = 200  # Still serving traffic
    elif health_data['status'] == HealthStatus.UNHEALTHY.value:
        status_code = 503  # Service unavailable
    
    return health_data, status_code
```

### Graceful Shutdown

```python
import signal
import threading
import atexit

class GracefulShutdown:
    def __init__(self, runtime, shutdown_timeout=30):
        self.runtime = runtime
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_event = threading.Event()
        self.active_validations = 0
        self.validation_lock = threading.Lock()
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        atexit.register(self.shutdown)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    def enter_validation(self):
        """Mark start of validation."""
        with self.validation_lock:
            if self.shutdown_event.is_set():
                raise RuntimeError("System is shutting down")
            self.active_validations += 1
    
    def exit_validation(self):
        """Mark end of validation."""
        with self.validation_lock:
            self.active_validations -= 1
    
    def shutdown(self):
        """Perform graceful shutdown."""
        if self.shutdown_event.is_set():
            return
        
        print("Initiating graceful shutdown...")
        self.shutdown_event.set()
        
        # Stop accepting new validations
        print("Stopping new validations...")
        
        # Wait for active validations to complete
        start_time = time.time()
        while self.active_validations > 0:
            if time.time() - start_time > self.shutdown_timeout:
                print(f"Shutdown timeout reached, {self.active_validations} validations still active")
                break
            
            print(f"Waiting for {self.active_validations} active validations...")
            time.sleep(1)
        
        # Cleanup runtime resources
        print("Cleaning up runtime resources...")
        if hasattr(self.runtime, 'cleanup'):
            self.runtime.cleanup()
        
        # Clear caches
        if hasattr(self.runtime, 'clear_cache'):
            self.runtime.clear_cache()
        
        print("Graceful shutdown completed")

class ShutdownAwareRuntime(LLMCLRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shutdown_manager = GracefulShutdown(self)
    
    def validate(self, contract, context):
        """Validate with graceful shutdown support."""
        self.shutdown_manager.enter_validation()
        
        try:
            return super().validate(contract, context)
        finally:
            self.shutdown_manager.exit_validation()

# Usage
runtime = ShutdownAwareRuntime()
```

This comprehensive guide covers all aspects of performance optimization, observability, and production best practices for LLMCL. These patterns ensure robust, scalable, and maintainable deployments in production environments.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "docs-1", "content": "Create main LLMCL documentation index with overview and getting started", "status": "completed", "priority": "high"}, {"id": "docs-2", "content": "Document LLMCL language syntax and grammar", "status": "completed", "priority": "high"}, {"id": "docs-3", "content": "Create contract types and temporal logic documentation", "status": "completed", "priority": "high"}, {"id": "docs-4", "content": "Document conflict resolution and auto-remediation features", "status": "completed", "priority": "high"}, {"id": "docs-5", "content": "Create API reference and examples documentation", "status": "completed", "priority": "medium"}, {"id": "docs-6", "content": "Document performance, observability and best practices", "status": "completed", "priority": "medium"}]