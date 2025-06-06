"""
Demonstration of Real-time Streaming Validation System

This example shows how to use the streaming validation system to:
1. Validate LLM responses in real-time as they stream
2. Terminate streams on critical violations
3. Apply auto-fixes for common issues
4. Monitor performance and optimize latency
5. Use specialized processors for different validation needs
"""

import asyncio
import time
from typing import List

# Import the streaming validation components
from llm_contracts.streaming import (
    StreamingValidator, StreamingClient, ChunkValidationMode,
    TokenProcessor, ContentFilter, PatternMatcher, JSONValidator,
    LengthMonitor, SentimentAnalyzer, ToxicityDetector,
    StreamingProfiler, LatencyOptimizer, benchmark_streaming_validation
)
from llm_contracts.providers import ImprovedOpenAIProvider


class DemoStreamingContract:
    """Demo contract that validates content doesn't contain secrets."""
    
    def __init__(self):
        self.name = "secret_detector"
        self.description = "Detects potential secrets in responses"
        self.supports_streaming = True
        
        # Common secret patterns
        self.secret_patterns = [
            r'api[_-]?key',
            r'password',
            r'secret',
            r'token',
            r'[A-Za-z0-9]{32,}',  # Long alphanumeric strings
        ]
    
    async def validate_chunk(self, chunk, context=None):
        """Validate chunk for secrets."""
        import re
        
        content = chunk.cumulative_content.lower()
        
        for pattern in self.secret_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                from llm_contracts.streaming.stream_validator import StreamValidationResult, ViolationSeverity
                return StreamValidationResult(
                    is_valid=False,
                    should_terminate=True,  # Critical violation
                    violation_severity=ViolationSeverity.CRITICAL,
                    violation_message=f"Potential secret detected: {pattern}",
                    auto_fix_suggestion="[REDACTED]"
                )
        
        from llm_contracts.streaming.stream_validator import StreamValidationResult
        return StreamValidationResult(is_valid=True)


async def demo_basic_streaming_validation():
    """Demonstrate basic streaming validation."""
    print("=== Basic Streaming Validation Demo ===")
    
    # Create processors
    contracts = [
        TokenProcessor(max_tokens=50, forbidden_tokens={"spam", "advertisement"}),
        ContentFilter(blocked_patterns=["buy now", "click here"], stop_on_violation=True),
        LengthMonitor(max_length=200, warn_threshold=150),
        DemoStreamingContract(),
    ]
    
    # Create streaming validator
    validator = StreamingValidator(
        contracts=contracts,
        validation_mode=ChunkValidationMode.IMMEDIATE,
        performance_monitoring=True,
        early_termination=True
    )
    
    # Simulate streaming content
    test_chunks = [
        "Hello! I'm here to help you with your questions. ",
        "Let me provide some useful information about our services. ",
        "We offer comprehensive solutions for your business needs. ",
        "Contact us today for more details!"
    ]
    
    print("Processing chunks:")
    for i, chunk in enumerate(test_chunks):
        print(f"  Chunk {i}: '{chunk.strip()}'")
        
        result = await validator.process_chunk(chunk, i)
        
        if result.is_valid:
            print(f"    ✅ Valid")
        else:
            print(f"    ❌ Violation: {result.violation_message}")
            if result.should_terminate:
                print(f"    🛑 Stream terminated!")
                break
    
    # Finalize validation
    final_result = await validator.finalize_validation()
    if final_result.is_valid:
        print("✅ Final validation passed")
    else:
        print(f"❌ Final validation failed: {final_result.violation_message}")
    
    # Show metrics
    metrics = validator.get_metrics().get_summary()
    print(f"\nMetrics:")
    print(f"  Chunks processed: {metrics['total_chunks']}")
    print(f"  Violations detected: {metrics['violations_detected']}")
    print(f"  Average latency: {metrics['average_validation_latency_ms']:.1f}ms")
    print(f"  Validation overhead: {metrics['validation_overhead_percentage']:.1f}%")


async def demo_critical_violation_termination():
    """Demonstrate critical violation termination."""
    print("\n=== Critical Violation Termination Demo ===")
    
    # Create validator with secret detection
    validator = StreamingValidator(
        contracts=[DemoStreamingContract()],
        early_termination=True
    )
    
    # Simulate content that will trigger termination
    dangerous_chunks = [
        "Here's some helpful information: ",
        "Your account details are secure. ",
        "Your api_key is: sk-abc123def456... ",  # This should trigger termination
        "Please keep it safe!"  # This should never be processed
    ]
    
    print("Processing potentially dangerous content:")
    for i, chunk in enumerate(dangerous_chunks):
        print(f"  Chunk {i}: '{chunk.strip()}'")
        
        result = await validator.process_chunk(chunk, i)
        
        if result.is_valid:
            print(f"    ✅ Valid")
        else:
            print(f"    ❌ CRITICAL VIOLATION: {result.violation_message}")
            if result.should_terminate:
                print(f"    🛑 Stream terminated immediately!")
                print(f"    💡 Auto-fix suggestion: {result.auto_fix_suggestion}")
                break


async def demo_specialized_processors():
    """Demonstrate specialized processors."""
    print("\n=== Specialized Processors Demo ===")
    
    # JSON Validation
    print("JSON Validation:")
    json_validator = JSONValidator(require_complete_json=True, required_fields=["status", "data"])
    
    json_chunks = ['{"status": "success"', ', "data": {"message": "Hello"}', ', "extra": "info"}']
    validator = StreamingValidator([json_validator])
    
    for i, chunk in enumerate(json_chunks):
        print(f"  Processing: '{chunk}'")
        result = await validator.process_chunk(chunk, i)
        print(f"    {'✅ Valid' if result.is_valid else '❌ Invalid: ' + (result.violation_message or 'Unknown')}")
    
    # Finalize JSON validation
    final_result = await validator.finalize_validation()
    print(f"  Final JSON validation: {'✅ Valid' if final_result.is_valid else '❌ Invalid: ' + (final_result.violation_message or 'Unknown')}")
    
    # Sentiment Analysis
    print("\nSentiment Analysis:")
    sentiment_analyzer = SentimentAnalyzer(min_sentiment=-0.3, max_sentiment=0.8)
    
    sentiment_test = "This product is absolutely terrible and I hate it completely. It's the worst thing ever!"
    validator = StreamingValidator([sentiment_analyzer])
    
    result = await validator.process_chunk(sentiment_test, 0)
    print(f"  Content: '{sentiment_test}'")
    print(f"  Result: {'✅ Valid' if result.is_valid else '❌ Invalid: ' + (result.violation_message or 'Unknown')}")
    
    # Toxicity Detection
    print("\nToxicity Detection:")
    toxicity_detector = ToxicityDetector(severity_escalation=True)
    
    toxic_content = "You're so stupid and I hate you"
    validator = StreamingValidator([toxicity_detector])
    
    result = await validator.process_chunk(toxic_content, 0)
    print(f"  Content: '{toxic_content}'")
    print(f"  Result: {'✅ Valid' if result.is_valid else '❌ Invalid: ' + (result.violation_message or 'Unknown')}")
    if result.auto_fix_suggestion:
        print(f"  Auto-fix: '{result.auto_fix_suggestion}'")


async def demo_performance_optimization():
    """Demonstrate performance profiling and optimization."""
    print("\n=== Performance Optimization Demo ===")
    
    # Create profiler and optimizer
    profiler = StreamingProfiler(enable_detailed_tracking=True)
    optimizer = LatencyOptimizer(target_latency_ms=5.0)
    
    # Create validator with multiple contracts
    contracts = [
        TokenProcessor(max_tokens=100),
        ContentFilter(blocked_patterns=["forbidden"]),
        LengthMonitor(max_length=500),
        PatternMatcher(required_patterns=["hello|hi|greetings"]),
    ]
    
    validator = StreamingValidator(contracts, performance_monitoring=True)
    
    # Start profiling
    profiler.start_profiling()
    
    # Simulate processing with performance monitoring
    test_content = "Hello there! This is a longer test message that will be processed in chunks to demonstrate the performance monitoring capabilities of our streaming validation system. " * 3
    
    chunks = [test_content[i:i+50] for i in range(0, len(test_content), 50)]
    
    print(f"Processing {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        start_time = time.time()
        
        result = await validator.process_chunk(chunk, i)
        
        latency_ms = (time.time() - start_time) * 1000
        profiler.record_chunk_processing(latency_ms)
        profiler.record_validation(latency_ms, f"batch_{i}", violated=not result.is_valid)
        
        # Get optimization recommendations
        recommendations = optimizer.record_latency(latency_ms)
        if recommendations:
            print(f"  Optimization recommendations: {recommendations}")
    
    # Stop profiling and get report
    report = profiler.stop_profiling()
    
    print(f"\nPerformance Report:")
    print(f"  Chunks processed: {report['chunks_processed']}")
    print(f"  Total validations: {report['validations_performed']}")
    print(f"  Average latency: {report['latency_metrics']['validation']['average_ms']:.2f}ms")
    print(f"  P95 latency: {report['latency_metrics']['validation']['p95_ms']:.2f}ms")
    print(f"  Peak latency: {report['latency_metrics']['validation']['peak_ms']:.2f}ms")
    print(f"  Throughput: {report['throughput']['chunks_per_second']:.1f} chunks/sec")
    
    # Show optimizer settings
    settings = optimizer.get_current_settings()
    print(f"\nOptimizer Settings:")
    print(f"  Validation frequency: {settings['validation_frequency']:.2f}")
    print(f"  Buffer size multiplier: {settings['buffer_size_multiplier']:.2f}")
    print(f"  Concurrent validation limit: {settings['concurrent_validation_limit']}")


async def demo_performance_benchmark():
    """Demonstrate performance benchmarking."""
    print("\n=== Performance Benchmark Demo ===")
    
    # Create a realistic set of contracts
    contracts = [
        TokenProcessor(max_tokens=200),
        ContentFilter(blocked_patterns=["spam", "advertisement", "click here"]),
        LengthMonitor(max_length=1000),
        SentimentAnalyzer(min_sentiment=-0.5, max_sentiment=0.8),
    ]
    
    validator = StreamingValidator(contracts, performance_monitoring=True)
    
    # Benchmark content
    test_content = """
    Hello! I'm here to provide you with helpful information about our services.
    We offer a wide range of solutions for businesses of all sizes.
    Our team is dedicated to delivering high-quality results that meet your needs.
    Please let me know how I can assist you today with your questions or concerns.
    We value your feedback and strive to continuously improve our offerings.
    """ * 2
    
    print(f"Running benchmark with {len(test_content)} characters...")
    print("This may take a moment...")
    
    # Run benchmark
    report = await benchmark_streaming_validation(
        validator=validator,
        test_content=test_content,
        chunk_size=25,
        iterations=20
    )
    
    print(f"\nBenchmark Results:")
    print(f"  Total iterations: {report['iteration_stats']['total_iterations']}")
    print(f"  Average iteration time: {report['iteration_stats']['avg_latency_ms']:.2f}ms")
    print(f"  P50 iteration time: {report['iteration_stats']['p50_latency_ms']:.2f}ms")
    print(f"  P95 iteration time: {report['iteration_stats']['p95_latency_ms']:.2f}ms")
    print(f"  Min iteration time: {report['iteration_stats']['min_latency_ms']:.2f}ms")
    print(f"  Max iteration time: {report['iteration_stats']['max_latency_ms']:.2f}ms")
    print(f"  Total chunks processed: {report['chunks_processed']}")
    print(f"  Validation throughput: {report['throughput']['validations_per_second']:.1f} validations/sec")


def demo_openai_integration():
    """Demonstrate integration with OpenAI provider (requires API key)."""
    print("\n=== OpenAI Integration Demo ===")
    print("Note: This demo requires an OpenAI API key to run.")
    print("It shows how streaming validation integrates with real OpenAI API calls.")
    
    # Example code (commented out since it requires API key)
    example_code = '''
# Create OpenAI provider with streaming validation
from llm_contracts.providers import ImprovedOpenAIProvider
from llm_contracts.streaming import TokenProcessor, ContentFilter, ToxicityDetector

client = ImprovedOpenAIProvider(api_key="your-api-key")

# Add streaming contracts
client.add_output_contract(TokenProcessor(max_tokens=100))
client.add_output_contract(ContentFilter(blocked_patterns=["harmful"]))
client.add_output_contract(ToxicityDetector())

# Create streaming completion with validation
async def stream_with_validation():
    stream = await client.chat.completions.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True
    )
    
    async for chunk in stream:
        # Validation happens automatically in real-time
        print(chunk.choices[0].delta.content, end="", flush=True)
    
    # Get validation metrics
    metrics = stream.get_validation_metrics()
    print(f"\\nValidation overhead: {metrics['validation_overhead_percentage']:.1f}%")

# Run the streaming completion
await stream_with_validation()
'''
    
    print(example_code)


async def main():
    """Run all demos."""
    print("🚀 Streaming Validation System Demo")
    print("=" * 50)
    
    try:
        await demo_basic_streaming_validation()
        await demo_critical_violation_termination()
        await demo_specialized_processors()
        await demo_performance_optimization()
        await demo_performance_benchmark()
        demo_openai_integration()
        
        print("\n✅ All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Real-time chunk validation with immediate feedback")
        print("• Critical violation detection and stream termination")
        print("• Specialized processors for different validation needs")
        print("• Performance monitoring and optimization")
        print("• Comprehensive benchmarking capabilities")
        print("• Integration with OpenAI streaming API")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())