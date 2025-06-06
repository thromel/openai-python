"""Demo of LLM Contract Language (LLMCL) with OpenAI integration."""

import asyncio
from openai import OpenAI
from llm_contracts.providers.openai_provider import ImprovedOpenAIProvider
from llm_contracts.language import LLMCLRuntime, ResolutionStrategy


# Example 1: Simple Safety Contract
safety_contract = """
contract ChatbotSafety(priority = critical) {
    # Input validation
    require len(content) > 0 and len(content) < 4000
        message: "Input must be between 1 and 4000 characters"
    
    require not match(content, "(?i)(injection|exploit)")
        message: "Potential security threat detected"
    
    # Output validation
    ensure not contains(response, "password")
        message: "Response contains sensitive information"
    
    ensure len(response) > 10
        message: "Response too short"
        auto_fix: "I'd be happy to help you. Could you please provide more details about what you're looking for?"
}
"""

# Example 2: JSON API Response Contract
api_contract = """
contract APIResponse {
    # Ensure valid JSON
    ensure json_valid(response)
        message: "Response must be valid JSON"
        auto_fix: '{"error": "Invalid response format", "original": "' + response + '"}'
    
    # Ensure required fields
    ensure contains(response, '"status"') and contains(response, '"data"')
        message: "Response must contain status and data fields"
}
"""

# Example 3: Conversation Quality Contract
quality_contract = """
contract ConversationQuality(priority = high) {
    # Ensure helpful responses
    ensure len(response) > 50 or contains(response, "?")
        message: "Provide detailed responses or ask clarifying questions"
    
    # Probabilistic quality check
    ensure_prob not startswith(response, "I don't know"), 0.8
        message: "Should provide helpful answers at least 80% of the time"
        window_size: 50
    
    # Temporal constraint
    temporal within 3 contains(response, "help") or contains(response, "assist")
        message: "Offer help within first 3 responses"
}
"""


async def main():
    """Run LLMCL demo."""
    print("🚀 LLM Contract Language (LLMCL) Demo\n")
    
    # Initialize runtime
    runtime = LLMCLRuntime()
    
    # Load contracts
    print("📝 Loading contracts...")
    safety_name = await runtime.load_contract(safety_contract)
    api_name = await runtime.load_contract(api_contract)
    quality_name = await runtime.load_contract(quality_contract)
    print(f"✅ Loaded contracts: {safety_name}, {api_name}, {quality_name}\n")
    
    # Create runtime context with conflict resolution
    context = runtime.create_context(
        "demo_session",
        conflict_strategy=ResolutionStrategy.MOST_RESTRICTIVE
    )
    
    # Add contracts to context
    runtime.add_contract_to_context("demo_session", safety_name)
    runtime.add_contract_to_context("demo_session", quality_name)
    
    # Demo 1: Basic validation
    print("🔍 Demo 1: Basic Validation")
    print("-" * 50)
    
    # Test valid input/output
    result = await runtime.validate(
        "This is a helpful and detailed response about Python programming. "
        "It provides useful information without any sensitive data.",
        "demo_session",
        validation_type="output",
        additional_context={"content": "Tell me about Python"}
    )
    print(f"Valid response: {result.is_valid}")
    
    # Test invalid output (too short)
    result = await runtime.validate(
        "OK",
        "demo_session",
        validation_type="output",
        additional_context={"content": "Explain quantum computing"}
    )
    print(f"Short response: {result.is_valid}")
    if not result.is_valid:
        print(f"  Violation: {result.message}")
    
    # Test security violation
    result = await runtime.validate(
        "Your password is stored in the database",
        "demo_session",
        validation_type="output",
        additional_context={"content": "How do I reset my password?"}
    )
    print(f"Security violation: {result.is_valid}")
    if not result.is_valid:
        print(f"  Violation: {result.message}\n")
    
    # Demo 2: Auto-remediation
    print("\n🔧 Demo 2: Auto-Remediation")
    print("-" * 50)
    
    # Test auto-fix for short response
    short_response = "OK"
    print(f"Original: {short_response}")
    fixed = await runtime.apply_auto_fix(short_response, "demo_session")
    print(f"Fixed: {fixed}\n")
    
    # Demo 3: JSON API validation
    print("📊 Demo 3: JSON API Validation")
    print("-" * 50)
    
    # Create API context
    api_context = runtime.create_context("api_session")
    runtime.add_contract_to_context("api_session", api_name)
    
    # Valid JSON
    valid_json = '{"status": "success", "data": {"id": 123, "name": "test"}}'
    result = await runtime.validate(valid_json, "api_session")
    print(f"Valid JSON: {result.is_valid}")
    
    # Invalid JSON
    invalid_json = "This is not JSON"
    result = await runtime.validate(invalid_json, "api_session")
    print(f"Invalid JSON: {result.is_valid}")
    if not result.is_valid:
        print(f"  Violation: {result.message}")
    
    # Apply auto-fix
    fixed_json = await runtime.apply_auto_fix(invalid_json, "api_session")
    print(f"  Auto-fixed: {fixed_json}\n")
    
    # Demo 4: OpenAI Integration
    print("🤖 Demo 4: OpenAI Integration with Contracts")
    print("-" * 50)
    
    # This would require an actual OpenAI API key
    # client = ImprovedOpenAIProvider(api_key="your-api-key")
    
    # Add contracts to OpenAI provider
    # safety = runtime.loaded_contracts[safety_name].contract_instance
    # quality = runtime.loaded_contracts[quality_name].contract_instance
    # client.add_output_contract(safety)
    # client.add_output_contract(quality)
    
    print("OpenAI integration example (requires API key):")
    print("""
    client = ImprovedOpenAIProvider(api_key="your-key")
    client.add_output_contract(safety_contract)
    client.add_output_contract(quality_contract)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    # Contracts automatically validated!
    """)
    
    # Show metrics
    print("\n📈 Runtime Metrics:")
    print("-" * 50)
    metrics = runtime.get_global_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Show context statistics
    stats = runtime.get_context_statistics("demo_session")
    print(f"\nSession Statistics:")
    print(f"Total validations: {stats['total_validations']}")
    print(f"Total violations: {stats['total_violations']}")
    print(f"Auto-fixes applied: {stats['total_auto_fixes']}")


if __name__ == "__main__":
    asyncio.run(main())