# LLM Contract Language (LLMCL) Documentation

**LLMCL** is a domain-specific language for defining and validating contracts in Large Language Model (LLM) applications. It provides a declarative, human-readable syntax for specifying constraints, temporal logic, probabilistic guarantees, and auto-remediation strategies.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Language Syntax](#language-syntax)
4. [Contract Types](#contract-types)
5. [Temporal Logic](#temporal-logic)
6. [Probabilistic Contracts](#probabilistic-contracts)
7. [Conflict Resolution](#conflict-resolution)
8. [Auto-Remediation](#auto-remediation)
9. [API Reference](#api-reference)
10. [Examples](#examples)
11. [Performance & Observability](#performance--observability)
12. [Best Practices](#best-practices)

## Overview

LLMCL transforms LLM validation from imperative programming to declarative contract specification. Instead of writing Python code to validate LLM inputs and outputs, you write human-readable contracts that specify what should be true.

### Key Features

- **Declarative Syntax**: Write "what" not "how"
- **Temporal Logic**: Multi-turn conversation validation
- **Probabilistic Contracts**: Statistical guarantees over time
- **Smart Conflict Resolution**: Automatic handling of conflicting constraints
- **Auto-Remediation**: Built-in fixing strategies
- **Performance Optimized**: Caching, lazy loading, async validation
- **Observability**: Metrics, tracing, and monitoring

### Architecture

```
LLMCL Source → Lexer → Parser → AST → Compiler → Runtime
                                        ↓
                                   Executable Contracts
```

## Quick Start

### Installation

```python
from llm_contracts.language import LLMCLRuntime, compile_contract

# Initialize runtime
runtime = LLMCLRuntime()
```

### Your First Contract

```llmcl
contract ChatbotSafety(priority = critical) {
    require len(content) < 4000
        message: "Input too long"
    
    ensure not contains(response, "password")
        message: "Response contains sensitive information"
    
    ensure_prob json_valid(response), 0.95
        message: "95% of responses should be valid JSON"
        window_size: 100
}
```

### Using the Contract

```python
# Compile contract
contract_source = """
contract ChatbotSafety(priority = critical) {
    require len(content) < 4000
        message: "Input too long"
    
    ensure not contains(response, "password")
        message: "Response contains sensitive information"
}
"""

contract = compile_contract(contract_source)

# Validate
context = {
    'content': 'User input here',
    'response': 'LLM response here'
}

result = runtime.validate(contract, context)
if not result.is_valid:
    print(f"Validation failed: {result.violations}")
```

## Language Syntax

### Contract Declaration

```llmcl
contract ContractName(priority = level, description = "text") {
    // Contract clauses
}
```

**Priority Levels**: `critical`, `high`, `medium`, `low`

### Clause Types

#### Preconditions (require)
```llmcl
require condition
    message: "Error message"
    auto_fix: fix_expression
```

#### Postconditions (ensure)
```llmcl
ensure condition
    message: "Error message" 
    auto_fix: fix_expression
```

#### Probabilistic Constraints (ensure_prob)
```llmcl
ensure_prob condition, probability
    message: "Error message"
    window_size: N
    auto_fix: fix_expression
```

#### Temporal Constraints
```llmcl
temporal operator condition
    message: "Error message"
```

### Expressions

#### Built-in Functions
- `len(string)` - String length
- `contains(string, substring)` - Substring check
- `startswith(string, prefix)` - Prefix check
- `endswith(string, suffix)` - Suffix check
- `match(string, pattern)` - Regex matching
- `json_valid(string)` - JSON validation

#### Operators
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `and`, `or`, `not`
- String: `+` (concatenation)

#### Literals
- Strings: `"text"` or `'text'`
- Numbers: `42`, `3.14`
- Booleans: `true`, `false`

### Comments

```llmcl
// Single line comment

/* Multi-line
   comment */
```

## Contract Types

### Input Validation Contracts

```llmcl
contract InputValidation(priority = high) {
    require len(content) > 0
        message: "Input cannot be empty"
    
    require len(content) <= 4000
        message: "Input too long"
        auto_fix: content[:4000]
    
    require not contains(content, "admin_password")
        message: "Input contains sensitive information"
}
```

### Output Quality Contracts

```llmcl
contract OutputQuality(priority = medium) {
    ensure len(response) > 10
        message: "Response too short"
        auto_fix: response + " Please let me know if you need more information."
    
    ensure not startswith(response, "I don't know")
        message: "Should provide helpful responses"
    
    ensure_prob contains(response, "helpful"), 0.8
        message: "80% of responses should be helpful"
        window_size: 50
}
```

### Content Policy Contracts

```llmcl
contract ContentPolicy(priority = critical) {
    ensure not contains(response, "password")
        message: "Must not expose passwords"
    
    ensure not contains(response, "API_KEY")
        message: "Must not expose API keys"
    
    ensure not match(response, r"\b\d{4}-\d{4}-\d{4}-\d{4}\b")
        message: "Must not expose credit card numbers"
}
```

### Format Contracts

```llmcl
contract JSONFormat(priority = high) {
    ensure json_valid(response)
        message: "Response must be valid JSON"
        auto_fix: '{"error": "Invalid format", "original": "' + response + '"}'
    
    ensure startswith(response, "{") and endswith(response, "}")
        message: "JSON must start with { and end with }"
}
```

## Temporal Logic

Temporal logic allows validation across multiple turns in a conversation.

### Temporal Operators

#### Always (□)
```llmcl
temporal always len(response) > 0
    message: "Response must never be empty"
```

#### Eventually (◇)
```llmcl
temporal eventually contains(response, "thank you")
    message: "Should eventually show gratitude"
```

#### Next (○)
```llmcl
temporal next contains(response, "follow-up")
    message: "Next response should follow up"
```

#### Within N Turns
```llmcl
temporal within 3 contains(response, "help")
    message: "Should offer help within 3 turns"
```

#### Until/Since
```llmcl
temporal contains(response, "error") until contains(response, "resolved")
    message: "Should keep showing error until resolved"

temporal since contains(response, "greeting") then len(response) > 20
    message: "After greeting, responses should be substantial"
```

### Conversation Context

```llmcl
contract ConversationFlow(priority = medium) {
    // First turn should be a greeting
    temporal next contains(response, "hello") or contains(response, "hi")
        message: "Should greet user initially"
    
    // Should not repeat the same response
    temporal always response != prev_response
        message: "Should not repeat exact responses"
    
    // Should conclude within 10 turns
    temporal within 10 contains(response, "goodbye") or contains(response, "conclusion")
        message: "Should conclude conversation reasonably"
}
```

## Probabilistic Contracts

Probabilistic contracts specify statistical guarantees over time windows.

### Basic Probabilistic Constraint

```llmcl
ensure_prob condition, probability
    message: "Error message"
    window_size: N
```

### Examples

```llmcl
contract StatisticalQuality(priority = medium) {
    // 90% of responses should be longer than 50 characters
    ensure_prob len(response) > 50, 0.9
        message: "Most responses should be substantial"
        window_size: 100
    
    // 95% should not start with "I don't know"
    ensure_prob not startswith(response, "I don't know"), 0.95
        message: "Should rarely claim ignorance"
        window_size: 50
    
    // 80% should contain helpful keywords
    ensure_prob contains(response, "help") or contains(response, "assist"), 0.8
        message: "Should frequently offer assistance"
        window_size: 25
}
```

### Window Management

- **Sliding Window**: Most recent N interactions
- **Session Window**: Current conversation session
- **Global Window**: All interactions ever

```llmcl
contract AdaptiveQuality(priority = medium) {
    // Short-term quality (last 10 responses)
    ensure_prob json_valid(response), 0.95
        window_size: 10
    
    // Medium-term engagement (last 50 responses)  
    ensure_prob len(response) > 100, 0.8
        window_size: 50
    
    // Long-term helpfulness (last 200 responses)
    ensure_prob contains(response, "helpful"), 0.7
        window_size: 200
}
```

## Conflict Resolution

When multiple contracts have conflicting requirements, LLMCL provides sophisticated resolution strategies.

### Conflict Types

1. **Format Conflicts**: JSON vs. plain text
2. **Length Conflicts**: Minimum vs. maximum length
3. **Content Conflicts**: Required vs. forbidden content
4. **Temporal Conflicts**: Conflicting temporal requirements
5. **Semantic Conflicts**: Contradictory meaning requirements

### Resolution Strategies

```llmcl
contract ConflictExample(priority = high, resolution = MOST_RESTRICTIVE) {
    require len(response) > 100      // Minimum length
        message: "Response too short"
    
    require len(response) < 50       // Conflicts with above!
        message: "Response too long"
}
```

#### Available Strategies

- `FIRST_WINS`: Use first matching contract
- `LAST_WINS`: Use last matching contract  
- `MOST_RESTRICTIVE`: Choose most restrictive constraint
- `LEAST_RESTRICTIVE`: Choose least restrictive constraint
- `MERGE`: Intelligently combine constraints
- `FAIL_ON_CONFLICT`: Raise error on conflicts

### Priority-Based Resolution

```llmcl
contract HighPriority(priority = critical) {
    ensure len(response) < 100
        message: "Critical: Keep responses short"
}

contract LowPriority(priority = low) {
    ensure len(response) > 200
        message: "Low: Prefer detailed responses"
}

// HighPriority wins due to higher priority
```

### Custom Conflict Resolution

```python
from llm_contracts.language.conflict_resolver import ConflictResolver

resolver = ConflictResolver()
resolver.add_strategy('custom', lambda conflicts: resolve_my_way(conflicts))

runtime = LLMCLRuntime(conflict_resolver=resolver)
```

## Auto-Remediation

LLMCL supports automatic fixing of contract violations through expression-based remediation.

### Basic Auto-Fix

```llmcl
contract AutoFixExample(priority = medium) {
    ensure len(response) <= 200
        message: "Response too long"
        auto_fix: response[:200] + "..."
    
    ensure json_valid(response)
        message: "Invalid JSON"
        auto_fix: '{"content": "' + response.replace('"', '\\"') + '"}'
    
    ensure not contains(response, "error")
        message: "Should not mention errors"
        auto_fix: response.replace("error", "issue")
}
```

### Complex Auto-Fix Logic

```llmcl
contract SmartAutoFix(priority = high) {
    ensure startswith(response, "{") and endswith(response, "}")
        message: "Must be JSON object"
        auto_fix: if startswith(response, "{") then 
                     response + "}" 
                   else if endswith(response, "}") then 
                     "{" + response 
                   else 
                     '{"content": "' + response + '"}'
    
    ensure len(response) > 20
        message: "Response too short"
        auto_fix: response + " Is there anything else I can help you with?"
}
```

### Auto-Fix Strategies

```llmcl
contract FixStrategies(priority = medium, fix_strategy = ALL_FIXES) {
    ensure len(response) <= 100
        auto_fix: response[:100]
    
    ensure contains(response, "helpful")
        auto_fix: response + " I hope this is helpful!"
}
```

**Fix Strategies**:
- `FIRST_FIX`: Apply only first applicable fix
- `ALL_FIXES`: Apply all applicable fixes in order
- `BEST_FIX`: Choose best fix based on confidence scores

## API Reference

### LLMCLRuntime

```python
class LLMCLRuntime:
    def __init__(self, 
                 conflict_resolver=None, 
                 telemetry_enabled=True,
                 cache_enabled=True):
        """Initialize LLMCL runtime."""
    
    def validate(self, contract, context):
        """Validate context against contract."""
    
    def validate_async(self, contract, context):
        """Async validation."""
    
    def compile_and_validate(self, source, context):
        """Compile source and validate in one step."""
    
    def add_contract(self, contract):
        """Add contract to runtime."""
    
    def remove_contract(self, contract_name):
        """Remove contract by name."""
    
    def get_statistics(self):
        """Get validation statistics."""
```

### Compilation

```python
def compile_contract(source: str) -> Contract:
    """Compile LLMCL source to executable contract."""

def parse_contract(source: str) -> AST:
    """Parse LLMCL source to AST."""

def validate_syntax(source: str) -> List[SyntaxError]:
    """Validate LLMCL syntax without compilation."""
```

### Validation Results

```python
class ValidationResult:
    is_valid: bool
    violations: List[Violation]
    auto_fixes: List[AutoFix]
    statistics: ValidationStatistics
    execution_time: float
    
class Violation:
    contract_name: str
    clause_type: str
    message: str
    severity: str
    context: dict
    
class AutoFix:
    violation: Violation
    fix_expression: str
    fixed_value: str
    confidence: float
```

## Examples

### Complete Application Example

```python
from llm_contracts.language import LLMCLRuntime, compile_contract

# Define contracts
chatbot_contracts = """
contract InputSafety(priority = critical) {
    require len(content) > 0
        message: "Input cannot be empty"
    
    require len(content) <= 4000
        message: "Input too long"
        auto_fix: content[:4000]
    
    require not contains(content, "admin_password")
        message: "Input contains sensitive information"
}

contract OutputQuality(priority = high) {
    ensure len(response) >= 20
        message: "Response too short"
        auto_fix: response + " Is there anything else I can help you with?"
    
    ensure_prob json_valid(response), 0.9
        message: "90% of responses should be valid JSON"
        window_size: 50
        auto_fix: '{"content": "' + response.replace('"', '\\"') + '"}'
    
    temporal always len(response) > 0
        message: "Response must never be empty"
}

contract ConversationFlow(priority = medium) {
    temporal within 5 contains(response, "help")
        message: "Should offer help within 5 turns"
    
    temporal always response != prev_response
        message: "Should not repeat responses"
}
"""

# Initialize runtime
runtime = LLMCLRuntime(
    conflict_resolver=ConflictResolver(strategy='MOST_RESTRICTIVE'),
    telemetry_enabled=True
)

# Compile contracts
contracts = compile_contract(chatbot_contracts)
runtime.add_contract(contracts)

# Usage in application
def process_user_input(user_input: str) -> str:
    # Pre-validation
    input_context = {'content': user_input}
    input_result = runtime.validate(contracts, input_context)
    
    if not input_result.is_valid:
        # Apply auto-fixes if available
        if input_result.auto_fixes:
            user_input = input_result.auto_fixes[0].fixed_value
        else:
            return "Sorry, I cannot process this input."
    
    # Generate LLM response (your LLM call here)
    llm_response = call_llm(user_input)
    
    # Post-validation
    output_context = {
        'content': user_input,
        'response': llm_response,
        'prev_response': get_previous_response()
    }
    
    output_result = runtime.validate(contracts, output_context)
    
    if not output_result.is_valid:
        # Apply auto-fixes
        if output_result.auto_fixes:
            llm_response = output_result.auto_fixes[0].fixed_value
        else:
            llm_response = "I apologize, but I cannot provide a proper response right now."
    
    return llm_response

# Example usage
response = process_user_input("Help me with Python")
print(response)

# Get statistics
stats = runtime.get_statistics()
print(f"Validation success rate: {stats.success_rate}")
print(f"Auto-fix success rate: {stats.auto_fix_rate}")
```

### Multi-Contract System

```python
# Define multiple specialized contracts
safety_contract = """
contract Safety(priority = critical) {
    ensure not contains(response, "password")
        message: "Must not expose passwords"
    
    ensure not match(response, r"\b\d{4}-\d{4}-\d{4}-\d{4}\b")
        message: "Must not expose credit card numbers"
}
"""

quality_contract = """
contract Quality(priority = high) {
    ensure_prob len(response) > 50, 0.8
        message: "80% of responses should be substantial"
        window_size: 20
    
    ensure_prob contains(response, "helpful"), 0.7
        message: "Should frequently be helpful"
        window_size: 30
}
"""

format_contract = """
contract Format(priority = medium) {
    ensure json_valid(response)
        message: "Response should be valid JSON"
        auto_fix: '{"content": "' + response + '"}'
}
"""

# Load contracts separately
runtime = LLMCLRuntime()
runtime.add_contract(compile_contract(safety_contract))
runtime.add_contract(compile_contract(quality_contract))
runtime.add_contract(compile_contract(format_contract))
```

## Performance & Observability

### Performance Features

- **Lazy Loading**: Contracts compiled on first use
- **Validation Caching**: Results cached based on context hash
- **Async Validation**: Non-blocking validation for high throughput
- **Circuit Breaker**: Fail-fast on repeated validation failures

### Metrics and Monitoring

```python
from llm_contracts.utils.telemetry import get_metrics

# Built-in metrics
metrics = get_metrics()
print(f"Total validations: {metrics.total_validations}")
print(f"Success rate: {metrics.success_rate}")
print(f"Average latency: {metrics.avg_latency_ms}ms")
print(f"Auto-fix rate: {metrics.auto_fix_rate}")

# Custom metrics
runtime.add_metric_callback(lambda result: log_to_datadog(result))
```

### OpenTelemetry Integration

```python
from opentelemetry import trace
from llm_contracts.language import LLMCLRuntime

# Automatic tracing
runtime = LLMCLRuntime(telemetry_enabled=True)

# Custom spans
with trace.get_tracer(__name__).start_as_current_span("contract_validation"):
    result = runtime.validate(contract, context)
```

### Performance Tuning

```python
# Configure performance settings
runtime = LLMCLRuntime(
    cache_size=10000,           # Validation result cache size
    cache_ttl=3600,             # Cache TTL in seconds  
    max_concurrent=100,         # Max concurrent validations
    circuit_breaker_threshold=5, # Failures before circuit opens
    enable_lazy_loading=True,   # Lazy contract compilation
    enable_async=True           # Enable async validation
)
```

## Best Practices

### Contract Design

1. **Keep Contracts Focused**: One concern per contract
2. **Use Appropriate Priorities**: Critical for safety, medium for quality
3. **Provide Clear Messages**: Help developers understand violations
4. **Include Auto-Fixes**: Automate common repairs
5. **Test Probabilistic Windows**: Choose appropriate window sizes

### Organization

```llmcl
// Good: Focused contracts
contract InputLength(priority = high) {
    require len(content) > 0 and len(content) <= 4000
        message: "Input must be 1-4000 characters"
        auto_fix: content[:4000]
}

contract OutputSafety(priority = critical) {
    ensure not contains(response, "password")
        message: "Must not expose passwords"
}

// Avoid: Monolithic contracts mixing concerns
contract Everything(priority = medium) {
    require len(content) > 0
    ensure json_valid(response)  
    ensure not contains(response, "password")
    temporal always len(response) > 0
    // ... too many mixed concerns
}
```

### Performance Optimization

1. **Cache Validation Results**: Enable caching for repeated validations
2. **Use Async Validation**: For high-throughput applications
3. **Batch Similar Contracts**: Group related validations
4. **Monitor Performance**: Track latency and success rates
5. **Tune Window Sizes**: Balance accuracy vs. performance

### Testing Contracts

```python
import pytest
from llm_contracts.language import compile_contract, LLMCLRuntime

def test_input_validation():
    contract = compile_contract("""
    contract TestInput(priority = high) {
        require len(content) > 0
            message: "Input required"
        
        require len(content) <= 100
            message: "Input too long"
            auto_fix: content[:100]
    }
    """)
    
    runtime = LLMCLRuntime()
    
    # Test valid input
    result = runtime.validate(contract, {'content': 'Hello world'})
    assert result.is_valid
    
    # Test empty input
    result = runtime.validate(contract, {'content': ''})
    assert not result.is_valid
    assert 'Input required' in result.violations[0].message
    
    # Test too long input with auto-fix
    long_input = 'x' * 200
    result = runtime.validate(contract, {'content': long_input})
    assert not result.is_valid
    assert len(result.auto_fixes[0].fixed_value) == 100

def test_probabilistic_contract():
    contract = compile_contract("""
    contract TestProb(priority = medium) {
        ensure_prob len(response) > 10, 0.8
            message: "80% should be substantial"
            window_size: 5
    }
    """)
    
    runtime = LLMCLRuntime()
    
    # Feed multiple responses to test probability
    responses = ['short', 'this is long enough', 'x', 'also long enough', 'y']
    
    for i, resp in enumerate(responses):
        result = runtime.validate(contract, {'response': resp})
        # Should pass until we have enough data points
```

### Error Handling

```python
from llm_contracts.core.exceptions import (
    ContractCompilationError,
    ContractValidationError,
    ConflictResolutionError
)

try:
    contract = compile_contract(contract_source)
    result = runtime.validate(contract, context)
    
    if not result.is_valid:
        # Handle validation failures
        for violation in result.violations:
            log_violation(violation)
        
        # Apply auto-fixes if available
        if result.auto_fixes:
            fixed_context = apply_fixes(context, result.auto_fixes)
            # Re-validate with fixes
            
except ContractCompilationError as e:
    logger.error(f"Contract compilation failed: {e}")
    # Handle syntax errors in contract
    
except ConflictResolutionError as e:
    logger.error(f"Cannot resolve conflicts: {e}")
    # Handle unresolvable contract conflicts
    
except Exception as e:
    logger.error(f"Unexpected validation error: {e}")
    # Fallback error handling
```

---

## Contributing

To contribute to LLMCL:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## License

LLMCL is released under the MIT License. See LICENSE file for details.