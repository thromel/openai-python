# LLM Contract Language (LLMCL)

A domain-specific language for defining and enforcing contracts on Large Language Model interactions. LLMCL provides a declarative syntax for specifying input validation, output constraints, temporal requirements, and probabilistic guarantees.

## Overview

LLMCL enables you to write contracts like:

```llmcl
contract ChatbotSafety(priority = critical) {
    require len(content) < 4000
        message: "Input too long"
    
    ensure not contains(response, "password")
        message: "Response contains sensitive info"
    
    ensure_prob json_valid(response), 0.95
        message: "95% of responses should be valid JSON"
    
    temporal always len(response) > 0
        message: "Never return empty responses"
}
```

## Key Features

### 1. **Declarative Contract Syntax**
- `require`: Input validation (preconditions)
- `ensure`: Output validation (postconditions)
- `ensure_prob`: Probabilistic guarantees
- `temporal`: Multi-turn conversation constraints

### 2. **Conflict Resolution**
- Automatic detection of conflicting contracts
- Multiple resolution strategies (first-wins, most-restrictive, merge)
- Priority-based precedence

### 3. **Auto-Remediation**
- Built-in fix suggestions
- Automatic response correction
- Customizable fix strategies

### 4. **Runtime Integration**
- Seamless OpenAI SDK integration
- Async validation support
- Real-time streaming validation

## Language Syntax

### Contract Declaration

```llmcl
contract ContractName(
    priority = high,              # critical, high, medium, low
    conflict_resolution = merge,  # first_wins, last_wins, most_restrictive, merge
    description = "Description"
) {
    # Contract body
}
```

### Require Clauses (Input Validation)

```llmcl
require condition
    message: "Error message"
    severity: "error"        # error, warning, info
    auto_fix: expression    # Optional fix suggestion
    tags: ["tag1", "tag2"]  # Optional tags
```

### Ensure Clauses (Output Validation)

```llmcl
ensure condition
    message: "Error message"
    severity: "error"
    auto_fix: expression
    tags: ["tag1", "tag2"]
```

### Probabilistic Ensures

```llmcl
ensure_prob condition, probability
    message: "Error message"
    window_size: 100        # Evaluation window
    tags: ["tag1", "tag2"]
```

### Temporal Constraints

```llmcl
temporal operator condition
    message: "Error message"
    tags: ["tag1", "tag2"]
```

Operators:
- `always`: Condition must always hold
- `eventually`: Condition must eventually be true
- `next`: Condition must be true in next turn
- `within N`: Condition must be true within N turns
- `until`: Condition true until another condition
- `since`: Condition true since another condition

### Built-in Functions

- `len(str)`: String length
- `contains(str, substring)`: Check substring
- `startswith(str, prefix)`: Check prefix
- `endswith(str, suffix)`: Check suffix
- `match(str, regex)`: Regex matching
- `json_valid(str)`: JSON validation
- `max(a, b)`, `min(a, b)`: Comparisons
- `abs(n)`: Absolute value

### Expressions

- Boolean: `and`, `or`, `not`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Membership: `in`
- Attribute access: `object.attribute`

## Usage Examples

### 1. Basic Safety Contract

```python
from llm_contracts.language import LLMCLRuntime

safety_contract = """
contract BasicSafety {
    require len(content) > 0
        message: "Input cannot be empty"
    
    ensure len(response) < 1000
        message: "Response too long"
}
"""

runtime = LLMCLRuntime()
await runtime.load_contract(safety_contract)
```

### 2. JSON API Contract

```python
api_contract = """
contract APIResponse {
    ensure json_valid(response)
        auto_fix: '{"error": "Invalid format"}'
    
    ensure contains(response, '"status"')
        message: "Missing status field"
}
"""
```

### 3. Conversation Quality Contract

```python
quality_contract = """
contract Quality(priority = high) {
    ensure_prob len(response) > 100, 0.7
        message: "70% of responses should be detailed"
    
    temporal within 3 contains(response, "help")
        message: "Offer help early in conversation"
}
"""
```

### 4. Conflict Resolution

```python
from llm_contracts.language import ResolutionStrategy

runtime = LLMCLRuntime()
context = runtime.create_context(
    "my_session",
    conflict_strategy=ResolutionStrategy.MOST_RESTRICTIVE
)
```

### 5. Integration with OpenAI

```python
from llm_contracts.providers import ImprovedOpenAIProvider

# Load contracts
contract_text = "contract Safety { ... }"
compiled = runtime.compiler.compile(contract_text)

# Add to OpenAI provider
client = ImprovedOpenAIProvider(api_key="...")
client.add_output_contract(compiled.contract_instance)

# Use normally - contracts enforced automatically
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Advanced Features

### 1. Contract Composition

```llmcl
import base_contracts

contract Extended extends BaseContract {
    # Additional constraints
    ensure custom_validation(response)
}
```

### 2. Conditional Contracts

```llmcl
contract Conditional {
    if context.user_type == "premium" {
        ensure len(response) > 200
    } else {
        ensure len(response) > 50
    }
}
```

### 3. Custom Variables

Access context variables in conditions:
- `content`: Input text
- `response`: Output text
- `context.*`: Custom context variables
- `context.conversation_history`: Previous turns

### 4. Contract Testing

```python
# Test framework support
from llm_contracts.language import LLMCLCompiler

compiler = LLMCLCompiler()
compiled = compiler.compile(contract_text)

# Test validation
result = await compiled.contract_instance.validate(
    "test response",
    {"content": "test input"}
)
assert result.is_valid
```

## Architecture

### Components

1. **Parser** (`parser.py`): Tokenizes and parses LLMCL syntax into AST
2. **AST Nodes** (`ast_nodes.py`): Abstract syntax tree representation
3. **Compiler** (`compiler.py`): Compiles AST into executable contracts
4. **Runtime** (`runtime.py`): Executes contracts with conflict resolution
5. **Conflict Resolver** (`conflict_resolver.py`): Handles contract conflicts

### Conflict Resolution Strategies

- **FIRST_WINS**: First registered contract takes precedence
- **LAST_WINS**: Last registered contract overrides
- **MOST_RESTRICTIVE**: Most constraining contract wins
- **LEAST_RESTRICTIVE**: Most permissive contract wins
- **MERGE**: Attempt to merge compatible constraints
- **FAIL_ON_CONFLICT**: Raise error on conflicts

## Performance Considerations

1. **Lazy Loading**: Contracts loaded on-demand
2. **Caching**: Validation results cached
3. **Async Execution**: Non-blocking validation
4. **Circuit Breaker**: Automatic degradation on failures

## Best Practices

1. **Use Specific Messages**: Provide clear error messages for debugging
2. **Set Appropriate Priorities**: Use priority levels to resolve conflicts
3. **Test Contracts**: Validate contracts with test cases
4. **Monitor Metrics**: Track validation performance and violations
5. **Use Auto-Fix Sparingly**: Only for safe, deterministic fixes

## Future Enhancements

1. **Visual Contract Editor**: GUI for contract creation
2. **Contract Marketplace**: Share and reuse contracts
3. **Static Analysis**: Compile-time contract verification
4. **IDE Support**: Syntax highlighting and auto-completion
5. **Contract Versioning**: Track contract evolution