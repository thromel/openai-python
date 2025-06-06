# LLMCL API Reference and Examples

This document provides comprehensive API documentation and practical examples for using LLMCL in your applications.

## Table of Contents

1. [Python API Overview](#python-api-overview)
2. [Core Classes](#core-classes)
3. [Compilation API](#compilation-api)
4. [Runtime API](#runtime-api)
5. [Validation Results](#validation-results)
6. [Configuration API](#configuration-api)
7. [Utilities and Helpers](#utilities-and-helpers)
8. [Complete Examples](#complete-examples)
9. [Integration Patterns](#integration-patterns)
10. [Error Handling](#error-handling)
11. [Advanced Usage](#advanced-usage)

## Python API Overview

The LLMCL Python API provides a clean interface for compiling contracts, validating data, and managing the runtime environment.

### Basic Import Structure

```python
# Core functionality
from llm_contracts.language import (
    LLMCLRuntime,
    compile_contract,
    parse_contract,
    validate_syntax
)

# Configuration and customization
from llm_contracts.language.conflict_resolver import ConflictResolver
from llm_contracts.language.auto_fix import AutoFixManager
from llm_contracts.core.exceptions import (
    ContractCompilationError,
    ContractValidationError,
    ConflictResolutionError
)

# Utilities
from llm_contracts.utils.telemetry import get_metrics
from llm_contracts.validators import (
    InputValidator,
    OutputValidator
)
```

## Core Classes

### LLMCLRuntime

The main runtime class for executing contracts and managing validation.

```python
class LLMCLRuntime:
    def __init__(self, 
                 conflict_resolver: Optional[ConflictResolver] = None,
                 auto_fix_manager: Optional[AutoFixManager] = None,
                 telemetry_enabled: bool = True,
                 cache_enabled: bool = True,
                 debug_mode: bool = False):
        """
        Initialize LLMCL runtime.
        
        Args:
            conflict_resolver: Custom conflict resolution strategy
            auto_fix_manager: Custom auto-fix behavior
            telemetry_enabled: Enable metrics collection
            cache_enabled: Enable validation result caching
            debug_mode: Enable detailed debugging information
        """

    def validate(self, 
                 contract: Contract, 
                 context: Dict[str, Any]) -> ValidationResult:
        """
        Validate context against a single contract.
        
        Args:
            contract: Compiled contract to validate against
            context: Data context for validation
            
        Returns:
            ValidationResult with success/failure and details
        """

    async def validate_async(self, 
                            contract: Contract, 
                            context: Dict[str, Any]) -> ValidationResult:
        """
        Asynchronous validation for high-throughput scenarios.
        """

    def validate_multiple(self, 
                         contracts: List[Contract], 
                         context: Dict[str, Any]) -> ValidationResult:
        """
        Validate context against multiple contracts.
        """

    def compile_and_validate(self, 
                            source: str, 
                            context: Dict[str, Any]) -> ValidationResult:
        """
        Compile contract source and validate in one step.
        """

    def add_contract(self, contract: Contract) -> None:
        """Add a contract to the runtime for reuse."""

    def remove_contract(self, contract_name: str) -> None:
        """Remove a contract by name."""

    def get_statistics(self) -> RuntimeStatistics:
        """Get comprehensive runtime statistics."""

    def clear_cache(self) -> None:
        """Clear validation result cache."""
```

### Contract

Represents a compiled LLMCL contract.

```python
class Contract:
    @property
    def name(self) -> str:
        """Contract name."""

    @property
    def priority(self) -> str:
        """Contract priority level."""

    @property
    def clauses(self) -> List[Clause]:
        """List of contract clauses."""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Contract parameters."""

    @property
    def dependencies(self) -> Set[str]:
        """Set of variable dependencies."""

    def get_clause_by_type(self, clause_type: str) -> List[Clause]:
        """Get clauses of specific type (require, ensure, etc.)."""

    def has_temporal_clauses(self) -> bool:
        """Check if contract contains temporal logic."""

    def has_probabilistic_clauses(self) -> bool:
        """Check if contract contains probabilistic constraints."""
```

### ValidationResult

Contains the results of contract validation.

```python
class ValidationResult:
    @property
    def is_valid(self) -> bool:
        """True if validation passed."""

    @property
    def violations(self) -> List[Violation]:
        """List of contract violations."""

    @property
    def auto_fixes(self) -> List[AutoFix]:
        """Available auto-fixes for violations."""

    @property
    def statistics(self) -> ValidationStatistics:
        """Detailed validation statistics."""

    @property
    def execution_time(self) -> float:
        """Validation execution time in seconds."""

    @property
    def conflicts_detected(self) -> List[Conflict]:
        """Any conflicts detected during validation."""

    def apply_auto_fixes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all auto-fixes to context and return updated context."""

    def get_violations_by_priority(self, priority: str) -> List[Violation]:
        """Get violations of specific priority level."""

    def get_violations_by_type(self, violation_type: str) -> List[Violation]:
        """Get violations of specific type."""
```

## Compilation API

### Basic Compilation

```python
def compile_contract(source: str, 
                    options: Optional[CompilationOptions] = None) -> Contract:
    """
    Compile LLMCL source code to executable contract.
    
    Args:
        source: LLMCL source code string
        options: Compilation options
        
    Returns:
        Compiled Contract object
        
    Raises:
        ContractCompilationError: If compilation fails
    """

def parse_contract(source: str) -> AST:
    """
    Parse LLMCL source to Abstract Syntax Tree.
    
    Args:
        source: LLMCL source code string
        
    Returns:
        AST representation of the contract
        
    Raises:
        SyntaxError: If parsing fails
    """

def validate_syntax(source: str) -> List[SyntaxError]:
    """
    Validate LLMCL syntax without full compilation.
    
    Args:
        source: LLMCL source code string
        
    Returns:
        List of syntax errors (empty if valid)
    """
```

### Advanced Compilation

```python
from llm_contracts.language.compiler import CompilationOptions

# Custom compilation options
options = CompilationOptions(
    optimize=True,
    strict_mode=True,
    allow_custom_functions=False,
    max_expression_depth=10,
    enable_type_checking=True
)

contract = compile_contract(source, options)

# Compile with custom context
compile_context = {
    'api_version': '2.0',
    'environment': 'production',
    'custom_functions': ['validate_email', 'check_profanity']
}

contract = compile_contract(source, options, compile_context)
```

## Runtime API

### Basic Runtime Usage

```python
# Initialize runtime
runtime = LLMCLRuntime()

# Compile contract
contract_source = """
contract Example(priority = high) {
    require len(content) > 0
        message: "Input required"
    
    ensure len(response) >= 20
        message: "Response should be substantial"
        auto_fix: response + " Please let me know if you need more information."
}
"""

contract = compile_contract(contract_source)

# Validate data
context = {
    'content': 'User input here',
    'response': 'Short reply'
}

result = runtime.validate(contract, context)

if not result.is_valid:
    print("Validation failed:")
    for violation in result.violations:
        print(f"- {violation.message}")
    
    # Apply auto-fixes
    if result.auto_fixes:
        fixed_context = result.apply_auto_fixes(context)
        # Re-validate with fixes
        result = runtime.validate(contract, fixed_context)
```

### Asynchronous Validation

```python
import asyncio

async def validate_batch(runtime, contract, contexts):
    """Validate multiple contexts asynchronously."""
    tasks = [
        runtime.validate_async(contract, context) 
        for context in contexts
    ]
    return await asyncio.gather(*tasks)

# Usage
contexts = [
    {'content': 'Input 1', 'response': 'Response 1'},
    {'content': 'Input 2', 'response': 'Response 2'},
    # ... more contexts
]

results = asyncio.run(validate_batch(runtime, contract, contexts))
```

### Runtime Configuration

```python
from llm_contracts.language.conflict_resolver import ConflictResolver
from llm_contracts.language.auto_fix import AutoFixManager

# Configure conflict resolution
conflict_resolver = ConflictResolver(
    default_strategy='MOST_RESTRICTIVE',
    priority_override=True
)

# Configure auto-fix behavior
auto_fix_manager = AutoFixManager(
    default_strategy='BEST_FIX',
    confidence_threshold=0.7,
    max_attempts=3
)

# Create configured runtime
runtime = LLMCLRuntime(
    conflict_resolver=conflict_resolver,
    auto_fix_manager=auto_fix_manager,
    telemetry_enabled=True,
    cache_enabled=True
)
```

## Validation Results

### Detailed Result Analysis

```python
# Perform validation
result = runtime.validate(contract, context)

# Basic result info
print(f"Valid: {result.is_valid}")
print(f"Execution time: {result.execution_time:.3f}s")
print(f"Violations: {len(result.violations)}")
print(f"Auto-fixes available: {len(result.auto_fixes)}")

# Analyze violations
for violation in result.violations:
    print(f"Contract: {violation.contract_name}")
    print(f"Clause: {violation.clause_type}")
    print(f"Priority: {violation.priority}")
    print(f"Message: {violation.message}")
    print(f"Context: {violation.context}")

# Analyze auto-fixes
for fix in result.auto_fixes:
    print(f"Fix for: {fix.violation.message}")
    print(f"Expression: {fix.fix_expression}")
    print(f"Preview: {fix.preview}")
    print(f"Confidence: {fix.confidence}")

# Statistics
stats = result.statistics
print(f"Clauses evaluated: {stats.clauses_evaluated}")
print(f"Conflicts resolved: {stats.conflicts_resolved}")
print(f"Cache hits: {stats.cache_hits}")
```

### Violation and Fix Classes

```python
class Violation:
    @property
    def contract_name(self) -> str:
        """Name of the contract that was violated."""

    @property
    def clause_type(self) -> str:
        """Type of clause (require, ensure, etc.)."""

    @property
    def message(self) -> str:
        """Human-readable violation message."""

    @property
    def priority(self) -> str:
        """Priority level of the violation."""

    @property
    def context(self) -> Dict[str, Any]:
        """Context data when violation occurred."""

    @property
    def actual_value(self) -> Any:
        """Actual value that caused violation."""

    @property
    def expected_value(self) -> Any:
        """Expected value according to contract."""

class AutoFix:
    @property
    def violation(self) -> Violation:
        """The violation this fix addresses."""

    @property
    def fix_expression(self) -> str:
        """LLMCL expression used for fixing."""

    @property
    def preview(self) -> str:
        """Preview of the fixed value."""

    @property
    def confidence(self) -> float:
        """Confidence score (0.0 to 1.0)."""

    @property
    def fix_type(self) -> str:
        """Type of fix (content, format, length, etc.)."""

    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this fix to the context."""
```

## Configuration API

### ConflictResolver Configuration

```python
from llm_contracts.language.conflict_resolver import ConflictResolver

# Basic resolver
resolver = ConflictResolver(
    default_strategy='MOST_RESTRICTIVE'
)

# Advanced resolver with custom strategies
resolver = ConflictResolver(
    default_strategy='MERGE',
    priority_override=True,
    strategy_mapping={
        'security': 'FAIL_ON_CONFLICT',
        'format': 'MERGE',
        'length': 'MOST_RESTRICTIVE',
        'content': 'LEAST_RESTRICTIVE'
    },
    custom_strategies={
        'business_logic': custom_business_resolver
    }
)

# Register custom resolution function
def custom_business_resolver(conflicts):
    # Custom business logic for conflict resolution
    if any(c.involves_api_format() for c in conflicts):
        return resolve_api_conflicts(conflicts)
    return apply_default_resolution(conflicts)

resolver.register_strategy('business_logic', custom_business_resolver)
```

### AutoFixManager Configuration

```python
from llm_contracts.language.auto_fix import AutoFixManager

# Basic auto-fix manager
fix_manager = AutoFixManager(
    default_strategy='FIRST_FIX'
)

# Advanced configuration
fix_manager = AutoFixManager(
    default_strategy='BEST_FIX',
    confidence_threshold=0.6,
    max_attempts=3,
    enable_cascading=True,
    custom_fix_functions={
        'smart_json_format': smart_json_formatter,
        'content_sanitizer': content_sanitizer
    }
)

# Register custom fix function
@fix_manager.register_function
def smart_truncate(text, max_length, preserve_words=True):
    """Smart truncation that preserves word boundaries."""
    if len(text) <= max_length:
        return text
    
    if preserve_words:
        # Find last complete word within limit
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # At least 80% of desired length
            return truncated[:last_space] + "..."
    
    return text[:max_length-3] + "..."
```

## Utilities and Helpers

### Telemetry and Monitoring

```python
from llm_contracts.utils.telemetry import (
    get_metrics,
    configure_telemetry,
    export_metrics
)

# Get current metrics
metrics = get_metrics()
print(f"Total validations: {metrics.total_validations}")
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Average latency: {metrics.avg_latency_ms:.1f}ms")

# Configure telemetry
configure_telemetry(
    enable_prometheus=True,
    enable_opentelemetry=True,
    custom_tags={'service': 'llm-validator', 'env': 'production'}
)

# Export metrics
export_metrics('metrics.json', format='json')
export_metrics('metrics.csv', format='csv')
```

### Input/Output Validators

```python
from llm_contracts.validators import InputValidator, OutputValidator

# Input validation
input_validator = InputValidator([
    'input_security_contract',
    'input_format_contract'
])

# Validate user input before processing
input_result = input_validator.validate({'content': user_input})
if not input_result.is_valid:
    return handle_invalid_input(input_result)

# Output validation
output_validator = OutputValidator([
    'output_quality_contract',
    'output_format_contract'
])

# Validate LLM response before returning
output_result = output_validator.validate({
    'content': user_input,
    'response': llm_response
})

if not output_result.is_valid and output_result.auto_fixes:
    llm_response = output_result.apply_auto_fixes({'response': llm_response})['response']
```

## Complete Examples

### Simple Chatbot Validation

```python
from llm_contracts.language import LLMCLRuntime, compile_contract

def create_chatbot_validator():
    """Create a complete chatbot validation system."""
    
    # Define contracts
    contracts_source = """
    contract InputSafety(priority = critical) {
        require len(content) > 0
            message: "Input cannot be empty"
        
        require len(content) <= 4000
            message: "Input too long"
            auto_fix: content[:4000]
        
        require not contains(lower(content), "password")
            message: "Input contains sensitive information"
    }
    
    contract OutputQuality(priority = high) {
        ensure len(response) >= 10
            message: "Response too short"
            auto_fix: response + " Is there anything else I can help you with?"
        
        ensure len(response) <= 2000
            message: "Response too long"
            auto_fix: response[:1950] + "... [truncated]"
        
        ensure not starts_with(response, "I don't know")
            message: "Should provide helpful responses"
            auto_fix: "Let me help you with that. " + response[12:]
    }
    
    contract ConversationFlow(priority = medium) {
        temporal always response != prev_response
            message: "Should not repeat responses"
        
        temporal within 5 contains(response, "help") or contains(response, "assist")
            message: "Should offer help within 5 turns"
            auto_fix: response + " How can I assist you further?"
    }
    """
    
    # Compile contracts
    contracts = compile_contract(contracts_source)
    
    # Initialize runtime
    runtime = LLMCLRuntime(
        telemetry_enabled=True,
        cache_enabled=True
    )
    
    return runtime, contracts

def process_conversation_turn(runtime, contracts, user_input, llm_response, prev_response=None):
    """Process a single conversation turn with validation."""
    
    # Validate input
    input_context = {'content': user_input}
    input_result = runtime.validate(contracts, input_context)
    
    if not input_result.is_valid:
        if input_result.auto_fixes:
            # Apply auto-fixes to input
            fixed_input = input_result.apply_auto_fixes(input_context)
            user_input = fixed_input['content']
        else:
            return None, "I cannot process this input."
    
    # Validate output
    output_context = {
        'content': user_input,
        'response': llm_response,
        'prev_response': prev_response
    }
    
    output_result = runtime.validate(contracts, output_context)
    
    if not output_result.is_valid:
        if output_result.auto_fixes:
            # Apply auto-fixes to output
            fixed_output = output_result.apply_auto_fixes(output_context)
            llm_response = fixed_output['response']
        else:
            llm_response = "I apologize, but I cannot provide a proper response."
    
    return user_input, llm_response

# Usage example
runtime, contracts = create_chatbot_validator()

conversation_history = []
prev_response = None

while True:
    user_input = input("User: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    
    # Simulate LLM response (replace with actual LLM call)
    llm_response = simulate_llm_response(user_input)
    
    # Validate and potentially fix
    validated_input, validated_response = process_conversation_turn(
        runtime, contracts, user_input, llm_response, prev_response
    )
    
    print(f"Bot: {validated_response}")
    
    conversation_history.append({
        'user': validated_input,
        'bot': validated_response
    })
    prev_response = validated_response

# Get final statistics
stats = runtime.get_statistics()
print(f"\nConversation statistics:")
print(f"Validation success rate: {stats.success_rate:.2%}")
print(f"Auto-fixes applied: {stats.auto_fixes_applied}")
```

### API Response Validation

```python
from llm_contracts.language import LLMCLRuntime, compile_contract
import json
from typing import Dict, Any

def create_api_validator():
    """Create API response validation system."""
    
    api_contracts = """
    contract APIResponseFormat(priority = critical) {
        ensure json_valid(response)
            message: "API response must be valid JSON"
            auto_fix: '{"error": "Invalid response format", "data": "' + response + '"}'
        
        ensure contains(response, '"status"')
            message: "API response must include status field"
            auto_fix: replace(response, '}', ', "status": "success"}')
        
        ensure if contains(response, '"error"') then contains(response, '"code"')
            message: "Error responses must include error code"
            auto_fix: replace(response, '"error":', '"error": {"code": 500, "message":') + '}'
    }
    
    contract APIPerformance(priority = high) {
        require processing_time_ms <= 5000
            message: "API response time exceeds 5 second limit"
        
        ensure_prob processing_time_ms <= 1000, 0.9
            message: "90% of API responses should complete under 1 second"
            window_size: 100
    }
    
    contract APIContent(priority = medium) {
        ensure if contains(response, '"data"') then len(json_extract(response, "data")) > 0
            message: "Data field should not be empty"
            auto_fix: replace(response, '"data": []', '"data": {"message": "No data available"}')
        
        ensure not contains(response, '"password"')
            message: "API responses must not contain password fields"
            auto_fix: json_remove_field(response, "password")
    }
    """
    
    contracts = compile_contract(api_contracts)
    runtime = LLMCLRuntime()
    
    return runtime, contracts

class APIValidator:
    def __init__(self):
        self.runtime, self.contracts = create_api_validator()
    
    def validate_api_response(self, response_data: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate API response and return result."""
        
        context = {
            'response': response_data,
            'processing_time_ms': metadata.get('processing_time_ms', 0),
            'endpoint': metadata.get('endpoint', ''),
            'method': metadata.get('method', 'GET')
        }
        
        result = self.runtime.validate(self.contracts, context)
        
        if not result.is_valid:
            # Apply auto-fixes if available
            if result.auto_fixes:
                fixed_context = result.apply_auto_fixes(context)
                response_data = fixed_context['response']
                
                # Re-validate after fixes
                result = self.runtime.validate(self.contracts, fixed_context)
        
        return {
            'response': response_data,
            'is_valid': result.is_valid,
            'violations': [v.message for v in result.violations],
            'fixes_applied': len(result.auto_fixes) if result.auto_fixes else 0,
            'execution_time': result.execution_time
        }

# Usage in API endpoint
validator = APIValidator()

def api_endpoint(request_data):
    import time
    start_time = time.time()
    
    # Process request (simulate)
    response_data = process_request(request_data)
    
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Validate response
    validation_result = validator.validate_api_response(
        response_data,
        {
            'processing_time_ms': processing_time,
            'endpoint': '/api/v1/data',
            'method': 'GET'
        }
    )
    
    if not validation_result['is_valid']:
        # Log validation issues
        logger.warning(f"API validation issues: {validation_result['violations']}")
    
    return validation_result['response']
```

### Batch Processing with Validation

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from llm_contracts.language import LLMCLRuntime, compile_contract

class BatchValidator:
    def __init__(self, max_workers=10):
        self.runtime = LLMCLRuntime()
        self.contracts = self._load_contracts()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def _load_contracts(self):
        contracts_source = """
        contract BatchProcessing(priority = high) {
            require len(content) > 0
                message: "Content cannot be empty"
            
            ensure len(response) >= 20
                message: "Response should be substantial"
                auto_fix: response + " [Auto-generated content to meet minimum length]"
            
            ensure_prob processing_successful, 0.95
                message: "95% of batch items should process successfully"
                window_size: 100
        }
        """
        return compile_contract(contracts_source)
    
    async def validate_batch_async(self, batch_items):
        """Process and validate a batch of items asynchronously."""
        
        async def validate_item(item):
            try:
                result = await self.runtime.validate_async(self.contracts, item)
                
                if not result.is_valid and result.auto_fixes:
                    # Apply auto-fixes
                    fixed_item = result.apply_auto_fixes(item)
                    # Re-validate
                    result = await self.runtime.validate_async(self.contracts, fixed_item)
                    return {'item': fixed_item, 'result': result, 'fixed': True}
                
                return {'item': item, 'result': result, 'fixed': False}
                
            except Exception as e:
                return {'item': item, 'error': str(e), 'fixed': False}
        
        # Process all items concurrently
        tasks = [validate_item(item) for item in batch_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def validate_batch_sync(self, batch_items):
        """Process and validate a batch of items synchronously."""
        
        def validate_item(item):
            try:
                result = self.runtime.validate(self.contracts, item)
                
                if not result.is_valid and result.auto_fixes:
                    fixed_item = result.apply_auto_fixes(item)
                    result = self.runtime.validate(self.contracts, fixed_item)
                    return {'item': fixed_item, 'result': result, 'fixed': True}
                
                return {'item': item, 'result': result, 'fixed': False}
                
            except Exception as e:
                return {'item': item, 'error': str(e), 'fixed': False}
        
        # Process using thread pool
        results = list(self.executor.map(validate_item, batch_items))
        return results

# Usage example
async def process_large_dataset():
    validator = BatchValidator(max_workers=20)
    
    # Sample batch data
    batch_items = [
        {
            'content': f'Input item {i}',
            'response': f'Response {i}',
            'processing_successful': i % 20 != 0  # 95% success rate
        }
        for i in range(1000)
    ]
    
    # Process asynchronously
    results = await validator.validate_batch_async(batch_items)
    
    # Analyze results
    valid_count = sum(1 for r in results if r.get('result', {}).is_valid)
    fixed_count = sum(1 for r in results if r.get('fixed', False))
    error_count = sum(1 for r in results if 'error' in r)
    
    print(f"Processed {len(batch_items)} items:")
    print(f"Valid: {valid_count}")
    print(f"Fixed: {fixed_count}")
    print(f"Errors: {error_count}")
    print(f"Success rate: {(valid_count / len(batch_items)):.2%}")

# Run the batch processing
asyncio.run(process_large_dataset())
```

## Integration Patterns

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException, Request
from llm_contracts.language import LLMCLRuntime, compile_contract
import time

app = FastAPI()

# Initialize LLMCL validator
runtime = LLMCLRuntime()
contracts = compile_contract("""
contract APIValidation(priority = high) {
    require len(content) > 0
        message: "Request content cannot be empty"
    
    ensure json_valid(response)
        message: "Response must be valid JSON"
        auto_fix: '{"data": ' + response + ', "status": "success"}'
    
    ensure processing_time_ms <= 5000
        message: "Response time should be under 5 seconds"
}
""")

@app.middleware("http")
async def validate_responses(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    processing_time = (time.time() - start_time) * 1000
    
    # Extract response content for validation
    response_content = getattr(response, 'body', '')
    
    # Validate response
    validation_context = {
        'content': getattr(request, 'body', ''),
        'response': response_content,
        'processing_time_ms': processing_time
    }
    
    result = runtime.validate(contracts, validation_context)
    
    if not result.is_valid:
        # Log validation issues
        app.logger.warning(f"Response validation failed: {result.violations}")
        
        # Apply auto-fixes if available
        if result.auto_fixes:
            fixed_context = result.apply_auto_fixes(validation_context)
            # Update response with fixed content
            response.body = fixed_context['response']
    
    return response

@app.post("/chat")
async def chat_endpoint(request: dict):
    # Your LLM processing logic here
    user_input = request.get('message', '')
    llm_response = process_with_llm(user_input)
    
    return {"response": llm_response}
```

### Flask Integration

```python
from flask import Flask, request, jsonify, g
from llm_contracts.language import LLMCLRuntime, compile_contract
import time
import functools

app = Flask(__name__)

# Initialize validator
runtime = LLMCLRuntime()
contracts = compile_contract("""
contract FlaskValidation(priority = high) {
    require len(content) > 0
        message: "Request content required"
    
    ensure json_valid(response)
        message: "Response must be JSON"
        auto_fix: '{"error": "Invalid response", "data": "' + response + '"}'
}
""")

def validate_llm_response(f):
    """Decorator for validating LLM responses."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        # Get request content
        content = request.get_json() or {}
        
        # Call the original function
        response = f(*args, **kwargs)
        
        # Validate response
        processing_time = (time.time() - start_time) * 1000
        
        validation_context = {
            'content': str(content),
            'response': str(response),
            'processing_time_ms': processing_time
        }
        
        result = runtime.validate(contracts, validation_context)
        
        if not result.is_valid:
            app.logger.warning(f"Validation failed: {result.violations}")
            
            if result.auto_fixes:
                fixed_context = result.apply_auto_fixes(validation_context)
                # Update response
                try:
                    import json
                    response = json.loads(fixed_context['response'])
                except:
                    response = {"error": "Response validation failed"}
        
        return response
    
    return decorated_function

@app.route('/api/chat', methods=['POST'])
@validate_llm_response
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    # Process with LLM
    llm_response = process_with_llm(user_message)
    
    return jsonify({"response": llm_response})
```

## Error Handling

### Exception Hierarchy

```python
from llm_contracts.core.exceptions import (
    LLMCLException,          # Base exception
    ContractCompilationError, # Compilation failures
    ContractValidationError,  # Runtime validation errors
    ConflictResolutionError,  # Conflict resolution failures
    AutoFixError,            # Auto-fix application errors
    TemporalLogicError       # Temporal validation errors
)

def safe_contract_execution(contract_source, context):
    """Demonstrate comprehensive error handling."""
    try:
        # Compile contract
        contract = compile_contract(contract_source)
        
        # Validate
        runtime = LLMCLRuntime()
        result = runtime.validate(contract, context)
        
        return result
        
    except ContractCompilationError as e:
        print(f"Compilation failed: {e.message}")
        print(f"Line {e.line_number}: {e.line_content}")
        print(f"Error details: {e.details}")
        return None
        
    except ConflictResolutionError as e:
        print(f"Conflict resolution failed: {e.message}")
        print(f"Conflicting contracts: {e.conflicting_contracts}")
        print(f"Suggested strategy: {e.suggested_strategy}")
        return None
        
    except ContractValidationError as e:
        print(f"Validation error: {e.message}")
        print(f"Context: {e.context}")
        return None
        
    except AutoFixError as e:
        print(f"Auto-fix failed: {e.message}")
        print(f"Fix expression: {e.fix_expression}")
        print(f"Original violation: {e.original_violation}")
        return None
        
    except LLMCLException as e:
        print(f"General LLMCL error: {e.message}")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None
```

### Graceful Degradation

```python
class RobustValidator:
    def __init__(self, primary_contracts, fallback_contracts=None):
        self.runtime = LLMCLRuntime()
        self.primary_contracts = primary_contracts
        self.fallback_contracts = fallback_contracts or []
    
    def validate_with_fallback(self, context):
        """Validate with fallback to simpler contracts if primary fails."""
        
        # Try primary contracts
        try:
            result = self.runtime.validate(self.primary_contracts, context)
            return result, 'primary'
            
        except ConflictResolutionError:
            # Try with most restrictive strategy
            try:
                runtime = LLMCLRuntime(
                    conflict_resolver=ConflictResolver(strategy='MOST_RESTRICTIVE')
                )
                result = runtime.validate(self.primary_contracts, context)
                return result, 'primary_with_conflict_resolution'
                
            except Exception:
                pass
        
        except Exception as e:
            self.logger.warning(f"Primary validation failed: {e}")
        
        # Fall back to simpler contracts
        if self.fallback_contracts:
            try:
                result = self.runtime.validate(self.fallback_contracts, context)
                return result, 'fallback'
            except Exception as e:
                self.logger.error(f"Fallback validation failed: {e}")
        
        # Ultimate fallback: basic validation
        return self._basic_validation(context), 'basic'
    
    def _basic_validation(self, context):
        """Basic validation when all else fails."""
        violations = []
        
        # Basic checks
        if not context.get('content'):
            violations.append(Violation(
                contract_name='basic',
                message='Content is required',
                priority='critical'
            ))
        
        if not context.get('response'):
            violations.append(Violation(
                contract_name='basic',
                message='Response is required',
                priority='critical'
            ))
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations
        )
```

## Advanced Usage

### Custom Function Registration

```python
from llm_contracts.language.runtime import register_function

@register_function
def custom_email_validator(email):
    """Custom email validation function."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

@register_function
def sentiment_score(text):
    """Calculate sentiment score for text."""
    # Integrate with sentiment analysis library
    from your_sentiment_lib import analyze_sentiment
    return analyze_sentiment(text).compound_score

# Use in contracts
contract_with_custom_functions = """
contract CustomValidation(priority = medium) {
    ensure if contains(content, "@") then custom_email_validator(content)
        message: "Email format is invalid"
    
    ensure sentiment_score(response) >= -0.1
        message: "Response should not be negative"
        auto_fix: improve_sentiment(response)
}
"""
```

### Plugin System

```python
from llm_contracts.language.plugins import Plugin, register_plugin

class SecurityPlugin(Plugin):
    def __init__(self):
        super().__init__('security')
    
    def get_functions(self):
        return {
            'check_pii': self.check_pii,
            'sanitize_data': self.sanitize_data,
            'encrypt_sensitive': self.encrypt_sensitive
        }
    
    def check_pii(self, text):
        """Check for personally identifiable information."""
        import re
        
        # Check for common PII patterns
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        
        return not (re.search(ssn_pattern, text) or re.search(phone_pattern, text))
    
    def sanitize_data(self, text):
        """Remove sensitive information."""
        import re
        
        # Replace SSNs
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
        
        # Replace phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)
        
        return text

# Register plugin
register_plugin(SecurityPlugin())

# Use plugin functions in contracts
security_contract = """
contract SecurityValidation(priority = critical) {
    ensure check_pii(response)
        message: "Response contains PII"
        auto_fix: sanitize_data(response)
}
"""
```

### Machine Learning Integration

```python
from llm_contracts.language.ml_integration import MLPredictor

class QualityPredictor(MLPredictor):
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
    
    def predict_quality(self, text):
        """Predict text quality score using ML model."""
        features = self.extract_features(text)
        return self.model.predict([features])[0]
    
    def predict_toxicity(self, text):
        """Predict toxicity score."""
        # Use pre-trained toxicity model
        return self.toxicity_model.predict(text)

# Register ML functions
quality_predictor = QualityPredictor('quality_model.pkl')

@register_function
def ml_quality_score(text):
    return quality_predictor.predict_quality(text)

@register_function
def ml_toxicity_score(text):
    return quality_predictor.predict_toxicity(text)

# Use in contracts
ml_contract = """
contract MLValidation(priority = high) {
    ensure ml_quality_score(response) >= 0.7
        message: "Response quality below threshold"
        auto_fix: improve_response_quality(response)
    
    ensure ml_toxicity_score(response) <= 0.1
        message: "Response may be toxic"
        auto_fix: remove_toxic_content(response)
}
"""
```

This comprehensive API reference and examples guide provides everything needed to integrate LLMCL into your applications effectively. The examples demonstrate real-world usage patterns and best practices for production systems.