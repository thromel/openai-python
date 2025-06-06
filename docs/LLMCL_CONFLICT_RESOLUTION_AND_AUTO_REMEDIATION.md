# LLMCL Conflict Resolution and Auto-Remediation

This document provides comprehensive guidance on conflict resolution strategies and auto-remediation capabilities in LLMCL.

## Table of Contents

1. [Conflict Resolution Overview](#conflict-resolution-overview)
2. [Types of Conflicts](#types-of-conflicts)
3. [Resolution Strategies](#resolution-strategies)
4. [Priority-Based Resolution](#priority-based-resolution)
5. [Conflict Detection](#conflict-detection)
6. [Auto-Remediation Framework](#auto-remediation-framework)
7. [Fix Strategies](#fix-strategies)
8. [Advanced Auto-Fix Patterns](#advanced-auto-fix-patterns)
9. [Configuration and Customization](#configuration-and-customization)
10. [Monitoring and Debugging](#monitoring-and-debugging)

## Conflict Resolution Overview

LLMCL's conflict resolution system automatically handles situations where multiple contracts have conflicting requirements. This ensures that your validation pipeline remains robust even when contracts evolve independently or when different teams contribute conflicting rules.

### Why Conflicts Occur

```llmcl
// Example conflict scenario
contract APIFormat(priority = high) {
    ensure json_valid(response)
        message: "API responses must be JSON"
        auto_fix: '{"content": "' + response + '"}'
}

contract UserFriendly(priority = medium) {
    ensure not contains(response, "{")
        message: "User responses should be plain text"
        auto_fix: extract_content_from_json(response)
}

// These contracts conflict: one requires JSON, other forbids it
```

### Conflict Resolution Benefits

- **Automatic Handling**: No manual intervention required
- **Predictable Behavior**: Consistent resolution across all conflicts
- **Priority Awareness**: Higher priority contracts take precedence
- **Strategy Flexibility**: Multiple resolution approaches available
- **Graceful Degradation**: System continues operating despite conflicts

## Types of Conflicts

### 1. Format Conflicts

Conflicts between different format requirements:

```llmcl
contract JSONRequired(priority = high) {
    ensure json_valid(response)
        message: "Must be valid JSON"
        auto_fix: '{"content": "' + response + '"}'
}

contract PlainTextRequired(priority = medium) {
    ensure not contains(response, "{") and not contains(response, "}")
        message: "Must be plain text"
        auto_fix: extract_text_content(response)
}

// Conflict: JSON format vs. plain text format
```

### 2. Length Conflicts

Conflicting length requirements:

```llmcl
contract MinimumLength(priority = medium) {
    ensure len(response) >= 100
        message: "Response must be detailed"
        auto_fix: response + " Please let me know if you need more information."
}

contract MaximumLength(priority = high) {
    ensure len(response) <= 50
        message: "Response must be concise"
        auto_fix: response[:50]
}

// Conflict: Minimum 100 chars vs. Maximum 50 chars
```

### 3. Content Conflicts

Conflicting content requirements:

```llmcl
contract RequireHelp(priority = medium) {
    ensure contains(response, "help")
        message: "Must offer help"
        auto_fix: response + " How can I help you?"
}

contract ForbidHelp(priority = low) {
    ensure not contains(response, "help")
        message: "Should not mention help"
        auto_fix: replace(response, "help", "assist")
}

// Conflict: Require vs. forbid "help" mention
```

### 4. Temporal Conflicts

Conflicting temporal requirements:

```llmcl
contract ImmediateResponse(priority = high) {
    temporal next contains(response, "immediate")
        message: "Next response must be immediate"
}

contract DelayedResponse(priority = medium) {
    temporal within 3 not contains(response, "immediate")
        message: "Should not respond immediately"
}

// Conflict: Immediate vs. delayed response timing
```

### 5. Semantic Conflicts

Conflicts in meaning or intent:

```llmcl
contract PositiveTone(priority = medium) {
    ensure contains(response, "great") or contains(response, "excellent")
        message: "Should maintain positive tone"
        auto_fix: response + " This is great!"
}

contract NeutralTone(priority = medium) {
    ensure not contains(response, "great") and not contains(response, "excellent")
        message: "Should maintain neutral tone"
        auto_fix: replace(replace(response, "great", "good"), "excellent", "adequate")
}

// Conflict: Positive vs. neutral tone requirements
```

## Resolution Strategies

### 1. FIRST_WINS Strategy

The first matching contract takes precedence:

```llmcl
contract ConflictExample(
    priority = medium,
    resolution = FIRST_WINS
) {
    ensure len(response) >= 100
        message: "Response should be detailed"
        auto_fix: response + " [Additional details...]"
    
    ensure len(response) <= 50
        message: "Response should be concise"
        auto_fix: response[:50]
    
    // First rule (>= 100) wins, second rule ignored
}
```

**Use Case**: When order of definition matters, or for legacy compatibility.

### 2. LAST_WINS Strategy

The last matching contract takes precedence:

```llmcl
contract LastWinsExample(
    priority = medium,
    resolution = LAST_WINS
) {
    ensure starts_with(response, "Hello")
        message: "Should start with greeting"
        auto_fix: "Hello! " + response
    
    ensure starts_with(response, "Hi")
        message: "Should start with casual greeting"
        auto_fix: "Hi! " + response
    
    // Last rule (starts with "Hi") wins
}
```

**Use Case**: When newer rules should override older ones, or for configuration overrides.

### 3. MOST_RESTRICTIVE Strategy

Applies the most restrictive constraint:

```llmcl
contract MostRestrictiveExample(
    priority = medium,
    resolution = MOST_RESTRICTIVE
) {
    ensure len(response) <= 200
        message: "Should not exceed 200 characters"
        auto_fix: response[:200]
    
    ensure len(response) <= 100
        message: "Should not exceed 100 characters"
        auto_fix: response[:100]
    
    // Most restrictive (<= 100) is applied
}
```

**Use Case**: When safety and security are paramount.

### 4. LEAST_RESTRICTIVE Strategy

Applies the least restrictive constraint:

```llmcl
contract LeastRestrictiveExample(
    priority = medium,
    resolution = LEAST_RESTRICTIVE
) {
    ensure len(response) >= 50
        message: "Should be at least 50 characters"
        auto_fix: response + " [Padding to meet minimum length]"
    
    ensure len(response) >= 20
        message: "Should be at least 20 characters"
        auto_fix: response + " [Short padding]"
    
    // Least restrictive (>= 20) is applied
}
```

**Use Case**: When flexibility and user experience are prioritized.

### 5. MERGE Strategy

Intelligently combines compatible constraints:

```llmcl
contract MergeExample(
    priority = medium,
    resolution = MERGE
) {
    ensure len(response) >= 50
        message: "Minimum length requirement"
        auto_fix: response + " [Additional content]"
    
    ensure len(response) <= 200
        message: "Maximum length requirement"
        auto_fix: response[:200]
    
    ensure contains(response, "help")
        message: "Must offer help"
        auto_fix: response + " How can I help?"
    
    // Merged: 50 <= len(response) <= 200 AND contains "help"
}
```

**Use Case**: When constraints are complementary and can be combined.

### 6. FAIL_ON_CONFLICT Strategy

Raises an error when conflicts are detected:

```llmcl
contract FailOnConflictExample(
    priority = critical,
    resolution = FAIL_ON_CONFLICT
) {
    ensure json_valid(response)
        message: "Must be valid JSON"
    
    ensure not contains(response, "{")
        message: "Must not contain JSON characters"
    
    // Throws ConflictResolutionError at compile time
}
```

**Use Case**: During development to catch conflicting requirements early.

## Priority-Based Resolution

### Priority Levels

LLMCL supports four priority levels:

1. **Critical**: Security, compliance, data protection
2. **High**: Core functionality, user experience
3. **Medium**: Quality improvements, optimizations
4. **Low**: Cosmetic improvements, nice-to-have features

### Priority Resolution Rules

```llmcl
contract HighPriorityRule(priority = critical) {
    ensure not contains(response, "password")
        message: "CRITICAL: Must not expose passwords"
}

contract MediumPriorityRule(priority = medium) {
    ensure contains(response, "password") 
        message: "Should include password reset info"
}

// Critical priority wins over medium priority
// Result: Password content is forbidden
```

### Mixed Priority Resolution

```llmcl
contract MixedPriorityExample(priority = high) {
    // High priority length requirement
    ensure len(response) <= 100
        message: "High priority: Keep responses concise"
        auto_fix: response[:100]
}

contract LowPriorityDetail(priority = low) {
    // Low priority detail requirement
    ensure len(response) >= 200
        message: "Low priority: Provide detailed responses"
        auto_fix: response + " [Additional details follow...]"
}

// High priority wins: response limited to 100 characters
```

### Priority Override Patterns

```llmcl
contract SecurityOverride(priority = critical) {
    // Security always wins
    ensure not contains(response, "admin_password")
        message: "Security override: No admin credentials"
}

contract UXOptimization(priority = medium) {
    // User experience optimization
    ensure contains(response, "admin") 
        message: "UX: Should mention admin features"
        auto_fix: response + " Admin features are available."
}

// Security (critical) overrides UX (medium)
// Result: Admin password forbidden, but admin mention allowed
```

## Conflict Detection

### Static Analysis

LLMCL performs static analysis during compilation to detect potential conflicts:

```python
from llm_contracts.language.conflict_resolver import detect_conflicts

contract_source = """
contract ConflictingRules(priority = medium) {
    ensure len(response) >= 100
    ensure len(response) <= 50
}
"""

conflicts = detect_conflicts(contract_source)
for conflict in conflicts:
    print(f"Conflict detected: {conflict.description}")
    print(f"Conflicting clauses: {conflict.clause1} vs {conflict.clause2}")
    print(f"Conflict type: {conflict.type}")
```

### Runtime Detection

Conflicts can also be detected at runtime:

```python
from llm_contracts.language import LLMCLRuntime, ConflictResolutionError

runtime = LLMCLRuntime(conflict_resolution='FAIL_ON_CONFLICT')

try:
    result = runtime.validate(conflicting_contract, context)
except ConflictResolutionError as e:
    print(f"Runtime conflict: {e.message}")
    print(f"Suggested resolution: {e.suggested_strategy}")
```

### Conflict Reporting

```python
# Detailed conflict analysis
conflict_report = runtime.analyze_conflicts(contracts)

print(f"Total conflicts: {conflict_report.total_conflicts}")
print(f"Critical conflicts: {conflict_report.critical_conflicts}")
print(f"Resolvable conflicts: {conflict_report.resolvable_conflicts}")

for conflict in conflict_report.conflicts:
    print(f"Type: {conflict.type}")
    print(f"Severity: {conflict.severity}")
    print(f"Contracts: {conflict.contract_names}")
    print(f"Resolution: {conflict.suggested_resolution}")
```

## Auto-Remediation Framework

### Auto-Fix Architecture

```
Validation Failure → Auto-Fix Detection → Fix Strategy Selection → Fix Application → Re-validation
```

### Basic Auto-Fix Syntax

```llmcl
ensure condition
    message: "Error description"
    auto_fix: fix_expression
    confidence: 0.8  // Optional confidence score
```

### Auto-Fix Examples

#### Simple String Fixes

```llmcl
contract SimpleStringFixes(priority = medium) {
    ensure len(response) <= 100
        message: "Response too long"
        auto_fix: response[:100] + "..."
    
    ensure not contains(response, "password")
        message: "Contains sensitive information"
        auto_fix: replace(response, "password", "[REDACTED]")
    
    ensure ends_with(response, ".")
        message: "Should end with period"
        auto_fix: response + "."
}
```

#### Conditional Auto-Fixes

```llmcl
contract ConditionalAutoFix(priority = medium) {
    ensure json_valid(response)
        message: "Invalid JSON format"
        auto_fix: if starts_with(response, "{") then 
                     response + "}" 
                   else if ends_with(response, "}") then 
                     "{" + response 
                   else 
                     '{"content": "' + response + '"}'
    
    ensure len(response) >= 20
        message: "Response too short"
        auto_fix: if len(response) < 10 then 
                     "I apologize, but I cannot provide a complete response." 
                   else 
                     response + " Please let me know if you need more information."
}
```

#### Complex Auto-Fixes

```llmcl
contract ComplexAutoFix(priority = high) {
    ensure starts_with(response, "{") and ends_with(response, "}") and json_valid(response)
        message: "Must be valid JSON object"
        auto_fix: if json_valid(response) then 
                     if not starts_with(response, "{") then 
                         "{" + response + "}" 
                     else 
                         response 
                   else 
                     '{"error": "Invalid response", "original": "' + 
                     response.replace('"', '\\"') + '"}'
}
```

## Fix Strategies

### 1. FIRST_FIX Strategy

Apply only the first applicable auto-fix:

```llmcl
contract FirstFixExample(
    priority = medium,
    fix_strategy = FIRST_FIX
) {
    ensure len(response) <= 100
        message: "Too long"
        auto_fix: response[:100]
    
    ensure contains(response, "help")
        message: "Should offer help"
        auto_fix: response + " How can I help?"
    
    // If both conditions fail, only first fix (truncation) is applied
}
```

### 2. ALL_FIXES Strategy

Apply all applicable auto-fixes in sequence:

```llmcl
contract AllFixesExample(
    priority = medium,
    fix_strategy = ALL_FIXES
) {
    ensure len(response) <= 100
        message: "Too long"
        auto_fix: response[:100]
    
    ensure ends_with(response, ".")
        message: "Should end with period"
        auto_fix: response + "."
    
    ensure not contains(response, "  ")  // Double spaces
        message: "Should not have double spaces"
        auto_fix: replace(response, "  ", " ")
    
    // All applicable fixes are applied in order
}
```

### 3. BEST_FIX Strategy

Choose the fix with the highest confidence score:

```llmcl
contract BestFixExample(
    priority = medium,
    fix_strategy = BEST_FIX
) {
    ensure json_valid(response)
        message: "Invalid JSON"
        auto_fix: '{"content": "' + response + '"}'
        confidence: 0.9
    
    ensure json_valid(response)
        message: "Invalid JSON"
        auto_fix: fix_json_syntax(response)
        confidence: 0.7
    
    ensure json_valid(response)
        message: "Invalid JSON"
        auto_fix: '{"error": "Could not parse", "raw": "' + response + '"}'
        confidence: 0.5
    
    // Fix with confidence 0.9 is chosen
}
```

### 4. INTERACTIVE_FIX Strategy

Prompt for user input when multiple fixes are available:

```python
# Python API for interactive fixing
runtime = LLMCLRuntime(fix_strategy='INTERACTIVE')

result = runtime.validate(contract, context)
if result.auto_fixes:
    print("Multiple fixes available:")
    for i, fix in enumerate(result.auto_fixes):
        print(f"{i+1}. {fix.description}: {fix.preview}")
    
    choice = input("Choose fix (1-{}): ".format(len(result.auto_fixes)))
    selected_fix = result.auto_fixes[int(choice) - 1]
    
    fixed_context = apply_fix(context, selected_fix)
```

## Advanced Auto-Fix Patterns

### Cascading Fixes

```llmcl
contract CascadingFixes(priority = medium) {
    ensure len(response) <= 200
        message: "Primary: Response too long"
        auto_fix: response[:200]
        cascade_to: "EnsureCompleteness"
    
    ensure ends_with(response, ".") or ends_with(response, "!") or ends_with(response, "?")
        message: "Secondary: Should end with punctuation"
        auto_fix: response + "."
        trigger_on: "EnsureCompleteness"
    
    // After truncation, ensure proper sentence ending
}
```

### Context-Aware Fixes

```llmcl
contract ContextAwareFixes(priority = medium) {
    ensure if user_type == "premium" then len(response) >= 100
        message: "Premium users need detailed responses"
        auto_fix: if len(response) < 100 then 
                     response + " As a premium user, here are additional details: " +
                     generate_premium_content(user_context)
                   else 
                     response
    
    ensure if api_mode then json_valid(response)
        message: "API mode requires JSON"
        auto_fix: convert_to_api_response(response, api_version)
}
```

### Machine Learning Enhanced Fixes

```llmcl
contract MLEnhancedFixes(priority = high) {
    ensure sentiment_score(response) > 0.1
        message: "Response should be positive or neutral"
        auto_fix: ml_enhance_sentiment(response, target_sentiment="neutral")
        confidence: ml_confidence_score(response)
    
    ensure coherence_score(response, content) > 0.7
        message: "Response should be coherent with input"
        auto_fix: ml_improve_coherence(response, content)
        confidence: coherence_confidence(response, content)
}
```

### Template-Based Fixes

```llmcl
contract TemplateFixes(priority = medium) {
    ensure api_response_format(response)
        message: "Should follow API response format"
        auto_fix: apply_template("api_response", {
            "status": "success",
            "data": response,
            "timestamp": current_timestamp(),
            "version": api_version
        })
    
    ensure error_response_format(response)
        message: "Error responses should follow standard format"
        auto_fix: apply_template("error_response", {
            "error": true,
            "message": response,
            "code": infer_error_code(response),
            "timestamp": current_timestamp()
        })
}
```

## Configuration and Customization

### Runtime Configuration

```python
from llm_contracts.language import LLMCLRuntime
from llm_contracts.language.conflict_resolver import ConflictResolver, AutoFixManager

# Configure conflict resolution
conflict_resolver = ConflictResolver(
    default_strategy='MOST_RESTRICTIVE',
    priority_override=True,
    strategy_mapping={
        'security': 'FAIL_ON_CONFLICT',
        'format': 'MERGE',
        'content': 'MOST_RESTRICTIVE'
    }
)

# Configure auto-fix behavior
auto_fix_manager = AutoFixManager(
    default_strategy='BEST_FIX',
    max_fix_attempts=3,
    confidence_threshold=0.6,
    enable_cascading=True
)

# Initialize runtime with custom configuration
runtime = LLMCLRuntime(
    conflict_resolver=conflict_resolver,
    auto_fix_manager=auto_fix_manager,
    fail_fast=False,
    enable_telemetry=True
)
```

### Custom Resolution Strategies

```python
from llm_contracts.language.conflict_resolver import ConflictResolver

def custom_resolution_strategy(conflicts):
    """Custom conflict resolution logic"""
    # Business logic specific resolution
    if any(c.type == 'security' for c in conflicts):
        return resolve_security_conflicts(conflicts)
    elif any(c.type == 'format' for c in conflicts):
        return resolve_format_conflicts(conflicts)
    else:
        return apply_priority_resolution(conflicts)

# Register custom strategy
resolver = ConflictResolver()
resolver.register_strategy('custom_business_logic', custom_resolution_strategy)

runtime = LLMCLRuntime(
    conflict_resolver=resolver,
    default_resolution_strategy='custom_business_logic'
)
```

### Custom Auto-Fix Functions

```python
from llm_contracts.language.auto_fix import register_fix_function

@register_fix_function
def smart_json_fix(response, context):
    """Intelligent JSON formatting with context awareness"""
    if context.get('api_mode'):
        return {
            "data": response,
            "status": "success",
            "api_version": context.get('api_version', '1.0')
        }
    else:
        return {"content": response}

@register_fix_function
def content_aware_truncation(response, max_length, context):
    """Truncate while preserving sentence boundaries"""
    if len(response) <= max_length:
        return response
    
    # Find last complete sentence within limit
    truncated = response[:max_length]
    last_sentence_end = max(
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?')
    )
    
    if last_sentence_end > max_length * 0.7:  # At least 70% of desired length
        return truncated[:last_sentence_end + 1]
    else:
        return truncated + "..."
```

## Monitoring and Debugging

### Conflict Resolution Metrics

```python
# Get conflict resolution statistics
stats = runtime.get_conflict_stats()

print(f"Total conflicts detected: {stats.total_conflicts}")
print(f"Conflicts resolved: {stats.resolved_conflicts}")
print(f"Resolution strategy distribution: {stats.strategy_usage}")
print(f"Average resolution time: {stats.avg_resolution_time_ms}ms")

# Conflict types breakdown
for conflict_type, count in stats.conflict_types.items():
    print(f"{conflict_type}: {count} conflicts")
```

### Auto-Fix Monitoring

```python
# Get auto-fix statistics
fix_stats = runtime.get_auto_fix_stats()

print(f"Total auto-fixes attempted: {fix_stats.total_attempts}")
print(f"Successful fixes: {fix_stats.successful_fixes}")
print(f"Fix success rate: {fix_stats.success_rate}")
print(f"Average fix confidence: {fix_stats.avg_confidence}")

# Fix strategy effectiveness
for strategy, stats in fix_stats.strategy_stats.items():
    print(f"{strategy}: {stats.success_rate}% success rate")
```

### Debug Mode

```python
# Enable detailed debugging
runtime = LLMCLRuntime(debug_mode=True)

result = runtime.validate(contract, context)

# Detailed debug information
print("Validation Debug Info:")
print(f"Contracts evaluated: {result.debug_info.contracts_evaluated}")
print(f"Conflicts detected: {result.debug_info.conflicts_detected}")
print(f"Resolution strategy used: {result.debug_info.resolution_strategy}")
print(f"Auto-fixes attempted: {result.debug_info.auto_fixes_attempted}")

# Step-by-step resolution process
for step in result.debug_info.resolution_steps:
    print(f"Step {step.order}: {step.action}")
    print(f"  Input: {step.input}")
    print(f"  Output: {step.output}")
    print(f"  Duration: {step.duration_ms}ms")
```

### Conflict Visualization

```python
# Generate conflict resolution report
report = runtime.generate_conflict_report(contracts)

# Export to various formats
report.export_html("conflict_report.html")
report.export_json("conflict_data.json")
report.export_csv("conflict_summary.csv")

# Visualize conflict graph
conflict_graph = report.generate_graph()
conflict_graph.render("conflict_visualization.png")
```

---

This comprehensive guide covers all aspects of conflict resolution and auto-remediation in LLMCL. Use these patterns and strategies to build robust validation systems that handle conflicts gracefully and automatically fix common issues.