# LLMCL Contract Types and Temporal Logic

This document provides comprehensive guidance on designing different types of contracts in LLMCL and using temporal logic for multi-turn conversation validation.

## Table of Contents

1. [Contract Design Principles](#contract-design-principles)
2. [Input Validation Contracts](#input-validation-contracts)
3. [Output Quality Contracts](#output-quality-contracts)
4. [Content Policy Contracts](#content-policy-contracts)
5. [Format and Structure Contracts](#format-and-structure-contracts)
6. [Performance Contracts](#performance-contracts)
7. [Business Logic Contracts](#business-logic-contracts)
8. [Temporal Logic Deep Dive](#temporal-logic-deep-dive)
9. [Advanced Temporal Patterns](#advanced-temporal-patterns)
10. [Contract Composition](#contract-composition)
11. [Testing Contracts](#testing-contracts)

## Contract Design Principles

### Single Responsibility Principle

Each contract should focus on one specific aspect of validation:

```llmcl
// Good: Focused on input length
contract InputLength(priority = high) {
    require len(content) > 0
        message: "Input cannot be empty"
    
    require len(content) <= 4000
        message: "Input exceeds maximum length"
        auto_fix: content[:4000]
}

// Good: Focused on content safety
contract ContentSafety(priority = critical) {
    require not contains(content, "password")
        message: "Input contains sensitive information"
    
    require not contains(content, "API_KEY")
        message: "Input contains API credentials"
}

// Avoid: Mixed concerns
contract Everything(priority = medium) {
    require len(content) > 0          // Input validation
    ensure json_valid(response)       // Format validation
    ensure not contains(response, "password")  // Security
    temporal always len(response) > 0 // Temporal logic
    // Too many different concerns in one contract
}
```

### Appropriate Priority Levels

Use priority levels to reflect business impact:

- **Critical**: Security, data protection, regulatory compliance
- **High**: Core functionality, user experience
- **Medium**: Quality improvements, performance optimizations
- **Low**: Nice-to-have features, cosmetic improvements

```llmcl
contract DataProtection(priority = critical) {
    ensure not contains(response, "SSN:")
        message: "Must not expose social security numbers"
    
    ensure not match(response, r"\b\d{4}-\d{4}-\d{4}-\d{4}\b")
        message: "Must not expose credit card numbers"
}

contract UserExperience(priority = high) {
    ensure len(response) >= 20
        message: "Response should be substantial"
        auto_fix: response + " Please let me know if you need more information."
}

contract ResponseStyle(priority = medium) {
    ensure_prob contains(response, "please") or contains(response, "thank"), 0.7
        message: "Should be polite most of the time"
        window_size: 20
}

contract Formatting(priority = low) {
    ensure not match(response, r"\s{2,}")
        message: "Should not have excessive whitespace"
        auto_fix: replace(response, r"\s+", " ")
}
```

### Clear and Actionable Messages

Error messages should help developers understand and fix issues:

```llmcl
// Good: Specific and actionable
contract APIResponseFormat(priority = high) {
    ensure json_valid(response)
        message: "API response must be valid JSON. Check for unescaped quotes or trailing commas."
        auto_fix: '{"error": "Invalid JSON format", "original": "' + response + '"}'
    
    ensure contains(response, "status")
        message: "API response must include a 'status' field for client error handling."
        auto_fix: '{"status": "success", "data": ' + response + '}'
}

// Avoid: Vague messages
contract VagueValidation(priority = medium) {
    ensure json_valid(response)
        message: "Invalid format"  // Too vague
    
    ensure len(response) > 10
        message: "Too short"       // Not helpful
}
```

### Meaningful Auto-Fixes

Auto-fixes should preserve intent while ensuring compliance:

```llmcl
contract SmartAutoFix(priority = medium) {
    // Preserve content while fixing format
    ensure json_valid(response)
        message: "Response must be valid JSON"
        auto_fix: '{"content": "' + response.replace('"', '\\"') + '", "format": "text"}'
    
    // Graceful truncation with indication
    ensure len(response) <= 200
        message: "Response too long for client display"
        auto_fix: response[:180] + "... [truncated]"
    
    // Content-aware replacement
    ensure not contains(response, "error")
        message: "Should not mention errors to users"
        auto_fix: replace(replace(response, "error", "issue"), "Error", "Issue")
}
```

## Input Validation Contracts

Input validation contracts ensure user inputs meet requirements before processing.

### Basic Input Validation

```llmcl
contract BasicInputValidation(priority = high) {
    require len(content) > 0
        message: "Input cannot be empty"
    
    require len(content) <= 4000
        message: "Input exceeds 4000 character limit"
        auto_fix: content[:4000]
    
    require not match(content, r"^\s*$")
        message: "Input cannot be only whitespace"
        auto_fix: trim(content)
}
```

### Content Type Validation

```llmcl
contract ContentTypeValidation(priority = medium) {
    require not contains(content, "<script")
        message: "HTML/JavaScript content not allowed"
    
    require not contains(content, "<?php")
        message: "PHP code not allowed"
    
    require not match(content, r"[\\/:*?\"<>|]")
        message: "Contains invalid filename characters"
        auto_fix: replace(replace(replace(content, "\\", "_"), "/", "_"), ":", "_")
}
```

### Language and Encoding Validation

```llmcl
contract LanguageValidation(priority = medium) {
    require match(content, r"^[\x00-\x7F]*$") or valid_utf8(content)
        message: "Content must be ASCII or valid UTF-8"
    
    require not contains(content, "\x00")
        message: "Null bytes not allowed"
        auto_fix: replace(content, "\x00", "")
    
    require len(content.encode()) <= 8192
        message: "Content exceeds byte limit"
        auto_fix: content[:8000]  // Conservative truncation
}
```

### Structured Input Validation

```llmcl
contract StructuredInputValidation(priority = high) {
    // JSON input validation
    require if starts_with(content, "{") then json_valid(content)
        message: "JSON-like input must be valid JSON"
        auto_fix: '{"content": "' + content + '"}'
    
    // Email validation
    require if contains(content, "@") then email_valid(content)
        message: "Email-like input must be valid email format"
    
    // URL validation  
    require if starts_with(content, "http") then url_valid(content)
        message: "URL-like input must be valid URL format"
}
```

### Security Input Validation

```llmcl
contract SecurityInputValidation(priority = critical) {
    require not contains(lower(content), "password")
        message: "Input should not contain password-related terms"
    
    require not contains(content, "admin")
        message: "Admin-related input not allowed"
    
    require not match(content, r"\b\d{3}-\d{2}-\d{4}\b")
        message: "Social Security Numbers not allowed"
    
    require not match(content, r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
        message: "Credit card numbers not allowed"
    
    require not match(content, r"(api[_-]?key|secret|token)\s*[:=]\s*\w+")
        message: "API keys or secrets not allowed"
}
```

## Output Quality Contracts

Output quality contracts ensure responses meet quality standards.

### Response Completeness

```llmcl
contract ResponseCompleteness(priority = high) {
    ensure len(response) >= 10
        message: "Response too short to be helpful"
        auto_fix: response + " Please let me know if you need more information."
    
    ensure len(response) <= 2000
        message: "Response too long for optimal user experience"
        auto_fix: response[:1950] + "... [Continue reading for more details]"
    
    ensure not match(response, r"^\s*$")
        message: "Response cannot be empty or whitespace only"
        auto_fix: "I apologize, but I cannot provide a response to that request."
}
```

### Response Relevance

```llmcl
contract ResponseRelevance(priority = medium) {
    // Check if response addresses the input
    ensure if contains(content, "help") then 
             contains(response, "help") or contains(response, "assist")
        message: "Should acknowledge help requests"
        auto_fix: "I'd be happy to help! " + response
    
    ensure if contains(content, "question") then 
             contains(response, "answer") or contains(response, "explain")
        message: "Should acknowledge questions appropriately"
    
    ensure if contains(content, "thank") then 
             contains(response, "welcome") or contains(response, "glad")
        message: "Should acknowledge gratitude"
        auto_fix: response + " You're welcome!"
}
```

### Response Helpfulness

```llmcl
contract ResponseHelpfulness(priority = medium) {
    ensure_prob contains(response, "help") or 
                 contains(response, "assist") or 
                 contains(response, "support"), 0.6
        message: "60% of responses should offer assistance"
        window_size: 25
    
    ensure not starts_with(response, "I don't know")
        message: "Should provide helpful responses instead of claiming ignorance"
        auto_fix: "Let me help you with that. " + response[12:]
    
    ensure not contains(response, "I cannot help")
        message: "Should offer alternative assistance"
        auto_fix: replace(response, "I cannot help", "I can suggest alternatives")
}
```

### Response Tone

```llmcl
contract ResponseTone(priority = low) {
    ensure_prob contains(response, "please") or 
                 contains(response, "thank") or 
                 contains(response, "appreciate"), 0.5
        message: "Should maintain polite tone"
        window_size: 10
    
    ensure not contains(response, "obviously")
        message: "Avoid condescending language"
        auto_fix: replace(response, "obviously", "")
    
    ensure not match(response, r"\b(stupid|dumb|idiotic)\b")
        message: "Should not use derogatory language"
        auto_fix: replace(response, match_group, "unclear")
}
```

## Content Policy Contracts

Content policy contracts enforce safety and compliance requirements.

### Data Protection

```llmcl
contract DataProtection(priority = critical) {
    ensure not match(response, r"\b\d{3}-\d{2}-\d{4}\b")
        message: "Must not expose Social Security Numbers"
    
    ensure not match(response, r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
        message: "Must not expose credit card numbers"
    
    ensure not match(response, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        message: "Must not expose email addresses without consent"
        auto_fix: replace(response, match_group, "[EMAIL_REDACTED]")
    
    ensure not match(response, r"\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b")
        message: "Must not expose phone numbers"
        auto_fix: replace(response, match_group, "[PHONE_REDACTED]")
}
```

### Credential Protection

```llmcl
contract CredentialProtection(priority = critical) {
    ensure not contains(lower(response), "password")
        message: "Must not mention passwords"
        auto_fix: replace(response, "password", "[CREDENTIAL]")
    
    ensure not contains(response, "API_KEY")
        message: "Must not expose API keys"
        auto_fix: replace(response, "API_KEY", "[API_KEY_REDACTED]")
    
    ensure not match(response, r"(bearer|token)\s+[A-Za-z0-9._-]+")
        message: "Must not expose bearer tokens"
        auto_fix: replace(response, match_group, "[TOKEN_REDACTED]")
    
    ensure not match(response, r"['\"]?[A-Za-z0-9]{20,}['\"]?")
        message: "Potential secret detected"
        auto_fix: replace(response, match_group, "[POTENTIAL_SECRET_REDACTED]")
}
```

### Content Appropriateness

```llmcl
contract ContentAppropriateness(priority = high) {
    ensure not match(response, r"\b(hate|violence|illegal)\b")
        message: "Must not promote harmful content"
    
    ensure not contains(response, "discriminat")
        message: "Should not contain discriminatory language"
    
    ensure not match(response, r"\b(kill|murder|harm)\b")
        message: "Must not contain violent language"
        auto_fix: replace(response, match_group, "[CONTENT_REMOVED]")
}
```

### Regulatory Compliance

```llmcl
contract RegulatoryCompliance(priority = critical) {
    // GDPR compliance
    ensure not contains(response, "personal data") or 
           contains(response, "consent")
        message: "Personal data mentions must include consent information"
    
    // COPPA compliance
    ensure not contains(response, "child") or 
           not contains(response, "collect")
        message: "Child-related data collection requires special handling"
    
    // HIPAA compliance
    ensure not match(response, r"\b(patient|medical|health)\b") or 
           contains(response, "confidential")
        message: "Medical information must include confidentiality notice"
}
```

## Format and Structure Contracts

Format contracts ensure responses follow required structural patterns.

### JSON Format Contracts

```llmcl
contract JSONFormat(priority = high) {
    ensure json_valid(response)
        message: "Response must be valid JSON"
        auto_fix: '{"content": "' + response.replace('"', '\\"') + '", "type": "text"}'
    
    ensure starts_with(response, "{") and ends_with(response, "}")
        message: "JSON response must be an object"
        auto_fix: if starts_with(response, "{") then 
                     response + "}" 
                   else if ends_with(response, "}") then 
                     "{" + response 
                   else 
                     '{"data": ' + response + '}'
    
    ensure contains(response, '"status"')
        message: "JSON response must include status field"
        auto_fix: replace(response, "}", ', "status": "success"}')
}
```

### API Response Format

```llmcl
contract APIResponseFormat(priority = high) {
    ensure json_valid(response)
        message: "API responses must be valid JSON"
        auto_fix: '{"error": "Invalid response format", "data": "' + response + '"}'
    
    ensure contains(response, '"data"') or contains(response, '"error"')
        message: "API response must contain data or error field"
        auto_fix: '{"data": ' + response + '}'
    
    ensure if contains(response, '"error"') then contains(response, '"code"')
        message: "Error responses must include error code"
        auto_fix: replace(response, '"error":', '"error":{"code": 500, "message":') + '}'
}
```

### Markdown Format

```llmcl
contract MarkdownFormat(priority = medium) {
    ensure if contains(response, "# ") then 
             match(response, r"^#\s+\w+")
        message: "Markdown headers must be properly formatted"
        auto_fix: replace(response, "#", "# ")
    
    ensure if contains(response, "```") then 
             count(response, "```") % 2 == 0
        message: "Code blocks must be properly closed"
        auto_fix: response + "\n```"
    
    ensure if contains(response, "[") and contains(response, "](") then
             match(response, r"\[([^\]]+)\]\(([^)]+)\)")
        message: "Links must be properly formatted"
}
```

### Structured Text Format

```llmcl
contract StructuredTextFormat(priority = medium) {
    ensure if contains(response, "\n\n") then 
             not match(response, r"\n{3,}")
        message: "Should not have excessive line breaks"
        auto_fix: replace(response, r"\n{3,}", "\n\n")
    
    ensure if len(response) > 100 then 
             contains(response, "\n") or contains(response, ". ")
        message: "Long responses should have proper sentence breaks"
        auto_fix: replace(response, ". ", ".\n")
    
    ensure ends_with(trim(response), ".") or 
           ends_with(trim(response), "!") or 
           ends_with(trim(response), "?")
        message: "Responses should end with proper punctuation"
        auto_fix: trim(response) + "."
}
```

## Performance Contracts

Performance contracts ensure responses meet timing and efficiency requirements.

### Response Time Contracts

```llmcl
contract ResponseTime(priority = medium) {
    require processing_time_ms <= 5000
        message: "Request processing time exceeds 5 second limit"
    
    ensure if len(response) > 1000 then processing_time_ms <= 10000
        message: "Long responses should still complete within 10 seconds"
    
    ensure_prob processing_time_ms <= 2000, 0.9
        message: "90% of responses should complete within 2 seconds"
        window_size: 50
}
```

### Resource Usage Contracts

```llmcl
contract ResourceUsage(priority = medium) {
    require memory_usage_mb <= 100
        message: "Memory usage exceeds 100MB limit"
    
    require cpu_usage_percent <= 80
        message: "CPU usage too high during processing"
    
    ensure_prob token_count <= 4000, 0.95
        message: "95% of responses should stay under token limit"
        window_size: 100
}
```

### Throughput Contracts

```llmcl
contract Throughput(priority = high) {
    ensure_prob requests_per_minute >= 60, 0.9
        message: "Should maintain at least 60 requests per minute"
        window_size: 10
    
    ensure concurrent_requests <= 50
        message: "Should not exceed concurrent request limit"
    
    temporal always queue_size <= 100
        message: "Request queue should never exceed 100 items"
}
```

## Business Logic Contracts

Business logic contracts enforce domain-specific rules and workflows.

### E-commerce Contracts

```llmcl
contract EcommerceValidation(priority = high) {
    require if contains(content, "price") then 
             match(content, r"\$\d+(\.\d{2})?")
        message: "Prices must be in valid currency format"
        auto_fix: "$" + match(content, r"\d+(\.\d{2})?")
    
    ensure if contains(content, "order") then 
           contains(response, "confirmation") or 
           contains(response, "tracking")
        message: "Order responses should include confirmation or tracking info"
    
    ensure if contains(content, "refund") then 
           contains(response, "policy") or 
           contains(response, "process")
        message: "Refund responses should reference policy or process"
}
```

### Support Ticket Contracts

```llmcl
contract SupportTicketValidation(priority = high) {
    ensure if contains(content, "urgent") or contains(content, "critical") then 
           len(response) >= 100
        message: "Urgent issues require detailed responses"
        auto_fix: response + " This has been escalated for immediate attention."
    
    ensure contains(response, "ticket") or contains(response, "reference")
        message: "Support responses should include ticket reference"
        auto_fix: response + " Reference ticket: #" + generate_ticket_id()
    
    temporal within 3 if contains(content, "follow-up") then 
                         contains(response, "status")
        message: "Follow-up requests should receive status updates within 3 turns"
}
```

### Financial Services Contracts

```llmcl
contract FinancialServicesValidation(priority = critical) {
    ensure if contains(content, "balance") or contains(content, "account") then 
           contains(response, "secure") or contains(response, "verify")
        message: "Financial information requests must include security notice"
    
    ensure not contains(response, "invest") or 
           contains(response, "risk")
        message: "Investment advice must include risk disclosure"
    
    ensure if match(content, r"\$[\d,]+") then 
           not contains(response, "guarantee")
        message: "Financial amounts should not be guaranteed"
        auto_fix: replace(response, "guarantee", "estimate")
}
```

## Temporal Logic Deep Dive

Temporal logic allows validation across conversation turns and time windows.

### Understanding Temporal Context

```llmcl
contract TemporalContext(priority = medium) {
    // Access to previous turns
    temporal always response != prev_response
        message: "Should not repeat exact previous response"
    
    // Access to turn numbers
    temporal if turn_number > 5 then 
             contains(response, "summary") or 
             contains(response, "conclusion")
        message: "Long conversations should include summaries"
    
    // Access to conversation history
    temporal if conversation_length > 10 then 
             eventually contains(response, "wrap up")
        message: "Extended conversations should eventually conclude"
}
```

### Temporal Operators in Detail

#### Always (□) - Universal Quantification

```llmcl
contract AlwaysExamples(priority = medium) {
    // Must be true in every turn
    temporal always len(response) > 0
        message: "Response must never be empty"
    
    // Must be true throughout conversation
    temporal always not contains(response, "password")
        message: "Must never expose passwords"
    
    // Conditional always
    temporal always if contains(content, "help") then 
                    len(response) >= 50
        message: "Help requests always get substantial responses"
}
```

#### Eventually (◇) - Existential Quantification

```llmcl
contract EventuallyExamples(priority = medium) {
    // Must happen at some point
    temporal eventually contains(response, "thank you")
        message: "Should show gratitude at some point"
    
    // Conditional eventually
    temporal eventually if turn_number > 3 then 
                        contains(response, "summary")
        message: "Should provide summary after several turns"
    
    // Within time bounds
    temporal eventually within 10 contains(response, "conclusion")
        message: "Should conclude conversation within 10 turns"
}
```

#### Next (○) - Immediate Future

```llmcl
contract NextExamples(priority = medium) {
    // Next turn requirement
    temporal next if contains(content, "continue") then 
                   starts_with(response, "Continuing")
        message: "Continue requests should be acknowledged immediately"
    
    // Chained next operations
    temporal next contains(response, "step 1")
    temporal next next contains(response, "step 2")
        message: "Should follow step sequence"
}
```

#### Within N - Bounded Future

```llmcl
contract WithinExamples(priority = medium) {
    // Time-bounded requirements
    temporal within 3 if contains(content, "urgent") then 
                        contains(response, "escalated")
        message: "Urgent issues should be escalated within 3 turns"
    
    // Multiple time bounds
    temporal within 5 contains(response, "help")
        message: "Should offer help within 5 turns"
    
    temporal within 10 contains(response, "resolution")
        message: "Should provide resolution within 10 turns"
}
```

#### Until/Since - Temporal Conditions

```llmcl
contract UntilSinceExamples(priority = medium) {
    // Until pattern
    temporal contains(response, "processing") until 
             contains(response, "complete")
        message: "Should show processing status until completion"
    
    // Since pattern
    temporal since contains(response, "error") then 
             contains(response, "retry") or 
             contains(response, "alternative")
        message: "After errors, should offer retry or alternatives"
    
    // Complex temporal relationships
    temporal since contains(content, "premium") then 
             always len(response) >= 100
        message: "Premium users always get detailed responses"
}
```

### Advanced Temporal Patterns

#### State Machine Patterns

```llmcl
contract ConversationStateMachine(priority = medium) {
    // Initial state
    temporal if turn_number == 1 then 
             contains(response, "hello") or contains(response, "hi")
        message: "First response should be a greeting"
    
    // State transitions
    temporal if contains(prev_response, "greeting") then 
             contains(response, "help")
        message: "After greeting, should offer help"
    
    // Terminal state
    temporal if contains(response, "goodbye") then 
             next not exists
        message: "Conversation should end after goodbye"
}
```

#### Conversation Flow Patterns

```llmcl
contract ConversationFlow(priority = medium) {
    // Progressive disclosure
    temporal if contains(content, "complex") then 
             within 3 contains(response, "step-by-step")
        message: "Complex requests should get step-by-step responses"
    
    // Information building
    temporal since contains(response, "first") then 
             eventually contains(response, "second") and 
             eventually contains(response, "third")
        message: "Should build information progressively"
    
    // Clarification patterns
    temporal if contains(response, "unclear") then 
             next contains(response, "clarification")
        message: "Unclear responses should prompt for clarification"
}
```

#### Quality Improvement Over Time

```llmcl
contract QualityImprovement(priority = medium) {
    // Increasing response quality
    temporal always len(response) >= len(prev_response) * 0.8
        message: "Response quality should not dramatically drop"
    
    // Learning from user feedback
    temporal if contains(content, "better") then 
             within 2 len(response) > avg_response_length * 1.2
        message: "Should improve responses after feedback"
    
    // Consistency over time
    temporal_prob always json_valid(response), 0.95
        message: "JSON validity should improve over time"
        window_size: 100
}
```

## Contract Composition

### Contract Inheritance

```llmcl
// Base contract
contract BaseValidation(priority = medium) {
    require len(content) > 0
        message: "Input required"
    
    ensure len(response) > 0
        message: "Response required"
}

// Specialized contracts
contract APIValidation extends BaseValidation(priority = high) {
    ensure json_valid(response)
        message: "API responses must be JSON"
        auto_fix: '{"data": ' + response + '}'
    
    ensure contains(response, "status")
        message: "API responses must include status"
}

contract ChatValidation extends BaseValidation(priority = medium) {
    ensure len(response) >= 20
        message: "Chat responses should be conversational"
        auto_fix: response + " How can I help you further?"
    
    ensure_prob contains(response, "help"), 0.6
        message: "Should offer help regularly"
        window_size: 10
}
```

### Contract Composition

```llmcl
// Compose multiple contracts
contract ComprehensiveValidation(
    priority = high,
    includes = [SecurityValidation, QualityValidation, FormatValidation]
) {
    // Additional specific requirements
    ensure all_included_contracts_pass()
        message: "All component validations must pass"
    
    // Override conflicts with higher priority rules
    ensure if conflicts_exist() then apply_priority_resolution()
        message: "Priority-based conflict resolution applied"
}
```

### Conditional Contract Activation

```llmcl
contract ConditionalValidation(priority = medium) {
    // Activate different validation based on context
    require if api_mode then apply_contract("APIValidation")
        message: "API mode requires API validation"
    
    require if chat_mode then apply_contract("ChatValidation")
        message: "Chat mode requires chat validation"
    
    require if production_env then apply_contract("ProductionValidation")
        message: "Production environment requires stricter validation"
}
```

## Testing Contracts

### Unit Testing Contracts

```python
import pytest
from llm_contracts.language import compile_contract, LLMCLRuntime

def test_input_validation_contract():
    contract_source = """
    contract InputValidation(priority = high) {
        require len(content) > 0
            message: "Input cannot be empty"
        
        require len(content) <= 100
            message: "Input too long"
            auto_fix: content[:100]
    }
    """
    
    contract = compile_contract(contract_source)
    runtime = LLMCLRuntime()
    
    # Test valid input
    result = runtime.validate(contract, {'content': 'Valid input'})
    assert result.is_valid
    
    # Test empty input
    result = runtime.validate(contract, {'content': ''})
    assert not result.is_valid
    assert any('empty' in v.message for v in result.violations)
    
    # Test input too long with auto-fix
    long_input = 'x' * 150
    result = runtime.validate(contract, {'content': long_input})
    assert not result.is_valid
    assert result.auto_fixes
    assert len(result.auto_fixes[0].fixed_value) == 100

def test_temporal_contract():
    contract_source = """
    contract TemporalValidation(priority = medium) {
        temporal always response != prev_response
            message: "Should not repeat responses"
        
        temporal within 3 contains(response, "help")
            message: "Should offer help within 3 turns"
    }
    """
    
    contract = compile_contract(contract_source)
    runtime = LLMCLRuntime()
    
    # Test conversation sequence
    contexts = [
        {'response': 'Hello', 'prev_response': None},
        {'response': 'How can I help?', 'prev_response': 'Hello'},
        {'response': 'Hello', 'prev_response': 'How can I help?'},  # Repeat!
    ]
    
    results = []
    for context in contexts:
        result = runtime.validate(contract, context)
        results.append(result)
    
    # First two should pass, third should fail (repeat)
    assert results[0].is_valid
    assert results[1].is_valid
    assert not results[2].is_valid

def test_probabilistic_contract():
    contract_source = """
    contract ProbabilisticValidation(priority = medium) {
        ensure_prob len(response) > 20, 0.8
            message: "80% of responses should be substantial"
            window_size: 5
    }
    """
    
    contract = compile_contract(contract_source)
    runtime = LLMCLRuntime()
    
    # Test with mixed response lengths
    responses = ['short', 'this is a much longer response', 'x', 'another long response', 'y']
    
    results = []
    for response in responses:
        result = runtime.validate(contract, {'response': response})
        results.append(result)
    
    # Should pass until we have enough data to evaluate probability
    # Last result might fail if probability threshold not met
```

### Integration Testing

```python
def test_contract_integration():
    """Test multiple contracts working together"""
    contracts_source = """
    contract Security(priority = critical) {
        ensure not contains(response, "password")
            message: "Must not expose passwords"
    }
    
    contract Quality(priority = high) {
        ensure len(response) >= 20
            message: "Response should be substantial"
            auto_fix: response + " Please let me know if you need more help."
    }
    
    contract Format(priority = medium) {
        ensure json_valid(response)
            message: "Response should be JSON"
            auto_fix: '{"content": "' + response + '"}'
    }
    """
    
    contracts = compile_contract(contracts_source)
    runtime = LLMCLRuntime(conflict_resolver=ConflictResolver(strategy='MOST_RESTRICTIVE'))
    
    # Test case that violates multiple contracts
    context = {
        'response': 'pwd:secret'  # Short, contains password, not JSON
    }
    
    result = runtime.validate(contracts, context)
    
    # Should have violations from multiple contracts
    assert not result.is_valid
    assert len(result.violations) >= 2  # Security and quality violations
    
    # Auto-fixes should be available
    assert result.auto_fixes
    
    # Apply fixes and re-validate
    fixed_context = apply_auto_fixes(context, result.auto_fixes)
    fixed_result = runtime.validate(contracts, fixed_context)
    
    # Should pass after fixes (except security violation which can't be auto-fixed)
    assert len(fixed_result.violations) == 1  # Only security violation remains

def test_temporal_conversation():
    """Test temporal logic across a full conversation"""
    contract_source = """
    contract ConversationFlow(priority = medium) {
        temporal if turn_number == 1 then contains(response, "hello")
            message: "Should greet on first turn"
        
        temporal within 5 contains(response, "help")
            message: "Should offer help within 5 turns"
        
        temporal always response != prev_response
            message: "Should not repeat responses"
    }
    """
    
    contract = compile_contract(contract_source)
    runtime = LLMCLRuntime()
    
    # Simulate conversation
    conversation = [
        {'response': 'Hello! How can I help you today?', 'turn_number': 1},
        {'response': 'I can assist with various tasks.', 'turn_number': 2, 'prev_response': 'Hello! How can I help you today?'},
        {'response': 'What would you like help with?', 'turn_number': 3, 'prev_response': 'I can assist with various tasks.'},
    ]
    
    for turn in conversation:
        result = runtime.validate(contract, turn)
        assert result.is_valid, f"Turn {turn['turn_number']} failed: {result.violations}"
```

---

This comprehensive guide covers the design and implementation of various contract types in LLMCL, along with detailed explanations of temporal logic capabilities. Use these patterns as building blocks for your specific validation requirements.