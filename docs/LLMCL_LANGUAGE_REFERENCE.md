# LLMCL Language Reference

This document provides a comprehensive reference for the LLM Contract Language (LLMCL) syntax, grammar, and semantics.

## Table of Contents

1. [Lexical Structure](#lexical-structure)
2. [Grammar Definition](#grammar-definition)
3. [Data Types](#data-types)
4. [Expressions](#expressions)
5. [Built-in Functions](#built-in-functions)
6. [Operators](#operators)
7. [Contract Syntax](#contract-syntax)
8. [Clause Types](#clause-types)
9. [Temporal Logic](#temporal-logic)
10. [Probabilistic Constructs](#probabilistic-constructs)
11. [Auto-Fix Expressions](#auto-fix-expressions)
12. [Error Handling](#error-handling)

## Lexical Structure

### Tokens

LLMCL recognizes the following token types:

#### Keywords
```
contract, require, ensure, ensure_prob, temporal, priority, message, auto_fix
window_size, description, resolution, fix_strategy
always, eventually, next, within, until, since
and, or, not, if, then, else
true, false
critical, high, medium, low
```

#### Identifiers
```
[a-zA-Z_][a-zA-Z0-9_]*
```

#### Literals

**String Literals**:
```
"double quoted string"
'single quoted string'
```

**Numeric Literals**:
```
42          // Integer
3.14        // Float
0.95        // Probability
```

**Boolean Literals**:
```
true
false
```

#### Operators
```
+  -  *  /  %           // Arithmetic
== != <  >  <= >=       // Comparison  
and or not              // Logical
=                       // Assignment
.                       // Attribute access
```

#### Delimiters
```
( ) { } [ ]             // Grouping
, ;                     // Separators
:                       // Key-value separator
```

#### Comments
```
// Single line comment
/* Multi-line comment */
```

### Whitespace

Whitespace (spaces, tabs, newlines) is ignored except within string literals.

## Grammar Definition

### Complete EBNF Grammar

```ebnf
program          ::= contract+

contract         ::= 'contract' identifier '(' parameters ')' '{' clause* '}'

parameters       ::= parameter (',' parameter)*
parameter        ::= identifier '=' literal

clause           ::= precondition | postcondition | probabilistic | temporal

precondition     ::= 'require' expression clause_options?
postcondition    ::= 'ensure' expression clause_options?
probabilistic    ::= 'ensure_prob' expression ',' probability clause_options?
temporal         ::= 'temporal' temporal_op expression clause_options?

clause_options   ::= 'message' ':' string_literal |
                     'auto_fix' ':' expression |
                     'window_size' ':' integer_literal

temporal_op      ::= 'always' | 'eventually' | 'next' | 
                     'within' integer_literal |
                     'until' | 'since'

expression       ::= logical_or

logical_or       ::= logical_and ('or' logical_and)*
logical_and      ::= equality ('and' equality)*
equality         ::= comparison (('==' | '!=') comparison)*
comparison       ::= arithmetic (('<' | '>' | '<=' | '>=') arithmetic)*
arithmetic       ::= term (('+' | '-') term)*
term             ::= factor (('*' | '/' | '%') factor)*
factor           ::= unary | primary
unary            ::= ('not' | '-') factor
primary          ::= literal | identifier | function_call | 
                     attribute_access | '(' expression ')'

function_call    ::= identifier '(' arguments? ')'
arguments        ::= expression (',' expression)*
attribute_access ::= primary '.' identifier

literal          ::= string_literal | numeric_literal | boolean_literal
probability      ::= numeric_literal  // Must be between 0.0 and 1.0

identifier       ::= [a-zA-Z_][a-zA-Z0-9_]*
string_literal   ::= '"' ([^"\\] | escape_sequence)* '"' |
                     "'" ([^'\\] | escape_sequence)* "'"
numeric_literal  ::= integer_literal | float_literal
integer_literal  ::= [0-9]+
float_literal    ::= [0-9]+ '.' [0-9]+
boolean_literal  ::= 'true' | 'false'

escape_sequence  ::= '\\' ('n' | 't' | 'r' | '\\' | '"' | "'")
```

### Precedence and Associativity

Operators in order of precedence (highest to lowest):

1. **Unary operators**: `not`, `-` (right-associative)
2. **Multiplicative**: `*`, `/`, `%` (left-associative)
3. **Additive**: `+`, `-` (left-associative)
4. **Comparison**: `<`, `>`, `<=`, `>=` (left-associative)
5. **Equality**: `==`, `!=` (left-associative)
6. **Logical AND**: `and` (left-associative)
7. **Logical OR**: `or` (left-associative)

## Data Types

### Primitive Types

#### String
```llmcl
"Hello, world!"
'Single quoted string'
"String with \"escaped\" quotes"
'String with \'escaped\' quotes'
```

#### Number
```llmcl
42          // Integer
3.14        // Float
-17         // Negative integer
0.0         // Zero
```

#### Boolean
```llmcl
true
false
```

### Composite Types

#### Object Attributes
```llmcl
user.name           // Access object attribute
response.length     // Built-in length attribute
context.metadata    // Nested object access
```

#### Lists (Implicit)
```llmcl
// Lists are accessed through built-in functions
len(responses)      // Length of list
contains(tags, "ai") // Check list membership
```

## Expressions

### Basic Expressions

#### Literals
```llmcl
"Hello"             // String literal
42                  // Integer literal
3.14                // Float literal
true                // Boolean literal
```

#### Variables
```llmcl
content             // Access context variable
response            // Access response variable
user_input          // Access user_input variable
```

#### Attribute Access
```llmcl
user.name           // Object attribute
response.metadata   // Nested attribute
request.headers     // Complex object access
```

### Function Calls

#### Built-in Functions
```llmcl
len(content)                    // String/list length
contains(response, "help")      // Substring check
startswith(text, "Hello")       // Prefix check
endswith(text, "!")            // Suffix check
match(text, r"\d+")            // Regex matching
json_valid(response)           // JSON validation
```

#### Custom Functions (if supported)
```llmcl
validate_email(email)          // Custom validation
sentiment_score(text)          // Custom analysis
```

### Arithmetic Expressions

```llmcl
len(content) + 10              // Addition
response_count * 2             // Multiplication
total_length / word_count      // Division
char_count % 100               // Modulo
-response_time                 // Unary minus
```

### Comparison Expressions

```llmcl
len(response) > 50             // Greater than
score <= 0.8                   // Less than or equal
status == "complete"           // Equality
type != "error"                // Inequality
priority >= "high"             // String comparison
```

### Logical Expressions

```llmcl
len(content) > 0 and len(content) < 1000    // AND
json_valid(response) or len(response) == 0   // OR
not contains(response, "error")              // NOT

// Complex logic
(priority == "high" or priority == "critical") and 
not contains(content, "spam")
```

### Conditional Expressions

```llmcl
// Ternary operator
if len(response) > 100 then "long" else "short"

// Nested conditionals
if score > 0.8 then 
    "excellent" 
else if score > 0.6 then 
    "good" 
else 
    "needs improvement"
```

## Built-in Functions

### String Functions

#### `len(string) -> number`
Returns the length of a string.
```llmcl
len("hello")        // Returns 5
len(response)       // Returns length of response
```

#### `contains(string, substring) -> boolean`
Checks if string contains substring.
```llmcl
contains("hello world", "world")    // Returns true
contains(response, "error")         // Check for error messages
```

#### `startswith(string, prefix) -> boolean`
Checks if string starts with prefix.
```llmcl
startswith("hello", "hel")          // Returns true
startswith(response, "Error:")      // Check error prefix
```

#### `endswith(string, suffix) -> boolean`
Checks if string ends with suffix.
```llmcl
endswith("hello", "lo")             // Returns true
endswith(response, ".")             // Check proper sentence ending
```

#### `match(string, pattern) -> boolean`
Checks if string matches regex pattern.
```llmcl
match("123-45-6789", r"\d{3}-\d{2}-\d{4}")     // SSN pattern
match(email, r"[^@]+@[^@]+\.[^@]+")            // Email pattern
```

#### `replace(string, old, new) -> string`
Replaces occurrences of old with new.
```llmcl
replace(response, "error", "issue")
replace(text, "\n", " ")                       // Remove newlines
```

#### `trim(string) -> string`
Removes leading and trailing whitespace.
```llmcl
trim("  hello  ")                              // Returns "hello"
trim(user_input)                               // Clean user input
```

#### `lower(string) -> string`
Converts string to lowercase.
```llmcl
lower("HELLO")                                 // Returns "hello"
lower(response)                                // Normalize case
```

#### `upper(string) -> string`
Converts string to uppercase.
```llmcl
upper("hello")                                 // Returns "HELLO"
upper(status)                                  // Normalize status
```

### Validation Functions

#### `json_valid(string) -> boolean`
Checks if string is valid JSON.
```llmcl
json_valid('{"key": "value"}')                 // Returns true
json_valid('invalid json')                     // Returns false
```

#### `email_valid(string) -> boolean`
Validates email format.
```llmcl
email_valid("user@domain.com")                 // Returns true
email_valid("invalid-email")                   // Returns false
```

#### `url_valid(string) -> boolean`
Validates URL format.
```llmcl
url_valid("https://example.com")               // Returns true
url_valid("not-a-url")                         // Returns false
```

### Numeric Functions

#### `abs(number) -> number`
Returns absolute value.
```llmcl
abs(-5)                                        // Returns 5
abs(score - target)                            // Distance from target
```

#### `min(number, number) -> number`
Returns minimum of two numbers.
```llmcl
min(len(response), 100)                        // Cap at 100
min(score, 1.0)                                // Cap score
```

#### `max(number, number) -> number`
Returns maximum of two numbers.
```llmcl
max(len(response), 10)                         // Minimum 10 chars
max(confidence, 0.0)                           // Non-negative
```

#### `round(number, digits?) -> number`
Rounds number to specified decimal places.
```llmcl
round(3.14159, 2)                              // Returns 3.14
round(score * 100)                             // Round percentage
```

### List Functions

#### `count(list, item) -> number`
Counts occurrences of item in list.
```llmcl
count(tags, "important")                       // Count important tags
count(responses, "error")                      // Count error responses
```

#### `first(list) -> any`
Returns first item in list.
```llmcl
first(responses)                               // First response
first(error_messages)                          // First error
```

#### `last(list) -> any`
Returns last item in list.
```llmcl
last(responses)                                // Most recent response
last(conversation_turns)                       // Latest turn
```

## Operators

### Arithmetic Operators

```llmcl
a + b               // Addition
a - b               // Subtraction  
a * b               // Multiplication
a / b               // Division
a % b               // Modulo
-a                  // Unary minus
```

**Type Rules**:
- Number + Number = Number
- String + String = String (concatenation)
- Number + String = Error

### Comparison Operators

```llmcl
a == b              // Equality
a != b              // Inequality
a < b               // Less than
a > b               // Greater than
a <= b              // Less than or equal
a >= b              // Greater than or equal
```

**Type Rules**:
- Numbers: numeric comparison
- Strings: lexicographic comparison
- Booleans: false < true
- Mixed types: Error

### Logical Operators

```llmcl
a and b             // Logical AND (short-circuit)
a or b              // Logical OR (short-circuit)
not a               // Logical NOT
```

**Type Rules**:
- Only work with boolean values
- Short-circuit evaluation

### String Operators

```llmcl
"hello" + " world"  // Concatenation: "hello world"
response + "!"      // Append exclamation
prefix + content    // Prepend prefix
```

## Contract Syntax

### Contract Declaration

```llmcl
contract ContractName(parameter1 = value1, parameter2 = value2) {
    // Contract clauses
}
```

#### Standard Parameters

```llmcl
contract Example(
    priority = high,                    // critical, high, medium, low
    description = "Contract description",
    resolution = MOST_RESTRICTIVE,      // Conflict resolution strategy
    fix_strategy = FIRST_FIX           // Auto-fix strategy
) {
    // Clauses
}
```

#### Custom Parameters

```llmcl
contract APIValidation(
    priority = high,
    api_version = "v1",
    timeout_ms = 5000,
    custom_param = "value"
) {
    // Can reference custom parameters in clauses
    require timeout < timeout_ms
        message: "Response took too long"
}
```

### Contract Body

The contract body contains zero or more clauses:

```llmcl
contract Example(priority = medium) {
    // Precondition
    require len(content) > 0
        message: "Input required"
    
    // Postcondition
    ensure json_valid(response)
        message: "Response must be JSON"
        auto_fix: '{"content": "' + response + '"}'
    
    // Probabilistic constraint
    ensure_prob len(response) > 50, 0.8
        message: "80% of responses should be substantial"
        window_size: 20
    
    // Temporal constraint
    temporal always len(response) > 0
        message: "Responses must never be empty"
}
```

## Clause Types

### Preconditions (require)

Preconditions validate input before processing.

```llmcl
require condition
    message: "Error message"
    auto_fix: fix_expression
```

**Examples**:
```llmcl
require len(content) > 0
    message: "Input cannot be empty"

require len(content) <= 4000
    message: "Input too long" 
    auto_fix: content[:4000]

require not contains(content, "password")
    message: "Input contains sensitive information"
```

### Postconditions (ensure)

Postconditions validate output after processing.

```llmcl
ensure condition
    message: "Error message"
    auto_fix: fix_expression
```

**Examples**:
```llmcl
ensure len(response) >= 10
    message: "Response too short"
    auto_fix: response + " Please let me know if you need more help."

ensure json_valid(response)
    message: "Response must be valid JSON"
    auto_fix: '{"content": "' + response.replace('"', '\\"') + '"}'

ensure not startswith(response, "Error:")
    message: "Should not return error messages"
```

### Probabilistic Constraints (ensure_prob)

Probabilistic constraints specify statistical requirements.

```llmcl
ensure_prob condition, probability
    message: "Error message"
    window_size: number
    auto_fix: fix_expression
```

**Examples**:
```llmcl
ensure_prob len(response) > 50, 0.8
    message: "80% of responses should be substantial"
    window_size: 20

ensure_prob json_valid(response), 0.95
    message: "95% of responses should be valid JSON"
    window_size: 100
    auto_fix: '{"content": "' + response + '"}'

ensure_prob not contains(response, "I don't know"), 0.9
    message: "Should rarely claim ignorance"
    window_size: 50
```

## Temporal Logic

Temporal logic allows validation across time and conversation turns.

### Temporal Operators

#### Always (□)
```llmcl
temporal always condition
    message: "Must always be true"
```

The condition must be true in every turn.

```llmcl
temporal always len(response) > 0
    message: "Response must never be empty"

temporal always not contains(response, "password")
    message: "Must never expose passwords"
```

#### Eventually (◇)
```llmcl
temporal eventually condition
    message: "Must eventually be true"
```

The condition must be true in at least one future turn.

```llmcl
temporal eventually contains(response, "thank you")
    message: "Should eventually show gratitude"

temporal eventually contains(response, "goodbye")
    message: "Should eventually conclude conversation"
```

#### Next (○)
```llmcl
temporal next condition
    message: "Must be true in next turn"
```

The condition must be true in the immediately following turn.

```llmcl
temporal next len(response) > len(prev_response)
    message: "Next response should be longer"

temporal next contains(response, "follow-up")
    message: "Should follow up in next response"
```

#### Within N Turns
```llmcl
temporal within N condition
    message: "Must be true within N turns"
```

The condition must become true within N turns.

```llmcl
temporal within 3 contains(response, "help")
    message: "Should offer help within 3 turns"

temporal within 5 contains(response, "resolution")
    message: "Should resolve issue within 5 turns"
```

#### Until/Since
```llmcl
temporal condition1 until condition2
temporal since condition1 then condition2
```

Complex temporal relationships.

```llmcl
temporal contains(response, "error") until contains(response, "resolved")
    message: "Should keep mentioning error until resolved"

temporal since contains(response, "greeting") then len(response) > 20
    message: "After greeting, responses should be substantial"
```

### Temporal Context Variables

Special variables available in temporal clauses:

- `prev_response`: Previous response
- `next_response`: Next response (in until/since)
- `turn_number`: Current turn number
- `conversation_length`: Total turns so far

```llmcl
temporal always response != prev_response
    message: "Should not repeat exact responses"

temporal turn_number <= 10 or contains(response, "conclusion")
    message: "Should conclude within 10 turns"
```

## Probabilistic Constructs

### Basic Probabilistic Constraints

```llmcl
ensure_prob condition, probability
    message: "Statistical requirement"
    window_size: N
```

**Parameters**:
- `condition`: Boolean expression to evaluate
- `probability`: Required success rate (0.0 to 1.0)
- `window_size`: Number of recent samples to consider

### Window Types

#### Sliding Window (Default)
```llmcl
ensure_prob len(response) > 50, 0.8
    window_size: 20  // Last 20 responses
```

#### Session Window
```llmcl
ensure_prob helpful_score > 0.7, 0.9
    window_size: session  // Current conversation only
```

#### Global Window
```llmcl
ensure_prob error_rate < 0.1, 0.95
    window_size: global  // All historical data
```

### Advanced Probabilistic Patterns

#### Adaptive Thresholds
```llmcl
ensure_prob len(response) > (50 + turn_number * 5), 0.8
    message: "Responses should get more detailed over time"
    window_size: 10
```

#### Conditional Probabilities
```llmcl
ensure_prob if contains(content, "help") then len(response) > 100, 0.9
    message: "Help requests should get detailed responses"
    window_size: 25
```

#### Multi-criteria Validation
```llmcl
ensure_prob (len(response) > 50 and contains(response, "helpful")), 0.7
    message: "Responses should be both long and helpful"
    window_size: 30
```

## Auto-Fix Expressions

Auto-fix expressions automatically repair contract violations.

### Basic Auto-Fix

```llmcl
ensure condition
    message: "Error message"
    auto_fix: fix_expression
```

**Examples**:
```llmcl
ensure len(response) <= 200
    message: "Response too long"
    auto_fix: response[:200] + "..."

ensure json_valid(response)
    message: "Invalid JSON"
    auto_fix: '{"content": "' + response.replace('"', '\\"') + '"}'

ensure not contains(response, "password")
    message: "Contains sensitive info"
    auto_fix: replace(response, "password", "[REDACTED]")
```

### Conditional Auto-Fix

```llmcl
ensure startswith(response, "{") and endswith(response, "}")
    message: "Must be JSON object"
    auto_fix: if startswith(response, "{") then 
                 response + "}" 
              else if endswith(response, "}") then 
                 "{" + response 
              else 
                 '{"content": "' + response + '"}'
```

### Multi-Step Auto-Fix

```llmcl
ensure json_valid(response) and len(response) <= 100
    message: "Must be valid JSON under 100 chars"
    auto_fix: '{"msg": "' + trim(response[:90]) + '"}'
```

### Auto-Fix Strategies

#### First Fix Strategy
```llmcl
contract Example(fix_strategy = FIRST_FIX) {
    ensure len(response) <= 100
        auto_fix: response[:100]
    
    ensure contains(response, "helpful")
        auto_fix: response + " (I hope this helps!)"
    
    // Only first applicable fix is applied
}
```

#### All Fixes Strategy
```llmcl
contract Example(fix_strategy = ALL_FIXES) {
    ensure len(response) <= 100
        auto_fix: response[:100]
    
    ensure endswith(response, ".")
        auto_fix: response + "."
    
    // Both fixes applied in order if both conditions fail
}
```

#### Best Fix Strategy
```llmcl
contract Example(fix_strategy = BEST_FIX) {
    ensure json_valid(response)
        auto_fix: '{"content": "' + response + '"}'
        confidence: 0.9
    
    ensure json_valid(response)
        auto_fix: '{"message": "' + response + '"}'
        confidence: 0.7
    
    // Fix with highest confidence is chosen
}
```

## Error Handling

### Syntax Errors

LLMCL provides detailed syntax error reporting:

```llmcl
// Error: Missing closing brace
contract BadContract(priority = high) {
    require len(content) > 0
// Missing }

// Error: Invalid operator
contract BadOperator(priority = medium) {
    require len(content) >> 0  // '>>' is not valid
        message: "Length check"
}

// Error: Unterminated string
contract BadString(priority = low) {
    require contains(content, "hello)  // Missing closing quote
        message: "Contains hello"
}
```

### Runtime Errors

#### Type Errors
```llmcl
// Error: Cannot compare string to number
ensure len(response) > "50"
    message: "Type mismatch error"

// Error: Cannot call len() on number
ensure len(42) > 0
    message: "Invalid function call"
```

#### Undefined Variables
```llmcl
// Error: 'unknown_var' is not defined
require len(unknown_var) > 0
    message: "Undefined variable error"
```

#### Invalid Function Calls
```llmcl
// Error: match() requires 2 arguments
ensure match(response)
    message: "Missing regex pattern"

// Error: Unknown function
ensure validate_email(email)  // Function not defined
    message: "Unknown function error"
```

### Error Recovery

#### Graceful Degradation
```llmcl
contract RobustValidation(priority = medium) {
    // Primary validation
    ensure json_valid(response)
        message: "Preferred: Response should be JSON"
        auto_fix: '{"content": "' + response + '"}'
    
    // Fallback validation if JSON parsing fails
    ensure len(response) > 0
        message: "Minimum: Response should not be empty"
        auto_fix: "No response provided"
}
```

#### Try-Catch Pattern (Future Enhancement)
```llmcl
// Proposed syntax for future versions
contract SafeValidation(priority = high) {
    try {
        ensure complex_validation(response)
            message: "Complex validation failed"
    } catch ValidationError {
        ensure simple_validation(response)
            message: "Fallback validation"
    }
}
```

### Debugging Support

#### Contract Introspection
```python
# Python API for debugging
contract = compile_contract(source)
print(contract.clauses)          # List all clauses
print(contract.dependencies)     # Variable dependencies
print(contract.syntax_tree)      # AST representation
```

#### Validation Debugging
```python
# Detailed validation results
result = runtime.validate(contract, context)
for violation in result.violations:
    print(f"Clause: {violation.clause}")
    print(f"Expression: {violation.expression}")
    print(f"Evaluated to: {violation.actual_value}")
    print(f"Expected: {violation.expected}")
```

---

This language reference provides the complete syntax and semantics of LLMCL. For examples and usage patterns, see the main LLMCL documentation.