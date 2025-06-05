# Task Integration Mapping - OpenAI SDK Changes

This document provides a detailed breakdown of how each task in our `tasks.json` relates to specific integration points with the OpenAI SDK and the exact changes required.

## Overview of Integration Strategy

**Key Principle**: We do NOT modify the OpenAI SDK. Instead, we create a **compatibility-preserving wrapper layer** that intercepts calls and adds contract enforcement while maintaining 100% API compatibility.

```python
# Original OpenAI Pattern (unchanged)
import openai
client = openai.OpenAI()
response = client.chat.completions.create(messages=[...])

# Our Contract-Enhanced Pattern (drop-in replacement)
from llm_contracts.providers import ImprovedOpenAIProvider
client = ImprovedOpenAIProvider()  # Same constructor as openai.OpenAI()
client.add_contract(PromptLengthContract())  # Add contracts

# EXACT SAME API - no code changes needed!
response = client.chat.completions.create(messages=[...])  # Contracts enforced transparently
```

## Task-by-Task Integration Details

### ✅ **COMPLETED TASKS (1-3)**

#### Task 1: Project Structure and Core Interfaces
**Status**: ✅ Done  
**OpenAI SDK Relationship**: Foundation layer - no direct SDK changes
**Integration Points**:
- `ProviderAdapter` interface defines contract points for any LLM provider
- `ContractBase` provides validation framework
- Exception hierarchy maps SDK errors to contract violations

```python
# Core interfaces enable this usage pattern:
from llm_contracts.core.interfaces import ProviderAdapter

class ImprovedOpenAIProvider:
    def __init__(self, **kwargs):
        self._client = openai.OpenAI(**kwargs)  # Wraps existing SDK
        self.contracts = []
    
    def __getattr__(self, name):
        # Proxy pattern - intercept specific methods for validation
        # Forward everything else unchanged
        pass
```

#### Task 2: Contract Taxonomy Base Classes
**Status**: ✅ Done  
**OpenAI SDK Relationship**: Contract definitions - no SDK changes needed
**Integration Points**:
- Contracts validate inputs/outputs independent of SDK
- Each contract type works with OpenAI response formats

```python
# Contracts work with OpenAI response structure:
class JSONFormatContract(OutputContract):
    def validate(self, response_content: str) -> ValidationResult:
        # Works with content extracted from openai.ChatCompletion
        try:
            json.loads(response_content)
            return ValidationResult(True, "Valid JSON")
        except:
            return ValidationResult(False, "Invalid JSON format")
```

#### Task 3: OpenAI API Provider Implementation  
**Status**: ✅ Done → **🔄 NEEDS REFACTOR FOR COMPATIBILITY**
**OpenAI SDK Relationship**: Primary wrapper implementation
**Current Issues**:
- ❌ Breaks API compatibility (`provider.call()` vs `client.chat.completions.create()`)
- ❌ Reimplements OpenAI methods instead of proxying
- ❌ Maintenance overhead for OpenAI updates

**Optimal Integration Points**:

1. **Transparent Client Wrapping**:
```python
class ImprovedOpenAIProvider:
    def __init__(self, **kwargs):
        # Create real OpenAI client with same parameters
        self._client = openai.OpenAI(**kwargs)
        self._async_client = openai.AsyncOpenAI(**kwargs)
        
        # Contract storage
        self.input_contracts = []
        self.output_contracts = []
```

2. **Proxy Pattern for API Compatibility**:
```python
def __getattr__(self, name):
    """Forward ALL unknown attributes to OpenAI client"""
    attr = getattr(self._client, name)
    
    # Only wrap 'chat' for contract enforcement
    if name == 'chat':
        return self._wrap_chat_namespace(attr)
    
    # Everything else passes through unchanged
    return attr
```

3. **Selective Method Interception**:
```python
def _wrap_chat_namespace(self, chat_attr):
    """Only wrap completions.create for validation"""
    class WrappedChat:
        def __init__(self, original_chat, provider):
            self._original_chat = original_chat
            self.completions = WrappedCompletions(
                original_chat.completions, provider
            )
        
        def __getattr__(self, name):
            # Forward all other chat methods unchanged
            return getattr(self._original_chat, name)
    
    return WrappedChat(chat_attr, self)
```

4. **Contract Enforcement at Call Site**:
```python
class WrappedCompletions:
    def create(self, **kwargs):
        # Pre-validation (Task 4 integration point)
        self._provider._validate_input(**kwargs)
        
        # Call original OpenAI method unchanged
        response = self._original_completions.create(**kwargs)
        
        # Post-validation (Task 5 integration point)
        return self._provider._validate_output(response)
    
    def __getattr__(self, name):
        # Forward all other completion methods unchanged
        return getattr(self._original_completions, name)
```

---

### 🔄 **PENDING TASKS (4-15)**

#### Task 4: Input Validation Stage
**Status**: 🔄 Pending  
**OpenAI SDK Relationship**: Pre-call validation layer
**Required Integration Changes**:

1. **Parameter Validation in Wrapped `create()` Method**:
```python
def create(self, **kwargs):  # In WrappedCompletions
    # Input contract validation
    validation_context = {
        "parameters": kwargs,
        "estimated_tokens": self._provider._estimate_tokens(kwargs.get('messages', []))
    }
    
    for contract in self._provider.input_contracts:
        result = contract.validate(kwargs, validation_context)
        if not result.is_valid:
            if result.auto_fix_suggestion:
                kwargs.update(result.auto_fix_suggestion)
            else:
                raise ContractViolationError(result.error_message)
    
    # Call original OpenAI method with potentially modified kwargs
    return self._original_completions.create(**kwargs)
```

2. **Token Estimation Integration**:
```python
def _estimate_tokens(self, messages):
    # Integration with tiktoken (OpenAI's tokenizer)
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model(self._client.model or "gpt-3.5-turbo")
        total_tokens = 0
        for message in messages:
            total_tokens += len(encoding.encode(message.get('content', '')))
        return total_tokens
    except:
        # Fallback estimation
        return sum(len(msg.get('content', '').split()) * 1.3 for msg in messages)
```

3. **Security Scanning Integration**:
```python
def _validate_input(self, **kwargs):
    # Extract messages for validation
    messages = kwargs.get('messages', [])
    if messages:
        latest_content = messages[-1].get('content', '')
        
        # Apply input contracts
        for contract in self.input_contracts:
            result = contract.validate(latest_content, kwargs)
            # Handle validation result...
```

#### Task 5: Output Validation with Auto-Remediation
**Status**: 🔄 Pending
**OpenAI SDK Relationship**: Post-call validation and retry logic
**Required Integration Changes**:

1. **Enhanced `_validate_output()` Method**:
```python
def _validate_output(self, response, context=None, max_retries=3):
    content = self._extract_content(response)
    
    for contract in self.output_contracts:
        result = contract.validate(content, context)
        if not result.is_valid:
            if result.auto_fix_suggestion:
                # Create corrected response maintaining OpenAI structure
                return self._create_corrected_response(response, result.auto_fix_suggestion)
            elif max_retries > 0:
                # Re-call OpenAI with correction instruction
                return self._retry_with_correction(response, result.error_message, max_retries - 1)
            else:
                raise ContractViolationError(result.error_message)
    
    return response  # Return original response if all contracts pass
```

2. **Retry Logic with OpenAI Integration**:
```python
def _retry_with_correction(self, original_response, error_message, retries_left):
    # Extract original request parameters
    original_messages = self._get_original_messages(original_response)
    
    # Add correction instruction
    correction_messages = original_messages + [{
        "role": "system",
        "content": f"Previous response violated contract: {error_message}. Please correct and retry."
    }]
    
    # Re-call OpenAI API
    retry_response = self._original_completions.create(
        messages=correction_messages,
        # Preserve other original parameters
    )
    
    # Validate retry response
    return self._validate_output(retry_response, max_retries=retries_left)
```

#### Task 6: Contract Specification Language (LLMCL)
**Status**: 🔄 Planned
**OpenAI SDK Relationship**: Declarative contract definition layer
**Integration Approach**:

```python
# LLMCL parser generates contract objects
from llm_contracts.llmcl import parse_contract

@parse_contract("""
    require: len(prompt) <= 4000 tokens
    require: not contains_profanity(prompt)  
    ensure: is_valid_json(response)
    ensure: response.length > 10
""")
def openai_json_call(prompt):
    # Standard OpenAI API usage - no changes needed!
    client = ImprovedOpenAIProvider()
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

# Parser automatically:
# 1. Creates contracts from require/ensure statements
# 2. Attaches contracts to the provider
# 3. Enforces contracts transparently
```

#### Task 7: Decorator API and Contract Annotations
**Status**: 🔄 Planned  
**OpenAI SDK Relationship**: Python decorator integration
**Integration Approach**:

```python
@contract
def summarize_text(text: str) -> str:
    require(len(text) <= 10000, "Text too long for summarization")
    require(text.strip() != "", "Empty text provided")
    
    # Use standard OpenAI API - decorator handles contract attachment
    client = ImprovedOpenAIProvider()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Summarize: {text}"}]
    )
    
    result = response.choices[0].message.content
    
    ensure(len(result) > 20, "Summary too short")
    ensure(len(result) < len(text), "Summary not shorter than original")
    return result

# Decorator automatically:
# 1. Extracts require() and ensure() statements
# 2. Creates input/output contracts
# 3. Wraps function to enforce contracts
# 4. Works with any OpenAI API usage inside function
```

#### Task 8: Streaming Response Support
**Status**: 🔄 Planned
**OpenAI SDK Relationship**: Streaming API wrapper with real-time validation
**Integration Changes**:

1. **Streaming Method Enhancement in WrappedCompletions**:
```python
def create(self, **kwargs):
    if kwargs.get('stream'):
        # Apply input validation
        self._provider._validate_input(**kwargs)
        
        # Get streaming response from OpenAI
        stream = self._original_completions.create(**kwargs)
        
        # Wrap stream for contract validation
        return self._provider._wrap_streaming_response(stream)
    else:
        # Regular non-streaming validation
        return self._regular_create(**kwargs)
```

2. **Streaming Validation Wrapper**:
```python
def _wrap_streaming_response(self, openai_stream):
    """Wrap OpenAI streaming response with contract validation"""
    accumulated_content = ""
    
    for chunk in openai_stream:
        # Extract content from chunk
        if chunk.choices and chunk.choices[0].delta.content:
            content_delta = chunk.choices[0].delta.content
            accumulated_content += content_delta
            
            # Check streaming contracts
            for contract in self.streaming_contracts:
                if contract.should_validate_now(accumulated_content):
                    result = contract.validate_partial(accumulated_content)
                    if not result.is_valid:
                        # Yield error chunk and stop
                        yield self._create_error_chunk(result.error_message)
                        return
        
        # Yield original chunk if validation passes
        yield chunk
    
    # Final validation on complete content
    self._validate_complete_stream(accumulated_content)
```

#### Task 9: Multi-turn Conversation Context
**Status**: 🔄 Planned
**OpenAI SDK Relationship**: Conversation state management wrapper
**Integration Approach**:

```python
class ConversationManager:
    def __init__(self, provider: ImprovedOpenAIProvider):
        self.provider = provider
        self.conversation_history = []
        self.conversation_contracts = []
    
    def send_message(self, message: str, **kwargs):
        # Build full conversation context for OpenAI
        full_messages = self.conversation_history + [
            {"role": "user", "content": message}
        ]
        
        # Validate conversation-level contracts
        for contract in self.conversation_contracts:
            result = contract.validate_conversation(full_messages)
            if not result.is_valid:
                raise ContractViolationError(result.error_message)
        
        # Call OpenAI with full context - same API!
        response = self.provider.chat.completions.create(
            messages=full_messages,
            **kwargs
        )
        
        # Update conversation history
        assistant_content = response.choices[0].message.content
        self.conversation_history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": assistant_content}
        ])
        
        return response
```

#### Task 10: LangChain Integration
**Status**: 🔄 Planned
**OpenAI SDK Relationship**: LangChain LLM wrapper with contracts
**Integration Approach**:

```python
from langchain.llms.base import LLM
from llm_contracts.providers import ImprovedOpenAIProvider

class ContractLLM(LLM):
    def __init__(self, contracts=None, **kwargs):
        # Create our OpenAI provider with contracts
        self.provider = ImprovedOpenAIProvider(**kwargs)
        if contracts:
            for contract in contracts:
                self.provider.add_contract(contract)
    
    def _call(self, prompt: str, stop=None, **kwargs):
        # LangChain calls this method
        # Use standard OpenAI API - contracts enforced transparently
        response = self.provider.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stop=stop,
            **kwargs
        )
        return response.choices[0].message.content

# Usage in LangChain chains - no changes to LangChain code:
llm = ContractLLM(contracts=[
    PromptLengthContract(max_tokens=4000),
    JSONFormatContract(schema=my_schema)
])

chain = LLMChain(llm=llm, prompt=prompt_template)
result = chain.run(input_text)  # Contracts enforced automatically
```

#### Tasks 11-15: Extended Ecosystem
**OpenAI SDK Relationship**: These tasks extend the pattern to other providers and tools

- **Task 11** (Multi-Platform): Same proxy pattern for Anthropic, Azure OpenAI, Cohere
- **Task 12** (Static Analysis): AST analysis of code using OpenAI providers  
- **Task 13** (VS Code): IDE integration showing contract violations in real-time
- **Task 14** (Jupyter): Magic commands for notebook-based contract enforcement
- **Task 15** (Benchmarking): Performance testing of contract overhead

## Integration Testing Strategy

### Unit Tests (Per Task)
Each task includes tests that verify integration with OpenAI SDK:

```python
# Task 4 example
def test_input_validation_with_openai():
    provider = ImprovedOpenAIProvider()
    provider.add_input_contract(PromptLengthContract(max_tokens=10))
    
    with pytest.raises(ContractViolationError):
        # Uses standard OpenAI API
        provider.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "A very long prompt that exceeds limit..."}]
        )

# Task 5 example  
def test_output_validation_with_retry():
    provider = ImprovedOpenAIProvider()
    provider.add_output_contract(JSONFormatContract())
    
    # Mock OpenAI to return invalid JSON first, then valid JSON
    with mock.patch.object(provider._client.chat.completions, 'create') as mock_create:
        mock_create.side_effect = [
            MockResponse("Invalid JSON"),  # First call
            MockResponse('{"valid": "json"}')  # Retry call
        ]
        
        result = provider.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Generate JSON"}]
        )
        assert json.loads(result.choices[0].message.content) == {"valid": "json"}
```

### Integration Tests
Full end-to-end tests with real OpenAI API:

```python
@pytest.mark.integration
def test_full_contract_workflow():
    provider = ImprovedOpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    provider.add_input_contract(PromptLengthContract(max_tokens=100))
    provider.add_output_contract(JSONFormatContract())
    
    # Use standard OpenAI API
    response = provider.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Generate a simple JSON object"}]
    )
    
    # Verify it's valid JSON
    content = response.choices[0].message.content
    parsed = json.loads(content)
    assert isinstance(parsed, dict)
```

## Migration Path for Existing OpenAI Code

### **Zero Breaking Changes Required** ⭐

```python
# Before - existing OpenAI code
import openai
client = openai.OpenAI(api_key="...")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=100
)

# After - simply replace import, same constructor, same API
from llm_contracts.providers import ImprovedOpenAIProvider
client = ImprovedOpenAIProvider(api_key="...")  # Same constructor
client.add_contract(PromptLengthContract())      # Add contracts

# EXACT SAME API CALL - no changes needed!
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=100
    # All parameters work unchanged!
)
```

### Step-by-Step Migration

1. **Drop-in Replacement**:
   - Replace `openai.OpenAI()` with `ImprovedOpenAIProvider()`
   - All existing code works unchanged

2. **Add Contracts Gradually**:
   ```python
   # Start with basic contracts
   client.add_input_contract(PromptLengthContract(max_tokens=4000))
   client.add_output_contract(ContentPolicyContract())
   
   # Add more sophisticated contracts over time
   client.add_output_contract(JSONFormatContract(schema=my_schema))
   ```

3. **Full Contract Specification**:
   ```python
   # Eventually migrate to declarative contracts
   @contract("""
       require: len(prompt) <= 4000 tokens
       ensure: is_valid_json(response)
       ensure: not contains_sensitive_data(response)
   """)
   def my_llm_function(prompt):
       return client.chat.completions.create(
           model="gpt-4",
           messages=[{"role": "user", "content": prompt}]
       )
   ```

## Key Benefits of This Integration Approach

1. **Zero Breaking Changes**: Existing OpenAI code works unchanged
2. **Future-Proof**: New OpenAI features automatically available
3. **Minimal Maintenance**: No need to update wrapper for OpenAI changes
4. **Perfect Compatibility**: All OpenAI parameters, methods, and patterns work
5. **Transparent Contracts**: Validation happens behind the scenes
6. **Easy Adoption**: Simple import change enables contract enforcement

This approach ensures that our contract framework integrates seamlessly with the OpenAI SDK while providing a **zero-friction upgrade path** for existing applications. 