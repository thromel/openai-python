# LLM Design by Contract Framework

A comprehensive Design by Contract framework for Large Language Model APIs that provides input/output validation, temporal contracts, streaming support, and multi-platform compatibility.

## 🚀 Features

- **Contract-Based Validation**: Input and output validation using design-by-contract principles
- **Multi-Provider Support**: Currently supports OpenAI with extensible architecture for other providers  
- **Comprehensive Contract Types**: 7 different contract categories covering various LLM reliability aspects
- **Streaming Support**: Real-time validation for streaming responses
- **Auto-Fix Suggestions**: Intelligent suggestions for contract violations
- **Async/Sync Support**: Both synchronous and asynchronous API calls

## 📋 Contract Types

The framework implements a comprehensive taxonomy of contracts:

### 1. Input Contracts
- **PromptLengthContract**: Validates token limits and length constraints
- **PromptInjectionContract**: Detects potential prompt injection attacks
- **ContentPolicyContract**: Ensures content compliance with policies

### 2. Output Contracts  
- **JSONFormatContract**: Validates JSON structure and schema compliance
- **ResponseTimeContract**: Monitors and validates performance metrics
- **ConversationConsistencyContract**: Ensures multi-turn conversation coherence
- **MedicalDisclaimerContract**: Domain-specific compliance validation

## 🔧 Installation

```bash
# Install the framework
pip install -e .

# Install OpenAI support (optional)
pip install openai
```

## 🏁 Quick Start

### Basic Usage with OpenAI

```python
from llm_contracts.providers import OpenAIProvider
from llm_contracts.validators import InputValidator, OutputValidator
from llm_contracts.contracts.base import PromptLengthContract, JSONFormatContract

# Create provider
provider = OpenAIProvider(model="gpt-3.5-turbo", api_key="your-key")

# Set up input validation
input_validator = InputValidator()
input_validator.add_contract(PromptLengthContract(max_tokens=100))
provider.set_input_validator(input_validator)

# Set up output validation  
output_validator = OutputValidator()
output_validator.add_contract(JSONFormatContract())
provider.set_output_validator(output_validator)

# Make API call with contract enforcement
response = provider.call("Generate a JSON response with user data")
```

### Advanced Contract Configuration

```python
from llm_contracts.contracts.base import (
    PromptInjectionContract, 
    ContentPolicyContract,
    ResponseTimeContract
)

# Create comprehensive input validator
input_validator = InputValidator("comprehensive")
input_validator.add_contract(PromptLengthContract(max_tokens=200))
input_validator.add_contract(PromptInjectionContract())
input_validator.add_contract(ContentPolicyContract())

# Create performance-aware output validator
output_validator = OutputValidator("performance")
output_validator.add_contract(JSONFormatContract(schema={"type": "object"}))
output_validator.add_contract(ResponseTimeContract(max_response_time=5.0))

provider.set_input_validator(input_validator)
provider.set_output_validator(output_validator)
```

### Async Usage

```python
import asyncio

async def async_example():
    provider = OpenAIProvider(model="gpt-4")
    
    # Same contract setup as sync
    input_validator = InputValidator()
    input_validator.add_contract(PromptLengthContract(max_tokens=150))
    provider.set_input_validator(input_validator)
    
    # Async API call
    response = await provider.acall("Write a short story")
    return response

# Run async
result = asyncio.run(async_example())
```

## 🔍 Contract Validation Examples

### Input Validation
```python
# This will pass validation
short_prompt = "Hello, how are you?"
response = provider.call(short_prompt)

# This will raise ContractViolationError if prompt is too long
long_prompt = "Very long prompt..." * 100
try:
    response = provider.call(long_prompt)
except ContractViolationError as e:
    print(f"Contract violated: {e}")
    print(f"Suggestion: {e.suggestion}")
```

### Output Validation
```python
# Set up JSON validation
json_contract = JSONFormatContract(schema={
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
    },
    "required": ["name", "age"]
})

output_validator = OutputValidator()
output_validator.add_contract(json_contract)
provider.set_output_validator(output_validator)

# This will validate the JSON structure
response = provider.call("Generate JSON with name and age fields")
```

## 🏗️ Architecture

The framework is built with a modular architecture:

```
llm_contracts/
├── core/                 # Core interfaces and exceptions
│   ├── interfaces.py     # Base classes and protocols
│   └── exceptions.py     # Exception hierarchy
├── contracts/            # Contract implementations
│   └── base.py          # All contract types
├── validators/           # Validation logic
│   └── basic_validators.py
├── providers/            # API provider adapters
│   └── openai_provider.py
└── utils/               # Utility functions
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests.test_openai_provider -v

# Run the demo
PYTHONPATH=. python examples/openai_demo.py
```

## 📚 Examples

Check the `examples/` directory for:
- `openai_demo.py`: Complete OpenAI provider demonstration
- More examples coming soon!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🔮 Roadmap

- [ ] Additional provider support (Anthropic, Google, etc.)
- [ ] LLMCL (LLM Contract Language) implementation
- [ ] Streaming validation support
- [ ] IDE extensions and tooling
- [ ] Performance optimization
- [ ] Advanced contract decorators

## 📞 Support

For questions and support, please open an issue on GitHub.

---

Built with ❤️ for reliable LLM applications.
