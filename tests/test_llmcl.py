"""Tests for LLM Contract Language (LLMCL)."""

import pytest
import asyncio
from llm_contracts.language import (
    LLMCLParser, LLMCLCompiler, LLMCLRuntime,
    ConflictResolver, ConflictType, ConflictAction,
    ResolutionStrategy, ContractPriority
)
from llm_contracts.language.ast_nodes import (
    ContractNode, RequireNode, EnsureNode, 
    EnsureProbNode, TemporalNode, TemporalOperator
)


class TestLLMCLParser:
    """Test LLMCL parser functionality."""
    
    def test_parse_simple_contract(self):
        """Test parsing a simple contract."""
        source = """
        contract SimpleContract {
            require len(content) > 0 message: "Content cannot be empty"
            ensure len(response) < 1000 message: "Response too long"
        }
        """
        
        parser = LLMCLParser()
        ast = parser.parse(source)
        
        assert isinstance(ast, ContractNode)
        assert ast.name == "SimpleContract"
        assert len(ast.requires) == 1
        assert len(ast.ensures) == 1
        
        # Check require clause
        require = ast.requires[0]
        assert isinstance(require, RequireNode)
        assert require.message == "Content cannot be empty"
        
        # Check ensure clause
        ensure = ast.ensures[0]
        assert isinstance(ensure, EnsureNode)
        assert ensure.message == "Response too long"
    
    def test_parse_contract_with_metadata(self):
        """Test parsing contract with metadata."""
        source = """
        contract PriorityContract(
            priority = high,
            conflict_resolution = most_restrictive,
            description = "High priority contract"
        ) {
            require temperature >= 0.0 and temperature <= 1.0
        }
        """
        
        parser = LLMCLParser()
        ast = parser.parse(source)
        
        assert ast.name == "PriorityContract"
        assert ast.priority == ContractPriority.HIGH
        assert ast.description == "High priority contract"
    
    def test_parse_probabilistic_ensure(self):
        """Test parsing probabilistic ensure clause."""
        source = """
        contract ProbContract {
            ensure_prob json_valid(response), 0.95
                message: "Response should be valid JSON 95% of the time"
                window_size: 200
        }
        """
        
        parser = LLMCLParser()
        ast = parser.parse(source)
        
        assert len(ast.ensures) == 1
        ensure_prob = ast.ensures[0]
        assert isinstance(ensure_prob, EnsureProbNode)
        assert ensure_prob.probability == 0.95
        assert ensure_prob.window_size == 200
    
    def test_parse_temporal_constraints(self):
        """Test parsing temporal constraints."""
        source = """
        contract TemporalContract {
            temporal always len(response) > 0
                message: "Response should never be empty"
            
            temporal within 5 contains(response, "thank you")
                message: "Should thank user within 5 turns"
        }
        """
        
        parser = LLMCLParser()
        ast = parser.parse(source)
        
        assert len(ast.temporal) == 2
        
        # Check always constraint
        temporal1 = ast.temporal[0]
        assert temporal1.operator == TemporalOperator.ALWAYS
        assert temporal1.message == "Response should never be empty"
        
        # Check within constraint
        temporal2 = ast.temporal[1]
        assert temporal2.operator == TemporalOperator.WITHIN
        assert temporal2.scope == 5
    
    def test_parse_complex_expressions(self):
        """Test parsing complex expressions."""
        source = """
        contract ComplexContract {
            require (len(content) > 10 and content.startswith("Please")) or 
                    (context.user_type == "admin")
            
            ensure match(response, "^[A-Za-z0-9\\s]+$") and
                   not contains(response, "error")
        }
        """
        
        parser = LLMCLParser()
        ast = parser.parse(source)
        
        assert len(ast.requires) == 1
        assert len(ast.ensures) == 1
    
    def test_parse_new_builtin_functions(self):
        """Test parsing new built-in functions."""
        source = """
        contract BuiltinFunctionsContract {
            require email_valid(context.email) 
                message: "Invalid email address"
            
            require url_valid(context.url)
                message: "Invalid URL"
            
            ensure count(response, "error") == 0
                message: "Response should not contain errors"
            
            ensure first(context.items) != last(context.items)
                message: "First and last items should be different"
        }
        """
        
        parser = LLMCLParser()
        ast = parser.parse(source)
        
        assert len(ast.requires) == 2
        assert len(ast.ensures) == 2


class TestLLMCLCompiler:
    """Test LLMCL compiler functionality."""
    
    @pytest.mark.asyncio
    async def test_compile_and_validate(self):
        """Test compiling and validating a contract."""
        source = """
        contract LengthContract {
            require len(content) > 0
            ensure len(response) < 100
        }
        """
        
        compiler = LLMCLCompiler()
        compiled = compiler.compile(source)
        
        assert compiled.name == "LengthContract"
        
        # Test validation
        contract = compiled.contract_instance
        
        # Valid case
        result = await contract.validate("Short response", {"content": "Hello"})
        assert result.is_valid
        
        # Invalid case - response too long
        long_response = "x" * 101
        result = await contract.validate(long_response, {"content": "Hello"})
        assert not result.is_valid
    
    @pytest.mark.asyncio
    async def test_compile_new_builtin_functions(self):
        """Test compiling contract with new built-in functions."""
        source = """
        contract ValidationContract {
            require email_valid(email)
                message: "Invalid email address"
            
            require url_valid(url)  
                message: "Invalid URL"
                
            ensure round(score, 2) >= 0.95
                message: "Score must be at least 0.95"
                
            ensure count(response, "error") == 0
                message: "Response should not contain errors"
                
            ensure first(words) == "Hello"
                message: "Response must start with Hello"
                
            ensure last(words) == "goodbye"
                message: "Response must end with goodbye"
        }
        """
        
        compiler = LLMCLCompiler()
        compiled = compiler.compile(source)
        contract = compiled.contract_instance
        
        # Test valid case
        context = {
            "email": "test@example.com",
            "url": "https://example.com",
            "score": 0.9567,
            "words": ["Hello", "world", "goodbye"]
        }
        result = await contract.validate("This is a clean response", context)
        assert result.is_valid
        
        # Test invalid email
        context["email"] = "invalid-email"
        result = await contract.validate("This is a clean response", context)
        assert not result.is_valid
        assert "Invalid email" in result.message
        
        # Test invalid URL
        context["email"] = "test@example.com"
        context["url"] = "not-a-url"
        result = await contract.validate("This is a clean response", context)
        assert not result.is_valid
        assert "Invalid URL" in result.message
        
        # Test count function
        context["url"] = "https://example.com"
        result = await contract.validate("This has an error in it", context)
        assert not result.is_valid
        assert "should not contain errors" in result.message
        
        # Test first/last functions
        context["words"] = ["Goodbye", "world", "hello"]
        result = await contract.validate("This is a clean response", context)
        assert not result.is_valid
    
    @pytest.mark.asyncio
    async def test_compile_with_auto_fix(self):
        """Test compiling contract with auto-fix."""
        source = """
        contract JSONContract {
            ensure json_valid(response)
                message: "Response must be valid JSON"
                auto_fix: '{"error": "Invalid response", "original": ' + str(response) + '}'
        }
        """
        
        compiler = LLMCLCompiler()
        compiled = compiler.compile(source)
        
        contract = compiled.contract_instance
        
        # Test invalid JSON
        result = await contract.validate("invalid json", {})
        assert not result.is_valid
        assert result.auto_fix_suggestion is not None
        assert "Invalid response" in result.auto_fix_suggestion


class TestConflictResolver:
    """Test conflict resolution functionality."""
    
    def test_detect_format_conflict(self):
        """Test detecting format conflicts."""
        from llm_contracts.contracts.base import JSONFormatContract
        
        contract1 = JSONFormatContract()
        contract1.required_format = "json"
        
        contract2 = JSONFormatContract()
        contract2.required_format = "xml"
        
        resolver = ConflictResolver()
        conflicts = resolver.detect_conflicts([contract1, contract2])
        
        assert len(conflicts) == 1
        assert conflicts[0].type == ConflictType.FORMAT
    
    def test_resolve_conflicts_first_wins(self):
        """Test first-wins conflict resolution."""
        from llm_contracts.contracts.base import PromptLengthContract
        
        contract1 = PromptLengthContract(max_tokens=100)
        contract1.max_length = 100
        
        contract2 = PromptLengthContract(max_tokens=200)
        contract2.max_length = 200
        
        resolver = ConflictResolver()
        resolved, conflicts = resolver.resolve_conflicts(
            [contract1, contract2],
            ResolutionStrategy.FIRST_WINS
        )
        
        assert len(resolved) == 1
        assert resolved[0].max_length == 100
    
    def test_resolve_conflicts_most_restrictive(self):
        """Test most restrictive conflict resolution."""
        from llm_contracts.contracts.base import PromptLengthContract
        
        contract1 = PromptLengthContract(max_tokens=100)
        contract1.max_length = 100
        contract1.min_length = 10
        
        contract2 = PromptLengthContract(max_tokens=200)
        contract2.max_length = 200
        contract2.min_length = 20
        
        resolver = ConflictResolver()
        resolved, conflicts = resolver.resolve_conflicts(
            [contract1, contract2],
            ResolutionStrategy.MOST_RESTRICTIVE
        )
        
        # Most restrictive should have min of 20 and max of 100
        assert len(resolved) >= 1


class TestLLMCLRuntime:
    """Test LLMCL runtime functionality."""
    
    @pytest.mark.asyncio
    async def test_runtime_basic_validation(self):
        """Test basic runtime validation."""
        source = """
        contract RuntimeContract {
            ensure len(response) > 10
                message: "Response too short"
        }
        """
        
        runtime = LLMCLRuntime()
        
        # Load contract
        contract_name = await runtime.load_contract(source)
        
        # Create context
        context = runtime.create_context("test_context")
        runtime.add_contract_to_context("test_context", contract_name)
        
        # Validate - should fail
        result = await runtime.validate("Short", "test_context")
        assert not result.is_valid
        assert "Response too short" in result.message
        
        # Validate - should pass
        result = await runtime.validate("This is a longer response", "test_context")
        assert result.is_valid
    
    @pytest.mark.asyncio
    async def test_runtime_conflict_resolution(self):
        """Test runtime conflict resolution."""
        source1 = """
        contract Contract1(priority = high) {
            ensure len(response) < 50
        }
        """
        
        source2 = """
        contract Contract2(priority = low) {
            ensure len(response) > 100
        }
        """
        
        runtime = LLMCLRuntime()
        
        # Load contracts
        contract1 = await runtime.load_contract(source1)
        contract2 = await runtime.load_contract(source2)
        
        # Create context with first-wins strategy
        context = runtime.create_context(
            "test_context",
            ResolutionStrategy.FIRST_WINS
        )
        runtime.add_contract_to_context("test_context", contract1)
        runtime.add_contract_to_context("test_context", contract2)
        
        # Validate - should use first contract (< 50)
        result = await runtime.validate("x" * 30, "test_context")
        assert result.is_valid
        
        result = await runtime.validate("x" * 60, "test_context")
        assert not result.is_valid
    
    @pytest.mark.asyncio
    async def test_runtime_probabilistic_validation(self):
        """Test probabilistic validation."""
        source = """
        contract ProbabilisticContract {
            ensure_prob json_valid(response), 0.8
                window_size: 10
        }
        """
        
        runtime = LLMCLRuntime()
        contract_name = await runtime.load_contract(source)
        
        context = runtime.create_context("test_context")
        runtime.add_contract_to_context("test_context", contract_name)
        
        # Validate multiple times
        valid_json = '{"valid": true}'
        invalid_json = 'invalid'
        
        # 8 valid, 2 invalid = 80% success rate
        for i in range(8):
            await runtime.validate(valid_json, "test_context")
        
        for i in range(2):
            await runtime.validate(invalid_json, "test_context")
        
        # Check statistics
        stats = runtime.get_context_statistics("test_context")
        assert stats['total_validations'] == 10
    
    @pytest.mark.asyncio
    async def test_runtime_auto_fix(self):
        """Test auto-fix functionality."""
        source = """
        contract AutoFixContract {
            ensure json_valid(response)
                auto_fix: '{"fixed": true, "original": "' + response + '"}'
        }
        """
        
        runtime = LLMCLRuntime()
        contract_name = await runtime.load_contract(source)
        
        context = runtime.create_context("test_context")
        runtime.add_contract_to_context("test_context", contract_name)
        
        # Apply auto-fix
        fixed = await runtime.apply_auto_fix("invalid json", "test_context")
        assert "fixed" in fixed
        assert "true" in fixed


class TestLLMCLExamples:
    """Test real-world LLMCL examples."""
    
    @pytest.mark.asyncio
    async def test_chatbot_safety_contract(self):
        """Test a comprehensive chatbot safety contract."""
        source = """
        contract ChatbotSafety(
            priority = critical,
            conflict_resolution = most_restrictive
        ) {
            # Input validation
            require len(content) > 0 and len(content) < 4000
                message: "Input must be between 1 and 4000 characters"
            
            require not match(content, "(?i)(hack|exploit|injection)")
                message: "Potential security threat detected"
            
            # Output validation
            ensure not contains(response, "I'm sorry") or len(response) > 50
                message: "Provide helpful responses, not just apologies"
            
            ensure not match(response, "(?i)(password|secret|token)")
                message: "Response contains sensitive information"
            
            # Temporal constraints
            temporal always not contains(response, "ERROR")
                message: "System errors should not be exposed"
            
            temporal within 3 contains(response, "help")
                message: "Offer help within 3 turns if user seems confused"
        }
        """
        
        runtime = LLMCLRuntime()
        contract_name = await runtime.load_contract(source)
        
        context = runtime.create_context("chatbot_session")
        runtime.add_contract_to_context("chatbot_session", contract_name)
        
        # Test various scenarios
        
        # Valid input and output
        result = await runtime.validate(
            "Hello! How can I help you today?",
            "chatbot_session",
            additional_context={"content": "Hi there!"}
        )
        assert result.is_valid
        
        # Invalid - contains sensitive info
        result = await runtime.validate(
            "Your password is: 12345",
            "chatbot_session",
            additional_context={"content": "What's my password?"}
        )
        assert not result.is_valid
        assert "sensitive information" in result.message
    
    @pytest.mark.asyncio
    async def test_api_response_contract(self):
        """Test API response formatting contract."""
        source = """
        contract APIResponse {
            # Ensure valid JSON structure
            ensure json_valid(response)
                message: "Response must be valid JSON"
                auto_fix: '{"error": "Invalid response format"}'
            
            # Ensure required fields
            ensure contains(response, '"status"') and contains(response, '"data"')
                message: "Response must contain status and data fields"
            
            # Probabilistic - most responses should be successful
            ensure_prob contains(response, '"status": "success"'), 0.9
                message: "90% of responses should be successful"
                window_size: 100
        }
        """
        
        runtime = LLMCLRuntime()
        contract_name = await runtime.load_contract(source)
        
        context = runtime.create_context("api_context")
        runtime.add_contract_to_context("api_context", contract_name)
        
        # Test valid response
        valid_response = '{"status": "success", "data": {"id": 123}}'
        result = await runtime.validate(valid_response, "api_context")
        assert result.is_valid
        
        # Test invalid response
        invalid_response = '{"incomplete": true}'
        result = await runtime.validate(invalid_response, "api_context")
        assert not result.is_valid


class TestLLMCLIntegration:
    """Test LLMCL integration helpers."""
    
    def test_llmcl_to_contract(self):
        """Test converting LLMCL to contract."""
        from llm_contracts.language import llmcl_to_contract
        
        source = """
        contract TestContract {
            ensure len(response) > 0
        }
        """
        
        contract = llmcl_to_contract(source)
        assert contract.name == "TestContract"
        assert contract.description == "Compiled from LLMCL: TestContract"
    
    @pytest.mark.asyncio
    async def test_llmcl_decorator(self):
        """Test LLMCL decorator."""
        from llm_contracts.language import llmcl_contract
        
        @llmcl_contract("""
            contract DecoratorTest {
                require len(prompt) > 0
                    message: "Prompt cannot be empty"
                ensure len(result) < 100
                    message: "Result too long"
            }
        """)
        async def process_prompt(prompt: str) -> str:
            return f"Processed: {prompt}"
        
        # Valid case
        result = await process_prompt("Hello")
        assert result == "Processed: Hello"
        
        # Invalid precondition
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await process_prompt("")
        
        # Invalid postcondition
        @llmcl_contract("""
            contract LongResponseTest {
                ensure len(result) < 10
            }
        """)
        async def long_response(prompt: str) -> str:
            return "This is a very long response"
        
        with pytest.raises(ValueError, match="Contract postcondition failed"):
            await long_response("test")
    
    def test_create_contract_bundle(self):
        """Test creating contract bundles."""
        from llm_contracts.language import create_contract_bundle
        
        sources = [
            'contract C1 { ensure len(response) > 0 }',
            'contract C2 { ensure json_valid(response) }'
        ]
        
        contracts = create_contract_bundle(sources)
        assert len(contracts) == 2
        assert contracts[0].name == "C1"
        assert contracts[1].name == "C2"
    
    def test_get_template_contract(self):
        """Test getting template contracts."""
        from llm_contracts.language import get_template_contract
        
        # Valid template
        contract = get_template_contract("safe_chat")
        assert contract.name == "SafeChat"
        
        # Invalid template
        with pytest.raises(ValueError, match="Unknown template"):
            get_template_contract("nonexistent")
    
    @pytest.mark.asyncio
    async def test_llmcl_enabled_client(self):
        """Test LLMCL-enabled client."""
        from llm_contracts.language import LLMCLEnabledClient
        from unittest.mock import MagicMock
        
        # Mock client
        client = LLMCLEnabledClient(api_key="test")
        
        # Add contract
        client.add_llmcl_contract("""
            contract ClientTest {
                ensure len(response) > 0
            }
        """)
        
        # Verify contract was added
        assert len(client.runtime.contracts) == 1
        assert "ClientTest" in client.runtime.contracts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])