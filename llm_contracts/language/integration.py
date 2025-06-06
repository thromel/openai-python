"""Integration helpers for LLMCL with OpenAI provider."""

from typing import Optional, Dict, Any, Union, List
from functools import wraps
import asyncio

from llm_contracts.language import LLMCLRuntime, LLMCLCompiler
from llm_contracts.contracts.base import ContractBase
from llm_contracts.providers.openai_provider import ImprovedOpenAIProvider


def llmcl_to_contract(llmcl_source: str) -> ContractBase:
    """Convert LLMCL source code to a contract instance.
    
    Args:
        llmcl_source: LLMCL contract source code
        
    Returns:
        ContractBase instance compiled from LLMCL
        
    Example:
        >>> contract = llmcl_to_contract('''
        ...     contract SafeResponse {
        ...         ensure not contains(response, "error")
        ...         ensure json_valid(response)
        ...     }
        ... ''')
    """
    compiler = LLMCLCompiler()
    compiled = compiler.compile(llmcl_source)
    return compiled.contract_instance


def llmcl_contract(contract_source: str):
    """Decorator to apply LLMCL contracts to functions.
    
    Args:
        contract_source: LLMCL contract source code
        
    Returns:
        Decorated function with contract validation
        
    Example:
        >>> @llmcl_contract('''
        ...     contract APIContract {
        ...         require len(prompt) < 4000
        ...         ensure json_valid(result)
        ...     }
        ... ''')
        ... async def call_api(prompt: str) -> str:
        ...     # Function implementation
        ...     return '{"status": "ok"}'
    """
    def decorator(func):
        # Compile contract once
        contract = llmcl_to_contract(contract_source)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Build context from function arguments
            context = {}
            
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Add all arguments to context
            context.update(bound_args.arguments)
            
            # Special handling for common parameter names
            if 'prompt' in context:
                context['content'] = context['prompt']
            elif 'content' not in context and args:
                context['content'] = str(args[0])
            
            # Validate preconditions
            if hasattr(contract, 'requires') and contract.requires:
                # Create a temporary result to check preconditions
                temp_result = await contract.validate("", context)
                if not temp_result.is_valid:
                    raise ValueError(f"Contract precondition failed: {temp_result.message}")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Validate postconditions
            validation_result = await contract.validate(str(result), context)
            if not validation_result.is_valid:
                if validation_result.auto_fix_suggestion:
                    # Apply auto-fix if available
                    return validation_result.auto_fix_suggestion
                raise ValueError(f"Contract postcondition failed: {validation_result.message}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Run async version in sync context
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class LLMCLEnabledClient:
    """OpenAI client with LLMCL contract support.
    
    This class wraps the ImprovedOpenAIProvider to add LLMCL contract support
    through a convenient interface.
    
    Example:
        >>> client = LLMCLEnabledClient(api_key="...")
        >>> 
        >>> # Add LLMCL contract
        >>> client.add_llmcl_contract('''
        ...     contract SafeChat {
        ...         require len(content) < 4000
        ...         ensure not contains(response, "error")
        ...     }
        ... ''')
        >>> 
        >>> # Use normally - contracts are enforced automatically
        >>> response = await client.chat.completions.create(...)
    """
    
    def __init__(self, **kwargs):
        """Initialize LLMCL-enabled OpenAI client.
        
        Args:
            **kwargs: Arguments passed to ImprovedOpenAIProvider
        """
        self.provider = ImprovedOpenAIProvider(**kwargs)
        self.runtime = LLMCLRuntime()
        self.context_name = "default"
        self.runtime.create_context(self.context_name)
    
    def add_llmcl_contract(self, contract_source: str, name: Optional[str] = None):
        """Add an LLMCL contract to the client.
        
        Args:
            contract_source: LLMCL contract source code
            name: Optional contract name (extracted from source if not provided)
        """
        # Load contract into runtime
        contract_name = asyncio.run(self.runtime.load_contract(contract_source, name))
        
        # Add to context
        self.runtime.add_contract_to_context(self.context_name, contract_name)
        
        # Also add to provider
        contract = llmcl_to_contract(contract_source)
        self.provider.add_contract(contract)
    
    def add_llmcl_file(self, file_path: str):
        """Add LLMCL contracts from a file.
        
        Args:
            file_path: Path to LLMCL file
        """
        with open(file_path, 'r') as f:
            self.add_llmcl_contract(f.read())
    
    def clear_contracts(self):
        """Clear all contracts."""
        self.runtime.clear_context(self.context_name)
        self.provider.contracts.clear()
    
    def __getattr__(self, name):
        """Delegate to provider."""
        return getattr(self.provider, name)


def create_contract_bundle(llmcl_sources: Union[str, List[str]]) -> List[ContractBase]:
    """Create a bundle of contracts from LLMCL sources.
    
    Args:
        llmcl_sources: Single LLMCL source or list of sources
        
    Returns:
        List of compiled contracts
        
    Example:
        >>> contracts = create_contract_bundle([
        ...     '''contract C1 { ensure len(response) > 0 }''',
        ...     '''contract C2 { ensure json_valid(response) }'''
        ... ])
    """
    if isinstance(llmcl_sources, str):
        llmcl_sources = [llmcl_sources]
    
    contracts = []
    for source in llmcl_sources:
        contracts.append(llmcl_to_contract(source))
    
    return contracts


# Example LLMCL contract templates
LLMCL_TEMPLATES = {
    "safe_chat": """
        contract SafeChat(priority = high) {
            require len(content) > 0 and len(content) < 4000
                message: "Input must be between 1 and 4000 characters"
            
            ensure not match(response, "(?i)(error|exception|failed)")
                message: "Response contains error indicators"
                
            temporal always len(response) > 0
                message: "Response should never be empty"
        }
    """,
    
    "json_api": """
        contract JSONAPIResponse {
            ensure json_valid(response)
                message: "Response must be valid JSON"
                auto_fix: '{"error": "Invalid response", "original": ' + str(response) + '}'
            
            ensure contains(response, '"status"')
                message: "Response must contain status field"
        }
    """,
    
    "length_limits": """
        contract LengthLimits(
            priority = medium,
            conflict_resolution = most_restrictive
        ) {
            require len(content) >= 10
                message: "Input too short (minimum 10 characters)"
                
            ensure len(response) <= 2000
                message: "Response too long (maximum 2000 characters)"
        }
    """,
    
    "no_pii": """
        contract NoPII(priority = critical) {
            ensure not match(response, "\\b\\d{3}-\\d{2}-\\d{4}\\b")
                message: "Response contains SSN pattern"
                
            ensure not match(response, "\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b")
                message: "Response contains email address"
                auto_fix: "[email redacted]"
        }
    """
}


def get_template_contract(template_name: str) -> ContractBase:
    """Get a pre-defined LLMCL contract template.
    
    Args:
        template_name: Name of the template (safe_chat, json_api, length_limits, no_pii)
        
    Returns:
        Compiled contract from template
        
    Example:
        >>> contract = get_template_contract("safe_chat")
    """
    if template_name not in LLMCL_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(LLMCL_TEMPLATES.keys())}")
    
    return llmcl_to_contract(LLMCL_TEMPLATES[template_name])