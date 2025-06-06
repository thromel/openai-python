"""LLMCL Runtime - Executes compiled contracts with conflict resolution."""

from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio
from collections import defaultdict
from dataclasses import dataclass
import time
from ..core.interfaces import ContractBase, ValidationResult
from .compiler import LLMCLCompiler, CompiledContract
from .conflict_resolver import ConflictResolver, ConflictResolutionStrategy
from .ast_nodes import ContractPriority, ConflictResolution


@dataclass
class RuntimeContext:
    """Runtime context for contract execution."""
    conversation_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    statistics: Dict[str, Any]
    active_contracts: List[CompiledContract]
    conflict_strategy: ConflictResolutionStrategy


class ProbabilisticValidator:
    """Handles probabilistic contract validation."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = defaultdict(list)
    
    def record_result(self, contract_name: str, condition_name: str, result: bool):
        """Record a validation result."""
        key = f"{contract_name}:{condition_name}"
        self.history[key].append(result)
        
        # Keep only last window_size results
        if len(self.history[key]) > self.window_size:
            self.history[key] = self.history[key][-self.window_size:]
    
    def get_success_rate(self, contract_name: str, condition_name: str) -> float:
        """Get success rate for a condition."""
        key = f"{contract_name}:{condition_name}"
        results = self.history[key]
        
        if not results:
            return 0.0
        
        return sum(results) / len(results)
    
    def check_probability(
        self, 
        contract_name: str, 
        condition_name: str, 
        required_probability: float
    ) -> Tuple[bool, float]:
        """Check if probability requirement is met."""
        actual_rate = self.get_success_rate(contract_name, condition_name)
        return actual_rate >= required_probability, actual_rate


class LLMCLRuntime:
    """Runtime for executing LLMCL contracts."""
    
    def __init__(self):
        self.compiler = LLMCLCompiler()
        self.conflict_resolver = ConflictResolver()
        self.probabilistic_validator = ProbabilisticValidator()
        self.loaded_contracts: Dict[str, CompiledContract] = {}
        self.runtime_contexts: Dict[str, RuntimeContext] = {}
        self.contract_cache = {}
        self.metrics = {
            'validations_run': 0,
            'violations_found': 0,
            'conflicts_resolved': 0,
            'auto_fixes_applied': 0
        }
    
    async def load_contract(self, source: str, name: Optional[str] = None) -> str:
        """Load a contract from source."""
        compiled = self.compiler.compile(source)
        
        contract_name = name or compiled.name
        self.loaded_contracts[contract_name] = compiled
        
        return contract_name
    
    async def load_contract_file(self, file_path: str, name: Optional[str] = None) -> str:
        """Load a contract from file."""
        compiled = self.compiler.compile_file(file_path)
        
        contract_name = name or compiled.name
        self.loaded_contracts[contract_name] = compiled
        
        return contract_name
    
    def create_context(
        self,
        context_id: str,
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.FAIL_ON_CONFLICT
    ) -> RuntimeContext:
        """Create a new runtime context."""
        context = RuntimeContext(
            conversation_history=[],
            metadata={},
            statistics=defaultdict(int),
            active_contracts=[],
            conflict_strategy=conflict_strategy
        )
        
        self.runtime_contexts[context_id] = context
        return context
    
    def add_contract_to_context(self, context_id: str, contract_name: str):
        """Add a contract to a runtime context."""
        if context_id not in self.runtime_contexts:
            raise ValueError(f"Unknown context: {context_id}")
        
        if contract_name not in self.loaded_contracts:
            raise ValueError(f"Unknown contract: {contract_name}")
        
        context = self.runtime_contexts[context_id]
        contract = self.loaded_contracts[contract_name]
        
        context.active_contracts.append(contract)
    
    async def validate(
        self,
        content: str,
        context_id: str,
        validation_type: str = "output",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate content against contracts in context."""
        if context_id not in self.runtime_contexts:
            raise ValueError(f"Unknown context: {context_id}")
        
        context = self.runtime_contexts[context_id]
        self.metrics['validations_run'] += 1
        
        # Prepare validation context
        validation_context = {
            'conversation_history': context.conversation_history,
            'metadata': context.metadata,
            'validation_type': validation_type,
            **(additional_context or {})
        }
        
        # Get contracts to validate
        contracts = [c.contract_instance for c in context.active_contracts]
        
        # Resolve conflicts
        resolved_contracts, conflicts = self.conflict_resolver.resolve_conflicts(
            contracts,
            context.conflict_strategy
        )
        
        if conflicts:
            self.metrics['conflicts_resolved'] += len(conflicts)
        
        # Run validation on resolved contracts
        violations = []
        auto_fixes = []
        
        for contract in resolved_contracts:
            try:
                result = await contract.validate(content, validation_context)
                
                if not result.is_valid:
                    violations.append(result)
                    self.metrics['violations_found'] += 1
                    
                    if result.auto_fix_suggestion:
                        auto_fixes.append({
                            'contract': contract.name,
                            'fix': result.auto_fix_suggestion
                        })
                
                # Handle probabilistic validation
                await self._handle_probabilistic_validation(
                    contract,
                    content,
                    validation_context
                )
                
            except Exception as e:
                violations.append(ValidationResult(
                    is_valid=False,
                    message=f"Error in contract {contract.name}: {str(e)}",
                    violation_type="error"
                ))
        
        # Update conversation history
        context.conversation_history.append({
            'timestamp': time.time(),
            'content': content,
            'validation_type': validation_type,
            'violations': len(violations),
            'auto_fixes': len(auto_fixes)
        })
        
        # Trim history if needed
        if len(context.conversation_history) > 100:
            context.conversation_history = context.conversation_history[-100:]
        
        # Return combined result
        if violations:
            return ValidationResult(
                is_valid=False,
                message="; ".join([v.message for v in violations]),
                violation_type="multiple" if len(violations) > 1 else violations[0].violation_type,
                metadata={
                    'violations': violations,
                    'auto_fixes': auto_fixes,
                    'conflicts_resolved': len(conflicts)
                }
            )
        
        return ValidationResult(
            is_valid=True,
            metadata={
                'contracts_validated': len(resolved_contracts),
                'conflicts_resolved': len(conflicts)
            }
        )
    
    async def _handle_probabilistic_validation(
        self,
        contract: ContractBase,
        content: str,
        context: Dict[str, Any]
    ):
        """Handle probabilistic contract validation."""
        # Check if contract has probabilistic ensures
        if hasattr(contract, 'ensures'):
            from .ast_nodes import EnsureProbNode
            from .compiler import ExpressionEvaluator
            
            evaluator = ExpressionEvaluator({
                'content': content,
                'response': content,
                'context': context,
                **context
            })
            
            for ensure in contract.ensures:
                if isinstance(ensure, EnsureProbNode):
                    # Evaluate condition
                    try:
                        condition_met = evaluator.evaluate(ensure.condition)
                        
                        # Record result
                        self.probabilistic_validator.record_result(
                            contract.name,
                            str(ensure.condition),
                            condition_met
                        )
                        
                        # Check if probability requirement is met
                        prob_met, actual_rate = self.probabilistic_validator.check_probability(
                            contract.name,
                            str(ensure.condition),
                            ensure.probability
                        )
                        
                        if not prob_met:
                            # Log warning but don't fail validation
                            print(f"Probabilistic constraint warning: {contract.name} - "
                                  f"Required: {ensure.probability}, Actual: {actual_rate}")
                    except Exception as e:
                        print(f"Error evaluating probabilistic constraint: {str(e)}")
    
    def get_context_statistics(self, context_id: str) -> Dict[str, Any]:
        """Get statistics for a runtime context."""
        if context_id not in self.runtime_contexts:
            raise ValueError(f"Unknown context: {context_id}")
        
        context = self.runtime_contexts[context_id]
        
        return {
            'total_validations': len(context.conversation_history),
            'active_contracts': len(context.active_contracts),
            'total_violations': sum(h['violations'] for h in context.conversation_history),
            'total_auto_fixes': sum(h['auto_fixes'] for h in context.conversation_history),
            'statistics': dict(context.statistics)
        }
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global runtime metrics."""
        return {
            **self.metrics,
            'loaded_contracts': len(self.loaded_contracts),
            'active_contexts': len(self.runtime_contexts),
            'probabilistic_checks': len(self.probabilistic_validator.history)
        }
    
    async def apply_auto_fix(
        self,
        content: str,
        context_id: str,
        fix_strategy: str = "first"
    ) -> str:
        """Apply auto-fix suggestions to content."""
        result = await self.validate(content, context_id)
        
        if result.is_valid:
            return content
        
        auto_fixes = result.metadata.get('auto_fixes', [])
        
        if not auto_fixes:
            return content
        
        # Apply fix based on strategy
        if fix_strategy == "first":
            fix = auto_fixes[0]['fix']
        elif fix_strategy == "all":
            # Apply all fixes sequentially
            fixed_content = content
            for fix_info in auto_fixes:
                fixed_content = fix_info['fix']
            fix = fixed_content
        else:
            return content
        
        self.metrics['auto_fixes_applied'] += 1
        return fix