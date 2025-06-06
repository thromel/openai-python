"""LLMCL Compiler - Compiles AST into executable contracts."""

from typing import Any, Dict, List, Optional, Type, Callable
from dataclasses import dataclass
import ast as python_ast
import operator
from ..core.interfaces import ContractBase, ValidationResult, ContractType
from ..contracts.base import (
    InputContract, OutputContract, TemporalContract,
    PromptLengthContract, JSONFormatContract, ContentPolicyContract
)
from .ast_nodes import (
    ASTNode, ASTVisitor, ContractNode, RequireNode, EnsureNode,
    EnsureProbNode, TemporalNode, ExpressionNode, BinaryOpNode,
    UnaryOpNode, FunctionCallNode, IdentifierNode, LiteralNode,
    AttributeAccessNode, ImportNode, CompositionNode, ConditionalNode,
    TemporalOperator
)
from .parser import LLMCLParser
from .conflict_resolver import ConflictResolver, ConflictResolutionStrategy


@dataclass
class CompiledContract:
    """A compiled contract ready for execution."""
    name: str
    contract_instance: ContractBase
    metadata: Dict[str, Any]
    source_ast: ContractNode


class ExpressionEvaluator:
    """Evaluates LLMCL expressions."""
    
    def __init__(self, context: Dict[str, Any] = None):
        self.context = context or {}
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '%': operator.mod,
            '==': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '>': operator.gt,
            '<=': operator.le,
            '>=': operator.ge,
            'and': operator.and_,
            'or': operator.or_,
            'in': operator.contains,
        }
        
        self.functions = {
            'len': len,
            'max': max,
            'min': min,
            'abs': abs,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'round': round,
            'contains': lambda s, sub: sub in s,
            'startswith': lambda s, prefix: s.startswith(prefix),
            'endswith': lambda s, suffix: s.endswith(suffix),
            'match': self._regex_match,
            'json_valid': self._json_valid,
            'email_valid': self._email_valid,
            'url_valid': self._url_valid,
            'count': self._count,
            'first': self._first,
            'last': self._last,
        }
    
    def evaluate(self, expr: ExpressionNode) -> Any:
        """Evaluate an expression node."""
        if isinstance(expr, LiteralNode):
            return expr.value
        
        elif isinstance(expr, IdentifierNode):
            if expr.name in self.context:
                return self.context[expr.name]
            raise NameError(f"Undefined variable: {expr.name}")
        
        elif isinstance(expr, BinaryOpNode):
            left = self.evaluate(expr.left)
            right = self.evaluate(expr.right)
            
            if expr.operator in self.operators:
                return self.operators[expr.operator](left, right)
            raise ValueError(f"Unknown operator: {expr.operator}")
        
        elif isinstance(expr, UnaryOpNode):
            operand = self.evaluate(expr.operand)
            
            if expr.operator == 'not':
                return not operand
            elif expr.operator == '-':
                return -operand
            raise ValueError(f"Unknown unary operator: {expr.operator}")
        
        elif isinstance(expr, FunctionCallNode):
            if expr.name in self.functions:
                args = [self.evaluate(arg) for arg in expr.args]
                kwargs = {k: self.evaluate(v) for k, v in expr.kwargs.items()}
                return self.functions[expr.name](*args, **kwargs)
            raise ValueError(f"Unknown function: {expr.name}")
        
        elif isinstance(expr, AttributeAccessNode):
            obj = self.evaluate(expr.object)
            return getattr(obj, expr.attribute)
        
        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")
    
    def _regex_match(self, text: str, pattern: str) -> bool:
        """Check if text matches regex pattern."""
        import re
        return bool(re.match(pattern, text))
    
    def _json_valid(self, text: str) -> bool:
        """Check if text is valid JSON."""
        import json
        try:
            json.loads(text)
            return True
        except:
            return False
    
    def _email_valid(self, text: str) -> bool:
        """Check if text is a valid email address."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, text))
    
    def _url_valid(self, text: str) -> bool:
        """Check if text is a valid URL."""
        import re
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, text))
    
    def _count(self, collection: Any, item: Any = None) -> int:
        """Count occurrences of item in collection or length if no item specified."""
        if item is None:
            return len(collection)
        if isinstance(collection, str):
            return collection.count(item)
        return sum(1 for x in collection if x == item)
    
    def _first(self, collection: Any, default: Any = None) -> Any:
        """Get first element of collection."""
        try:
            if isinstance(collection, (list, tuple, str)):
                return collection[0] if collection else default
            # For iterables
            return next(iter(collection), default)
        except:
            return default
    
    def _last(self, collection: Any, default: Any = None) -> Any:
        """Get last element of collection."""
        try:
            if isinstance(collection, (list, tuple, str)):
                return collection[-1] if collection else default
            # For iterables, we need to consume it
            item = default
            for item in collection:
                pass
            return item
        except:
            return default


class LLMCLCompiler(ASTVisitor):
    """Compiles LLMCL AST into executable contracts."""
    
    def __init__(self):
        self.parser = LLMCLParser()
        self.conflict_resolver = ConflictResolver()
        self.compiled_contracts = {}
        self.import_cache = {}
    
    def compile(self, source: str) -> CompiledContract:
        """Compile LLMCL source code into executable contract."""
        # Parse source into AST
        ast = self.parser.parse(source)
        
        # Visit AST to compile
        return self.visit_contract(ast)
    
    def compile_file(self, file_path: str) -> CompiledContract:
        """Compile LLMCL from file."""
        with open(file_path, 'r') as f:
            source = f.read()
        return self.compile(source)
    
    def visit_contract(self, node: ContractNode) -> CompiledContract:
        """Compile a contract node."""
        # Process imports first
        for import_node in node.imports:
            self.visit_import(import_node)
        
        # Create dynamic contract class
        class DynamicContract(ContractBase):
            def __init__(self):
                super().__init__(
                    name=node.name,
                    description=node.description or f"Compiled from LLMCL: {node.name}",
                    contract_type=self._determine_contract_type(node),
                    priority=node.priority
                )
                self.requires = node.requires
                self.ensures = node.ensures
                self.temporal = node.temporal
                self.conflict_resolution = node.conflict_resolution
                self.evaluator = ExpressionEvaluator()
            
            async def validate(self, content: str, context: Optional[Dict] = None) -> ValidationResult:
                """Validate content against contract."""
                # Set up evaluation context
                self.evaluator.context = {
                    'content': content,
                    'response': content,  # Alias
                    'context': context or {},
                    **(context or {})
                }
                
                # Check preconditions
                for require in self.requires:
                    result = self._validate_require(require)
                    if not result.is_valid:
                        return result
                
                # Check postconditions
                for ensure in self.ensures:
                    if isinstance(ensure, EnsureNode):
                        result = self._validate_ensure(ensure)
                        if not result.is_valid:
                            return result
                    elif isinstance(ensure, EnsureProbNode):
                        # Probabilistic validation handled separately
                        pass
                
                # Check temporal constraints
                for temporal in self.temporal:
                    result = self._validate_temporal(temporal, context)
                    if not result.is_valid:
                        return result
                
                return ValidationResult(is_valid=True)
            
            def _validate_require(self, require: RequireNode) -> ValidationResult:
                """Validate a require clause."""
                try:
                    condition_met = self.evaluator.evaluate(require.condition)
                    if not condition_met:
                        return ValidationResult(
                            is_valid=False,
                            message=require.message or f"Precondition failed: {require.condition}",
                            violation_type=require.severity,
                            auto_fix_suggestion=self._evaluate_auto_fix(require.auto_fix)
                        )
                    return ValidationResult(is_valid=True)
                except Exception as e:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Error evaluating precondition: {str(e)}",
                        violation_type="error"
                    )
            
            def _validate_ensure(self, ensure: EnsureNode) -> ValidationResult:
                """Validate an ensure clause."""
                try:
                    condition_met = self.evaluator.evaluate(ensure.condition)
                    if not condition_met:
                        return ValidationResult(
                            is_valid=False,
                            message=ensure.message or f"Postcondition failed: {ensure.condition}",
                            violation_type=ensure.severity,
                            auto_fix_suggestion=self._evaluate_auto_fix(ensure.auto_fix)
                        )
                    return ValidationResult(is_valid=True)
                except Exception as e:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Error evaluating postcondition: {str(e)}",
                        violation_type="error"
                    )
            
            def _validate_temporal(self, temporal: TemporalNode, context: Dict) -> ValidationResult:
                """Validate temporal constraint."""
                # This is a simplified implementation
                # Real temporal validation would require conversation history
                try:
                    if temporal.operator == TemporalOperator.ALWAYS:
                        # Check if condition holds in current state
                        condition_met = self.evaluator.evaluate(temporal.condition)
                        if not condition_met:
                            return ValidationResult(
                                is_valid=False,
                                message=temporal.message or f"Temporal constraint violated: always {temporal.condition}",
                                violation_type="error"
                            )
                    # Add other temporal operators as needed
                    return ValidationResult(is_valid=True)
                except Exception as e:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Error evaluating temporal constraint: {str(e)}",
                        violation_type="error"
                    )
            
            def _evaluate_auto_fix(self, auto_fix_expr: Optional[ExpressionNode]) -> Optional[str]:
                """Evaluate auto-fix expression."""
                if not auto_fix_expr:
                    return None
                try:
                    return str(self.evaluator.evaluate(auto_fix_expr))
                except:
                    return None
            
            def _determine_contract_type(self, node: ContractNode) -> ContractType:
                """Determine contract type from node."""
                if node.requires and not node.ensures:
                    return ContractType.INPUT
                elif node.ensures and not node.requires:
                    return ContractType.OUTPUT
                elif node.temporal:
                    return ContractType.TEMPORAL
                else:
                    return ContractType.SEMANTIC_CONSISTENCY
        
        # Create instance
        contract_instance = DynamicContract()
        
        # Create compiled contract
        compiled = CompiledContract(
            name=node.name,
            contract_instance=contract_instance,
            metadata={
                'priority': node.priority,
                'conflict_resolution': node.conflict_resolution,
                'source_file': node.source_file,
                'tags': self._extract_tags(node)
            },
            source_ast=node
        )
        
        # Cache for reuse
        self.compiled_contracts[node.name] = compiled
        
        return compiled
    
    def visit_import(self, node: ImportNode) -> None:
        """Handle import statements."""
        # This is a simplified implementation
        # Real implementation would load contracts from modules
        if node.module in self.import_cache:
            return
        
        # Try to load module
        try:
            # For now, just mark as imported
            self.import_cache[node.module] = True
        except Exception as e:
            raise ImportError(f"Failed to import {node.module}: {str(e)}")
    
    def _extract_tags(self, node: ContractNode) -> List[str]:
        """Extract all tags from contract."""
        tags = []
        for require in node.requires:
            tags.extend(require.tags)
        for ensure in node.ensures:
            if hasattr(ensure, 'tags'):
                tags.extend(ensure.tags)
        for temporal in node.temporal:
            tags.extend(temporal.tags)
        return list(set(tags))
    
    # Implement other visitor methods as needed
    def visit_require(self, node: RequireNode) -> Any:
        pass
    
    def visit_ensure(self, node: EnsureNode) -> Any:
        pass
    
    def visit_ensure_prob(self, node: EnsureProbNode) -> Any:
        pass
    
    def visit_temporal(self, node: TemporalNode) -> Any:
        pass
    
    def visit_binary_op(self, node: BinaryOpNode) -> Any:
        pass
    
    def visit_unary_op(self, node: UnaryOpNode) -> Any:
        pass
    
    def visit_function_call(self, node: FunctionCallNode) -> Any:
        pass
    
    def visit_identifier(self, node: IdentifierNode) -> Any:
        pass
    
    def visit_literal(self, node: LiteralNode) -> Any:
        pass
    
    def visit_attribute_access(self, node: AttributeAccessNode) -> Any:
        pass
    
    def visit_composition(self, node: CompositionNode) -> Any:
        pass
    
    def visit_conditional(self, node: ConditionalNode) -> Any:
        pass