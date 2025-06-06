"""Abstract Syntax Tree nodes for LLMCL."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum, auto


class ContractPriority(Enum):
    """Priority levels for contract precedence."""
    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    DEFAULT = 50


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    FIRST_WINS = auto()
    LAST_WINS = auto()
    MOST_RESTRICTIVE = auto()
    LEAST_RESTRICTIVE = auto()
    MERGE = auto()
    FAIL_ON_CONFLICT = auto()
    CUSTOM = auto()


class TemporalOperator(Enum):
    """Temporal operators for multi-turn contracts."""
    ALWAYS = auto()          # □ (box) - always true
    EVENTUALLY = auto()      # ◇ (diamond) - eventually true
    NEXT = auto()           # ○ (circle) - true in next turn
    UNTIL = auto()          # U - true until condition
    SINCE = auto()          # S - true since condition
    WITHIN = auto()         # true within N turns


class ASTNode(ABC):
    """Base class for all AST nodes."""
    def __init__(self, line: int = 0, column: int = 0, source_file: Optional[str] = None):
        self.line = line
        self.column = column
        self.source_file = source_file
    
    @abstractmethod
    def accept(self, visitor: "ASTVisitor") -> Any:
        """Accept a visitor for processing."""
        pass


@dataclass
class ContractNode(ASTNode):
    """Root node representing a complete contract."""
    name: str
    description: Optional[str] = None
    priority: ContractPriority = ContractPriority.DEFAULT
    conflict_resolution: ConflictResolution = ConflictResolution.FAIL_ON_CONFLICT
    requires: List["RequireNode"] = field(default_factory=list)
    ensures: List[Union["EnsureNode", "EnsureProbNode"]] = field(default_factory=list)
    temporal: List["TemporalNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    imports: List["ImportNode"] = field(default_factory=list)
    line: int = 0
    column: int = 0
    source_file: Optional[str] = None
    
    def __post_init__(self):
        super().__init__(self.line, self.column, self.source_file)
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_contract(self)


@dataclass
class ImportNode(ASTNode):
    """Import statement for reusing contracts."""
    module: str
    contracts: List[str] = field(default_factory=list)  # Empty means import all
    alias: Optional[str] = None
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_import(self)


@dataclass
class ExpressionNode(ASTNode):
    """Base class for expressions."""
    pass


@dataclass
class BinaryOpNode(ExpressionNode):
    """Binary operation node."""
    operator: str
    left: ExpressionNode
    right: ExpressionNode
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_binary_op(self)


@dataclass
class UnaryOpNode(ExpressionNode):
    """Unary operation node."""
    operator: str
    operand: ExpressionNode
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_unary_op(self)


@dataclass
class FunctionCallNode(ExpressionNode):
    """Function call node."""
    name: str
    args: List[ExpressionNode] = field(default_factory=list)
    kwargs: Dict[str, ExpressionNode] = field(default_factory=dict)
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_function_call(self)


@dataclass
class IdentifierNode(ExpressionNode):
    """Variable or identifier reference."""
    name: str
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_identifier(self)


@dataclass
class LiteralNode(ExpressionNode):
    """Literal value node."""
    value: Any
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_literal(self)


@dataclass
class AttributeAccessNode(ExpressionNode):
    """Attribute access (e.g., response.content)."""
    object: ExpressionNode
    attribute: str
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_attribute_access(self)


@dataclass
class RequireNode(ASTNode):
    """Precondition (input validation)."""
    condition: ExpressionNode
    message: Optional[str] = None
    severity: str = "error"
    auto_fix: Optional[ExpressionNode] = None
    tags: List[str] = field(default_factory=list)
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_require(self)


@dataclass
class EnsureNode(ASTNode):
    """Postcondition (output validation)."""
    condition: ExpressionNode
    message: Optional[str] = None
    severity: str = "error"
    auto_fix: Optional[ExpressionNode] = None
    tags: List[str] = field(default_factory=list)
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_ensure(self)


@dataclass
class EnsureProbNode(ASTNode):
    """Probabilistic postcondition."""
    condition: ExpressionNode
    probability: float
    message: Optional[str] = None
    window_size: int = 100  # Number of calls to evaluate over
    tags: List[str] = field(default_factory=list)
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_ensure_prob(self)


@dataclass
class TemporalNode(ASTNode):
    """Temporal constraint for multi-turn conversations."""
    operator: TemporalOperator
    condition: ExpressionNode
    message: Optional[str] = None
    scope: Optional[int] = None  # For WITHIN operator
    tags: List[str] = field(default_factory=list)
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_temporal(self)


@dataclass
class CompositionNode(ASTNode):
    """Contract composition node."""
    base_contracts: List[str]
    override_priority: Optional[ContractPriority] = None
    merge_strategy: ConflictResolution = ConflictResolution.MERGE
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_composition(self)


@dataclass
class ConditionalNode(ASTNode):
    """Conditional contract application."""
    condition: ExpressionNode
    then_contracts: List[Union[RequireNode, EnsureNode]]
    else_contracts: List[Union[RequireNode, EnsureNode]] = field(default_factory=list)
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_conditional(self)


class ASTVisitor(ABC):
    """Visitor interface for AST traversal."""
    
    @abstractmethod
    def visit_contract(self, node: ContractNode) -> Any:
        pass
    
    @abstractmethod
    def visit_import(self, node: ImportNode) -> Any:
        pass
    
    @abstractmethod
    def visit_require(self, node: RequireNode) -> Any:
        pass
    
    @abstractmethod
    def visit_ensure(self, node: EnsureNode) -> Any:
        pass
    
    @abstractmethod
    def visit_ensure_prob(self, node: EnsureProbNode) -> Any:
        pass
    
    @abstractmethod
    def visit_temporal(self, node: TemporalNode) -> Any:
        pass
    
    @abstractmethod
    def visit_binary_op(self, node: BinaryOpNode) -> Any:
        pass
    
    @abstractmethod
    def visit_unary_op(self, node: UnaryOpNode) -> Any:
        pass
    
    @abstractmethod
    def visit_function_call(self, node: FunctionCallNode) -> Any:
        pass
    
    @abstractmethod
    def visit_identifier(self, node: IdentifierNode) -> Any:
        pass
    
    @abstractmethod
    def visit_literal(self, node: LiteralNode) -> Any:
        pass
    
    @abstractmethod
    def visit_attribute_access(self, node: AttributeAccessNode) -> Any:
        pass
    
    @abstractmethod
    def visit_composition(self, node: CompositionNode) -> Any:
        pass
    
    @abstractmethod
    def visit_conditional(self, node: ConditionalNode) -> Any:
        pass