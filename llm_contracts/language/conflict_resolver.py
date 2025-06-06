"""Conflict resolution system for LLMCL contracts."""

from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
from dataclasses import dataclass
from ..core.interfaces import ContractBase, ValidationResult
from .ast_nodes import (
    ContractNode, RequireNode, EnsureNode, 
    ContractPriority, ConflictResolution as ConflictResolutionStrategy
)


class ConflictType(Enum):
    """Types of conflicts between contracts."""
    FORMAT = auto()          # Different format requirements
    LENGTH = auto()          # Conflicting length constraints
    CONTENT_POLICY = auto()  # Different content restrictions
    TEMPORAL = auto()        # Conflicting temporal constraints
    SEMANTIC = auto()        # Semantic contradictions
    PRIORITY = auto()        # Priority conflicts
    UNKNOWN = auto()         # Unclassified conflicts


class ConflictAction(Enum):
    """Actions to take when conflicts are detected."""
    MERGE = auto()           # Merge contracts into unified constraint
    OVERRIDE = auto()        # Higher priority overrides lower
    SKIP = auto()           # Skip conflicting contract
    FAIL = auto()           # Fail on conflict
    CUSTOM = auto()         # Use custom resolution


class ResolutionStrategy(Enum):
    """High-level resolution strategies."""
    FIRST_WINS = auto()
    LAST_WINS = auto()
    MOST_RESTRICTIVE = auto()
    LEAST_RESTRICTIVE = auto()
    MERGE = auto()
    FAIL_ON_CONFLICT = auto()


@dataclass
class ConflictInfo:
    """Information about a detected conflict."""
    type: ConflictType
    contract1: ContractBase
    contract2: ContractBase
    description: str
    severity: str = "warning"
    resolvable: bool = True
    suggested_action: Optional[ConflictAction] = None


@dataclass
class ResolutionResult:
    """Result of conflict resolution."""
    action: ConflictAction
    merged_contract: Optional[ContractBase] = None
    selected_contract: Optional[ContractBase] = None
    skip_contracts: List[ContractBase] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class ConflictResolver:
    """Resolves conflicts between contracts."""
    
    def __init__(self):
        self.conflict_detectors = {
            ConflictType.FORMAT: self._detect_format_conflict,
            ConflictType.LENGTH: self._detect_length_conflict,
            ConflictType.CONTENT_POLICY: self._detect_content_policy_conflict,
            ConflictType.TEMPORAL: self._detect_temporal_conflict,
            ConflictType.SEMANTIC: self._detect_semantic_conflict,
        }
        
        self.resolution_handlers = {
            ConflictResolutionStrategy.FIRST_WINS: self._resolve_first_wins,
            ConflictResolutionStrategy.LAST_WINS: self._resolve_last_wins,
            ConflictResolutionStrategy.MOST_RESTRICTIVE: self._resolve_most_restrictive,
            ConflictResolutionStrategy.LEAST_RESTRICTIVE: self._resolve_least_restrictive,
            ConflictResolutionStrategy.MERGE: self._resolve_merge,
            ConflictResolutionStrategy.FAIL_ON_CONFLICT: self._resolve_fail,
        }
    
    def detect_conflicts(self, contracts: List[ContractBase]) -> List[ConflictInfo]:
        """Detect all conflicts between contracts."""
        conflicts = []
        
        # Check each pair of contracts
        for i in range(len(contracts)):
            for j in range(i + 1, len(contracts)):
                contract1 = contracts[i]
                contract2 = contracts[j]
                
                # Run all conflict detectors
                for conflict_type, detector in self.conflict_detectors.items():
                    conflict = detector(contract1, contract2)
                    if conflict:
                        conflicts.append(conflict)
        
        return conflicts
    
    def resolve_conflicts(
        self,
        contracts: List[ContractBase],
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.FAIL_ON_CONFLICT
    ) -> Tuple[List[ContractBase], List[ConflictInfo]]:
        """Resolve conflicts using the specified strategy."""
        # Detect conflicts
        conflicts = self.detect_conflicts(contracts)
        
        if not conflicts:
            return contracts, []
        
        # Apply resolution strategy
        handler = self.resolution_handlers.get(strategy)
        if not handler:
            raise ValueError(f"Unknown resolution strategy: {strategy}")
        
        resolved_contracts = handler(contracts, conflicts)
        return resolved_contracts, conflicts
    
    def _detect_format_conflict(
        self, 
        contract1: ContractBase, 
        contract2: ContractBase
    ) -> Optional[ConflictInfo]:
        """Detect format conflicts between contracts."""
        # Check if both contracts specify format requirements
        format1 = getattr(contract1, 'required_format', None)
        format2 = getattr(contract2, 'required_format', None)
        
        if format1 and format2 and format1 != format2:
            return ConflictInfo(
                type=ConflictType.FORMAT,
                contract1=contract1,
                contract2=contract2,
                description=f"Format conflict: {contract1.name} requires {format1}, "
                          f"{contract2.name} requires {format2}",
                severity="error",
                resolvable=False,
                suggested_action=ConflictAction.OVERRIDE
            )
        
        return None
    
    def _detect_length_conflict(
        self,
        contract1: ContractBase,
        contract2: ContractBase
    ) -> Optional[ConflictInfo]:
        """Detect length conflicts between contracts."""
        # Check min/max length constraints
        min1 = getattr(contract1, 'min_length', None)
        max1 = getattr(contract1, 'max_length', None)
        min2 = getattr(contract2, 'min_length', None)
        max2 = getattr(contract2, 'max_length', None)
        
        # Check for impossible constraints
        if min1 and max2 and min1 > max2:
            return ConflictInfo(
                type=ConflictType.LENGTH,
                contract1=contract1,
                contract2=contract2,
                description=f"Length conflict: {contract1.name} requires min {min1}, "
                          f"{contract2.name} requires max {max2}",
                severity="error",
                resolvable=True,
                suggested_action=ConflictAction.MERGE
            )
        
        if min2 and max1 and min2 > max1:
            return ConflictInfo(
                type=ConflictType.LENGTH,
                contract1=contract1,
                contract2=contract2,
                description=f"Length conflict: {contract2.name} requires min {min2}, "
                          f"{contract1.name} requires max {max1}",
                severity="error",
                resolvable=True,
                suggested_action=ConflictAction.MERGE
            )
        
        return None
    
    def _detect_content_policy_conflict(
        self,
        contract1: ContractBase,
        contract2: ContractBase
    ) -> Optional[ConflictInfo]:
        """Detect content policy conflicts."""
        # Check if contracts have conflicting content policies
        policy1 = getattr(contract1, 'content_policy', None)
        policy2 = getattr(contract2, 'content_policy', None)
        
        if policy1 and policy2:
            # Check for mutually exclusive policies
            if hasattr(policy1, 'allowed_topics') and hasattr(policy2, 'forbidden_topics'):
                overlap = set(policy1.allowed_topics) & set(policy2.forbidden_topics)
                if overlap:
                    return ConflictInfo(
                        type=ConflictType.CONTENT_POLICY,
                        contract1=contract1,
                        contract2=contract2,
                        description=f"Content policy conflict: topics {overlap} are allowed "
                                  f"by {contract1.name} but forbidden by {contract2.name}",
                        severity="error",
                        resolvable=True,
                        suggested_action=ConflictAction.MOST_RESTRICTIVE
                    )
        
        return None
    
    def _detect_temporal_conflict(
        self,
        contract1: ContractBase,
        contract2: ContractBase
    ) -> Optional[ConflictInfo]:
        """Detect temporal constraint conflicts."""
        # Check if both contracts have temporal constraints
        temporal1 = getattr(contract1, 'temporal_constraint', None)
        temporal2 = getattr(contract2, 'temporal_constraint', None)
        
        if temporal1 and temporal2:
            # Check for contradictory temporal requirements
            if temporal1.operator == 'always' and temporal2.operator == 'never':
                if temporal1.condition == temporal2.condition:
                    return ConflictInfo(
                        type=ConflictType.TEMPORAL,
                        contract1=contract1,
                        contract2=contract2,
                        description=f"Temporal conflict: {contract1.name} requires condition "
                                  f"to always hold, {contract2.name} requires it to never hold",
                        severity="error",
                        resolvable=False,
                        suggested_action=ConflictAction.FAIL
                    )
        
        return None
    
    def _detect_semantic_conflict(
        self,
        contract1: ContractBase,
        contract2: ContractBase
    ) -> Optional[ConflictInfo]:
        """Detect semantic conflicts between contracts."""
        # This is a placeholder for more sophisticated semantic analysis
        # In a real implementation, this could use NLP or logical reasoning
        
        # Check if contracts have semantic tags
        tags1 = getattr(contract1, 'semantic_tags', set())
        tags2 = getattr(contract2, 'semantic_tags', set())
        
        # Define known conflicting tag pairs
        conflicting_pairs = [
            ({'formal', 'professional'}, {'casual', 'informal'}),
            ({'technical', 'detailed'}, {'simple', 'high-level'}),
            ({'optimistic', 'positive'}, {'pessimistic', 'negative'}),
        ]
        
        for pair1, pair2 in conflicting_pairs:
            if (tags1 & pair1 and tags2 & pair2) or (tags1 & pair2 and tags2 & pair1):
                return ConflictInfo(
                    type=ConflictType.SEMANTIC,
                    contract1=contract1,
                    contract2=contract2,
                    description=f"Semantic conflict: {contract1.name} requires {tags1 & (pair1 | pair2)}, "
                              f"{contract2.name} requires {tags2 & (pair1 | pair2)}",
                    severity="warning",
                    resolvable=True,
                    suggested_action=ConflictAction.OVERRIDE
                )
        
        return None
    
    def _resolve_first_wins(
        self,
        contracts: List[ContractBase],
        conflicts: List[ConflictInfo]
    ) -> List[ContractBase]:
        """Resolution: first contract wins in conflicts."""
        resolved = []
        skip_contracts = set()
        
        for contract in contracts:
            # Skip if this contract lost a conflict
            if contract in skip_contracts:
                continue
            
            # Check if this contract has conflicts with already resolved contracts
            has_conflict = False
            for conflict in conflicts:
                if conflict.contract2 == contract and conflict.contract1 in resolved:
                    skip_contracts.add(contract)
                    has_conflict = True
                    break
            
            if not has_conflict:
                resolved.append(contract)
        
        return resolved
    
    def _resolve_last_wins(
        self,
        contracts: List[ContractBase],
        conflicts: List[ConflictInfo]
    ) -> List[ContractBase]:
        """Resolution: last contract wins in conflicts."""
        # Reverse the list and apply first-wins logic
        reversed_contracts = list(reversed(contracts))
        resolved = self._resolve_first_wins(reversed_contracts, conflicts)
        return list(reversed(resolved))
    
    def _resolve_most_restrictive(
        self,
        contracts: List[ContractBase],
        conflicts: List[ConflictInfo]
    ) -> List[ContractBase]:
        """Resolution: most restrictive contract wins."""
        resolved = []
        processed = set()
        
        # Group contracts by conflict
        conflict_groups = {}
        for conflict in conflicts:
            key = (conflict.type, frozenset([conflict.contract1.name, conflict.contract2.name]))
            if key not in conflict_groups:
                conflict_groups[key] = []
            conflict_groups[key].append(conflict)
        
        # Process each contract
        for contract in contracts:
            if contract in processed:
                continue
            
            # Find all contracts that conflict with this one
            conflicting_contracts = [contract]
            for conflict in conflicts:
                if conflict.contract1 == contract:
                    conflicting_contracts.append(conflict.contract2)
                elif conflict.contract2 == contract:
                    conflicting_contracts.append(conflict.contract1)
            
            # Select the most restrictive from the group
            most_restrictive = self._select_most_restrictive(conflicting_contracts)
            if most_restrictive not in resolved:
                resolved.append(most_restrictive)
            
            # Mark all as processed
            processed.update(conflicting_contracts)
        
        return resolved
    
    def _resolve_least_restrictive(
        self,
        contracts: List[ContractBase],
        conflicts: List[ConflictInfo]
    ) -> List[ContractBase]:
        """Resolution: least restrictive contract wins."""
        # Similar to most restrictive but inverted selection
        resolved = []
        processed = set()
        
        for contract in contracts:
            if contract in processed:
                continue
            
            conflicting_contracts = [contract]
            for conflict in conflicts:
                if conflict.contract1 == contract:
                    conflicting_contracts.append(conflict.contract2)
                elif conflict.contract2 == contract:
                    conflicting_contracts.append(conflict.contract1)
            
            least_restrictive = self._select_least_restrictive(conflicting_contracts)
            if least_restrictive not in resolved:
                resolved.append(least_restrictive)
            
            processed.update(conflicting_contracts)
        
        return resolved
    
    def _resolve_merge(
        self,
        contracts: List[ContractBase],
        conflicts: List[ConflictInfo]
    ) -> List[ContractBase]:
        """Resolution: merge conflicting contracts."""
        resolved = []
        processed = set()
        
        for contract in contracts:
            if contract in processed:
                continue
            
            # Find all contracts that conflict with this one
            conflicting_contracts = [contract]
            for conflict in conflicts:
                if conflict.contract1 == contract and conflict.contract2 not in processed:
                    conflicting_contracts.append(conflict.contract2)
                elif conflict.contract2 == contract and conflict.contract1 not in processed:
                    conflicting_contracts.append(conflict.contract1)
            
            # Merge the conflicting contracts
            if len(conflicting_contracts) > 1:
                merged = self._merge_contracts(conflicting_contracts)
                resolved.append(merged)
            else:
                resolved.append(contract)
            
            processed.update(conflicting_contracts)
        
        return resolved
    
    def _resolve_fail(
        self,
        contracts: List[ContractBase],
        conflicts: List[ConflictInfo]
    ) -> List[ContractBase]:
        """Resolution: fail on any conflict."""
        if conflicts:
            conflict_desc = "\n".join([c.description for c in conflicts])
            raise ValueError(f"Contract conflicts detected:\n{conflict_desc}")
        return contracts
    
    def _select_most_restrictive(self, contracts: List[ContractBase]) -> ContractBase:
        """Select the most restrictive contract from a group."""
        # Simple heuristic: contract with most constraints
        def restrictiveness_score(contract):
            score = 0
            # Check various constraint attributes
            if hasattr(contract, 'max_length'):
                score += 1
            if hasattr(contract, 'min_length'):
                score += 1
            if hasattr(contract, 'required_format'):
                score += 2
            if hasattr(contract, 'forbidden_patterns'):
                score += len(getattr(contract, 'forbidden_patterns', []))
            if hasattr(contract, 'required_patterns'):
                score += len(getattr(contract, 'required_patterns', []))
            # Priority also affects restrictiveness
            if hasattr(contract, 'priority'):
                score += contract.priority.value / 10
            return score
        
        return max(contracts, key=restrictiveness_score)
    
    def _select_least_restrictive(self, contracts: List[ContractBase]) -> ContractBase:
        """Select the least restrictive contract from a group."""
        # Inverse of most restrictive
        def permissiveness_score(contract):
            return -self.restrictiveness_score(contract)
        
        return max(contracts, key=permissiveness_score)
    
    def _merge_contracts(self, contracts: List[ContractBase]) -> ContractBase:
        """Merge multiple contracts into one."""
        # This is a simplified merge - real implementation would be more sophisticated
        from ..contracts.base import CompositeContract
        
        # Create a composite contract that enforces all constraints
        merged = CompositeContract(
            name=f"Merged({', '.join([c.name for c in contracts])})",
            description=f"Merged contract from: {', '.join([c.name for c in contracts])}",
            contracts=contracts
        )
        
        # Merge specific attributes
        if all(hasattr(c, 'max_length') for c in contracts):
            merged.max_length = min(c.max_length for c in contracts)
        
        if all(hasattr(c, 'min_length') for c in contracts):
            merged.min_length = max(c.min_length for c in contracts)
        
        # Use highest priority
        if all(hasattr(c, 'priority') for c in contracts):
            priorities = [c.priority for c in contracts]
            merged.priority = max(priorities, key=lambda p: p.value)
        
        return merged