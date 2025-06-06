"""Core conversation state management system."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from enum import Enum, auto
from collections import deque
import logging
import json
import copy

logger = logging.getLogger(__name__)


class ConversationPhase(Enum):
    """Phases of a conversation."""
    INITIALIZATION = "initialization"
    ACTIVE = "active"
    WRAPPING_UP = "wrapping_up"
    CONCLUDED = "concluded"
    PAUSED = "paused"
    ERROR = "error"


class TurnRole(Enum):
    """Roles in conversation turns."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


class StateChangeType(Enum):
    """Types of state changes."""
    TURN_ADDED = auto()
    CONTEXT_UPDATED = auto()
    INVARIANT_VIOLATED = auto()
    INVARIANT_RESTORED = auto()
    PHASE_CHANGED = auto()
    MEMORY_UPDATED = auto()
    TEMPORAL_CONTRACT_TRIGGERED = auto()


@dataclass
class TurnState:
    """State information for a single conversation turn."""
    turn_id: str
    role: TurnRole
    content: str
    timestamp: float
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    contracts_validated: List[str] = field(default_factory=list)
    violations_detected: List[str] = field(default_factory=list)
    auto_fixes_applied: List[str] = field(default_factory=list)
    
    # Context information
    context_window_position: int = 0
    importance_score: float = 1.0
    compressed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "turn_id": self.turn_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "contracts_validated": self.contracts_validated,
            "violations_detected": self.violations_detected,
            "auto_fixes_applied": self.auto_fixes_applied,
            "context_window_position": self.context_window_position,
            "importance_score": self.importance_score,
            "compressed": self.compressed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurnState":
        """Create from dictionary representation."""
        return cls(
            turn_id=data["turn_id"],
            role=TurnRole(data["role"]),
            content=data["content"],
            timestamp=data["timestamp"],
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {}),
            contracts_validated=data.get("contracts_validated", []),
            violations_detected=data.get("violations_detected", []),
            auto_fixes_applied=data.get("auto_fixes_applied", []),
            context_window_position=data.get("context_window_position", 0),
            importance_score=data.get("importance_score", 1.0),
            compressed=data.get("compressed", False),
        )


@dataclass
class StateTransition:
    """Represents a state transition in the conversation."""
    transition_id: str
    timestamp: float
    change_type: StateChangeType
    from_state: Optional[Dict[str, Any]]
    to_state: Dict[str, Any]
    trigger: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "transition_id": self.transition_id,
            "timestamp": self.timestamp,
            "change_type": self.change_type.name,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "trigger": self.trigger,
            "metadata": self.metadata,
        }


@dataclass
class StateSnapshot:
    """Snapshot of conversation state at a point in time."""
    snapshot_id: str
    timestamp: float
    conversation_id: str
    phase: ConversationPhase
    turn_count: int
    total_tokens: int
    current_context_size: int
    active_invariants: List[str]
    temporal_contracts_status: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "conversation_id": self.conversation_id,
            "phase": self.phase.value,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "current_context_size": self.current_context_size,
            "active_invariants": self.active_invariants,
            "temporal_contracts_status": self.temporal_contracts_status,
            "metadata": self.metadata,
        }


@dataclass
class ConversationState:
    """Complete state of a conversation."""
    conversation_id: str
    created_at: float
    updated_at: float
    phase: ConversationPhase = ConversationPhase.INITIALIZATION
    
    # Turn history
    turns: List[TurnState] = field(default_factory=list)
    turn_count: int = 0
    total_tokens: int = 0
    
    # Context management
    context_window_size: int = 4096
    active_context_tokens: int = 0
    compressed_context: Optional[str] = None
    
    # State tracking
    current_topic: Optional[str] = None
    conversation_summary: Optional[str] = None
    key_facts: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Validation tracking
    active_invariants: Set[str] = field(default_factory=set)
    violated_invariants: Set[str] = field(default_factory=set)
    temporal_contract_states: Dict[str, Any] = field(default_factory=dict)
    
    # History and snapshots
    state_transitions: List[StateTransition] = field(default_factory=list)
    snapshots: List[StateSnapshot] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_turn(self, turn: TurnState) -> None:
        """Add a new turn to the conversation."""
        self.turns.append(turn)
        self.turn_count += 1
        self.total_tokens += turn.token_count
        self.active_context_tokens += turn.token_count
        self.updated_at = time.time()
        
        # Update context window position
        turn.context_window_position = len(self.turns) - 1
        
        # Log state transition
        self._record_state_transition(
            StateChangeType.TURN_ADDED,
            f"Added turn from {turn.role.value}",
            {"turn_id": turn.turn_id, "role": turn.role.value}
        )
    
    def get_recent_turns(self, count: int) -> List[TurnState]:
        """Get the most recent N turns."""
        return self.turns[-count:] if count < len(self.turns) else self.turns
    
    def get_turns_by_role(self, role: TurnRole) -> List[TurnState]:
        """Get all turns by a specific role."""
        return [turn for turn in self.turns if turn.role == role]
    
    def get_context_window(self, max_tokens: Optional[int] = None) -> List[TurnState]:
        """Get turns that fit within the context window."""
        max_tokens = max_tokens or self.context_window_size
        
        context_turns = []
        token_count = 0
        
        # Add turns from most recent backwards
        for turn in reversed(self.turns):
            if token_count + turn.token_count <= max_tokens:
                context_turns.insert(0, turn)
                token_count += turn.token_count
            else:
                break
        
        return context_turns
    
    def update_phase(self, new_phase: ConversationPhase, reason: str = "") -> None:
        """Update conversation phase."""
        old_phase = self.phase
        self.phase = new_phase
        self.updated_at = time.time()
        
        self._record_state_transition(
            StateChangeType.PHASE_CHANGED,
            f"Phase changed from {old_phase.value} to {new_phase.value}: {reason}",
            {"old_phase": old_phase.value, "new_phase": new_phase.value, "reason": reason}
        )
    
    def add_invariant_violation(self, invariant_name: str, details: str = "") -> None:
        """Record an invariant violation."""
        self.violated_invariants.add(invariant_name)
        if invariant_name in self.active_invariants:
            self.active_invariants.remove(invariant_name)
        
        self._record_state_transition(
            StateChangeType.INVARIANT_VIOLATED,
            f"Invariant {invariant_name} violated: {details}",
            {"invariant": invariant_name, "details": details}
        )
    
    def restore_invariant(self, invariant_name: str, details: str = "") -> None:
        """Restore a previously violated invariant."""
        if invariant_name in self.violated_invariants:
            self.violated_invariants.remove(invariant_name)
        self.active_invariants.add(invariant_name)
        
        self._record_state_transition(
            StateChangeType.INVARIANT_RESTORED,
            f"Invariant {invariant_name} restored: {details}",
            {"invariant": invariant_name, "details": details}
        )
    
    def create_snapshot(self, reason: str = "") -> StateSnapshot:
        """Create a snapshot of current state."""
        snapshot = StateSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=time.time(),
            conversation_id=self.conversation_id,
            phase=self.phase,
            turn_count=self.turn_count,
            total_tokens=self.total_tokens,
            current_context_size=self.active_context_tokens,
            active_invariants=list(self.active_invariants),
            temporal_contracts_status=copy.deepcopy(self.temporal_contract_states),
            metadata={"reason": reason}
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def _record_state_transition(self, change_type: StateChangeType, trigger: str, metadata: Dict[str, Any]) -> None:
        """Record a state transition."""
        transition = StateTransition(
            transition_id=str(uuid.uuid4()),
            timestamp=time.time(),
            change_type=change_type,
            from_state=None,  # Could be populated with previous state if needed
            to_state={"phase": self.phase.value, "turn_count": self.turn_count},
            trigger=trigger,
            metadata=metadata
        )
        
        self.state_transitions.append(transition)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "phase": self.phase.value,
            "turns": [turn.to_dict() for turn in self.turns],
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "context_window_size": self.context_window_size,
            "active_context_tokens": self.active_context_tokens,
            "compressed_context": self.compressed_context,
            "current_topic": self.current_topic,
            "conversation_summary": self.conversation_summary,
            "key_facts": self.key_facts,
            "user_preferences": self.user_preferences,
            "active_invariants": list(self.active_invariants),
            "violated_invariants": list(self.violated_invariants),
            "temporal_contract_states": self.temporal_contract_states,
            "state_transitions": [t.to_dict() for t in self.state_transitions],
            "snapshots": [s.to_dict() for s in self.snapshots],
            "metadata": self.metadata,
        }


class ConversationStateManager:
    """Manages conversation state across multiple turns with temporal contracts and context management."""
    
    def __init__(self, 
                 conversation_id: Optional[str] = None,
                 context_window_size: int = 4096,
                 max_history_length: int = 1000,
                 auto_snapshot_interval: int = 10,
                 enable_compression: bool = True):
        """Initialize the conversation state manager.
        
        Args:
            conversation_id: Unique identifier for the conversation
            context_window_size: Maximum tokens in context window
            max_history_length: Maximum number of turns to keep in memory
            auto_snapshot_interval: Create snapshots every N turns
            enable_compression: Enable context compression when needed
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.max_history_length = max_history_length
        self.auto_snapshot_interval = auto_snapshot_interval
        self.enable_compression = enable_compression
        
        # Initialize conversation state
        self.state = ConversationState(
            conversation_id=self.conversation_id,
            created_at=time.time(),
            updated_at=time.time(),
            context_window_size=context_window_size
        )
        
        # Registered components
        self.temporal_contracts: List[Any] = []
        self.invariant_trackers: List[Any] = []
        self.context_manager: Optional[Any] = None
        self.memory_store: Optional[Any] = None
        
        # Event handlers
        self.state_change_handlers: List[Callable[[StateTransition], None]] = []
        self.violation_handlers: List[Callable[[str, str], None]] = []
        
        # Performance tracking
        self.validation_metrics = {
            "turns_processed": 0,
            "contracts_evaluated": 0,
            "violations_detected": 0,
            "auto_fixes_applied": 0,
            "context_compressions": 0,
        }
        
        logger.info(f"ConversationStateManager initialized for conversation {self.conversation_id}")
    
    def add_turn(self, 
                 role: Union[TurnRole, str], 
                 content: str, 
                 metadata: Optional[Dict[str, Any]] = None) -> TurnState:
        """Add a new turn to the conversation.
        
        Args:
            role: Role of the speaker (user, assistant, system, etc.)
            content: Content of the turn
            metadata: Additional metadata for the turn
            
        Returns:
            The created TurnState object
        """
        if isinstance(role, str):
            role = TurnRole(role.lower())
        
        # Create turn state
        turn = TurnState(
            turn_id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            token_count=self._estimate_tokens(content),
            metadata=metadata or {}
        )
        
        # Add to conversation state
        self.state.add_turn(turn)
        
        # Update metrics
        self.validation_metrics["turns_processed"] += 1
        
        # Process the turn through validation pipeline
        self._process_turn_validation(turn)
        
        # Check if we need to manage context window
        if self.context_manager:
            self._manage_context_window()
        
        # Update conversation phase if needed
        self._update_conversation_phase()
        
        # Create automatic snapshot if needed
        if (self.state.turn_count % self.auto_snapshot_interval == 0 and 
            self.state.turn_count > 0):
            self.state.create_snapshot(f"Auto-snapshot at turn {self.state.turn_count}")
        
        # Notify handlers
        self._notify_state_change_handlers()
        
        logger.debug(f"Turn added: {turn.turn_id} ({role.value})")
        return turn
    
    def get_conversation_state(self) -> ConversationState:
        """Get the current conversation state."""
        return self.state
    
    def get_context_window(self, max_tokens: Optional[int] = None) -> List[TurnState]:
        """Get the current context window."""
        return self.state.get_context_window(max_tokens)
    
    def get_conversation_history(self, 
                                include_metadata: bool = False,
                                role_filter: Optional[TurnRole] = None) -> List[Dict[str, Any]]:
        """Get conversation history in OpenAI format.
        
        Args:
            include_metadata: Include turn metadata in output
            role_filter: Only include turns from specific role
            
        Returns:
            List of message dictionaries compatible with OpenAI API
        """
        turns = self.state.turns
        if role_filter:
            turns = [t for t in turns if t.role == role_filter]
        
        history = []
        for turn in turns:
            message = {
                "role": turn.role.value,
                "content": turn.content
            }
            
            if include_metadata:
                message["metadata"] = {
                    "turn_id": turn.turn_id,
                    "timestamp": turn.timestamp,
                    "token_count": turn.token_count,
                    "importance_score": turn.importance_score,
                    **turn.metadata
                }
            
            history.append(message)
        
        return history
    
    def register_temporal_contract(self, contract: Any) -> None:
        """Register a temporal contract for validation."""
        self.temporal_contracts.append(contract)
        logger.info(f"Temporal contract registered: {getattr(contract, 'name', 'unnamed')}")
    
    def register_invariant_tracker(self, tracker: Any) -> None:
        """Register an invariant tracker."""
        self.invariant_trackers.append(tracker)
        logger.info(f"Invariant tracker registered: {getattr(tracker, 'name', 'unnamed')}")
    
    def set_context_manager(self, manager: Any) -> None:
        """Set the context window manager."""
        self.context_manager = manager
        logger.info("Context manager set")
    
    def set_memory_store(self, store: Any) -> None:
        """Set the memory store."""
        self.memory_store = store
        logger.info("Memory store set")
    
    def add_state_change_handler(self, handler: Callable[[StateTransition], None]) -> None:
        """Add a state change event handler."""
        self.state_change_handlers.append(handler)
    
    def add_violation_handler(self, handler: Callable[[str, str], None]) -> None:
        """Add a violation event handler."""
        self.violation_handlers.append(handler)
    
    def create_snapshot(self, reason: str = "") -> StateSnapshot:
        """Create a manual snapshot of the current state."""
        return self.state.create_snapshot(reason)
    
    def restore_from_snapshot(self, snapshot_id: str) -> bool:
        """Restore state from a snapshot (if supported by implementation)."""
        # This would require more complex state restoration logic
        # For now, just find and return the snapshot
        for snapshot in self.state.snapshots:
            if snapshot.snapshot_id == snapshot_id:
                logger.info(f"Found snapshot {snapshot_id} for restoration")
                return True
        
        logger.warning(f"Snapshot {snapshot_id} not found")
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get conversation and validation metrics."""
        return {
            **self.validation_metrics,
            "conversation_id": self.conversation_id,
            "turn_count": self.state.turn_count,
            "total_tokens": self.state.total_tokens,
            "active_context_tokens": self.state.active_context_tokens,
            "conversation_phase": self.state.phase.value,
            "active_invariants": len(self.state.active_invariants),
            "violated_invariants": len(self.state.violated_invariants),
            "snapshots_created": len(self.state.snapshots),
            "state_transitions": len(self.state.state_transitions),
        }
    
    def export_conversation(self, format: str = "json") -> str:
        """Export conversation to various formats."""
        if format.lower() == "json":
            return json.dumps(self.state.to_dict(), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _process_turn_validation(self, turn: TurnState) -> None:
        """Process a turn through all validation components."""
        # Validate with temporal contracts
        for contract in self.temporal_contracts:
            try:
                if hasattr(contract, 'validate_turn'):
                    result = contract.validate_turn(turn, self.state)
                    turn.contracts_validated.append(getattr(contract, 'name', 'unnamed'))
                    self.validation_metrics["contracts_evaluated"] += 1
                    
                    if hasattr(result, 'is_valid') and not result.is_valid:
                        violation_msg = getattr(result, 'message', 'Temporal contract violation')
                        turn.violations_detected.append(violation_msg)
                        self.validation_metrics["violations_detected"] += 1
                        self._notify_violation_handlers(getattr(contract, 'name', 'unnamed'), violation_msg)
                        
                        # Apply auto-fix if available
                        if hasattr(result, 'auto_fix') and result.auto_fix:
                            turn.auto_fixes_applied.append(result.auto_fix)
                            self.validation_metrics["auto_fixes_applied"] += 1
                        
            except Exception as e:
                logger.error(f"Error validating turn with temporal contract: {e}")
        
        # Validate with invariant trackers
        for tracker in self.invariant_trackers:
            try:
                if hasattr(tracker, 'check_turn'):
                    violations = tracker.check_turn(turn, self.state)
                    if violations:
                        for violation in violations:
                            self.state.add_invariant_violation(
                                getattr(tracker, 'name', 'unnamed'), 
                                str(violation)
                            )
            except Exception as e:
                logger.error(f"Error checking invariants: {e}")
    
    def _manage_context_window(self) -> None:
        """Manage context window size and compression."""
        if not self.context_manager:
            return
        
        try:
            if hasattr(self.context_manager, 'optimize_context'):
                optimized = self.context_manager.optimize_context(self.state)
                if optimized:
                    self.validation_metrics["context_compressions"] += 1
                    logger.debug("Context window optimized")
        except Exception as e:
            logger.error(f"Error managing context window: {e}")
    
    def _update_conversation_phase(self) -> None:
        """Update conversation phase based on current state."""
        current_phase = self.state.phase
        
        # Simple phase transition logic
        if current_phase == ConversationPhase.INITIALIZATION and self.state.turn_count > 0:
            self.state.update_phase(ConversationPhase.ACTIVE, "First turn received")
        
        # More sophisticated phase logic could be added here
        # based on content analysis, time, turn patterns, etc.
    
    def _notify_state_change_handlers(self) -> None:
        """Notify all state change handlers."""
        if self.state.state_transitions:
            latest_transition = self.state.state_transitions[-1]
            for handler in self.state_change_handlers:
                try:
                    handler(latest_transition)
                except Exception as e:
                    logger.error(f"Error in state change handler: {e}")
    
    def _notify_violation_handlers(self, contract_name: str, violation_msg: str) -> None:
        """Notify all violation handlers."""
        for handler in self.violation_handlers:
            try:
                handler(contract_name, violation_msg)
            except Exception as e:
                logger.error(f"Error in violation handler: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (simple approximation)."""
        # Simple approximation: ~4 characters per token for English
        return max(1, len(text) // 4)
    
    def __str__(self) -> str:
        return f"ConversationStateManager(id={self.conversation_id}, turns={self.state.turn_count}, phase={self.state.phase.value})"
    
    def __repr__(self) -> str:
        return self.__str__()