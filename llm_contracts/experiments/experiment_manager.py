"""Experiment management system for A/B testing contracts."""

import asyncio
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from llm_contracts.contracts.base import ContractBase, ValidationResult
from llm_contracts.core.exceptions import ContractViolation


class ExperimentStatus(Enum):
    """Status of an experiment."""
    DRAFT = auto()      # Experiment created but not started
    RUNNING = auto()    # Experiment is actively running
    PAUSED = auto()     # Experiment temporarily paused
    COMPLETED = auto()  # Experiment completed successfully
    ABORTED = auto()    # Experiment aborted due to issues


class TrafficAllocation(Enum):
    """Traffic allocation strategies."""
    RANDOM = auto()           # Random assignment
    USER_BASED = auto()       # Consistent per user
    SESSION_BASED = auto()    # Consistent per session
    PERCENTAGE_BASED = auto() # Fixed percentage split


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    control_contract: Optional[ContractBase] = None
    treatment_contracts: List[ContractBase] = field(default_factory=list)
    traffic_allocation: TrafficAllocation = TrafficAllocation.RANDOM
    traffic_percentages: List[float] = field(default_factory=list)  # [control%, treatment1%, ...]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_participants: Optional[int] = None
    min_participants_per_variant: int = 100
    confidence_level: float = 0.95
    enable_early_stopping: bool = True
    rollback_on_degradation: bool = True
    degradation_threshold: float = 0.1  # 10% performance drop triggers rollback
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParticipantInfo:
    """Information about an experiment participant."""
    participant_id: str
    variant: str  # "control" or "treatment_N"
    timestamp: datetime
    context: Dict[str, Any]
    session_id: Optional[str] = None


@dataclass
class ExperimentMetrics:
    """Metrics collected for an experiment variant."""
    variant: str
    total_requests: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    contract_violations: int = 0
    auto_fixes_applied: int = 0
    average_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_validations / self.total_requests
    
    @property
    def violation_rate(self) -> float:
        """Calculate contract violation rate."""
        if self.total_requests == 0:
            return 0.0
        return self.contract_violations / self.total_requests
    
    def update_latency(self, latency_ms: float):
        """Update latency metrics."""
        self.latencies.append(latency_ms)
        self.average_latency_ms = sum(self.latencies) / len(self.latencies)


@dataclass
class ExperimentResult:
    """Results of an experiment."""
    experiment_id: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]
    metrics_by_variant: Dict[str, ExperimentMetrics]
    winner: Optional[str] = None
    confidence: Optional[float] = None
    p_value: Optional[float] = None
    statistical_significance: bool = False
    recommendations: List[str] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.name,
            "duration": str(self.end_time - self.start_time) if self.end_time else "ongoing",
            "variants": list(self.metrics_by_variant.keys()),
            "total_participants": sum(m.total_requests for m in self.metrics_by_variant.values()),
            "winner": self.winner,
            "confidence": self.confidence,
            "statistical_significance": self.statistical_significance,
            "recommendations": self.recommendations,
        }


class TrafficSplitter:
    """Handles traffic splitting for experiments."""
    
    def __init__(self, allocation_strategy: TrafficAllocation = TrafficAllocation.RANDOM):
        self.allocation_strategy = allocation_strategy
        self.user_assignments: Dict[str, str] = {}
        self.session_assignments: Dict[str, str] = {}
    
    def assign_variant(
        self,
        experiment_config: ExperimentConfig,
        participant_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Assign a participant to a variant."""
        variants = ["control"] + [f"treatment_{i}" for i in range(len(experiment_config.treatment_contracts))]
        
        if self.allocation_strategy == TrafficAllocation.USER_BASED and participant_id:
            if participant_id in self.user_assignments:
                return self.user_assignments[participant_id]
            variant = self._select_variant(variants, experiment_config.traffic_percentages)
            self.user_assignments[participant_id] = variant
            return variant
        
        elif self.allocation_strategy == TrafficAllocation.SESSION_BASED and session_id:
            if session_id in self.session_assignments:
                return self.session_assignments[session_id]
            variant = self._select_variant(variants, experiment_config.traffic_percentages)
            self.session_assignments[session_id] = variant
            return variant
        
        else:  # RANDOM or PERCENTAGE_BASED
            return self._select_variant(variants, experiment_config.traffic_percentages)
    
    def _select_variant(self, variants: List[str], percentages: List[float]) -> str:
        """Select variant based on traffic percentages."""
        if not percentages:
            # Equal split if no percentages specified
            return random.choice(variants)
        
        # Ensure percentages sum to 100
        total = sum(percentages)
        if abs(total - 100.0) > 0.01:
            percentages = [p * 100.0 / total for p in percentages]
        
        rand = random.uniform(0, 100)
        cumulative = 0.0
        
        for variant, percentage in zip(variants, percentages):
            cumulative += percentage
            if rand <= cumulative:
                return variant
        
        return variants[-1]  # Fallback to last variant


class MetricCollector:
    """Collects and aggregates metrics for experiments."""
    
    def __init__(self):
        self.metrics: Dict[str, Dict[str, ExperimentMetrics]] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    async def record_validation(
        self,
        experiment_id: str,
        variant: str,
        result: ValidationResult,
        latency_ms: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Record a validation result."""
        async with self._lock:
            if variant not in self.metrics[experiment_id]:
                self.metrics[experiment_id][variant] = ExperimentMetrics(variant=variant)
            
            metrics = self.metrics[experiment_id][variant]
            metrics.total_requests += 1
            
            if result.is_valid:
                metrics.successful_validations += 1
            else:
                metrics.failed_validations += 1
                if isinstance(result.violation_type, str):
                    metrics.error_types[result.violation_type] += 1
            
            if result.contract_violations:
                metrics.contract_violations += len(result.contract_violations)
            
            if result.auto_fix_suggestion:
                metrics.auto_fixes_applied += 1
            
            metrics.update_latency(latency_ms)
    
    def get_metrics(self, experiment_id: str) -> Dict[str, ExperimentMetrics]:
        """Get metrics for an experiment."""
        return dict(self.metrics.get(experiment_id, {}))
    
    def clear_metrics(self, experiment_id: str):
        """Clear metrics for an experiment."""
        if experiment_id in self.metrics:
            del self.metrics[experiment_id]


@dataclass
class Experiment:
    """Represents an A/B testing experiment."""
    id: str
    config: ExperimentConfig
    status: ExperimentStatus
    participants: List[ParticipantInfo] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        if self.status != ExperimentStatus.RUNNING:
            return False
        
        now = datetime.now()
        
        if self.config.start_time and now < self.config.start_time:
            return False
        
        if self.config.end_time and now > self.config.end_time:
            return False
        
        if self.config.max_participants and len(self.participants) >= self.config.max_participants:
            return False
        
        return True
    
    def can_conclude(self, metrics: Dict[str, ExperimentMetrics]) -> bool:
        """Check if experiment has enough data to conclude."""
        for variant_metrics in metrics.values():
            if variant_metrics.total_requests < self.config.min_participants_per_variant:
                return False
        return True


class ExperimentManager:
    """Manages A/B testing experiments for contracts."""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.traffic_splitter = TrafficSplitter()
        self.metric_collector = MetricCollector()
        self._lock = asyncio.Lock()
        self._background_tasks: Set[asyncio.Task] = set()
    
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment."""
        async with self._lock:
            experiment_id = str(uuid.uuid4())
            experiment = Experiment(
                id=experiment_id,
                config=config,
                status=ExperimentStatus.DRAFT,
            )
            self.experiments[experiment_id] = experiment
            return experiment_id
    
    async def start_experiment(self, experiment_id: str):
        """Start an experiment."""
        async with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError(f"Experiment {experiment_id} is not in DRAFT status")
            
            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.now()
            
            # Start background monitoring
            task = asyncio.create_task(self._monitor_experiment(experiment_id))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def pause_experiment(self, experiment_id: str):
        """Pause a running experiment."""
        async with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            if experiment.status != ExperimentStatus.RUNNING:
                raise ValueError(f"Experiment {experiment_id} is not running")
            
            experiment.status = ExperimentStatus.PAUSED
    
    async def resume_experiment(self, experiment_id: str):
        """Resume a paused experiment."""
        async with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            if experiment.status != ExperimentStatus.PAUSED:
                raise ValueError(f"Experiment {experiment_id} is not paused")
            
            experiment.status = ExperimentStatus.RUNNING
    
    async def stop_experiment(self, experiment_id: str) -> ExperimentResult:
        """Stop an experiment and get results."""
        async with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            experiment.status = ExperimentStatus.COMPLETED
            experiment.ended_at = datetime.now()
            
            # Get final metrics
            metrics = self.metric_collector.get_metrics(experiment_id)
            
            # Analyze results
            result = await self._analyze_results(experiment, metrics)
            
            return result
    
    async def get_contract_for_request(
        self,
        experiment_id: str,
        participant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ContractBase], str]:
        """Get the appropriate contract for a request based on experiment assignment."""
        if experiment_id not in self.experiments:
            return None, "control"
        
        experiment = self.experiments[experiment_id]
        if not experiment.is_active():
            return experiment.config.control_contract, "control"
        
        # Assign variant
        variant = self.traffic_splitter.assign_variant(
            experiment.config,
            participant_id,
            session_id,
        )
        
        # Record participant
        participant = ParticipantInfo(
            participant_id=participant_id or str(uuid.uuid4()),
            variant=variant,
            timestamp=datetime.now(),
            context=context or {},
            session_id=session_id,
        )
        experiment.participants.append(participant)
        
        # Return appropriate contract
        if variant == "control":
            return experiment.config.control_contract, variant
        else:
            treatment_idx = int(variant.split("_")[1])
            if treatment_idx < len(experiment.config.treatment_contracts):
                return experiment.config.treatment_contracts[treatment_idx], variant
            else:
                return experiment.config.control_contract, "control"
    
    async def record_validation_result(
        self,
        experiment_id: str,
        variant: str,
        result: ValidationResult,
        latency_ms: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Record validation result for an experiment."""
        await self.metric_collector.record_validation(
            experiment_id,
            variant,
            result,
            latency_ms,
            context,
        )
    
    async def _monitor_experiment(self, experiment_id: str):
        """Background task to monitor experiment progress."""
        while True:
            try:
                experiment = self.experiments.get(experiment_id)
                if not experiment or experiment.status != ExperimentStatus.RUNNING:
                    break
                
                # Check if experiment should end
                if experiment.config.end_time and datetime.now() > experiment.config.end_time:
                    await self.stop_experiment(experiment_id)
                    break
                
                # Check for early stopping conditions
                if experiment.config.enable_early_stopping:
                    metrics = self.metric_collector.get_metrics(experiment_id)
                    if await self._should_stop_early(experiment, metrics):
                        await self.stop_experiment(experiment_id)
                        break
                
                # Check for degradation
                if experiment.config.rollback_on_degradation:
                    metrics = self.metric_collector.get_metrics(experiment_id)
                    if await self._check_degradation(experiment, metrics):
                        experiment.status = ExperimentStatus.ABORTED
                        break
                
                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error monitoring experiment {experiment_id}: {e}")
                break
    
    async def _should_stop_early(
        self,
        experiment: Experiment,
        metrics: Dict[str, ExperimentMetrics],
    ) -> bool:
        """Check if experiment should stop early based on statistical significance."""
        if not experiment.can_conclude(metrics):
            return False
        
        # Simple check: if one variant is significantly better
        if "control" in metrics and any(v.startswith("treatment_") for v in metrics):
            control_metrics = metrics["control"]
            
            for variant, variant_metrics in metrics.items():
                if variant.startswith("treatment_"):
                    # Check if treatment is significantly better or worse
                    if variant_metrics.total_requests >= experiment.config.min_participants_per_variant:
                        success_diff = abs(variant_metrics.success_rate - control_metrics.success_rate)
                        if success_diff > 0.1:  # 10% difference
                            return True
        
        return False
    
    async def _check_degradation(
        self,
        experiment: Experiment,
        metrics: Dict[str, ExperimentMetrics],
    ) -> bool:
        """Check if any treatment shows significant degradation."""
        if "control" not in metrics:
            return False
        
        control_metrics = metrics["control"]
        
        for variant, variant_metrics in metrics.items():
            if variant.startswith("treatment_"):
                # Check various degradation indicators
                success_degradation = control_metrics.success_rate - variant_metrics.success_rate
                latency_degradation = (variant_metrics.average_latency_ms - control_metrics.average_latency_ms) / control_metrics.average_latency_ms if control_metrics.average_latency_ms > 0 else 0
                
                if success_degradation > experiment.config.degradation_threshold:
                    return True
                
                if latency_degradation > experiment.config.degradation_threshold:
                    return True
        
        return False
    
    async def _analyze_results(
        self,
        experiment: Experiment,
        metrics: Dict[str, ExperimentMetrics],
    ) -> ExperimentResult:
        """Analyze experiment results and determine winner."""
        result = ExperimentResult(
            experiment_id=experiment.id,
            status=experiment.status,
            start_time=experiment.started_at or experiment.created_at,
            end_time=experiment.ended_at,
            metrics_by_variant=metrics,
        )
        
        # Simple analysis - in real implementation would use proper statistical tests
        if "control" in metrics:
            control_metrics = metrics["control"]
            best_variant = "control"
            best_success_rate = control_metrics.success_rate
            
            for variant, variant_metrics in metrics.items():
                if variant.startswith("treatment_"):
                    if variant_metrics.success_rate > best_success_rate:
                        best_variant = variant
                        best_success_rate = variant_metrics.success_rate
            
            # Check if difference is significant (simplified)
            if best_variant != "control":
                improvement = (best_success_rate - control_metrics.success_rate) / control_metrics.success_rate
                if improvement > 0.05:  # 5% improvement
                    result.winner = best_variant
                    result.confidence = min(0.95, improvement * 10)  # Simplified confidence
                    result.statistical_significance = True
                    result.recommendations.append(
                        f"Recommend adopting {best_variant} - {improvement:.1%} improvement in success rate"
                    )
                else:
                    result.recommendations.append(
                        "No significant improvement found - recommend keeping control"
                    )
            else:
                result.recommendations.append(
                    "Control performs best - no changes recommended"
                )
        
        return result
    
    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status of an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        metrics = self.metric_collector.get_metrics(experiment_id)
        
        return {
            "id": experiment.id,
            "name": experiment.config.name,
            "status": experiment.status.name,
            "is_active": experiment.is_active(),
            "created_at": experiment.created_at.isoformat(),
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "total_participants": len(experiment.participants),
            "variants": {
                variant: {
                    "requests": m.total_requests,
                    "success_rate": m.success_rate,
                    "average_latency_ms": m.average_latency_ms,
                }
                for variant, m in metrics.items()
            },
        }
    
    async def cleanup(self):
        """Clean up resources."""
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)