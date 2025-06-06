"""A/B Testing and Experimentation framework for LLM contracts."""

from .experiment_manager import (
    ExperimentManager,
    Experiment,
    ExperimentStatus,
    ExperimentResult,
    TrafficSplitter,
    MetricCollector,
    ExperimentConfig,
)

from .decorators import (
    ab_test_contract,
    experiment_contract,
    gradual_rollout_contract,
)

from .metrics import (
    ContractMetric,
    MetricType,
    MetricAggregator,
    StatisticalAnalyzer,
)

__all__ = [
    "ExperimentManager",
    "Experiment",
    "ExperimentStatus",
    "ExperimentResult",
    "TrafficSplitter",
    "MetricCollector",
    "ExperimentConfig",
    "ab_test_contract",
    "experiment_contract",
    "gradual_rollout_contract",
    "ContractMetric",
    "MetricType",
    "MetricAggregator",
    "StatisticalAnalyzer",
]