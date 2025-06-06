"""Metrics collection and analysis for contract experiments."""

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

# Optional imports for advanced statistics
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class MetricType(Enum):
    """Types of metrics that can be collected."""
    SUCCESS_RATE = auto()      # Binary success/failure
    LATENCY = auto()          # Response time in milliseconds
    ERROR_RATE = auto()       # Rate of errors
    VIOLATION_RATE = auto()   # Rate of contract violations
    AUTO_FIX_RATE = auto()    # Rate of auto-fixes applied
    CUSTOM = auto()           # Custom metric


@dataclass
class ContractMetric:
    """A metric measurement for contract validation."""
    name: str
    metric_type: MetricType
    value: float
    timestamp: float
    variant: str
    experiment_id: str
    context: Dict[str, Any] = field(default_factory=dict)


class MetricAggregator:
    """Aggregates metrics across multiple measurements."""
    
    def __init__(self):
        self.metrics: Dict[str, List[ContractMetric]] = defaultdict(list)
    
    def add_metric(self, metric: ContractMetric):
        """Add a metric measurement."""
        key = f"{metric.experiment_id}:{metric.variant}:{metric.name}"
        self.metrics[key].append(metric)
    
    def get_aggregated_metrics(
        self,
        experiment_id: str,
        variant: str,
        metric_name: str,
    ) -> Dict[str, float]:
        """Get aggregated statistics for a metric."""
        key = f"{experiment_id}:{variant}:{metric_name}"
        metrics = self.metrics.get(key, [])
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "p95": np.percentile(values, 95) if HAS_NUMPY and values else (sorted(values)[int(0.95 * len(values))] if values else 0.0),
            "p99": np.percentile(values, 99) if HAS_NUMPY and values else (sorted(values)[int(0.99 * len(values))] if values else 0.0),
        }
    
    def get_time_series(
        self,
        experiment_id: str,
        variant: str,
        metric_name: str,
        window_minutes: int = 60,
    ) -> List[Tuple[float, float]]:
        """Get time series data for a metric."""
        key = f"{experiment_id}:{variant}:{metric_name}"
        metrics = self.metrics.get(key, [])
        
        if not metrics:
            return []
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        
        # Group into time windows
        window_seconds = window_minutes * 60
        time_series = []
        
        if sorted_metrics:
            start_time = sorted_metrics[0].timestamp
            window_start = start_time
            window_metrics = []
            
            for metric in sorted_metrics:
                if metric.timestamp >= window_start + window_seconds:
                    # Process current window
                    if window_metrics:
                        window_value = statistics.mean([m.value for m in window_metrics])
                        time_series.append((window_start, window_value))
                    
                    # Start new window
                    window_start = metric.timestamp
                    window_metrics = [metric]
                else:
                    window_metrics.append(metric)
            
            # Process final window
            if window_metrics:
                window_value = statistics.mean([m.value for m in window_metrics])
                time_series.append((window_start, window_value))
        
        return time_series
    
    def clear_metrics(self, experiment_id: str):
        """Clear all metrics for an experiment."""
        keys_to_remove = [key for key in self.metrics.keys() if key.startswith(f"{experiment_id}:")]
        for key in keys_to_remove:
            del self.metrics[key]


class StatisticalAnalyzer:
    """Performs statistical analysis on experiment metrics."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compare_success_rates(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
    ) -> Dict[str, Any]:
        """Compare success rates between control and treatment using two-proportion z-test."""
        if control_total == 0 or treatment_total == 0:
            return {
                "p_value": None,
                "confidence_interval": None,
                "effect_size": None,
                "significant": False,
                "error": "Insufficient data",
            }
        
        control_rate = control_successes / control_total
        treatment_rate = treatment_successes / treatment_total
        
        # Two-proportion z-test
        pooled_rate = (control_successes + treatment_successes) / (control_total + treatment_total)
        se_pooled = math.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_total + 1/treatment_total))
        
        if se_pooled == 0:
            return {
                "p_value": None,
                "confidence_interval": None,
                "effect_size": treatment_rate - control_rate,
                "significant": False,
                "error": "No variance in data",
            }
        
        z_score = (treatment_rate - control_rate) / se_pooled
        if HAS_SCIPY:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            # Simple approximation without scipy
            p_value = 2 * (1 - (0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2)))))
        
        # Confidence interval for difference
        se_diff = math.sqrt(
            (control_rate * (1 - control_rate) / control_total) +
            (treatment_rate * (1 - treatment_rate) / treatment_total)
        )
        
        if HAS_SCIPY:
            z_critical = stats.norm.ppf(1 - self.alpha / 2)
        else:
            # Approximation for 95% confidence (1.96)
            z_critical = 1.96
        margin_error = z_critical * se_diff
        diff = treatment_rate - control_rate
        
        return {
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "difference": diff,
            "relative_improvement": (diff / control_rate) if control_rate > 0 else None,
            "p_value": p_value,
            "z_score": z_score,
            "confidence_interval": (diff - margin_error, diff + margin_error),
            "significant": p_value < self.alpha,
            "effect_size": diff,
        }
    
    def compare_means(
        self,
        control_values: List[float],
        treatment_values: List[float],
    ) -> Dict[str, Any]:
        """Compare means between control and treatment using t-test."""
        if len(control_values) < 2 or len(treatment_values) < 2:
            return {
                "p_value": None,
                "confidence_interval": None,
                "effect_size": None,
                "significant": False,
                "error": "Insufficient data for t-test",
            }
        
        try:
            # Two-sample t-test
            if HAS_SCIPY:
                t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
            else:
                # Simple t-test approximation
                control_mean = statistics.mean(control_values)
                treatment_mean = statistics.mean(treatment_values)
                control_var = statistics.variance(control_values)
                treatment_var = statistics.variance(treatment_values)
                
                pooled_se = math.sqrt(control_var/len(control_values) + treatment_var/len(treatment_values))
                if pooled_se > 0:
                    t_stat = (treatment_mean - control_mean) / pooled_se
                    # Simple approximation for p-value
                    p_value = 2 * (1 - (0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2)))))
                else:
                    t_stat = 0
                    p_value = 1.0
            
            control_mean = statistics.mean(control_values)
            treatment_mean = statistics.mean(treatment_values)
            
            # Cohen's d for effect size
            control_std = statistics.stdev(control_values)
            treatment_std = statistics.stdev(treatment_values)
            pooled_std = math.sqrt(
                ((len(control_values) - 1) * control_std**2 + 
                 (len(treatment_values) - 1) * treatment_std**2) /
                (len(control_values) + len(treatment_values) - 2)
            )
            
            cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for difference
            se_diff = math.sqrt(
                (control_std**2 / len(control_values)) +
                (treatment_std**2 / len(treatment_values))
            )
            
            df = len(control_values) + len(treatment_values) - 2
            if HAS_SCIPY:
                t_critical = stats.t.ppf(1 - self.alpha / 2, df)
            else:
                # Approximation for t-critical (roughly 2.0 for large samples)
                t_critical = 2.0
            margin_error = t_critical * se_diff
            diff = treatment_mean - control_mean
            
            return {
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "difference": diff,
                "relative_improvement": (diff / control_mean) if control_mean != 0 else None,
                "p_value": p_value,
                "t_statistic": t_stat,
                "degrees_of_freedom": df,
                "confidence_interval": (diff - margin_error, diff + margin_error),
                "significant": p_value < self.alpha,
                "effect_size": cohens_d,
                "cohens_d_interpretation": self._interpret_cohens_d(cohens_d),
            }
        
        except Exception as e:
            return {
                "p_value": None,
                "confidence_interval": None,
                "effect_size": None,
                "significant": False,
                "error": str(e),
            }
    
    def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        power: float = 0.8,
    ) -> int:
        """Calculate required sample size for detecting a minimum effect."""
        if baseline_rate <= 0 or baseline_rate >= 1:
            raise ValueError("Baseline rate must be between 0 and 1")
        
        if minimum_detectable_effect <= 0:
            raise ValueError("Minimum detectable effect must be positive")
        
        if power <= 0 or power >= 1:
            raise ValueError("Power must be between 0 and 1")
        
        # Calculate required sample size for proportion test
        treatment_rate = baseline_rate + minimum_detectable_effect
        
        if treatment_rate >= 1:
            treatment_rate = 1 - 0.001  # Cap at 99.9%
        
        # Use normal approximation for sample size calculation
        if HAS_SCIPY:
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            z_beta = stats.norm.ppf(power)
        else:
            # Standard approximations
            z_alpha = 1.96  # 95% confidence
            z_beta = 0.84   # 80% power
        
        p_avg = (baseline_rate + treatment_rate) / 2
        
        # Sample size per group
        n = (
            (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) +
             z_beta * math.sqrt(baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate)))**2
        ) / (minimum_detectable_effect**2)
        
        return math.ceil(n)
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def run_sequential_analysis(
        self,
        control_data: List[float],
        treatment_data: List[float],
        look_frequency: int = 100,
    ) -> Dict[str, Any]:
        """Run sequential analysis to determine when to stop experiment early."""
        results = []
        
        min_sample_size = min(len(control_data), len(treatment_data))
        
        for i in range(look_frequency, min_sample_size + 1, look_frequency):
            control_subset = control_data[:i]
            treatment_subset = treatment_data[:i]
            
            analysis_result = self.compare_means(control_subset, treatment_subset)
            analysis_result["sample_size"] = i
            analysis_result["look_number"] = len(results) + 1
            
            results.append(analysis_result)
            
            # Check for early stopping
            if analysis_result["significant"] and analysis_result.get("effect_size"):
                effect_size = abs(analysis_result["effect_size"])
                if effect_size > 0.2:  # Meaningful effect size
                    analysis_result["early_stop_recommendation"] = True
                    break
        
        return {
            "sequential_results": results,
            "final_recommendation": results[-1] if results else None,
        }