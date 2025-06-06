"""Tests for A/B testing and experimentation framework."""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from llm_contracts.contracts.base import ValidationResult
from llm_contracts.experiments import (
    ExperimentManager,
    ExperimentConfig,
    ExperimentStatus,
    TrafficSplitter,
    TrafficAllocation,
    MetricCollector,
    StatisticalAnalyzer,
    ContractMetric,
    MetricType,
    ab_test_contract,
    experiment_contract,
    gradual_rollout_contract,
)


class TestTrafficSplitter:
    """Test traffic splitting functionality."""
    
    def test_random_allocation(self):
        """Test random traffic allocation."""
        splitter = TrafficSplitter(TrafficAllocation.RANDOM)
        config = ExperimentConfig(
            name="test",
            description="test",
            traffic_percentages=[50, 50]
        )
        
        # Test multiple assignments
        variants = []
        for _ in range(100):
            variant = splitter.assign_variant(config)
            variants.append(variant)
        
        # Should have both control and treatment_0
        assert "control" in variants
        assert "treatment_0" in variants
    
    def test_user_based_allocation(self):
        """Test user-based consistent allocation."""
        splitter = TrafficSplitter(TrafficAllocation.USER_BASED)
        config = ExperimentConfig(
            name="test",
            description="test",
            traffic_percentages=[50, 50]
        )
        
        # Same user should get same variant
        variant1 = splitter.assign_variant(config, participant_id="user123")
        variant2 = splitter.assign_variant(config, participant_id="user123")
        assert variant1 == variant2
        
        # Different users might get different variants
        variant3 = splitter.assign_variant(config, participant_id="user456")
        # Note: variant3 might be same as variant1, but that's ok due to randomness
    
    def test_percentage_based_allocation(self):
        """Test percentage-based allocation."""
        splitter = TrafficSplitter(TrafficAllocation.PERCENTAGE_BASED)
        config = ExperimentConfig(
            name="test",
            description="test",
            traffic_percentages=[70, 30]
        )
        
        # Test allocation over many samples
        control_count = 0
        treatment_count = 0
        
        for _ in range(1000):
            variant = splitter.assign_variant(config)
            if variant == "control":
                control_count += 1
            elif variant == "treatment_0":
                treatment_count += 1
        
        # Should roughly match percentages (within 10%)
        control_percentage = control_count / 1000 * 100
        treatment_percentage = treatment_count / 1000 * 100
        
        assert 60 <= control_percentage <= 80
        assert 20 <= treatment_percentage <= 40


class TestMetricCollector:
    """Test metric collection functionality."""
    
    @pytest.mark.asyncio
    async def test_record_validation(self):
        """Test recording validation results."""
        collector = MetricCollector()
        
        # Create mock validation result
        result = ValidationResult(is_valid=True)
        
        await collector.record_validation(
            experiment_id="exp1",
            variant="control",
            result=result,
            latency_ms=100.0,
        )
        
        # Check metrics
        metrics = collector.get_metrics("exp1")
        assert "control" in metrics
        assert metrics["control"].total_requests == 1
        assert metrics["control"].successful_validations == 1
        assert metrics["control"].failed_validations == 0
        assert metrics["control"].average_latency_ms == 100.0
    
    @pytest.mark.asyncio
    async def test_record_failed_validation(self):
        """Test recording failed validation results."""
        collector = MetricCollector()
        
        # Create mock failed validation result
        result = ValidationResult(
            is_valid=False,
            message="Validation failed",
        )
        
        await collector.record_validation(
            experiment_id="exp1",
            variant="treatment_0",
            result=result,
            latency_ms=150.0,
        )
        
        # Check metrics
        metrics = collector.get_metrics("exp1")
        assert "treatment_0" in metrics
        assert metrics["treatment_0"].total_requests == 1
        assert metrics["treatment_0"].successful_validations == 0
        assert metrics["treatment_0"].failed_validations == 1
        assert metrics["treatment_0"].success_rate == 0.0


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality."""
    
    def test_compare_success_rates(self):
        """Test success rate comparison."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        # Test case with clear difference
        result = analyzer.compare_success_rates(
            control_successes=80,
            control_total=100,
            treatment_successes=90,
            treatment_total=100,
        )
        
        assert result["control_rate"] == 0.8
        assert result["treatment_rate"] == 0.9
        assert result["difference"] == 0.1
        assert result["relative_improvement"] == 0.125  # 12.5% improvement
        assert "p_value" in result
        assert "confidence_interval" in result
    
    def test_compare_means(self):
        """Test means comparison."""
        analyzer = StatisticalAnalyzer()
        
        control_values = [100, 110, 105, 95, 108, 102, 98, 115, 92, 107]
        treatment_values = [85, 90, 88, 95, 82, 87, 91, 86, 89, 84]
        
        result = analyzer.compare_means(control_values, treatment_values)
        
        assert "control_mean" in result
        assert "treatment_mean" in result
        assert "p_value" in result
        assert "t_statistic" in result
        assert "confidence_interval" in result
        assert "effect_size" in result
    
    def test_sample_size_calculation(self):
        """Test sample size calculation."""
        analyzer = StatisticalAnalyzer()
        
        sample_size = analyzer.calculate_sample_size(
            baseline_rate=0.8,
            minimum_detectable_effect=0.05,  # 5 percentage point improvement
            power=0.8,
        )
        
        assert isinstance(sample_size, int)
        assert sample_size > 0


class TestExperimentManager:
    """Test experiment management functionality."""
    
    @pytest.mark.asyncio
    async def test_create_experiment(self):
        """Test experiment creation."""
        manager = ExperimentManager()
        
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
        )
        
        experiment_id = await manager.create_experiment(config)
        
        assert experiment_id in manager.experiments
        experiment = manager.experiments[experiment_id]
        assert experiment.config.name == "test_experiment"
        assert experiment.status == ExperimentStatus.DRAFT
    
    @pytest.mark.asyncio
    async def test_start_experiment(self):
        """Test starting an experiment."""
        manager = ExperimentManager()
        
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
        )
        
        experiment_id = await manager.create_experiment(config)
        await manager.start_experiment(experiment_id)
        
        experiment = manager.experiments[experiment_id]
        assert experiment.status == ExperimentStatus.RUNNING
        assert experiment.started_at is not None
    
    @pytest.mark.asyncio
    async def test_get_contract_for_request(self):
        """Test getting contract for a request."""
        manager = ExperimentManager()
        
        # Create mock contracts
        control_contract = MagicMock()
        treatment_contract = MagicMock()
        
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            control_contract=control_contract,
            treatment_contracts=[treatment_contract],
            traffic_percentages=[50, 50],
        )
        
        experiment_id = await manager.create_experiment(config)
        await manager.start_experiment(experiment_id)
        
        # Get contract for request
        contract, variant = await manager.get_contract_for_request(
            experiment_id,
            participant_id="user123",
        )
        
        assert contract is not None
        assert variant in ["control", "treatment_0"]
        
        # Same user should get same variant
        contract2, variant2 = await manager.get_contract_for_request(
            experiment_id,
            participant_id="user123",
        )
        
        assert variant == variant2
    
    @pytest.mark.asyncio
    async def test_stop_experiment(self):
        """Test stopping an experiment."""
        manager = ExperimentManager()
        
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
        )
        
        experiment_id = await manager.create_experiment(config)
        await manager.start_experiment(experiment_id)
        
        result = await manager.stop_experiment(experiment_id)
        
        experiment = manager.experiments[experiment_id]
        assert experiment.status == ExperimentStatus.COMPLETED
        assert experiment.ended_at is not None
        assert result.experiment_id == experiment_id


class TestDecorators:
    """Test experimental decorators."""
    
    @pytest.mark.asyncio
    async def test_ab_test_decorator_basic(self):
        """Test basic A/B test decorator functionality."""
        
        # Mock the global experiment manager
        with patch('llm_contracts.experiments.decorators._experiment_manager') as mock_manager:
            mock_manager.create_experiment = AsyncMock(return_value="exp123")
            mock_manager.start_experiment = AsyncMock()
            mock_manager.get_contract_for_request = AsyncMock(return_value=(None, "control"))
            mock_manager.record_validation_result = AsyncMock()
            
            @ab_test_contract(
                control_contract="contract Control { ensure len(response) > 0 }",
                treatment_contract="contract Treatment { ensure json_valid(response) }",
                experiment_name="test_ab",
                auto_start=False,  # Don't auto-start for testing
            )
            async def test_function(prompt: str) -> str:
                return '{"result": "success"}'
            
            # Test function execution
            result = await test_function("test prompt")
            assert result == '{"result": "success"}'
            
            # Verify experiment manager was called
            assert mock_manager.create_experiment.called
    
    @pytest.mark.asyncio 
    async def test_experiment_decorator_multivariant(self):
        """Test multi-variant experiment decorator."""
        
        with patch('llm_contracts.experiments.decorators._experiment_manager') as mock_manager:
            mock_manager.create_experiment = AsyncMock(return_value="exp456")
            mock_manager.start_experiment = AsyncMock()
            mock_manager.get_contract_for_request = AsyncMock(return_value=(None, "control"))
            mock_manager.record_validation_result = AsyncMock()
            
            @experiment_contract(
                contracts={
                    "control": "contract Control { ensure len(response) > 0 }",
                    "variant_a": "contract A { ensure json_valid(response) }",
                    "variant_b": "contract B { ensure len(response) < 1000 }",
                },
                experiment_name="multi_variant_test",
                traffic_percentages=[40, 30, 30],
                auto_start=False,
            )
            async def test_function(prompt: str) -> str:
                return '{"result": "test"}'
            
            result = await test_function("test prompt")
            assert result == '{"result": "test"}'
            
            # Check that variants were set up correctly
            assert hasattr(test_function, 'variant_names')
            assert test_function.variant_names == ["control", "variant_a", "variant_b"]
    
    @pytest.mark.asyncio
    async def test_gradual_rollout_decorator(self):
        """Test gradual rollout decorator."""
        
        with patch('llm_contracts.experiments.decorators._experiment_manager') as mock_manager:
            mock_manager.create_experiment = AsyncMock(return_value="rollout789")
            mock_manager.start_experiment = AsyncMock()
            mock_manager.get_contract_for_request = AsyncMock(return_value=(None, "control"))
            mock_manager.record_validation_result = AsyncMock()
            mock_manager.get_experiment_status = AsyncMock(return_value={
                "variants": {
                    "control": {"requests": 100, "success_rate": 0.9},
                    "treatment_0": {"requests": 10, "success_rate": 0.95},
                }
            })
            
            @gradual_rollout_contract(
                old_contract="contract Old { ensure len(response) > 0 }",
                new_contract="contract New { ensure json_valid(response) }",
                rollout_name="gradual_test",
                initial_percentage=5.0,
                increment_percentage=10.0,
                increment_interval_hours=0.001,  # Very short for testing
            )
            async def test_function(prompt: str) -> str:
                return '{"result": "rollout"}'
            
            result = await test_function("test prompt")
            assert result == '{"result": "rollout"}'
            
            # Check rollout status
            status = await test_function.get_rollout_status()
            assert status["rollout_name"] == "gradual_test"
            assert status["current_percentage"] == 5.0


class TestRealWorldScenarios:
    """Test real-world experiment scenarios."""
    
    @pytest.mark.asyncio
    async def test_api_response_validation_experiment(self):
        """Test A/B testing API response validation."""
        
        manager = ExperimentManager()
        
        # Create contracts
        control_contract = MagicMock()
        control_contract.validate = AsyncMock(return_value=ValidationResult(is_valid=True))
        
        treatment_contract = MagicMock()
        treatment_contract.validate = AsyncMock(return_value=ValidationResult(is_valid=True))
        
        config = ExperimentConfig(
            name="api_validation_test",
            description="Test API response validation improvements",
            control_contract=control_contract,
            treatment_contracts=[treatment_contract],
            traffic_percentages=[70, 30],
            min_participants_per_variant=50,
        )
        
        experiment_id = await manager.create_experiment(config)
        await manager.start_experiment(experiment_id)
        
        # Simulate API calls
        for i in range(100):
            participant_id = f"user_{i}"
            contract, variant = await manager.get_contract_for_request(
                experiment_id,
                participant_id=participant_id,
            )
            
            # Simulate API response
            response = '{"data": "test"}'
            
            # Validate response
            validation_result = await contract.validate(response, {})
            
            # Record metrics
            await manager.record_validation_result(
                experiment_id,
                variant,
                validation_result,
                latency_ms=100.0,
            )
        
        # Get experiment status
        status = await manager.get_experiment_status(experiment_id)
        
        assert status["total_participants"] == 100
        assert "control" in status["variants"]
        assert "treatment_0" in status["variants"]
        
        # Stop experiment and get results
        result = await manager.stop_experiment(experiment_id)
        
        assert result.status == ExperimentStatus.COMPLETED
        assert len(result.metrics_by_variant) >= 1
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self):
        """Test detection of performance degradation."""
        
        manager = ExperimentManager()
        
        # Create contracts with different performance characteristics
        control_contract = MagicMock()
        control_contract.validate = AsyncMock(return_value=ValidationResult(is_valid=True))
        
        degraded_contract = MagicMock()
        degraded_contract.validate = AsyncMock(return_value=ValidationResult(is_valid=False, message="Degraded"))
        
        config = ExperimentConfig(
            name="performance_test",
            description="Test performance degradation detection",
            control_contract=control_contract,
            treatment_contracts=[degraded_contract],
            traffic_percentages=[50, 50],
            rollback_on_degradation=True,
            degradation_threshold=0.2,  # 20% degradation threshold
        )
        
        experiment_id = await manager.create_experiment(config)
        await manager.start_experiment(experiment_id)
        
        # Simulate requests that show degradation
        for i in range(100):
            participant_id = f"user_{i}"
            contract, variant = await manager.get_contract_for_request(
                experiment_id,
                participant_id=participant_id,
            )
            
            # Validate response
            validation_result = await contract.validate("test", {})
            
            # Record metrics
            await manager.record_validation_result(
                experiment_id,
                variant,
                validation_result,
                latency_ms=100.0,
            )
        
        # Check if degradation was detected
        metrics = manager.metric_collector.get_metrics(experiment_id)
        
        if "control" in metrics and "treatment_0" in metrics:
            control_success = metrics["control"].success_rate
            treatment_success = metrics["treatment_0"].success_rate
            
            # Treatment should have lower success rate due to mocked failures
            assert treatment_success < control_success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])