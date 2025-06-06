"""Experimental decorators for A/B testing contracts."""

import asyncio
import time
import uuid
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

from llm_contracts.contracts.base import ContractBase, ValidationResult
from llm_contracts.language.integration import llmcl_to_contract
from .experiment_manager import ExperimentManager, ExperimentConfig, TrafficAllocation


# Global experiment manager instance
_experiment_manager = ExperimentManager()


def ab_test_contract(
    control_contract: Union[str, ContractBase],
    treatment_contract: Union[str, ContractBase],
    experiment_name: str,
    traffic_split: float = 0.5,
    allocation_strategy: TrafficAllocation = TrafficAllocation.RANDOM,
    auto_start: bool = True,
    min_participants: int = 100,
    max_participants: Optional[int] = None,
    duration_hours: Optional[int] = None,
    confidence_level: float = 0.95,
):
    """
    Decorator for A/B testing two contracts.
    
    Args:
        control_contract: Control contract (LLMCL string or ContractBase)
        treatment_contract: Treatment contract (LLMCL string or ContractBase)
        experiment_name: Name of the experiment
        traffic_split: Percentage of traffic to send to treatment (0.0-1.0)
        allocation_strategy: How to allocate traffic
        auto_start: Whether to start experiment immediately
        min_participants: Minimum participants per variant before conclusions
        max_participants: Maximum total participants
        duration_hours: Maximum duration in hours
        confidence_level: Statistical confidence level
    
    Example:
        >>> @ab_test_contract(
        ...     control_contract='''
        ...         contract ControlContract {
        ...             ensure len(response) > 0
        ...         }
        ...     ''',
        ...     treatment_contract='''
        ...         contract TreatmentContract {
        ...             ensure len(response) > 0
        ...             ensure json_valid(response)
        ...         }
        ...     ''',
        ...     experiment_name="json_validation_test",
        ...     traffic_split=0.3,
        ... )
        ... async def my_llm_function(prompt: str) -> str:
        ...     return '{"result": "success"}'
    """
    def decorator(func: Callable) -> Callable:
        # Convert contracts if they're strings
        control = llmcl_to_contract(control_contract) if isinstance(control_contract, str) else control_contract
        treatment = llmcl_to_contract(treatment_contract) if isinstance(treatment_contract, str) else treatment_contract
        
        # Create experiment config
        config = ExperimentConfig(
            name=experiment_name,
            description=f"A/B test for {func.__name__}",
            control_contract=control,
            treatment_contracts=[treatment],
            traffic_allocation=allocation_strategy,
            traffic_percentages=[100 * (1 - traffic_split), 100 * traffic_split],
            min_participants_per_variant=min_participants,
            max_participants=max_participants,
            confidence_level=confidence_level,
        )
        
        if duration_hours:
            from datetime import datetime, timedelta
            config.end_time = datetime.now() + timedelta(hours=duration_hours)
        
        # Create and start experiment
        experiment_id = None
        
        async def setup_experiment():
            nonlocal experiment_id
            experiment_id = await _experiment_manager.create_experiment(config)
            if auto_start:
                await _experiment_manager.start_experiment(experiment_id)
        
        # Run setup in background
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(setup_experiment())
        else:
            asyncio.run(setup_experiment())
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not experiment_id:
                # Fallback to control if experiment not ready
                result = await func(*args, **kwargs)
                if control:
                    validation_result = await control.validate(str(result), {})
                    if not validation_result.is_valid:
                        raise ValueError(f"Contract validation failed: {validation_result.message}")
                return result
            
            # Get participant info
            participant_id = kwargs.get('participant_id') or str(uuid.uuid4())
            session_id = kwargs.get('session_id')
            
            # Get contract for this request
            start_time = time.time()
            contract, variant = await _experiment_manager.get_contract_for_request(
                experiment_id,
                participant_id,
                session_id,
                context={"args": args, "kwargs": kwargs},
            )
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Validate with assigned contract
            validation_result = ValidationResult(is_valid=True)
            if contract:
                validation_result = await contract.validate(str(result), {})
                if not validation_result.is_valid and variant == "control":
                    # Only raise errors for control group to maintain backwards compatibility
                    raise ValueError(f"Contract validation failed: {validation_result.message}")
            
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            await _experiment_manager.record_validation_result(
                experiment_id,
                variant,
                validation_result,
                latency_ms,
                context={"function": func.__name__},
            )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Add experiment management methods to the decorated function
        if asyncio.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        
        async def get_experiment_status():
            if experiment_id:
                return await _experiment_manager.get_experiment_status(experiment_id)
            return None
        
        async def stop_experiment():
            if experiment_id:
                return await _experiment_manager.stop_experiment(experiment_id)
            return None
        
        wrapper.get_experiment_status = get_experiment_status
        wrapper.stop_experiment = stop_experiment
        wrapper.experiment_id = experiment_id
        
        return wrapper
    
    return decorator


def experiment_contract(
    contracts: Dict[str, Union[str, ContractBase]],
    experiment_name: str,
    traffic_percentages: Optional[List[float]] = None,
    allocation_strategy: TrafficAllocation = TrafficAllocation.RANDOM,
    auto_start: bool = True,
    min_participants: int = 100,
    max_participants: Optional[int] = None,
    duration_hours: Optional[int] = None,
    confidence_level: float = 0.95,
):
    """
    Decorator for multi-variant contract testing.
    
    Args:
        contracts: Dictionary mapping variant names to contracts
        experiment_name: Name of the experiment
        traffic_percentages: Percentage split for each variant (must sum to 100)
        allocation_strategy: How to allocate traffic
        auto_start: Whether to start experiment immediately
        min_participants: Minimum participants per variant
        max_participants: Maximum total participants
        duration_hours: Maximum duration in hours
        confidence_level: Statistical confidence level
    
    Example:
        >>> @experiment_contract(
        ...     contracts={
        ...         "control": "contract Control { ensure len(response) > 0 }",
        ...         "basic_validation": "contract Basic { ensure json_valid(response) }",
        ...         "strict_validation": "contract Strict { ensure json_valid(response) and len(response) < 1000 }",
        ...     },
        ...     experiment_name="validation_comparison",
        ...     traffic_percentages=[50, 30, 20],
        ... )
        ... async def my_function(prompt: str) -> str:
        ...     return process_prompt(prompt)
    """
    def decorator(func: Callable) -> Callable:
        # Convert contracts
        parsed_contracts = {}
        for name, contract in contracts.items():
            if isinstance(contract, str):
                parsed_contracts[name] = llmcl_to_contract(contract)
            else:
                parsed_contracts[name] = contract
        
        # Determine control and treatments
        variant_names = list(parsed_contracts.keys())
        control_name = variant_names[0]
        treatment_names = variant_names[1:]
        
        control_contract = parsed_contracts[control_name]
        treatment_contracts = [parsed_contracts[name] for name in treatment_names]
        
        # Set default traffic percentages
        if traffic_percentages is None:
            equal_split = 100.0 / len(variant_names)
            traffic_percentages = [equal_split] * len(variant_names)
        
        # Create experiment config
        config = ExperimentConfig(
            name=experiment_name,
            description=f"Multi-variant test for {func.__name__}",
            control_contract=control_contract,
            treatment_contracts=treatment_contracts,
            traffic_allocation=allocation_strategy,
            traffic_percentages=traffic_percentages,
            min_participants_per_variant=min_participants,
            max_participants=max_participants,
            confidence_level=confidence_level,
        )
        
        if duration_hours:
            from datetime import datetime, timedelta
            config.end_time = datetime.now() + timedelta(hours=duration_hours)
        
        # Create and start experiment
        experiment_id = None
        
        async def setup_experiment():
            nonlocal experiment_id
            experiment_id = await _experiment_manager.create_experiment(config)
            if auto_start:
                await _experiment_manager.start_experiment(experiment_id)
        
        # Run setup
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(setup_experiment())
        else:
            asyncio.run(setup_experiment())
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not experiment_id:
                # Fallback to control
                result = await func(*args, **kwargs)
                if control_contract:
                    validation_result = await control_contract.validate(str(result), {})
                    if not validation_result.is_valid:
                        raise ValueError(f"Contract validation failed: {validation_result.message}")
                return result
            
            # Get participant info
            participant_id = kwargs.get('participant_id') or str(uuid.uuid4())
            session_id = kwargs.get('session_id')
            
            # Get contract for this request
            start_time = time.time()
            contract, variant = await _experiment_manager.get_contract_for_request(
                experiment_id,
                participant_id,
                session_id,
                context={"args": args, "kwargs": kwargs, "variant_names": variant_names},
            )
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Validate with assigned contract
            validation_result = ValidationResult(is_valid=True)
            if contract:
                validation_result = await contract.validate(str(result), {})
                # Only raise errors for control to maintain compatibility
                if not validation_result.is_valid and variant == "control":
                    raise ValueError(f"Contract validation failed: {validation_result.message}")
            
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            await _experiment_manager.record_validation_result(
                experiment_id,
                variant,
                validation_result,
                latency_ms,
                context={"function": func.__name__, "variant_name": variant},
            )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Choose wrapper and add management methods
        if asyncio.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        
        async def get_experiment_status():
            if experiment_id:
                return await _experiment_manager.get_experiment_status(experiment_id)
            return None
        
        async def stop_experiment():
            if experiment_id:
                return await _experiment_manager.stop_experiment(experiment_id)
            return None
        
        wrapper.get_experiment_status = get_experiment_status
        wrapper.stop_experiment = stop_experiment
        wrapper.experiment_id = experiment_id
        wrapper.variant_names = variant_names
        
        return wrapper
    
    return decorator


def gradual_rollout_contract(
    old_contract: Union[str, ContractBase],
    new_contract: Union[str, ContractBase],
    rollout_name: str,
    initial_percentage: float = 5.0,
    max_percentage: float = 100.0,
    increment_percentage: float = 10.0,
    increment_interval_hours: int = 24,
    success_threshold: float = 0.95,
    rollback_threshold: float = 0.1,
    allocation_strategy: TrafficAllocation = TrafficAllocation.USER_BASED,
):
    """
    Decorator for gradual rollout of new contracts.
    
    This decorator gradually increases traffic to a new contract based on success metrics.
    If the new contract performs well, traffic gradually increases. If it performs poorly,
    the rollout is halted or rolled back.
    
    Args:
        old_contract: Current/old contract
        new_contract: New contract to roll out
        rollout_name: Name of the rollout
        initial_percentage: Starting percentage for new contract
        max_percentage: Maximum percentage for new contract
        increment_percentage: How much to increase percentage each interval
        increment_interval_hours: Hours between percentage increases
        success_threshold: Success rate threshold to continue rollout
        rollback_threshold: Performance drop threshold to trigger rollback
        allocation_strategy: How to allocate traffic (USER_BASED recommended for consistency)
    
    Example:
        >>> @gradual_rollout_contract(
        ...     old_contract="contract Old { ensure len(response) > 0 }",
        ...     new_contract="contract New { ensure len(response) > 0 and json_valid(response) }",
        ...     rollout_name="json_validation_rollout",
        ...     initial_percentage=5.0,
        ...     increment_percentage=15.0,
        ...     increment_interval_hours=12,
        ... )
        ... async def api_function(prompt: str) -> str:
        ...     return process_request(prompt)
    """
    def decorator(func: Callable) -> Callable:
        # Convert contracts
        old = llmcl_to_contract(old_contract) if isinstance(old_contract, str) else old_contract
        new = llmcl_to_contract(new_contract) if isinstance(new_contract, str) else new_contract
        
        # State for gradual rollout
        current_percentage = initial_percentage
        rollout_active = True
        
        # Create initial experiment config
        config = ExperimentConfig(
            name=rollout_name,
            description=f"Gradual rollout for {func.__name__}",
            control_contract=old,
            treatment_contracts=[new],
            traffic_allocation=allocation_strategy,
            traffic_percentages=[100 - current_percentage, current_percentage],
            min_participants_per_variant=50,  # Lower threshold for rollouts
            confidence_level=0.9,  # Slightly lower confidence for faster decisions
            enable_early_stopping=True,
            rollback_on_degradation=True,
            degradation_threshold=rollback_threshold,
        )
        
        experiment_id = None
        
        async def setup_rollout():
            nonlocal experiment_id
            experiment_id = await _experiment_manager.create_experiment(config)
            await _experiment_manager.start_experiment(experiment_id)
            
            # Start background rollout management
            asyncio.create_task(manage_rollout())
        
        async def manage_rollout():
            """Background task to manage gradual rollout."""
            nonlocal current_percentage, rollout_active
            
            while rollout_active and current_percentage < max_percentage:
                await asyncio.sleep(increment_interval_hours * 3600)  # Convert to seconds
                
                if not experiment_id:
                    break
                
                # Get current metrics
                status = await _experiment_manager.get_experiment_status(experiment_id)
                
                if "control" in status["variants"] and "treatment_0" in status["variants"]:
                    control_metrics = status["variants"]["control"]
                    treatment_metrics = status["variants"]["treatment_0"]
                    
                    # Check if we have enough data
                    if (control_metrics["requests"] >= 50 and 
                        treatment_metrics["requests"] >= 50):
                        
                        # Check success rates
                        if treatment_metrics["success_rate"] >= success_threshold:
                            # Increase percentage
                            current_percentage = min(
                                current_percentage + increment_percentage,
                                max_percentage
                            )
                            
                            # Update experiment configuration
                            new_config = config
                            new_config.traffic_percentages = [
                                100 - current_percentage,
                                current_percentage
                            ]
                            
                            # Create new experiment with updated percentages
                            await _experiment_manager.stop_experiment(experiment_id)
                            experiment_id = await _experiment_manager.create_experiment(new_config)
                            await _experiment_manager.start_experiment(experiment_id)
                            
                            print(f"Rollout {rollout_name}: Increased traffic to {current_percentage}%")
                            
                            if current_percentage >= max_percentage:
                                print(f"Rollout {rollout_name}: Completed successfully at {max_percentage}%")
                                rollout_active = False
                        
                        else:
                            # Performance below threshold, halt rollout
                            print(f"Rollout {rollout_name}: Halted due to low success rate: {treatment_metrics['success_rate']}")
                            rollout_active = False
        
        # Setup rollout
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(setup_rollout())
        else:
            asyncio.run(setup_rollout())
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not experiment_id or not rollout_active:
                # Fallback to old contract
                result = await func(*args, **kwargs)
                if old:
                    validation_result = await old.validate(str(result), {})
                    if not validation_result.is_valid:
                        raise ValueError(f"Contract validation failed: {validation_result.message}")
                return result
            
            # Get participant info
            participant_id = kwargs.get('participant_id') or str(uuid.uuid4())
            session_id = kwargs.get('session_id')
            
            # Get contract for this request
            start_time = time.time()
            contract, variant = await _experiment_manager.get_contract_for_request(
                experiment_id,
                participant_id,
                session_id,
                context={"rollout": True, "current_percentage": current_percentage},
            )
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Validate with assigned contract
            validation_result = ValidationResult(is_valid=True)
            if contract:
                validation_result = await contract.validate(str(result), {})
                # Only raise errors for control to maintain stability
                if not validation_result.is_valid and variant == "control":
                    raise ValueError(f"Contract validation failed: {validation_result.message}")
            
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            await _experiment_manager.record_validation_result(
                experiment_id,
                variant,
                validation_result,
                latency_ms,
                context={"function": func.__name__, "rollout": True},
            )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Choose wrapper and add management methods
        if asyncio.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        
        async def get_rollout_status():
            status = {
                "rollout_name": rollout_name,
                "current_percentage": current_percentage,
                "max_percentage": max_percentage,
                "rollout_active": rollout_active,
            }
            if experiment_id:
                exp_status = await _experiment_manager.get_experiment_status(experiment_id)
                status.update(exp_status)
            return status
        
        async def stop_rollout():
            nonlocal rollout_active
            rollout_active = False
            if experiment_id:
                return await _experiment_manager.stop_experiment(experiment_id)
            return None
        
        wrapper.get_rollout_status = get_rollout_status
        wrapper.stop_rollout = stop_rollout
        wrapper.experiment_id = experiment_id
        
        return wrapper
    
    return decorator


# Utility functions to access the global experiment manager
async def get_experiment_manager() -> ExperimentManager:
    """Get the global experiment manager instance."""
    return _experiment_manager


async def get_all_experiments() -> Dict[str, Dict[str, Any]]:
    """Get status of all running experiments."""
    results = {}
    for exp_id in _experiment_manager.experiments:
        try:
            results[exp_id] = await _experiment_manager.get_experiment_status(exp_id)
        except Exception as e:
            results[exp_id] = {"error": str(e)}
    return results


async def cleanup_experiments():
    """Clean up all experiments and resources."""
    await _experiment_manager.cleanup()