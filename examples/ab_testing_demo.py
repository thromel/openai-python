#!/usr/bin/env python3
"""Demonstration of A/B testing capabilities with contract experiments."""

import asyncio
import random
import time

from llm_contracts.experiments import (
    ab_test_contract,
    experiment_contract,
    gradual_rollout_contract,
    get_all_experiments,
)


@ab_test_contract(
    control_contract="""
        contract BasicValidation {
            ensure len(response) > 0
                message: "Response cannot be empty"
        }
    """,
    treatment_contract="""
        contract EnhancedValidation {
            ensure len(response) > 0
                message: "Response cannot be empty"
            ensure len(response) < 1000
                message: "Response too long"
            ensure not contains(response, "error")
                message: "Response contains error"
        }
    """,
    experiment_name="basic_vs_enhanced_validation",
    traffic_split=0.3,  # 30% to treatment
    auto_start=True,
    min_participants=10,
    duration_hours=1,
)
async def process_user_request(user_id: str, prompt: str) -> str:
    """Simulate processing a user request with contract validation."""
    # Simulate some processing time
    await asyncio.sleep(0.1)
    
    # Simulate different response types
    responses = [
        '{"status": "success", "result": "Task completed"}',
        "Simple text response that works fine",
        "error: Something went wrong",  # This will fail enhanced validation
        "x" * 1200,  # This will fail length validation
        "",  # This will fail basic validation
    ]
    
    # Weight responses to favor success
    weights = [0.5, 0.3, 0.1, 0.05, 0.05]
    return random.choices(responses, weights=weights)[0]


@experiment_contract(
    contracts={
        "minimal": "contract Minimal { ensure len(response) > 0 }",
        "standard": """
            contract Standard {
                ensure len(response) > 0
                ensure len(response) < 500
            }
        """,
        "comprehensive": """
            contract Comprehensive {
                ensure len(response) > 0
                ensure len(response) < 500
                ensure not contains(response, "error")
                ensure not contains(response, "failed")
            }
        """,
    },
    experiment_name="validation_level_comparison",
    traffic_percentages=[40, 35, 25],  # 40% minimal, 35% standard, 25% comprehensive
    auto_start=True,
    min_participants=15,
)
async def generate_api_response(request_id: str, data: dict) -> str:
    """Simulate API response generation with different validation levels."""
    await asyncio.sleep(0.05)
    
    # Simulate various response scenarios
    scenarios = [
        '{"success": true, "data": {"id": 123, "name": "test"}}',
        "OK - Request processed successfully",
        "Request failed due to invalid input",
        "error: Database connection timeout",
        "x" * 600,  # Too long
    ]
    
    weights = [0.6, 0.25, 0.08, 0.05, 0.02]
    return random.choices(scenarios, weights=weights)[0]


@gradual_rollout_contract(
    old_contract="contract Old { ensure len(response) > 0 }",
    new_contract="""
        contract New {
            ensure len(response) > 0
            ensure json_valid(response)
                message: "Response must be valid JSON"
        }
    """,
    rollout_name="json_validation_rollout",
    initial_percentage=10.0,
    increment_percentage=20.0,
    increment_interval_hours=0.01,  # Very short for demo (36 seconds)
    success_threshold=0.8,
)
async def json_api_endpoint(endpoint: str, payload: dict) -> str:
    """Simulate JSON API endpoint with gradual rollout of JSON validation."""
    await asyncio.sleep(0.02)
    
    # Simulate JSON and non-JSON responses
    responses = [
        '{"result": "success", "timestamp": "2024-01-01T12:00:00Z"}',
        '{"error": false, "data": [1, 2, 3]}',
        "Plain text response",  # Will fail JSON validation
        "Success: Operation completed",
        '{"malformed": json}',  # Invalid JSON
    ]
    
    # Favor JSON responses
    weights = [0.5, 0.3, 0.1, 0.05, 0.05]
    return random.choices(responses, weights=weights)[0]


async def simulate_traffic():
    """Simulate realistic traffic patterns for the experiments."""
    print("🚀 Starting A/B testing simulation...")
    print("=" * 60)
    
    tasks = []
    
    # Simulate users making requests
    for i in range(50):
        user_id = f"user_{i % 10}"  # 10 different users
        
        # Basic vs Enhanced validation experiment
        task1 = asyncio.create_task(
            process_user_request(user_id, f"Request {i}")
        )
        tasks.append(task1)
        
        # Multi-variant validation experiment
        task2 = asyncio.create_task(
            generate_api_response(f"req_{i}", {"data": f"value_{i}"})
        )
        tasks.append(task2)
        
        # Gradual rollout experiment
        task3 = asyncio.create_task(
            json_api_endpoint(f"/api/endpoint_{i % 3}", {"param": i})
        )
        tasks.append(task3)
        
        # Add some realistic timing
        if i % 5 == 0:
            await asyncio.sleep(0.1)
    
    print(f"📊 Running {len(tasks)} simulated requests...")
    
    # Execute all tasks concurrently
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = sum(1 for r in results if isinstance(r, Exception))
        
        print(f"✅ Completed: {successes} successful, {failures} failed requests")
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")


async def show_experiment_status():
    """Display the status of all running experiments."""
    print("\n📈 Experiment Status Report")
    print("=" * 60)
    
    try:
        experiments = await get_all_experiments()
        
        for exp_id, status in experiments.items():
            if "error" in status:
                print(f"❌ Experiment {exp_id}: {status['error']}")
                continue
                
            print(f"\n🧪 {status.get('name', 'Unknown Experiment')}")
            print(f"   Status: {status.get('status', 'Unknown')}")
            print(f"   Participants: {status.get('total_participants', 0)}")
            
            variants = status.get('variants', {})
            for variant_name, metrics in variants.items():
                requests = metrics.get('requests', 0)
                success_rate = metrics.get('success_rate', 0) * 100
                avg_latency = metrics.get('average_latency_ms', 0)
                
                print(f"   📊 {variant_name}:")
                print(f"      Requests: {requests}")
                print(f"      Success Rate: {success_rate:.1f}%")
                print(f"      Avg Latency: {avg_latency:.1f}ms")
        
        if not experiments:
            print("No active experiments found.")
            
    except Exception as e:
        print(f"❌ Error getting experiment status: {e}")


async def main():
    """Main demo function."""
    print("🔬 LLM Contract A/B Testing Demo")
    print("=" * 60)
    print()
    
    # Run traffic simulation
    await simulate_traffic()
    
    # Wait a bit for metrics to be collected
    await asyncio.sleep(1)
    
    # Show experiment status
    await show_experiment_status()
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print()
    print("In a real scenario, you would:")
    print("• Monitor experiments over longer periods")
    print("• Collect more detailed metrics")
    print("• Use statistical significance testing")
    print("• Implement automated decision making")
    print("• Set up alerting for performance degradation")


if __name__ == "__main__":
    asyncio.run(main())