#!/usr/bin/env python3
"""
ADVERSARIAL OpenAI Provider Demo - Testing Contract Violations

This script creates CORNER CASES and FAILURE SCENARIOS to test that our 
ImprovedOpenAIProvider actually catches contract violations. We're not testing
happy paths - we're trying to break things and verify our system catches it.

Task 3 Validation - Stress Testing Contract Enforcement
======================================================
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Callable

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Make sure OPENAI_API_KEY is in environment.")

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from llm_contracts.providers.openai_provider import (
        ImprovedOpenAIProvider,
        ContractCircuitBreaker,
        CircuitState,
    )
    from llm_contracts.core.exceptions import ContractViolationError
    print("✅ Successfully imported our ImprovedOpenAIProvider")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Use faster, cheaper model for aggressive testing
MODEL = "gpt-4o-mini"  # Fast and cheap for testing failures


class AdversarialContract:
    """Contracts designed to catch specific violations."""

    def __init__(self, name: str, validation_func: Callable, description: str = "", auto_remediation_enabled: bool = True):
        self.name = name
        self.description = description
        self.validation_func = validation_func
        self.supports_streaming = False
        # NEW: Per-contract auto-remediation control
        self.auto_remediation_enabled = auto_remediation_enabled

    def validate(self, content: Any) -> 'ValidationResult':
        """Perform strict validation designed to catch violations."""
        try:
            is_valid, message, auto_fix = self.validation_func(content)

            # If auto-remediation is disabled for this contract, remove auto-fix suggestion
            if not self.auto_remediation_enabled:
                auto_fix = None

            return ValidationResult(
                is_valid=is_valid,
                message=message,
                auto_fix_suggestion=auto_fix
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Validation error: {str(e)}",
                auto_fix_suggestion=None
            )

    def validate_input(self, kwargs: Dict[str, Any]) -> 'ValidationResult':
        """Validate input parameters."""
        return self.validate(kwargs)


class ValidationResult:
    """Validation result container."""

    def __init__(self, is_valid: bool, message: str, auto_fix_suggestion: Any = None):
        self.is_valid = is_valid
        self.message = message
        self.auto_fix_suggestion = auto_fix_suggestion


def create_strict_contracts() -> List[AdversarialContract]:
    """Create contracts that will catch violations - not lenient."""

    def strict_prompt_length(content):
        """STRICT prompt length - will fail on long prompts."""
        if isinstance(content, dict) and 'messages' in content:
            total_length = sum(len(msg.get('content', ''))
                               for msg in content['messages'])
            if total_length > 200:  # Very strict limit
                # NO AUTO-FIX - just fail
                return False, f"Prompt too long: {total_length} chars (max: 200)", None
            return True, f"Prompt length OK: {total_length} chars", None
        return True, "No messages to validate", None

    def strict_json_format(content):
        """STRICT JSON validation - will fail on non-JSON."""
        if isinstance(content, str):
            content = content.strip()
            if not (content.startswith('{') and content.endswith('}')):
                # NO AUTO-FIX - just fail
                return False, "Response must be JSON object starting with { and ending with }", None
            try:
                import json
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    return False, "JSON must be an object, not array or primitive", None
                if len(parsed) < 2:
                    return False, "JSON object must have at least 2 fields", None
                return True, "Valid JSON format", None
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON syntax: {e}", None
        return False, "Response is not a string", None

    def strict_response_length(content):
        """STRICT length check - will fail on short responses."""
        if isinstance(content, str):
            if len(content.strip()) < 50:
                # NO AUTO-FIX - just fail
                return False, f"Response too short: {len(content)} chars (min: 50)", None
            if len(content.strip()) > 300:
                return False, f"Response too long: {len(content)} chars (max: 300)", None
            return True, f"Response length OK: {len(content)} chars", None
        return False, "Response is not a string", None

    def strict_content_safety(content):
        """STRICT safety check - will flag many words."""
        if isinstance(content, str):
            # Expanded list of "forbidden" words for testing
            forbidden_words = ['kill', 'hack', 'virus', 'malware', 'exploit', 'bomb', 'weapon',
                               'violence', 'attack', 'destroy', 'break', 'crack', 'illegal', 'steal', 'robbery']
            content_lower = content.lower()
            violations = [
                word for word in forbidden_words if word in content_lower]
            if violations:
                # NO AUTO-FIX - just fail
                return False, f"Content safety violation: forbidden words {violations}", None
        return True, "Content appears safe", None

    def strict_no_code(content):
        """Forbid any code-like content."""
        if isinstance(content, str):
            code_indicators = ['def ', 'function', 'import ', 'class ', '```',
                               'print(', 'console.log', 'SELECT ', 'FROM ', 'WHERE ', '&&', '||', 'if (', 'for (']
            content_check = content.lower()
            violations = [
                indicator for indicator in code_indicators if indicator.lower() in content_check]
            if violations:
                # NO AUTO-FIX - just fail
                return False, f"No code allowed, found: {violations}", None
        return True, "No code detected", None

    return [
        AdversarialContract("StrictPromptLength", strict_prompt_length,
                            "Max 200 chars total - will fail on long prompts", auto_remediation_enabled=False),
        AdversarialContract("StrictJSONFormat", strict_json_format,
                            "Must be valid JSON object with 2+ fields", auto_remediation_enabled=False),
        AdversarialContract("StrictResponseLength", strict_response_length,
                            "Must be 50-300 chars - will fail outside range", auto_remediation_enabled=False),
        AdversarialContract("StrictContentSafety", strict_content_safety,
                            "Flags many common words as unsafe", auto_remediation_enabled=False),
        AdversarialContract("StrictNoCode", strict_no_code,
                            "Forbids any code-like content", auto_remediation_enabled=False)
    ]


def create_lenient_contracts() -> List[AdversarialContract]:
    """Create contracts that provide auto-fixes for violations."""

    def lenient_prompt_length(content):
        """Lenient prompt length - provides auto-fix for long prompts."""
        if isinstance(content, dict) and 'messages' in content:
            total_length = sum(len(msg.get('content', ''))
                               for msg in content['messages'])
            if total_length > 200:  # Provide auto-fix
                return False, f"Prompt too long: {total_length} chars (max: 200)", {"max_tokens": 50}
            return True, f"Prompt length OK: {total_length} chars", None
        return True, "No messages to validate", None

    def lenient_response_length(content):
        """Lenient length check - provides auto-fix for length issues."""
        if isinstance(content, str):
            if len(content.strip()) < 50:
                return False, f"Response too short: {len(content)} chars (min: 50)", {"min_tokens": 100}
            if len(content.strip()) > 300:
                return False, f"Response too long: {len(content)} chars (max: 300)", {"max_tokens": 50}
            return True, f"Response length OK: {len(content)} chars", None
        return False, "Response is not a string", None

    return [
        AdversarialContract("LenientPromptLength", lenient_prompt_length,
                            "Max 200 chars - provides auto-fix", auto_remediation_enabled=True),
        AdversarialContract("LenientResponseLength", lenient_response_length,
                            "50-300 chars - provides auto-fix", auto_remediation_enabled=True)
    ]


def test_prompt_length_violation():
    """Test 1: Deliberately trigger prompt length violation."""
    print("🔥 Test 1: Prompt Length Violation (SHOULD FAIL)")
    print("----------------------------------------")

    try:
        provider = ImprovedOpenAIProvider()

        # DISABLE auto-remediation for strict testing
        provider.auto_remediation = False
        provider.max_retries = 0

        # Add STRICT prompt length contract (max 200 chars)
        length_contract = create_strict_contracts()[0]
        provider.add_input_contract(length_contract)

        print(f"✅ Added strict prompt length contract (max: 200 chars)")
        print(f"🚫 Auto-remediation DISABLED for strict testing")

        # Create a VERY LONG prompt that should violate the contract
        long_prompt = """This is an extremely long prompt that is designed to test our contract validation system by exceeding the maximum allowed character limit. I am intentionally making this prompt much longer than the allowed 200 character maximum to see if our input validation properly catches this violation and either rejects it or applies auto-remediation. This prompt should definitely trigger a contract violation because it contains way more than 200 characters and our strict contract should catch this and prevent the API call or modify the request parameters."""

        print(
            f"🎯 Testing with {len(long_prompt)} character prompt (should violate 200 char limit)")
        print(f"📝 Prompt preview: {long_prompt[:100]}...")

        # This should trigger input validation failure
        response = provider.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": long_prompt}
            ],
            max_tokens=50
        )

        print(f"❌ UNEXPECTED: API call succeeded when it should have failed!")
        print(f"📝 Response: {response.choices[0].message.content}")
        return False

    except ContractViolationError as e:
        print(f"✅ SUCCESS: Caught expected contract violation: {e}")
        return True
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        return False


def test_json_format_violation():
    """Test 2: Force non-JSON response when JSON is required."""
    print("\n🔥 Test 2: JSON Format Violation (SHOULD FAIL)")
    print("----------------------------------------")

    try:
        provider = ImprovedOpenAIProvider()

        # DISABLE auto-remediation for strict testing
        provider.auto_remediation = False
        provider.max_retries = 0

        # Add STRICT JSON format contract
        json_contract = create_strict_contracts()[1]
        provider.add_output_contract(json_contract)

        print("✅ Added strict JSON format contract (requires valid JSON object)")
        print("🚫 Auto-remediation DISABLED for strict testing")

        # Explicitly ask for NON-JSON response
        response = provider.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Write a simple paragraph about cats. DO NOT use JSON format. Just write normal text sentences."}
            ],
            max_tokens=100
        )

        content = response.choices[0].message.content
        print(f"📝 Response received: {content}")

        # Manually check if it's valid JSON (it shouldn't be)
        try:
            import json
            json.loads(content)
            print(f"❌ UNEXPECTED: Response is valid JSON when we asked for plain text!")
            return False
        except json.JSONDecodeError:
            print(
                f"❌ UNEXPECTED: Got non-JSON response but no contract violation was caught!")
            print(f"This suggests our output validation isn't working properly.")
            return False

    except ContractViolationError as e:
        print(f"✅ SUCCESS: Caught expected JSON format violation: {e}")
        return True
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        return False


def test_response_length_violation():
    """Test 3: Get response that's too short or too long."""
    print("\n🔥 Test 3: Response Length Violation (SHOULD FAIL)")
    print("----------------------------------------")

    try:
        provider = ImprovedOpenAIProvider()

        # DISABLE auto-remediation for strict testing
        provider.auto_remediation = False
        provider.max_retries = 0

        # Add STRICT response length contract (50-300 chars)
        length_contract = create_strict_contracts()[2]
        provider.add_output_contract(length_contract)

        print("✅ Added strict response length contract (50-300 chars)")
        print("🚫 Auto-remediation DISABLED for strict testing")

        # Ask for a very short response that should violate minimum length
        print("🎯 Requesting very short response (should violate 50 char minimum)")
        response = provider.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Say just 'Hi' and nothing else. Keep it as short as possible."}
            ],
            max_tokens=5  # Force very short response
        )

        content = response.choices[0].message.content
        print(f"📝 Response: '{content}' (length: {len(content)} chars)")

        if len(content) < 50:
            print(f"❌ UNEXPECTED: Got short response but no contract violation!")
            return False
        else:
            print(f"❌ UNEXPECTED: Response is long enough, should test differently")
            return False

    except ContractViolationError as e:
        print(f"✅ SUCCESS: Caught expected length violation: {e}")
        return True
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        return False


def test_content_safety_violation():
    """Test 4: Trigger content safety violations."""
    print("\n🔥 Test 4: Content Safety Violation (SHOULD FAIL)")
    print("----------------------------------------")

    try:
        provider = ImprovedOpenAIProvider()

        # DISABLE auto-remediation for strict testing
        provider.auto_remediation = False
        provider.max_retries = 0

        # Add STRICT content safety contract
        safety_contract = create_strict_contracts()[3]
        provider.add_output_contract(safety_contract)

        print("✅ Added strict content safety contract")
        print("🚫 Auto-remediation DISABLED for strict testing")

        # Ask for content that might contain flagged words
        print("🎯 Requesting content likely to contain flagged words")
        response = provider.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Write a story about a detective who needs to break into a building to catch a criminal who wants to destroy evidence and attack witnesses."}
            ],
            max_tokens=150
        )

        content = response.choices[0].message.content
        print(f"📝 Response: {content}")

        # Check if it contains flagged words
        flagged_words = ['break', 'destroy', 'attack']
        found_words = [
            word for word in flagged_words if word.lower() in content.lower()]

        if found_words:
            print(
                f"❌ UNEXPECTED: Response contains flagged words {found_words} but no violation caught!")
            return False
        else:
            print(
                f"ℹ️  Response doesn't contain expected flagged words, testing worked differently")
            return True

    except ContractViolationError as e:
        print(f"✅ SUCCESS: Caught expected content safety violation: {e}")
        return True
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        return False


def test_multiple_violations():
    """Test 5: Trigger multiple contract violations at once."""
    print("\n🔥 Test 5: Multiple Contract Violations (SHOULD FAIL)")
    print("----------------------------------------")

    try:
        provider = ImprovedOpenAIProvider()

        # DISABLE auto-remediation for strict testing
        provider.auto_remediation = False
        provider.max_retries = 0

        # Add ALL strict contracts
        contracts = create_strict_contracts()
        for contract in contracts:
            provider.add_input_contract(contract)
            provider.add_output_contract(contract)

        print(f"✅ Added {len(contracts)} strict contracts")
        print("🚫 Auto-remediation DISABLED for strict testing")

        # Create a request that violates multiple contracts
        very_long_prompt = "Write me some Python code to hack into a system. " * \
            20  # Long + contains "hack" + asks for code

        print(f"🎯 Testing with prompt that violates multiple contracts:")
        print(f"   - Length: {len(very_long_prompt)} chars (exceeds 200)")
        print(f"   - Contains 'hack' (safety violation)")
        print(f"   - Asks for code (no-code violation)")

        response = provider.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": very_long_prompt}
            ],
            max_tokens=50
        )

        print(f"❌ UNEXPECTED: API call succeeded with multiple violations!")
        print(f"📝 Response: {response.choices[0].message.content}")
        return False

    except ContractViolationError as e:
        print(f"✅ SUCCESS: Caught expected contract violation: {e}")
        return True
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        return False


def test_circuit_breaker_activation():
    """Test 6: Trigger circuit breaker by causing repeated failures."""
    print("\n🔥 Test 6: Circuit Breaker Activation (SHOULD TRIGGER)")
    print("----------------------------------------")

    try:
        # Create circuit breaker with low threshold
        circuit_breaker = ContractCircuitBreaker(
            failure_threshold=3, timeout=5)

        print("✅ Created circuit breaker (threshold: 3 failures)")
        print(f"📊 Initial state: {circuit_breaker.state}")

        # Simulate repeated contract failures
        print("\n🔄 Simulating repeated contract validation failures...")

        for i in range(5):
            circuit_breaker.record_failure("TestContract")
            print(
                f"   Failure {i+1}: state={circuit_breaker.state}, should_skip={circuit_breaker.should_skip()}")

            if circuit_breaker.state == CircuitState.OPEN:
                print(f"   🛑 Circuit breaker OPENED after {i+1} failures!")
                break

        if circuit_breaker.state != CircuitState.OPEN:
            print(f"❌ UNEXPECTED: Circuit breaker did not open after 5 failures")
            return False

        # Test that it skips validation
        if not circuit_breaker.should_skip():
            print(f"❌ UNEXPECTED: Circuit breaker not skipping when it should")
            return False

        print(f"✅ SUCCESS: Circuit breaker properly activated and skipping validation")

        # Test recovery
        print(f"\n⏳ Testing circuit breaker recovery...")
        time.sleep(1)  # Short wait

        # Should still be open
        if circuit_breaker.should_skip():
            print(f"✅ Circuit breaker still open before timeout")
        else:
            print(f"⚠️  Circuit breaker recovered early")

        # Record success to reset
        circuit_breaker.record_success()
        print(
            f"📊 After success: state={circuit_breaker.state}, should_skip={circuit_breaker.should_skip()}")

        if circuit_breaker.state == CircuitState.CLOSED:
            print(f"✅ SUCCESS: Circuit breaker properly reset after success")
            return True
        else:
            print(f"❌ UNEXPECTED: Circuit breaker did not reset properly")
            return False

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        return False


def test_auto_remediation():
    """Test 7: Verify auto-remediation works for lenient contracts but not strict ones."""
    print("\n🔥 Test 7: Contract-Level Auto-Remediation Control")
    print("----------------------------------------")

    try:
        # Test 1: Strict contract (should fail without auto-fix)
        print("📋 Part A: Testing STRICT contract (auto-remediation disabled)")
        provider = ImprovedOpenAIProvider()
        provider.auto_remediation = True  # Provider allows it, but contract doesn't
        provider.max_retries = 2

        strict_contract = create_strict_contracts()[0]  # StrictPromptLength
        provider.add_input_contract(strict_contract)

        print(f"✅ Added strict contract: {strict_contract.name}")
        print(
            f"🚫 Contract auto-remediation: {strict_contract.auto_remediation_enabled}")

        long_prompt = "This is a very long prompt that exceeds the 200 character limit and should trigger a contract violation that cannot be auto-fixed because this contract has auto-remediation disabled."

        print(
            f"🎯 Testing with {len(long_prompt)} char prompt (exceeds 200 limit)")

        try:
            response = provider.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": long_prompt}],
                max_tokens=50
            )
            print(f"❌ UNEXPECTED: Strict contract allowed call to succeed!")
            return False
        except ContractViolationError as e:
            print(f"✅ SUCCESS: Strict contract properly blocked call: {e}")

        # Test 2: Lenient contract (should auto-fix)
        print(f"\n📋 Part B: Testing LENIENT contract (auto-remediation enabled)")
        provider2 = ImprovedOpenAIProvider()
        provider2.auto_remediation = True
        provider2.max_retries = 2

        lenient_contract = create_lenient_contracts()[0]  # LenientPromptLength
        provider2.add_input_contract(lenient_contract)

        print(f"✅ Added lenient contract: {lenient_contract.name}")
        print(
            f"🔧 Contract auto-remediation: {lenient_contract.auto_remediation_enabled}")

        print(f"🎯 Testing same long prompt with lenient contract")

        response = provider2.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": long_prompt}],
            max_tokens=100  # Should be auto-fixed to 50
        )

        print(f"✅ SUCCESS: Lenient contract allowed call with auto-fix applied")
        print(f"📝 Response: {response.choices[0].message.content}")

        # Check metrics for auto-fix attempts
        health_report = provider2.get_metrics()
        print(f"🔧 Auto-fix metrics: {health_report}")

        return True

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        return False


def main():
    """Run all adversarial tests to verify contract enforcement."""
    print("🔥 ADVERSARIAL CONTRACT TESTING")
    print("=" * 50)
    print("Goal: BREAK things and verify our contracts catch violations")
    print(f"Using model: {MODEL}")
    print("")

    # Check for OpenAI package and API key
    try:
        import openai
        print(f"✅ OpenAI package version: {openai.__version__}")
    except ImportError:
        print("❌ OpenAI package not installed. Please install with:")
        print("   pip install openai")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return

    print("✅ OpenAI API key loaded")
    print("")

    # Run adversarial tests
    tests = [
        test_prompt_length_violation,
        test_json_format_violation,
        test_response_length_violation,
        test_content_safety_violation,
        test_multiple_violations,
        test_circuit_breaker_activation,
        test_auto_remediation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print("")  # Spacing between tests
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            print("")

    # Final results
    print("🏆 ADVERSARIAL TESTING COMPLETE")
    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED - Contract enforcement is working!")
    elif passed > total // 2:
        print("⚠️  MOST TESTS PASSED - Contract system mostly working")
    else:
        print("❌ MANY TESTS FAILED - Contract system needs improvement")

    print("")
    print("🚀 Task 3 Implementation Stress Tested!")


if __name__ == "__main__":
    main()
