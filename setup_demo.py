#!/usr/bin/env python3
"""
Setup script for the OpenAI Provider Demo

This script helps you set up the environment to run the real OpenAI API demo.
"""

import os
import sys
import subprocess


def check_openai_package():
    """Check if openai package is installed."""
    try:
        import openai
        print(f"✅ OpenAI package is installed (version: {openai.__version__})")
        return True
    except ImportError:
        print("❌ OpenAI package not found")
        return False


def install_openai():
    """Install the openai package."""
    print("📦 Installing OpenAI package...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "openai"])
        print("✅ OpenAI package installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install OpenAI package")
        return False


def create_env_file():
    """Create .env file with OpenAI API key."""
    env_path = ".env"

    if os.path.exists(env_path):
        print("✅ .env file already exists")
        with open(env_path, 'r') as f:
            content = f.read()
            if 'OPENAI_API_KEY' in content and 'your_' not in content:
                print("✅ OpenAI API key appears to be set")
                return True
            else:
                print("⚠️  OpenAI API key needs to be set in .env file")

    print("\n🔑 Creating .env file...")
    api_key = input("Please enter your OpenAI API key: ").strip()

    if not api_key or api_key.startswith('your_'):
        print("❌ Invalid API key. Please provide a real OpenAI API key.")
        return False

    env_content = f"""# OpenAI API Configuration
OPENAI_API_KEY={api_key}

# Optional: Customize behavior
# MODEL=gpt-3.5-turbo
# MAX_TOKENS=4096
# TEMPERATURE=0.7
# DEBUG=false
"""

    with open(env_path, 'w') as f:
        f.write(env_content)

    print("✅ .env file created successfully")
    return True


def install_optional_packages():
    """Install optional packages for better experience."""
    optional_packages = ['python-dotenv']

    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"⚠️  Failed to install {package} (optional)")


def main():
    """Main setup function."""
    print("🚀 OpenAI Provider Demo Setup")
    print("=" * 40)

    # Check and install OpenAI package
    if not check_openai_package():
        if not install_openai():
            print("❌ Setup failed: Could not install OpenAI package")
            return False

    # Install optional packages
    install_optional_packages()

    # Create .env file
    if not create_env_file():
        print("❌ Setup failed: Could not set up API key")
        return False

    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run the demo: python3 demo_openai_provider.py")
    print("2. Or run individual tests: python3 -c 'import llm_contracts.providers.openai_provider'")
    print("\n📋 What the demo will show:")
    print("  - Real OpenAI API calls with contract enforcement")
    print("  - Circuit breaker pattern demonstration")
    print("  - Performance metrics collection")
    print("  - Streaming response validation")
    print("  - Auto-remediation features")
    print("  - 100% OpenAI SDK compatibility")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
