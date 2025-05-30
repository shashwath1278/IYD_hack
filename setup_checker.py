"""
Ramayana Fact Checker Setup Script
This script helps you set up and test your Ramayana fact-checking system.
"""

import requests
import time
import subprocess
import sys
from pathlib import Path

def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_models():
    """Get list of available models in Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json()
        return [model['name'] for model in models.get('models', [])]
    except:
        return []

def check_model_download_status(model_name="llama3.1:8b"):
    """Check if a model is downloaded."""
    models = get_available_models()
    return model_name in models

def create_custom_model():
    """Create the custom Ramayana checker model."""
    print("Creating custom Ramayana fact-checker model...")
    try:
        result = subprocess.run(
            ["ollama", "create", "ramayana-checker", "-f", "RamayanaModelfile"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        if result.returncode == 0:
            print("✓ Custom model 'ramayana-checker' created successfully!")
            return True
        else:
            print(f"✗ Failed to create custom model: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error creating custom model: {e}")
        return False

def main():
    """Main setup function."""
    print("Ramayana Fact Checker Setup")
    print("=" * 40)
    
    # Step 1: Check if Ollama is running
    print("1. Checking Ollama server...")
    if not check_ollama_running():
        print("✗ Ollama server is not running!")
        print("   Please start Ollama with: ollama serve")
        print("   Then run this setup script again.")
        return
    print("✓ Ollama server is running")
    
    # Step 2: Check available models
    print("\n2. Checking available models...")
    models = get_available_models()
    print(f"   Available models: {models}")
    
    # Step 3: Check if base model is downloaded
    print("\n3. Checking base model (llama3.1:8b)...")
    if not check_model_download_status("llama3.1:8b"):
        print("✗ llama3.1:8b is not downloaded yet")
        print("   Please wait for download to complete: ollama pull llama3.1:8b")
        print("   Then run this setup script again.")
        return
    print("✓ llama3.1:8b is available")
    
    # Step 4: Check if custom model exists
    print("\n4. Checking custom model...")
    if "ramayana-checker" in models:
        print("✓ Custom model 'ramayana-checker' already exists")
    else:
        print("! Custom model 'ramayana-checker' not found")
        
        # Check if Modelfile exists
        if Path("RamayanaModelfile").exists():
            print("   Found RamayanaModelfile, creating custom model...")
            if create_custom_model():
                pass  # Success message already printed
            else:
                print("   You can create it manually with:")
                print("   ollama create ramayana-checker -f RamayanaModelfile")
        else:
            print("   RamayanaModelfile not found. Run ramayana_training_data.py first")
            return
    
    # Step 5: Test the fact checker
    print("\n5. Testing fact checker...")
    try:
        from fact_checker_improved import RamayanaFactChecker, test_ollama_connection
        
        if test_ollama_connection():
            checker = RamayanaFactChecker("./data", "ramayana-checker")
            
            # Quick test
            test_statement = "Rama was the son of King Dasharatha"
            print(f"   Testing: '{test_statement}'")
            result = checker.check_fact(test_statement)
            
            if result is True:
                print("   ✓ Test passed - returned True (correct)")
            elif result is False:
                print("   ! Test returned False (unexpected)")
            elif result is None:
                print("   ! Test returned None (may need model adjustment)")
            
            print("\n✓ Setup complete! You can now use:")
            print("   - python fact_checker_improved.py (for testing)")
            print("   - python fact_checker_improved.py (uncomment interactive_mode)")
        else:
            print("   ✗ Ollama connection test failed")
    
    except Exception as e:
        print(f"   ✗ Error testing fact checker: {e}")
    
    print("\n" + "=" * 40)
    print("Setup Summary:")
    print(f"✓ Ollama running: {check_ollama_running()}")
    print(f"✓ Base model ready: {check_model_download_status('llama3.1:8b')}")
    print(f"✓ Custom model ready: {'ramayana-checker' in get_available_models()}")
    print(f"✓ Data files: {len(list(Path('./data').glob('*.txt')))} files found")

if __name__ == "__main__":
    main()
