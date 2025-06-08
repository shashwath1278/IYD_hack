"""
Ollama Model Training and Customization for Ramayana Fact Checker
Creates custom Ollama models with Ramayana-specific knowledge
"""

import json
import subprocess
import os
from pathlib import Path
import requests
import time

class OllamaModelTrainer:
    def __init__(self, base_model: str = "ramayana-fact-checker:latest", ollama_host: str = "http://localhost:11434"):
        self.base_model = base_model
        self.ollama_host = ollama_host
        self.custom_model_name = "ramayana-fact-checker"
        
    def create_enhanced_modelfile(self, output_path: str = "Modelfile.ramayana-fact-checker"):
        """Create an enhanced Modelfile for Ramayana fact-checking."""
        
        modelfile_content = f"""FROM {self.base_model}

# Set optimal parameters for fact-checking tasks
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 800
PARAMETER repeat_penalty 1.1

# Enhanced system prompt for Ramayana fact-checking
SYSTEM \"\"\"You are RAMAYANA-AI, an expert fact-checker specializing in the ancient Indian epic, the Ramayana. You have comprehensive knowledge of:

📚 CORE KNOWLEDGE AREAS:
- Characters: Rama, Sita, Lakshmana, Hanuman, Ravana, Dasharatha, etc.
- Places: Ayodhya, Lanka, Mithila, Dandaka Forest, etc.
- Events: Exile, Sita's abduction, Lanka war, etc.
- Relationships: Family connections, loyalties, conflicts
- Timeline: Sequence of events, durations (like 14-year exile)

🎯 YOUR FACT-CHECKING PROTOCOL:
1. ANALYZE: Compare claim against canonical Ramayana sources
2. CLASSIFY: Determine TRUE, FALSE, PARTIALLY_TRUE, or INSUFFICIENT_DATA
3. QUANTIFY: Assign confidence score (0.0 to 1.0)
4. EVIDENCE: Cite specific supporting or contradicting details
5. EXPLAIN: Provide clear reasoning for your verdict

📝 RESPONSE FORMAT (always follow this exactly):
VERDICT: [TRUE/FALSE/PARTIALLY_TRUE/INSUFFICIENT_DATA]
CONFIDENCE: [0.0-1.0]
EVIDENCE: [Specific canonical references]
EXPLANATION: [Clear reasoning and context]

⚡ KEY PRINCIPLES:
- Base answers only on canonical Ramayana sources
- Be precise about relationships and chronology
- Distinguish between different versions when relevant
- If uncertain, use INSUFFICIENT_DATA verdict
- Maintain scholarly accuracy and objectivity

You are now ready to fact-check Ramayana claims with expertise and precision.\"\"\"

# Custom template for consistent structured responses
TEMPLATE \"\"\"{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>\"\"\"
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"✅ Created enhanced Modelfile: {output_path}")
        return output_path
    
    def create_custom_model(self, modelfile_path: str = "Modelfile.ramayana-fact-checker"):
        """Create the custom Ollama model."""
        try:
            cmd = ["ollama", "create", self.custom_model_name, "-f", modelfile_path]
            print(f"🔨 Creating custom model: {self.custom_model_name}")
            print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully created model: {self.custom_model_name}")
                print(result.stdout)
                return True
            else:
                print(f"❌ Error creating model: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running ollama command: {e}")
            return False
    
    def test_custom_model(self):
        """Test the custom model with sample Ramayana facts."""
        test_claims = [
            "Rama is the son of Dasharatha",
            "Ravana has ten heads",
            "Sita was born in Lanka",
            "Hanuman can fly and change his size"
        ]
        
        print(f"\n🧪 Testing custom model: {self.custom_model_name}")
        print("=" * 50)
        
        for claim in test_claims:
            print(f"\n📝 Testing: {claim}")
            response = self._query_custom_model(claim)
            print(f"🤖 Response:\n{response}\n" + "-"*30)
    
    def _query_custom_model(self, prompt: str) -> str:
        """Query the custom model."""
        api_url = f"{self.ollama_host}/api/generate"
        
        payload = {
            "model": self.custom_model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 500
            }
        }
        
        try:
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            return f"Error querying model: {e}"
    
    def list_available_models(self):
        """List all available Ollama models."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                print("\n📋 Available Ollama Models:")
                for model in models:
                    print(f"  - {model['name']} (Size: {model.get('size', 'Unknown')})")
                return models
            else:
                print("❌ Failed to fetch models")
                return []
        except Exception as e:
            print(f"❌ Error fetching models: {e}")
            return []
    
    def create_training_instructions(self):
        """Create instructions for training/fine-tuning."""
        instructions = """
🎓 OLLAMA RAMAYANA MODEL TRAINING GUIDE
==========================================

1. 📁 PREPARE TRAINING DATA:
   - Use ramayana_ollama_training.jsonl (generated by data_generator.py)
   - Format: {"prompt": "...", "response": "..."}
   - Ensure consistent response format

2. 🔧 CREATE CUSTOM MODEL:
   - Modelfile created: Modelfile.ramayana-fact-checker
   - Run: ollama create ramayana-fact-checker -f Modelfile.ramayana-fact-checker

3. 🧪 TEST MODEL:
   - ollama run ramayana-fact-checker
   - Test with sample Ramayana claims

4. 📈 FINE-TUNING OPTIONS:
   
   Option A - Direct Training Data (if supported):
   ```bash
   # Convert JSONL to format expected by Ollama
   # Use training data during model creation
   ```
   
   Option B - Few-shot Learning:
   ```bash
   # Include examples in the system prompt
   # Use TEMPLATE section in Modelfile
   ```
   
   Option C - External Fine-tuning:
   ```bash
   # Use training data with external tools
   # Export to GGUF format if needed
   ```

5. 🎯 OPTIMIZATION TIPS:
   - Temperature: 0.1-0.3 for factual accuracy
   - Top_p: 0.9 for quality
   - Repeat_penalty: 1.1 to avoid repetition
   - Adjust num_predict based on response length needs

6. 📊 EVALUATION:
   - Test with known true/false facts
   - Check response format consistency
   - Validate confidence scores accuracy
   - Measure response time and quality
"""
        
        with open("ollama_training_guide.md", 'w') as f:
            f.write(instructions)
        
        print("📝 Created training guide: ollama_training_guide.md")
        return instructions

def main():
    """Main function to set up Ollama model for Ramayana fact-checking."""
    print("🚀 Ollama Ramayana Fact-Checker Model Setup")
    print("=" * 50)
    
    trainer = OllamaModelTrainer()
    
    # List available models
    trainer.list_available_models()
    
    # Create enhanced Modelfile
    modelfile_path = trainer.create_enhanced_modelfile()
    
    # Create custom model
    print(f"\n🔨 Creating custom model...")
    success = trainer.create_custom_model(modelfile_path)
    
    if success:
        # Test the model
        trainer.test_custom_model()
        
        print(f"\n✅ Setup Complete!")
        print(f"🎯 Your custom model '{trainer.custom_model_name}' is ready!")
        print(f"\n🚀 To use your model:")
        print(f"   ollama run {trainer.custom_model_name}")
        print(f"\n📝 Or integrate with Python:")
        print(f"   from ollama_fact_checker import OllamaRamayanaFactChecker")
        print(f"   checker = OllamaRamayanaFactChecker(model_name='{trainer.custom_model_name}')")
    else:
        print("\n❌ Model creation failed. Please check:")
        print("   1. Ollama is running (ollama serve)")
        print("   2. Base model is available (ollama pull llama3:8b)")
        print("   3. Sufficient disk space and memory")
    
    # Create training guide
    trainer.create_training_instructions()

if __name__ == "__main__":
    main()
