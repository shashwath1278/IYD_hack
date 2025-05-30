#!/bin/bash

# Setup script for Ollama Ramayana Fact Checker
echo "🏛️ Setting up Ollama Ramayana Fact Checker"
echo "=========================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install Ollama first:"
    echo "   Visit: https://ollama.ai"
    exit 1
fi

echo "✅ Ollama found"

# Check if Ollama service is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "🚀 Starting Ollama service..."
    ollama serve &
    sleep 5
else
    echo "✅ Ollama service is running"
fi

# Pull Llama 3.8B model if not available
echo "📥 Checking for Llama 3.8B model..."
if ! ollama list | grep -q "llama3:8b"; then
    echo "📥 Downloading Llama 3.8B model (this may take a while)..."
    ollama pull llama3:8b
else
    echo "✅ Llama 3.8B model found"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Generate training data
echo "🎓 Generating training data..."
python src/data_generator.py

# Set up custom model
echo "🔧 Setting up custom Ramayana model..."
python src/ollama_trainer.py

# Test the setup
echo "🧪 Testing the fact checker..."
python src/ollama_fact_checker.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 Quick start commands:"
echo "   # Use the custom model directly:"
echo "   ollama run ramayana-fact-checker"
echo ""
echo "   # Use Python interface:"
echo "   python src/ollama_fact_checker.py"
echo ""
echo "📝 Check ollama_training_guide.md for advanced usage"
