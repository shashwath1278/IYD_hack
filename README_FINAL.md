# Ramayana Fact Checker - Essential Files

## Core System (Keep These Files)

### Main Implementation
- `src/ollama_fact_checker.py` - Main Ollama RAG implementation
- `model_report.py` - Evaluation system
- `requirements.txt` - Python dependencies

### Documentation  
- `docs/Algorithm_Pipeline_Report.md` - Technical documentation
- `docs/create_pdf_report_alternative.py` - PDF report generator

### Data
- `data/` directory - Ramayana text files (if present)

## Removed Files (No Longer Needed)

### Transformer Components
- `transformer_fact_checker.py` - Complex transformer implementation
- `fact_checker_improved.py` - DiffLlama implementation
- `fact_checker.py` - Compatibility wrapper

### Training & Setup
- `src/ollama_trainer.py` - Custom model creation
- `src/data_generator.py` - Training data generation
- `ramayana_training_data.py` - Training data generator
- `setup_ollama.sh` - Linux setup script
- `setup_env.bat` - Environment setup
- `setup_checker.py` - Setup validation
- `run_training.bat` - Training runner

## Current System Architecture

The cleaned-up system focuses on the working Ollama RAG approach:

1. **Core**: `src/ollama_fact_checker.py` - Main implementation
2. **Evaluation**: `model_report.py` - Testing framework  
3. **Documentation**: Algorithm report and PDF generator
4. **Dependencies**: Minimal requirements (Ollama, NLTK, requests)

This provides a clean, focused codebase for the Ramayana fact-checking system.
