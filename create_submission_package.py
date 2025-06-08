"""
Create submission package for hackathon
"""

import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def create_submission_package():
    """Create a comprehensive submission package"""
    
    # Create submission directory
    submission_dir = Path("submission_package")
    submission_dir.mkdir(exist_ok=True)
    
    # Files to include
    files_to_include = [
        # Main application files
        "src/ollama_fact_checker.py",
        "model_report.py", 
        "test_single_statement.py",
        
        # Data files
        "data/valmiki_verses_initial_data.jsonl",
        "data/ramayana_simple_test_data.jsonl",
        "data/my_detailed_results.jsonl",
        
        # Model files
        "Modelfile.ramayana-fact-checker",
        "RamayanaModelfile",
        
        # Documentation
        "docs/Algorithm_Pipeline_Report.md",
        "README.md",
        "requirements.txt",
        
        # Training and setup
        "src/ollama_trainer.py",
        "src/data_generator.py",
        "setup_env.bat",
        "run_training.bat"
    ]
    
    # Copy files
    for file_path in files_to_include:
        src = Path(file_path)
        if src.exists():
            dst = submission_dir / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"✅ Copied: {file_path}")
        else:
            print(f"⚠️  Missing: {file_path}")
    
    # Create README for submission
    submission_readme = submission_dir / "SUBMISSION_README.md"
    with open(submission_readme, 'w') as f:
        f.write(f"""# Ramayana Fact Checker - Submission Package

## Quick Start
1. Install Ollama and pull llama3:8b
2. Install Python requirements: `pip install -r requirements.txt`
3. Create custom model: `ollama create ramayana-fact-checker:latest -f Modelfile.ramayana-fact-checker`
4. Run evaluation: `python model_report.py --eval_data_file "data/valmiki_verses_initial_data.jsonl" --output_log_file "results.jsonl"`

## Key Files
- `src/ollama_fact_checker.py` - Main fact checking implementation
- `model_report.py` - Evaluation script
- `docs/Algorithm_Pipeline_Report.md` - Technical documentation
- `data/` - Test datasets and results

## Algorithm Pipeline
See `docs/Algorithm_Pipeline_Report.md` for complete technical details.

## Submitted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    
    # Create ZIP file
    zip_filename = f"ramayana_fact_checker_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in submission_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(submission_dir)
                zipf.write(file_path, arcname)
                
    print(f"\n✅ Submission package created: {zip_filename}")
    print(f"📦 Package size: {Path(zip_filename).stat().st_size / (1024*1024):.1f} MB")
    
    # Cleanup
    shutil.rmtree(submission_dir)
    
    return zip_filename

if __name__ == "__main__":
    package_file = create_submission_package()
    print(f"\n🎯 Ready for submission: {package_file}")
