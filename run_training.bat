@echo off
echo Ramayana Fact Checker - Advanced DiffLlama Training
echo ====================================================
echo.

REM Set Ollama models directory to D: drive
set OLLAMA_MODELS=D:\ollama\models

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting advanced model training with DiffLlama architecture...
echo Using differential attention mechanisms for enhanced fact verification...
echo.

REM Run the improved fact checker with DiffLlama
python fact_checker_improved.py --train --epochs 4 --lr 1e-5 --batch-size 6

echo.
echo Training completed!
pause
