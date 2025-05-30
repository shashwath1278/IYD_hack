@echo off
echo Ramayana Fact Checker - Interactive Mode
echo ========================================

:: Check if virtual environment exists
if not exist "env\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call env\Scripts\activate.bat

:: Run interactive mode
python transformer_fact_checker.py --interactive

:: Keep window open
pause
