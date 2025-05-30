@echo off
echo Setting up Ramayana Fact Checker Environment...
echo.

REM Set Ollama models directory to D: drive to save C: drive space
set OLLAMA_MODELS=D:\ollama\models
echo Ollama models will be stored in: %OLLAMA_MODELS%

REM Create the directory if it doesn't exist
if not exist "D:\ollama\models" mkdir "D:\ollama\models"

:: Create virtual environment
echo Creating virtual environment...
python -m venv env

:: Activate virtual environment
echo Activating virtual environment...
call env\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt

echo.
echo ================================================
echo Environment setup complete!
echo.
echo To activate the environment, run:
echo   env\Scripts\activate.bat
echo.
echo To train the model, run:
echo   python transformer_fact_checker.py --train
echo.
echo To deactivate, run:
echo   deactivate
echo ================================================
pause
