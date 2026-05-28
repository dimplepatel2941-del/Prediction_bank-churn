@echo off
echo.
echo ========================================
echo Bank Customer Churn Predictor
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed
    echo Please install Python with pip
    pause
    exit /b 1
)

REM Install dependencies
echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting Streamlit Application...
echo ========================================
echo.
echo The app will open at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

REM Run Streamlit app
streamlit run app.py

pause
