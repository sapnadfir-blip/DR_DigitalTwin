@echo off
echo ============================================
echo  DR Digital Twin - Windows Setup
echo  RTX 5090 + i9 Ultra Workstation
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

:: Create virtual environment
echo [1/4] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

:: Install PyTorch with CUDA
echo [2/4] Installing PyTorch with CUDA support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

:: Install other dependencies
echo [3/4] Installing project dependencies...
pip install -r requirements.txt

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo To run:
echo   1. Activate venv:  .venv\Scripts\activate
echo   2. Demo mode:      set DR_WORKSTATION=1 ^& python run_pipeline.py --demo
echo   3. Streamlit UI:   set DR_WORKSTATION=1 ^& streamlit run ui/app.py
echo   4. Full training:  set DR_WORKSTATION=1 ^& python run_pipeline.py
echo.
pause
