@echo off
call .venv\Scripts\activate.bat
set DR_WORKSTATION=1
set PYTHONPATH=.
echo Starting full training on OLIVES dataset...
echo Make sure OLIVES dataset is available or set env vars:
echo   OLIVES_LABEL_CSV_OVERRIDE=path\to\labels.csv
echo   OLIVES_PRIME_ROOT_OVERRIDE=path\to\Prime_FULL
echo   OLIVES_TREX_ROOT_OVERRIDE=path\to\TREX_DME
echo.
python run_pipeline.py
pause
