@echo off
call .venv\Scripts\activate.bat
set DR_WORKSTATION=1
set PYTHONPATH=.
streamlit run ui/app.py
pause
