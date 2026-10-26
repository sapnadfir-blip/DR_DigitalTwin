@echo off
call .venv\Scripts\activate.bat
set PYTHONPATH=.
echo Starting DR Digital Twin Metrics Server for Grafana...
echo.
echo Open Grafana at http://localhost:3000
echo Metrics API at http://localhost:3001
echo.
python grafana/metrics_server.py
pause
