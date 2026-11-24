@echo off
cd /d C:\Users\sapna\DR_DigitalTwin
set PYTHONPATH=C:\Users\sapna\DR_DigitalTwin
"C:\Program Files\Python310\python.exe" metrics_server.py > metrics_server_log.txt 2>&1
