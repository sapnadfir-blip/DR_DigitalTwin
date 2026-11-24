@echo off
cd /d C:\Users\sapna\DR_DigitalTwin
set PYTHONPATH=C:\Users\sapna\DR_DigitalTwin;C:\Users\sapna\AppData\Roaming\Python\Python310\site-packages
set DR_WORKSTATION=1
"C:\Program Files\Python310\python.exe" -m streamlit run ui\app.py --server.port 8501 --server.headless true > streamlit_log.txt 2>&1
