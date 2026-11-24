@echo off
cd /d C:\Users\sapna\grafana-v11.4.0
bin\grafana-server.exe --homepath C:\Users\sapna\grafana-v11.4.0 --config conf\custom.ini > C:\Users\sapna\DR_DigitalTwin\grafana_log.txt 2>&1
