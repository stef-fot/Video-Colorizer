@echo off
rem Starts the colorization server and the Stremio addon in two windows.
rem Run from the project folder after: python -m venv .venv ; .venv\Scripts\pip install -r requirements-directml.txt
cd /d %~dp0\..
start "ColorStream server" cmd /k ".venv\Scripts\python -m colorstream serve"
start "ColorStream addon" cmd /k "cd addon && node index.js"
echo Dashboard: http://127.0.0.1:7001
echo Stremio addon URL: http://127.0.0.1:7000/manifest.json
