@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 py -3 -m venv .venv
 .venv\Scripts\python.exe -m pip install -r requirements.txt
)
.venv\Scripts\python.exe predict.py %*
if "%~1"=="" pause
