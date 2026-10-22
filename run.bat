@echo off
cd /d "%~dp0"
py -3 predict.py %*
