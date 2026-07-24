@echo off
title Smart Motor Health Dashboard
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python environment...
    py -m venv .venv
)

echo Installing or checking required packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo Starting the revised dashboard on port 8502...
echo Open this address if the browser does not open automatically:
echo http://localhost:8502
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8502
pause
