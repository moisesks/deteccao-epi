@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Ambiente nao encontrado. Rode instalar.bat primeiro.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -c "import PySide6" 2>nul
if errorlevel 1 (
    echo Instalando a interface grafica ^(PySide6, ~100 MB^) - so na primeira vez...
    .venv\Scripts\python.exe -m pip install "PySide6>=6.6"
)
start "" .venv\Scripts\pythonw.exe app_epi.py
