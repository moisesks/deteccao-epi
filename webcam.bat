@echo off
chcp 65001 >nul
call .venv\Scripts\activate.bat
python detectar.py --fonte webcam
pause
