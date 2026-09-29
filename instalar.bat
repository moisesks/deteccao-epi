@echo off
chcp 65001 >nul
echo ============================================
echo  Detector de EPI - instalacao
echo ============================================
echo.
where python >nul 2>&1
if errorlevel 1 (
    echo Python nao encontrado. Instale o Python 3.10 ou superior em
    echo https://www.python.org/downloads/ marcando "Add Python to PATH".
    pause
    exit /b 1
)
echo Criando ambiente virtual em .venv ...
python -m venv .venv
call .venv\Scripts\activate.bat
echo.
echo Instalando dependencias. Na primeira vez baixa ~2 GB (PyTorch). Aguarde.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo ============================================
echo  Pronto. Use webcam.bat, tela.bat ou:
echo    .venv\Scripts\activate
echo    python detectar.py --fonte foto.jpg
echo ============================================
pause
