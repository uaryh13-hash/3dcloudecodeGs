@echo off
chcp 65001 >nul
title 3DGS 训练工作室

set SCRIPT_DIR=%~dp0
set PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo [错误] 未找到 Python 环境，请先运行 setup.bat
    pause
    exit /b 1
)

echo =============================================
echo  3DGS 训练工作室
echo =============================================
echo.
echo  启动 Web 服务...
echo.
"%PYTHON%" "%SCRIPT_DIR%app.py"
pause
