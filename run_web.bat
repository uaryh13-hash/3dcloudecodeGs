@echo off
chcp 65001 >nul
title 3DGS 训练工作室

set SCRIPT_DIR=%~dp0
set PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe

:: 首次运行：自动安装环境
if not exist "%PYTHON%" (
    echo =============================================
    echo  首次运行 - 正在安装环境...
    echo  需要联网，约 5-10 分钟
    echo =============================================
    echo.
    call "%SCRIPT_DIR%setup.bat"
    if errorlevel 1 (
        echo.
        echo [错误] 环境安装失败
        pause
        exit /b 1
    )
)

echo =============================================
echo  3DGS 训练工作室
echo =============================================
echo.
echo  浏览器请访问: http://localhost:8080
echo  按 Ctrl+C 停止服务
echo.
"%PYTHON%" "%SCRIPT_DIR%app.py"
pause
