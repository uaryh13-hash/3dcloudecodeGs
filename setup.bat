@echo off
chcp 65001 >nul
title 3DGS 环境安装

setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0

:: ─── 快速跳过检测 ──────────────────────────────────
if exist "%SCRIPT_DIR%\.installed" (
    echo 环境已安装，跳过。
    exit /b 0
)

echo =============================================
echo  3DGS 环境安装
echo =============================================
echo.

:: ─── 检查 CUDA 显卡 ──────────────────────────────
echo [检测] NVIDIA GPU / CUDA 驱动...
set HAS_CUDA=0
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    set HAS_CUDA=1
    for /f "tokens=2 delims= " %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do set GPU_NAME=%%i
    if not defined GPU_NAME set GPU_NAME=NVIDIA
    echo    GPU: !GPU_NAME! | findstr /v "GPU"
    nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>nul | findstr /v " MiB"
    echo    CUDA 驱动可用
) else (
    echo    [!] 未检测到 NVIDIA GPU
    echo    本软件需要 NVIDIA 显卡 + CUDA 驱动
    echo.
)

:: ─── 检查 Python 3.10 ─────────────────────────────
echo.
echo [1/5] 检查 Python...

set PYTHON_EXE=
for %%v in (3.10 3.11 3.12) do (
    where python%%v >nul 2>&1
    if !errorlevel! equ 0 (
        if "%%v"=="3.10" set "PYTHON_EXE=python3.10"
    )
)
if "!PYTHON_EXE!"=="" (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
        if "!PY_VER:~0,4!"=="3.10" set "PYTHON_EXE=python"
    )
)
if "!PYTHON_EXE!"=="" (
    for /f "tokens=2" %%i in ('py -3.10 --version 2^>^&1') do set PY_VER=%%i
    if "!PY_VER:~0,4!"=="3.10" set "PYTHON_EXE=py -3.10"
)
if "!PYTHON_EXE!"=="" (
    echo [!] 未找到 Python！
    echo.
    for %%v in (3.10 3.11 3.12) do (
        where python%%v >nul 2>&1 && (
            echo    找到 Python %%v，请改用 3.10
            goto :need_python
        )
    )
    :need_python
    echo     请安装 Python 3.10: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo    Python: !PYTHON_EXE!

:: ─── 虚拟环境 ────────────────────────────────────
echo.
echo [2/5] 虚拟环境...

set VENV_DIR=%SCRIPT_DIR%venv
set VENV_PY=%VENV_DIR%\Scripts\python.exe
set VENV_PIP=%VENV_DIR%\Scripts\pip.exe

if exist "%VENV_DIR%" (
    echo    虚拟环境已存在
) else (
    !PYTHON_EXE! -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [!] 创建失败
        pause
        exit /b 1
    )
    echo    创建成功
)

:: ─── PyTorch ────────────────────────────────────
echo.
echo [3/5] PyTorch...

"%VENV_PY%" -c "import torch; exit(0)" >nul 2>&1
if !errorlevel! equ 0 (
    echo    PyTorch 已安装
) else (
    if !HAS_CUDA! equ 1 (
        echo    安装 PyTorch CUDA 12.4...
        "%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    )
    if !errorlevel! neq 0 (
        echo    安装 CPU 版 PyTorch...
        "%VENV_PIP%" install torch torchvision torchaudio
    )
)

:: ─── 其他依赖 ────────────────────────────────────
echo.
echo [4/5] 依赖库...

"%VENV_PY%" -c "import gsplat; import fastapi; import PIL; exit(0)" >nul 2>&1
if !errorlevel! equ 0 (
    echo    依赖库已安装
) else (
    "%VENV_PIP%" install numpy>=1.26.0 Pillow>=10.0.0 tqdm>=4.66.0 fastapi uvicorn
    "%VENV_PIP%" install gsplat --only-binary gsplat 2>nul
    if !errorlevel! neq 0 (
        "%VENV_PIP%" install gsplat
    )
    "%VENV_PY%" -c "import gsplat" >nul 2>&1 || (
        echo [!] gsplat 安装失败
        pause
    )
)

:: ─── COLMAP ──────────────────────────────────────
echo.
echo [5/5] COLMAP...

if exist "%SCRIPT_DIR%colmap\bin\colmap.exe" (
    echo    COLMAP 已存在
) else (
    echo    下载 COLMAP 4.0.4 (CUDA)...
    curl -L --progress-bar -o "%SCRIPT_DIR%colmap.zip" "https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip"
    if exist "%SCRIPT_DIR%colmap.zip" (
        powershell -Command "Expand-Archive -Path '%SCRIPT_DIR%colmap.zip' -DestinationPath '%SCRIPT_DIR%colmap' -Force"
        del "%SCRIPT_DIR%colmap.zip"
    ) else (
        echo [!] 下载失败，请手动下载:
        echo     https://github.com/colmap/colmap/releases/tag/4.0.4
    )
)

:: ─── 验证 ────────────────────────────────────────
echo.
echo =============================================
echo  验证...
echo =============================================
"%VENV_PY%" -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
"%VENV_PY%" -c "import gsplat; print('gsplat:', gsplat.__version__)"
if exist "%SCRIPT_DIR%colmap\bin\colmap.exe" (
    "%SCRIPT_DIR%colmap\bin\colmap.exe" -h 2>&1 | findstr "COLMAP"
)

:: 标记完成
echo . > "%SCRIPT_DIR%\.installed"

echo.
echo =============================================
echo  安装完成！
echo.
echo  Web 界面: 双击 run_web.bat，浏览器打开 http://localhost:8080
echo  命令行: 把照片文件夹拖到 run.bat 上
echo =============================================
pause
