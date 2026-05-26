@echo off
chcp 65001 >nul
title 3DGS 环境安装

setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0

echo =============================================
echo  3D Gaussian Splatting 本地管线 - 环境安装
echo =============================================
echo.

:: ============================================
:: 步骤 1: 检查 Python 3.10
:: ============================================
echo [1/5] 检查 Python 3.10...

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
    echo [!] 未找到 Python 3.10！
    echo.
    echo     请先从 https://www.python.org/downloads/ 下载 Python 3.10
    echo     安装时务必勾选 "Add Python to PATH"
    pause
    exit /b 1
)
echo    Python 3.10 已找到: !PYTHON_EXE!

:: ============================================
:: 步骤 2: 创建虚拟环境
:: ============================================
echo.
echo [2/5] 创建虚拟环境...

if exist "%SCRIPT_DIR%venv" (
    echo    虚拟环境已存在，跳过
) else (
    !PYTHON_EXE! -m venv "%SCRIPT_DIR%venv"
    if !errorlevel! neq 0 (
        echo [!] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo    虚拟环境创建成功
)

set VENV_PY=%SCRIPT_DIR%venv\Scripts\python.exe
set VENV_PIP=%SCRIPT_DIR%venv\Scripts\pip.exe

:: ============================================
:: 步骤 3: 安装 PyTorch CUDA
:: ============================================
echo.
echo [3/5] 安装 PyTorch (CUDA 12.4)...

"%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if !errorlevel! neq 0 (
    echo [!] PyTorch CUDA 安装失败，尝试 CPU 版本...
    "%VENV_PIP%" install torch torchvision torchaudio
)

:: ============================================
:: 步骤 4: 安装 gsplat 和其他依赖
:: ============================================
echo.
echo [4/5] 安装 gsplat 和其他依赖...

:: 先安装基础依赖
"%VENV_PIP%" install numpy>=1.26.0 Pillow>=10.0.0 tqdm>=4.66.0 fastapi uvicorn

:: 安装 gsplat - 优先使用预编译 wheel
echo    下载 gsplat 预编译 wheel...
"%VENV_PIP%" install gsplat --only-binary gsplat 2>nul
if !errorlevel! neq 0 (
    echo    尝试从镜像下载预编译 wheel...
    "%VENV_PIP%" install gsplat
)

:: 验证 gsplat
echo    验证 gsplat...
"%VENV_PY%" -c "import gsplat; print('gsplat:', gsplat.__version__)" 2>nul
if !errorlevel! neq 0 (
    echo.
    echo [!] gsplat 安装失败！
    echo     请手动运行以下命令安装 gsplat
    echo     "%VENV_PIP%" install gsplat
    pause
)

:: ============================================
:: 步骤 5: 下载 COLMAP
:: ============================================
echo.
echo [5/5] 安装 COLMAP...

if exist "%SCRIPT_DIR%colmap\bin\colmap.exe" (
    echo    COLMAP 已存在
) else (
    echo    下载 COLMAP 4.0.4 (CUDA)...
    set ZIP_PATH=%SCRIPT_DIR%colmap.zip
    curl -L --progress-bar -o "!ZIP_PATH!" "https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip"
    if exist "!ZIP_PATH!" (
        echo    解压中...
        powershell -Command "Expand-Archive -Path '!ZIP_PATH!' -DestinationPath '%SCRIPT_DIR%colmap' -Force"
        del "!ZIP_PATH!"
    ) else (
        echo [!] COLMAP 下载失败，请手动下载:
        echo     https://github.com/colmap/colmap/releases/tag/4.0.4
        pause
    )
)

:: 最终验证
echo.
echo =============================================
echo  验证安装...
echo =============================================
"%VENV_PY%" -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
"%VENV_PY%" -c "import gsplat; print('gsplat:', gsplat.__version__)"
if exist "%SCRIPT_DIR%colmap\bin\colmap.exe" (
    "%SCRIPT_DIR%colmap\bin\colmap.exe" -h 2>&1 | findstr "COLMAP"
)

echo.
echo =============================================
echo  安装完成！
echo.
echo  命令行: 把照片文件夹拖到 run.bat 上
echo  Web 界面: 双击 run_web.bat，浏览器打开 http://localhost:8080
echo  查看: 在 Web 界面点击查看按钮或打开 viewer.html
echo =============================================
pause
