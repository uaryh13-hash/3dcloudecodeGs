@echo off
chcp 65001 >nul
title 3DGS 训练管线

setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0

:: 设置 Python (优先使用内置 venv)
if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        echo [!] 找不到 Python！
        echo     请先运行 setup.bat 安装环境
        pause
        exit /b 1
    )
)

:: 设置 COLMAP 路径
if exist "%SCRIPT_DIR%colmap\bin\colmap.exe" (
    set "COLMAP_EXE=%SCRIPT_DIR%colmap\bin\colmap.exe"
) else (
    where colmap >nul 2>&1
    if !errorlevel! equ 0 (
        set "COLMAP_EXE=colmap"
    ) else (
        echo [!] 找不到 COLMAP！
        echo     请先运行 setup.bat 安装环境
        pause
        exit /b 1
    )
)

:: 检查参数（支持拖拽）
if "%1"=="" (
    echo.
    echo =============================================
    echo  3DGS 训练管线
    echo =============================================
    echo.
    echo  把照片文件夹拖到这个 bat 文件上
    echo.
    echo  或者: run.bat ^<照片文件夹路径^>
    echo.
    pause
    exit /b 1
)

set IMAGES_DIR=%1
set OUTPUT_DIR=%SCRIPT_DIR%output

:: 去除引号
set IMAGES_DIR=%IMAGES_DIR:"=%

if not exist "%IMAGES_DIR%" (
    echo [!] 文件夹不存在: %IMAGES_DIR%
    pause
    exit /b 1
)

:: 统计照片
set COUNT=0
for %%f in ("%IMAGES_DIR%\*.jpg" "%IMAGES_DIR%\*.png" "%IMAGES_DIR%\*.jpeg") do set /a COUNT+=1
echo 照片数量: %COUNT%
if %COUNT% lss 10 (
    echo [!] 建议 30 张以上，覆盖每个角度
)

echo.
echo =============================================
echo  开始处理
echo  照片: %IMAGES_DIR%
echo  输出: %OUTPUT_DIR%
echo =============================================
echo.

:: 设置 CUDA 环境变量并运行
set "COLMAP_EXE=%COLMAP_EXE%"
set "CUDA_PATH=%SCRIPT_DIR%venv\Lib\site-packages\nvidia\cu13"
"%PYTHON_EXE%" "%SCRIPT_DIR%train.py" "%IMAGES_DIR%" --output "%OUTPUT_DIR%" --steps 30000

if !errorlevel! neq 0 (
    echo.
    echo [!] 训练出错，检查上面信息
    pause
    exit /b 1
)

:: 完成
echo.
echo  模型文件: %OUTPUT_DIR%\model.splat
echo  打开 viewer.html 拖入 .splat 文件即可查看
pause
