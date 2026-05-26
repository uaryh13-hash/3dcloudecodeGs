@echo off
chcp 65001 >nul
title 打包 3DGS 便携包

setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
set PARENT_DIR=%SCRIPT_DIR%..
set PKG_NAME=3dgs_pipeline_portable
set PKG_DIR=%PARENT_DIR%\%PKG_NAME%

echo =============================================
echo  打包 3DGS 便携部署包
echo =============================================
echo.

:: 创建临时打包目录
if exist "%PKG_DIR%" (
    echo 删除旧打包目录...
    rmdir /s /q "%PKG_DIR%"
)

echo 创建打包目录...
mkdir "%PKG_DIR%"
mkdir "%PKG_DIR%\colmap"
mkdir "%PKG_DIR%\venv"

:: 复制核心脚本
echo 复制脚本文件...
copy "%SCRIPT_DIR%train.py" "%PKG_DIR%\" >nul
copy "%SCRIPT_DIR%app.py" "%PKG_DIR%\" >nul
copy "%SCRIPT_DIR%viewer.html" "%PKG_DIR%\" >nul
copy "%SCRIPT_DIR%run.bat" "%PKG_DIR%\" >nul
copy "%SCRIPT_DIR%run_web.bat" "%PKG_DIR%\" >nul
copy "%SCRIPT_DIR%setup.bat" "%PKG_DIR%\" >nul
copy "%SCRIPT_DIR%download_colmap.bat" "%PKG_DIR%\" >nul
copy "%SCRIPT_DIR%requirements.txt" "%PKG_DIR%\" >nul
xcopy "%SCRIPT_DIR%templates" "%PKG_DIR%\templates\" /E /I /Q >nul 2>&1
xcopy "%SCRIPT_DIR%static" "%PKG_DIR%\static\" /E /I /Q >nul 2>&1

:: 复制 COLMAP (排除大文件中的无用内容)
echo 复制 COLMAP (便携版)...
xcopy "%SCRIPT_DIR%colmap\bin" "%PKG_DIR%\colmap\bin\" /E /I /Q >nul 2>&1
if exist "%SCRIPT_DIR%colmap\lib" (
    xcopy "%SCRIPT_DIR%colmap\lib" "%PKG_DIR%\colmap\lib\" /E /I /Q >nul 2>&1
)

:: 复制 venv (排除缓存和过大的无用文件)
echo 复制 Python 环境 (可能需要几分钟)...
xcopy "%SCRIPT_DIR%venv\Scripts" "%PKG_DIR%\venv\Scripts\" /E /I /Q >nul 2>&1
xcopy "%SCRIPT_DIR%venv\Lib" "%PKG_DIR%\venv\Lib\" /E /I /Q >nul 2>&1
xcopy "%SCRIPT_DIR%venv\pyvenv.cfg" "%PKG_DIR%\" >nul 2>&1

:: 清理 venv 中的缓存和测试文件
echo 清理临时文件...
if exist "%PKG_DIR%\venv\Lib\site-packages\*.pyc" del /s /q "%PKG_DIR%\venv\Lib\site-packages\*.pyc" >nul 2>&1
if exist "%PKG_DIR%\venv\Scripts\*.exe" (
    :: 保留核心 exe，清理不需要的
)

:: 创建说明文件
echo 创建使用说明...
(
    echo 3D Gaussian Splatting - 便携式部署包
    echo ======================================
    echo.
    echo 系统要求:
    echo   - Windows 10/11
    echo   - NVIDIA GPU (8GB+ VRAM 推荐)
    echo   - 已安装 NVIDIA 显卡驱动
    echo.
    echo 首次使用:
    echo   1. 右键管理员运行 setup.bat（安装/更新 Python 依赖）
    echo      （需要网络，约 5-10 分钟）
    echo   2. 准备 30-100 张照片（覆盖每个角度）
    echo   3. 双击 run_web.bat 启动 Web 界面，浏览器打开 http://localhost:8080
    echo   4. 选择照片 → 开始训练 → 查看 3D 结果
    echo.
    echo   也可以直接把照片文件夹拖到 run.bat 上（命令行模式）
    echo.
    echo 文件结构:
    echo   train.py              - 训练管线
    echo   app.py                - Web 界面后端
    echo   viewer.html           - 3DGS 查看器
    echo   run_web.bat           - Web 界面入口（推荐）
    echo   run.bat               - 命令行模式入口
    echo   setup.bat             - 环境安装
    echo   colmap\               - COLMAP SfM
    echo   venv\                 - Python 环境
    echo   output\               - 输出文件夹（运行后自动创建）
) > "%PKG_DIR%\README.txt"

echo.
echo =============================================
echo  打包完成！
echo  目录: %PKG_DIR%
echo.
echo  可以直接压缩 %PKG_NAME% 文件夹分享给其他电脑
echo  其他电脑上需先运行 setup.bat
echo =============================================
pause
