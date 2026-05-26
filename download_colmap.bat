@echo off
chcp 65001 >nul
title 下载 COLMAP

set SCRIPT_DIR=%~dp0
set ZIP_PATH=%SCRIPT_DIR%colmap.zip

echo 下载 COLMAP 4.0.4 (CUDA) Windows 版...
echo 大小约 410MB，请耐心等待...

curl -L --progress-bar -o "%ZIP_PATH%" "https://github.com/colmap/colmap/releases/download/4.0.4/colmap-x64-windows-cuda.zip"

echo 解压中...
powershell -Command "Expand-Archive -Path '%ZIP_PATH%' -DestinationPath '%SCRIPT_DIR%colmap' -Force"

del "%ZIP_PATH%"

echo.
echo COLMAP 安装完成！
echo 路径: %SCRIPT_DIR%colmap\bin\colmap.exe

"%SCRIPT_DIR%colmap\bin\colmap.exe" -h | findstr "COLMAP"

pause
