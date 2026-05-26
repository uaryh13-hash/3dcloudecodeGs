@echo off
chcp 65001 >nul
title 构建 3DGS 安装程序

set SCRIPT_DIR=%~dp0

echo =============================================
echo  构建 3DGS 训练工作室 安装程序
echo =============================================
echo.

:: 查找 Inno Setup 编译器
set ISCC=
for %%P in (
    "%ProgramFiles%\Inno Setup 6\ISCC.exe"
    "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    "%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"
) do (
    if exist %%P set ISCC=%%P
)

if not defined ISCC (
    echo [1/2] 未找到 Inno Setup，正在通过 winget 安装...
    echo.
    winget install --id JRSoftware.InnoSetup --exact --silent --accept-package-agreements
    if errorlevel 1 (
        echo.
        echo [!] 安装失败，请手动下载安装 Inno Setup 6:
        echo     https://jrsoftware.org/isdl.php
        echo.
        echo     安装完成后重新运行本脚本。
        pause
        exit /b 1
    )
    :: 重新查找
    for %%P in (
        "%ProgramFiles%\Inno Setup 6\ISCC.exe"
        "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
        "%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"
    ) do (
        if exist %%P set ISCC=%%P
    )
    if not defined ISCC (
        echo [!] Inno Setup 安装完成但未找到编译器
        pause
        exit /b 1
    )
    echo   Inno Setup 安装完成
) else (
    echo [1/2] Inno Setup 已找到
)

:: 编译安装程序
echo [2/2] 编译安装程序...
echo.
%ISCC% "installer.iss"
if errorlevel 1 (
    echo [!] 编译失败
    pause
    exit /b 1
)

for %%F in ("%SCRIPT_DIR%dist\3DGS_Setup.exe") do set FILESIZE=%%~zF
set /a SIZEMB=%FILESIZE% / 1024 / 1024

echo.
echo =============================================
echo  构建成功！
echo.
echo  安装程序: %SCRIPT_DIR%dist\3DGS_Setup.exe (%SIZEMB% MB)
echo.
echo  将此文件分发给其他电脑即可安装使用。
echo  用户首次启动时需联网（自动下载 PyTorch 等依赖）
echo =============================================
pause
