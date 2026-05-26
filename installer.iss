; 3DGS 训练工作室 - Inno Setup 安装脚本
; 使用 Inno Setup 6 编译

#define MyAppName "3DGS 训练工作室"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "3DCloud"
#define MyAppURL "https://github.com/uaryh13-hash/3dcloudecodeGs"
#define MyAppExeName "run_web.bat"

[Setup]
AppId={{8A1E5B3C-2D4F-4E6A-8B7C-9D0E1F2A3B4C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=3DGS_Setup
SetupIconFile=static\favicon.ico
UninstallDisplayIcon={app}\static\favicon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
DisableDirPage=no
AllowNoIcons=yes
UninstallFilesDir={app}\uninstall

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
; 让英文版安装程序显示中文文本
AppName=3DGS 训练工作室
NameAndVersion=%1 版本 %2
AdditionalIcons=附加图标
CreateDesktopIcon=创建桌面快捷方式
DesktopIcon=桌面快捷方式
ProgramFolderIcon=开始菜单图标
InstallRun=安装后运行 3DGS 训练工作室
SetupAppTitle=安装
SetupWindowTitle=安装 - 3DGS 训练工作室

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式:"; Flags: checkedonce

[Files]
Source: "train.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "app.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "viewer.html"; DestDir: "{app}"; Flags: ignoreversion
Source: "run.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "run_web.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "download_colmap.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: ".gitignore"; DestDir: "{app}"; Flags: ignoreversion
Source: "templates\*"; DestDir: "{app}\templates"; Flags: ignoreversion recursesubdirs
Source: "static\*"; DestDir: "{app}\static"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\static\favicon.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\static\favicon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 3DGS 训练工作室"; Flags: postinstall nowait skipifsilent shellexec

[UninstallDelete]
Type: filesandordirs; Name: "{app}\output"
Type: filesandordirs; Name: "{app}\colmap"
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\__pycache__"
