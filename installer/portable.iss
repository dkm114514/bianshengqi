#ifndef BundleDir
  #error Pass /DBundleDir=<offline application directory>
#endif
#ifndef ReleaseDir
  #define ReleaseDir "..\dist\portable-release"
#endif
#ifndef AppVersion
  #define AppVersion "0.1.1"
#endif
#ifndef PatchOnly
  #define PatchOnly 0
#endif

[Setup]
AppId=bianshengqi-portable
AppName=实时变声器绿色版
AppVersion={#AppVersion}
AppPublisher=dkm114514
DefaultDirName={sd}\变声器
UsePreviousAppDir=no
UsePreviousGroup=no
UsePreviousTasks=no
UsePreviousLanguage=no
UsePreviousSetupType=no
Uninstallable=no
CreateUninstallRegKey=no
DisableProgramGroupPage=yes
DisableDirPage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#ReleaseDir}
#if Int(PatchOnly)
OutputBaseFilename=bianshengqi-{#AppVersion}-windows-x64-update
DiskSpanning=no
#else
OutputBaseFilename=bianshengqi-{#AppVersion}-windows-x64-portable
DiskSpanning=yes
DiskSliceSize=2000000000
SlicesPerDisk=1
#endif
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
DisableWelcomePage=no
DisableReadyPage=yes
CloseApplications=no
RestartApplications=no
SetupLogging=no

[Languages]
Name: "chinesesimplified"; MessagesFile: "ChineseSimplified.isl"

[Messages]
chinesesimplified.SelectDirLabel3=请选择变声器文件夹。所有程序文件、配置和声纹都会保存在这个文件夹里。
chinesesimplified.ButtonInstall=解压(&E)
chinesesimplified.FinishedHeadingLabel=解压完成
chinesesimplified.FinishedLabelNoIcons=双击文件夹里的“启动变声器.bat”即可使用。程序不创建系统安装项；关闭后删除整个文件夹即可移除软件。VB-CABLE 驱动需要单独管理。

[Files]
#if Int(PatchOnly)
Source: "{#BundleDir}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BundleDir}\engine\*"; DestDir: "{app}\engine"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BundleDir}\dsp\*"; DestDir: "{app}\dsp"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BundleDir}\tools\*"; DestDir: "{app}\tools"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BundleDir}\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\selfcheck.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\*.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\offline_bundle.json"; DestDir: "{app}"; Flags: ignoreversion
#else
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif

[Code]
function OpenExclusiveFile(FileName: String; DesiredAccess, ShareMode: LongWord;
  SecurityAttributes: LongWord; CreationDisposition, Flags: LongWord;
  TemplateFile: THandle): THandle;
  external 'CreateFileW@kernel32.dll stdcall';
function CloseNativeHandle(Handle: THandle): Boolean;
  external 'CloseHandle@kernel32.dll stdcall';

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = wpSelectDir then begin
#if Int(PatchOnly)
    if not FileExists(ExpandConstant('{app}\offline_bundle.json')) or
       not FileExists(ExpandConstant('{app}\RVC\runtime\python.exe')) then begin
      MsgBox('更新包需要已有的离线完整版。请选择原来的变声器文件夹；首次使用请下载绿色完整包。', mbError, MB_OK);
      Result := False;
    end;
#else
    if FileExists(ExpandConstant('{app}\unins000.exe')) then begin
      MsgBox('这里是旧安装版的目录。请为绿色版选择另一个文件夹，避免与旧卸载器混用。', mbError, MB_OK);
      Result := False;
    end;
#endif
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  FileHandle: THandle;
begin
  Result := '';
#if Int(PatchOnly)
  if not FileExists(ExpandConstant('{app}\offline_bundle.json')) or
     not FileExists(ExpandConstant('{app}\RVC\runtime\python.exe')) then begin
    Result := '更新包需要已有的离线完整版。请指定原来的变声器文件夹。';
    exit;
  end;
#else
  if FileExists(ExpandConstant('{app}\unins000.exe')) then begin
    Result := '绿色版需要独立的文件夹，请不要覆盖旧安装版的目录。';
    exit;
  end;
#endif
  if FileExists(ExpandConstant('{app}\data\.instance.lock')) then begin
    FileHandle := OpenExclusiveFile(ExpandConstant('{app}\data\.instance.lock'), $80000000 or $40000000,
                                    0, 0, 3, $80, 0);
    if FileHandle = THandle(-1) then
      Result := '请先关闭这个文件夹里的变声器，再进行解压或更新。'
    else
      CloseNativeHandle(FileHandle);
  end;
end;
