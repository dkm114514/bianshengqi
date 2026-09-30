#ifndef BundleDir
  #error Pass /DBundleDir=<offline application directory>
#endif
#ifndef ReleaseDir
  #define ReleaseDir "..\dist\release"
#endif
#ifndef Split
  #define Split "no"
#endif

[Setup]
AppId={{2CAB5558-25C5-4D7C-BE7D-C92AA574E990}
AppName=实时变声器
AppVersion=0.1.0
AppPublisher=dkm114514
AppPublisherURL=https://github.com/dkm114514/bianshengqi
DefaultDirName={localappdata}\Programs\BianShengQi
DefaultGroupName=实时变声器
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#ReleaseDir}
OutputBaseFilename=bianshengqi-0.1.0-windows-x64-setup
Compression=lzma2/fast
SolidCompression=yes
DiskSpanning={#Split}
DiskSliceSize=2000000000
SlicesPerDisk=1
WizardStyle=modern
UninstallDisplayName=实时变声器
InfoBeforeFile={#BundleDir}\THIRD_PARTY_NOTICES.md

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; Flags: checkedonce

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\实时变声器"; Filename: "{app}\启动变声器.bat"; WorkingDir: "{app}"
Name: "{userdesktop}\实时变声器"; Filename: "{app}\启动变声器.bat"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{group}\安装 VB-CABLE 虚拟声卡"; Filename: "{app}\drivers\vbcable\VBCABLE_Setup_x64.exe"; WorkingDir: "{app}\drivers\vbcable"
Name: "{group}\卸载实时变声器"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\drivers\vbcable\VBCABLE_Setup_x64.exe"; WorkingDir: "{app}\drivers\vbcable"; Verb: "runas"; Description: "安装 VB-CABLE 虚拟声卡（VB-Audio donationware，首次使用需安装）"; Flags: postinstall shellexec skipifsilent unchecked
Filename: "{app}\启动变声器.bat"; WorkingDir: "{app}"; Description: "启动实时变声器"; Flags: postinstall shellexec skipifsilent unchecked
