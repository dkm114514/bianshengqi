@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0drivers\vbcable"
if not exist "VBCABLE_Setup_x64.exe" (
  echo [错误] 未找到随包的 VB-CABLE 驱动程序，请重新解压完整绿色包。
  pause
  exit /b 1
)
echo 将打开 VB-CABLE 原版驱动程序，需要管理员权限。
echo Install Driver 用于安装，Remove Driver 用于卸载。完成后请重启电脑。
echo 其他软件还在使用 VB-CABLE 时，请不要卸载它。
powershell -NoProfile -Command "Start-Process -FilePath '.\VBCABLE_Setup_x64.exe' -Verb RunAs -WorkingDirectory (Get-Location).Path"
exit /b %ERRORLEVEL%
