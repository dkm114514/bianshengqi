@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if exist "%~dp0offline_bundle.json" (
  set "BSQ_RVC_ROOT=%~dp0RVC"
  set "RVC_ROOT=%~dp0RVC"
  set "RVC_RUNTIME=%~dp0RVC\runtime\python.exe"
  set "BSQ_OFFLINE=1"
  goto run
)
if defined BSQ_RVC_ROOT (
  set "RVC_RUNTIME=%BSQ_RVC_ROOT%\runtime\python.exe"
  goto run
)
if defined RVC_ROOT (
  set "RVC_RUNTIME=%RVC_ROOT%\runtime\python.exe"
  goto run
)
if defined RVC_RUNTIME goto run
if exist "%~dp0runtime.local.txt" set /p "RVC_RUNTIME="<"%~dp0runtime.local.txt"
if not defined RVC_RUNTIME set "RVC_RUNTIME=%~dp0RVC\runtime\python.exe"
:run
if not exist "%RVC_RUNTIME%" (
  echo [错误] 未找到 RVC runtime: %RVC_RUNTIME%
  echo 请设置 RVC_ROOT，或在 runtime.local.txt 写入 runtime\python.exe 的完整路径。
  pause
  exit /b 1
)

"%RVC_RUNTIME%" -B -X utf8 "%~dp0main.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo [变声器] 进程退出码 %RC%
  pause
)
exit /b %RC%
