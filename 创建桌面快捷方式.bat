@echo off
rem Creates a desktop shortcut that points at this folder's start.bat.
rem The shortcut is generated from the current location, so moving the folder
rem and re-running this file is enough.
rem Keep this file ASCII-only (cmd.exe reads .bat as GBK on Chinese Windows).
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

if not exist "%ROOT%\start.bat" (
  echo [x] start.bat not found under %ROOT%
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$w = New-Object -ComObject WScript.Shell; $p = Join-Path ([Environment]::GetFolderPath('Desktop')) 'FastSAM Bench.lnk'; $s = $w.CreateShortcut($p); $s.TargetPath = Join-Path '%ROOT%' 'start.bat'; $s.WorkingDirectory = '%ROOT%'; $s.WindowStyle = 7; $s.Description = 'FastSAM segmentation bench'; $s.Save(); Write-Host ('[=] created: ' + $p)"
if errorlevel 1 (
  echo [x] failed to create the shortcut
  pause
  exit /b 1
)
echo     double-click it any time to open the bench
pause
