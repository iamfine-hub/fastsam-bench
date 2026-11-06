@echo off
rem Builds a clean zip of this folder for handing to someone else.
rem Leaves out what must not travel: the local .venv (absolute paths are baked
rem in, and it is 1.7 GB), .git history, caches, screenshots, personal notes.
rem The recipient installs the environment per the manifest.
rem
rem Two steps on purpose:
rem   1. robocopy stages a filtered copy (robocopy handles the exclude lists)
rem   2. .NET ZipFile writes the archive - unlike tar.exe it marks non-ASCII
rem      file names as UTF-8, so the Chinese-named .bat files survive unzipping
rem      on a machine whose system locale is not Chinese.
rem Keep this file ASCII-only (cmd.exe reads .bat as GBK on Chinese Windows).
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "STAGE=%TEMP%\fastsam-dist"
set "ZIP=%ROOT%\..\FastSAM-bench.zip"

if not exist "%ROOT%\webapp\server.py" (
  echo [x] run this from inside the FastSAM folder
  pause
  exit /b 1
)

echo [*] staging a filtered copy ...
if exist "%STAGE%" rmdir /s /q "%STAGE%"
robocopy "%ROOT%" "%STAGE%" /E /NFL /NDL /NJH /NJS /NP /XD .venv .git .idea __pycache__ _shot .cache /XF fastsam_coco_preds.json env_setup.log *.zip
if errorlevel 8 goto failed

echo [*] compressing ...
if exist "%ZIP%" del /q "%ZIP%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; $ok = $false; for ($i = 0; $i -lt 3 -and -not $ok; $i++) { try { [System.IO.Compression.ZipFile]::CreateFromDirectory('%STAGE%', '%ZIP%', [System.IO.Compression.CompressionLevel]::Optimal, $false); $ok = $true } catch { Start-Sleep -Seconds 2 } }; if (-not $ok) { exit 1 }"
if errorlevel 1 goto failed
if not exist "%ZIP%" goto failed

rmdir /s /q "%STAGE%"
for %%A in ("%ZIP%") do set "SIZE=%%~zA"
powershell -NoProfile -Command "Write-Host ('[=] done: ' + '%ZIP%' + '  (' + [math]::Round(%SIZE%/1MB,1) + ' MB)')"
echo.
echo     send this zip. the recipient unzips it anywhere, installs the
echo     environment following the manifest, then double-clicks the launcher bat.
pause
exit /b 0

:failed
if exist "%STAGE%" rmdir /s /q "%STAGE%"
echo [x] packing failed - check free disk space and try again
pause
exit /b 1
