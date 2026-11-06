@echo off
rem ---------------------------------------------------------------------------
rem  FastSAM segmentation bench - one click launcher
rem  Usage: double click, or:  start.bat [port]
rem  NO_BROWSER=1 skips opening the browser (used by tests).
rem  Works from any folder: everything is resolved relative to this file.
rem  First time on a new machine? See the environment manifest in this folder.
rem ---------------------------------------------------------------------------
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"
set "URL=http://127.0.0.1:%PORT%/"

if not exist "%ROOT%\webapp\server.py" (
  echo [x] webapp\server.py not found under %ROOT%
  echo     keep this file in the FastSAM folder, next to the webapp directory
  pause
  exit /b 1
)

set "LAUNCH="
if exist "%ROOT%\.venv\Scripts\python.exe" set "LAUNCH="%ROOT%\.venv\Scripts\python.exe""
if defined LAUNCH goto havepy
where python >nul 2>&1 && set "LAUNCH=python"
if defined LAUNCH goto havepy
where py >nul 2>&1 && set "LAUNCH=py -3.12"
if defined LAUNCH goto havepy

echo [x] no python found - neither .venv nor python on PATH
echo     see the environment manifest for the one-time setup steps
pause
exit /b 1

:havepy
call :probe
if not errorlevel 1 (
  echo [=] server already running on port %PORT%
  call :open
  exit /b 0
)

echo [*] starting server on port %PORT% ...
echo     python: %LAUNCH%
start "FastSAM server" /min %LAUNCH% "%ROOT%\webapp\server.py" %PORT%

for /l %%i in (1,1,120) do (
  call :probe
  if not errorlevel 1 (
    echo [=] ready: %URL%
    call :open
    exit /b 0
  )
  ping -n 2 -w 1000 127.0.0.1 >nul 2>&1
)

echo [x] server did not answer on %URL%
echo     check the "FastSAM server" window for the error
echo     common causes: torch not installed, or setuptools 81 or newer
pause
exit /b 1

:open
if defined NO_BROWSER (
  echo     [NO_BROWSER] would open %URL%
  exit /b 0
)
start "" "%URL%"
exit /b 0

:probe
powershell -NoProfile -ExecutionPolicy Bypass -Command "try{ $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%PORT%/api/health'; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
exit /b %ERRORLEVEL%
