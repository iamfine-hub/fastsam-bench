@echo off
rem Double-click entry point. The real logic lives in start.bat next to this file.
rem Keep this file ASCII-only: cmd.exe reads .bat as the OEM code page (GBK on
rem Chinese Windows) and would garble any UTF-8 text written here.
call "%~dp0start.bat" %*
