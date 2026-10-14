@echo off
rem One-click development loop on Windows: pull, build viproc into .venv, test.
rem Double-click it or run "dev.cmd [-NoPull] [-NoTest] [-Python <exe>]".
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\dev.ps1" %*
set VIPROC_DEV_EXIT=%ERRORLEVEL%
rem Opened by double-click (cmd /c "...dev.cmd"): keep the window open.
echo %CMDCMDLINE% | find /i "/c" >nul && pause
exit /b %VIPROC_DEV_EXIT%
