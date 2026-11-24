@echo off
rem KK-VIM GUI: double-click to start (SSSS and CCCC, N = 1-6; normal mode, not the demonstration or the N = 1 preview).
rem Every start runs scripts\windows\build_gui.ps1 -IfStale: it compares the source fingerprint (src, pyproject.toml,
rem uv.lock, the build script) with the stamp of the installed application in dist\kkvim-gui and rebuilds it with uv
rem when it is missing or stale, so an obsolete build is never started. A failed rebuild keeps the previous installation
rem and starts nothing. Then dist\kkvim-gui\Scripts\kkvim-gui.exe (windowed, no console) is started with the arguments.
rem Optional for tests: KKVIM_GUI_TARGET (another installation folder), KKVIM_GUI_NO_PAUSE (no pause after an error).
rem ASCII only and CRLF line ends: cmd.exe misreads LF-only or UTF-8 batch files.
setlocal
set "ROOT=%~dp0"
set "TARGET=%ROOT%dist\kkvim-gui"
if defined KKVIM_GUI_TARGET set "TARGET=%KKVIM_GUI_TARGET%"
set "EXE=%TARGET%\Scripts\kkvim-gui.exe"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\windows\build_gui.ps1" -IfStale -Target "%TARGET%"
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" goto failed
if not exist "%EXE%" goto missing
start "" "%EXE%" %*
endlocal & exit /b 0
:missing
set "CODE=1"
echo ERROR: %EXE% does not exist after the build.
:failed
echo.
echo KK-VIM GUI was not started (code %CODE%). See the messages above.
echo The previous installation, if any, is unchanged; it is not started because it is not built from the current sources.
echo The first build and every rebuild need uv on PATH: https://docs.astral.sh/uv/
if not defined KKVIM_GUI_NO_PAUSE pause
endlocal & exit /b %CODE%
