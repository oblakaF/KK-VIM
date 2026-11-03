@echo off
rem KK-VIM GUI functional preview (SSSS, N = 1): double-click to start.
rem The first start builds the application into dist\kkvim-gui-preview (scripts\windows\build_gui_preview.ps1,
rem versions from uv.lock); later starts open dist\kkvim-gui-preview\Scripts\kkvim-gui.exe directly.
rem ASCII only and CRLF line ends: cmd.exe misreads LF-only or UTF-8 batch files.
setlocal
set "ROOT=%~dp0"
set "EXE=%ROOT%dist\kkvim-gui-preview\Scripts\kkvim-gui.exe"
if exist "%EXE%" goto start
echo First start: building KK-VIM GUI into dist\kkvim-gui-preview ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\windows\build_gui_preview.ps1"
if errorlevel 1 goto failed
:start
start "" "%EXE%" %*
endlocal
exit /b 0
:failed
echo ERROR: the build failed. uv must be on PATH; see the messages above.
pause
endlocal
exit /b 1
