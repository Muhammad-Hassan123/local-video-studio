@echo off
rem Makes a stickman episode from stickman\episodes\<name>\screenplay.txt. Double-click it, or: make_stickman.bat <name> [stages]
set "EP=%~1"
if "%EP%"=="" set /p "EP=Episode folder name (in stickman\episodes): "
if not exist "%~dp0stickman\episodes\%EP%\screenplay.txt" (
    echo No screenplay at stickman\episodes\%EP%\screenplay.txt. Copy stickman\templates\whatif_v2.txt there first.
    pause
    exit /b 1
)
set "STAGES=%~2 %~3 %~4 %~5 %~6"
if "%~2"=="" set "STAGES=all"
cd /d "%~dp0stickman"
"%~dp0venv\Scripts\python.exe" make_video.py %EP% %STAGES%
echo.
echo Finished files: stickman\episodes\%EP%\final    Review sheets: stickman\episodes\%EP%\qa
pause
