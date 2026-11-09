@echo off
rem Makes a cartoon episode from cartoon\episodes\<name>\screenplay.txt. Double-click it, or: make_cartoon.bat <name> [stages]
rem Safe to run again after any error or edit: finished work is kept and only what changed is redone.
set "EP=%~1"
if "%EP%"=="" set /p "EP=Episode folder name (in cartoon\episodes): "
if not exist "%~dp0cartoon\episodes\%EP%\screenplay.txt" (
    echo No screenplay at cartoon\episodes\%EP%\screenplay.txt. Make one with new_cartoon_episode.bat first.
    pause
    exit /b 1
)
set "STAGES=%~2 %~3 %~4 %~5 %~6"
if "%~2"=="" set "STAGES=all"
cd /d "%~dp0cartoon"
"%~dp0venv\Scripts\python.exe" make_cartoon.py %EP% %STAGES%
echo.
echo Finished files: cartoon\episodes\%EP%\final    Review sheets: cartoon\episodes\%EP%\qa
pause
