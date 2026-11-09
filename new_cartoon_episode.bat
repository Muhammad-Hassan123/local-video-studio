@echo off
rem Creates cartoon\episodes\<name>\screenplay.txt from the template and opens it in Notepad.
set "EP=%~1"
if "%EP%"=="" set /p "EP=New episode folder name (for example ep02_the_cookies): "
set "DIR=%~dp0cartoon\episodes\%EP%"
if exist "%DIR%\screenplay.txt" (
    echo %DIR%\screenplay.txt already exists. Opening it.
) else (
    mkdir "%DIR%" 2>nul
    copy "%~dp0cartoon\templates\story_v1.txt" "%DIR%\screenplay.txt" >nul
    echo Created %DIR%\screenplay.txt from the template.
)
start notepad "%DIR%\screenplay.txt"
