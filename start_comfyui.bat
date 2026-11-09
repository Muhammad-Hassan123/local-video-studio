@echo off
rem Starts ComfyUI, the engine that draws characters, locations and music. Keep this window open while making episodes.
if "%COMFYUI_DIR%"=="" set "COMFYUI_DIR=%USERPROFILE%\comfyui"
set "PY=%COMFYUI_DIR%\venv\Scripts\python.exe"
if exist "%COMFYUI_DIR%\python_embeded\python.exe" set "PY=%COMFYUI_DIR%\python_embeded\python.exe"
set "APP=%COMFYUI_DIR%"
if exist "%COMFYUI_DIR%\ComfyUI\main.py" set "APP=%COMFYUI_DIR%\ComfyUI"
if not exist "%APP%\main.py" (
    echo ComfyUI was not found in %COMFYUI_DIR%. Run setup.ps1 with -ComfyUI ^<its folder^>.
    pause
    exit /b 1
)
set TRITON_DEFAULT_FP_FUSION=1
set SAGE=
"%PY%" -c "import sageattention" 2>nul && set SAGE=--use-sage-attention
cd /d "%APP%"
"%PY%" main.py --listen 127.0.0.1 --port 8188 %SAGE%
pause
