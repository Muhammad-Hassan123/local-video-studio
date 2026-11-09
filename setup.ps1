# One-time setup on a new Windows PC (NVIDIA GPU with 8 GB+ VRAM, Python 3.12, git).
#   powershell -ExecutionPolicy Bypass -File setup.ps1                 # venv + voice/caption models
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -ComfyUI D:\comfyui -DownloadImageModels
# ComfyUI itself is installed separately (https://github.com/comfyanonymous/ComfyUI, portable or git + venv).
param(
    [string]$ComfyUI = "",
    [switch]$DownloadImageModels
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if (-not (Test-Path "$root\venv\Scripts\python.exe")) {
    py -3.12 -m venv "$root\venv"
}
& "$root\venv\Scripts\python.exe" -m pip install --upgrade pip
& "$root\venv\Scripts\python.exe" -m pip install -r "$root\requirements.txt"

# Kokoro TTS (Apache-2.0) from the kokoro-onnx releases; faster-whisper fetches its own model on first use.
New-Item -ItemType Directory -Force "$root\models" | Out-Null
$kokoro = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
foreach ($f in "kokoro-v1.0.onnx", "voices-v1.0.bin") {
    if (-not (Test-Path "$root\models\$f")) {
        Write-Host "Downloading $f"
        Invoke-WebRequest "$kokoro/$f" -OutFile "$root\models\$f"
    }
}

# Models used through ComfyUI (about 20 GB in total): FLUX.2 klein for pictures, an anime upscaler, ACE-Step 1.5 for music.
$ace = "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/split_files"
$models = @(
    @("diffusion_models", "https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors"),
    @("text_encoders", "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors"),
    @("vae", "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"),
    @("upscale_models", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"),
    @("diffusion_models", "$ace/diffusion_models/acestep_v1.5_turbo.safetensors"),
    @("text_encoders", "$ace/text_encoders/qwen_0.6b_ace15.safetensors"),
    @("text_encoders", "$ace/text_encoders/qwen_1.7b_ace15.safetensors"),
    @("vae", "$ace/vae/ace_1.5_vae.safetensors")
)
if ($ComfyUI) {
    [Environment]::SetEnvironmentVariable("COMFYUI_DIR", $ComfyUI, "User")
    Write-Host "COMFYUI_DIR set to $ComfyUI (open a new terminal to pick it up)"
}
foreach ($m in $models) {
    $name = Split-Path $m[1] -Leaf
    if ($DownloadImageModels -and $ComfyUI) {
        $dest = Join-Path $ComfyUI "models\$($m[0])\$name"
        if (-not (Test-Path $dest)) {
            Write-Host "Downloading $name"
            Invoke-WebRequest $m[1] -OutFile $dest
        }
    } else {
        Write-Host "ComfyUI needs models\$($m[0])\$name  <-  $($m[1])"
    }
}
Write-Host "Done. Start ComfyUI (python main.py), then: venv\Scripts\python cartoon\make_cartoon.py <episode> all"
