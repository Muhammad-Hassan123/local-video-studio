"""Tiny ComfyUI API client + workflow builders shared by the stickman and cartoon pipelines.

Workflows mirror the official ComfyUI templates (v0.42):
  image_flux2_klein_image_edit_4b_distilled  -> keyframes (4 steps, cfg 1)
  video_wan2_2_14B_i2v (lightx2v branch)     -> animation (4 steps, 2 high + 2 low, cfg 1, shift 5)
Wan models are the QuantStack Q4_K_S GGUFs, loaded with ComfyUI-GGUF's UnetLoaderGGUF.
"""
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# on another PC set COMFYUI_DIR (the ComfyUI folder), and COMFYUI_URL if it listens elsewhere
COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFY_DIR = Path(os.environ.get("COMFYUI_DIR", Path.home() / "comfyui"))

KLEIN_UNET = "flux-2-klein-4b-fp8.safetensors"
KLEIN_TE = "qwen_3_4b_fp8_mixed.safetensors"   # ~24% faster edits than the bf16 file on 16 GB RAM, same look (A/B 2026-11-08)
KLEIN_VAE = "flux2-vae.safetensors"

WAN_HIGH = "Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf"
WAN_LOW = "Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf"
WAN_LORA_HIGH = "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
WAN_LORA_LOW = "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
WAN_TE = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = "wan_2.1_vae.safetensors"
RIFE_MODEL = "rife_v4.26.safetensors"
UPSCALE_MODEL = "realesr-animevideov3.pth"
ACE_UNET = "acestep_v1.5_turbo.safetensors"
ACE_TE = ("qwen_0.6b_ace15.safetensors", "qwen_1.7b_ace15.safetensors")
ACE_VAE = "ace_1.5_vae.safetensors"
# Official template negative prompt (Chinese, as Wan was trained with it)
WAN_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
                "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
                "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


# ---------------------------------------------------------------- API helpers

def _get(path):
    with urllib.request.urlopen(COMFY_URL + path) as r:
        return json.loads(r.read())


def queue(prompt):
    body = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(COMFY_URL + "/prompt", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI rejected prompt: {e.read().decode()[:2000]}") from None
    except urllib.error.URLError:
        raise SystemExit(f"ComfyUI is not running at {COMFY_URL}. Start it (start_comfyui.bat), wait for 'To see the GUI go to', "
                         "then run the same command again: finished work is kept.") from None


def free():
    """Ask ComfyUI to unload its models so another process can use the GPU."""
    req = urllib.request.Request(COMFY_URL + "/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req).read()


def wait(prompt_id, poll=2.0):
    """Block until the prompt finishes; return (history entry, seconds)."""
    t0 = time.time()
    while True:
        h = _get(f"/history/{prompt_id}")
        if prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "error":
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    raise RuntimeError(f"Execution error: {json.dumps(msgs)[:3000]}")
                return entry, time.time() - t0
        time.sleep(poll)


def fetch_outputs(entry, dest_dir, prefix):
    """Copy every output file of a finished prompt into dest_dir; return paths."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for node_out in entry.get("outputs", {}).values():
        for key in ("images", "videos", "gifs", "animated", "audio"):
            for f in node_out.get(key, []) or []:
                if not isinstance(f, dict) or "filename" not in f:
                    continue
                q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
                with urllib.request.urlopen(f"{COMFY_URL}/view?{q}") as r:
                    out = dest_dir / f"{prefix}{Path(f['filename']).suffix}"
                    n = 1
                    while out.exists():
                        out = dest_dir / f"{prefix}_{n}{Path(f['filename']).suffix}"
                        n += 1
                    out.write_bytes(r.read())
                    saved.append(out)
    return saved


def to_input(path):
    """Copy a local image into ComfyUI's input folder so LoadImage can read it; return its name."""
    path = Path(path)
    name = f"stk_{path.name}"
    shutil.copy(path, COMFY_DIR / "input" / name)
    return name


def run(prompt, dest_dir, prefix):
    pid = queue(prompt)
    entry, secs = wait(pid)
    return fetch_outputs(entry, dest_dir, prefix), secs


# ---------------------------------------------------------------- workflows

def klein_image(prompt_text, width=1280, height=720, seed=0, refs=(), steps=4, filename_prefix="stickman/key"):
    """FLUX.2 klein 4B distilled. refs = ComfyUI input-image names used as reference images."""
    p = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": KLEIN_UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": KLEIN_TE, "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN_VAE}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["clip", 0]}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sched": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    }
    pos, neg = ["pos", 0], ["neg", 0]
    for i, ref in enumerate(refs):
        p[f"ref_load{i}"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        p[f"ref_scale{i}"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"upscale_method": "nearest-exact", "megapixels": 1, "resolution_steps": 1, "image": [f"ref_load{i}", 0]}}
        p[f"ref_enc{i}"] = {"class_type": "VAEEncode", "inputs": {"pixels": [f"ref_scale{i}", 0], "vae": ["vae", 0]}}
        p[f"ref_pos{i}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [f"ref_enc{i}", 0]}}
        p[f"ref_neg{i}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [f"ref_enc{i}", 0]}}
        pos, neg = [f"ref_pos{i}", 0], [f"ref_neg{i}", 0]
    p["guider"] = {"class_type": "CFGGuider", "inputs": {"cfg": 1, "model": ["unet", 0], "positive": pos, "negative": neg}}
    p["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sched", 0], "latent_image": ["latent", 0]}}
    p["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    p["save"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["decode", 0]}}
    return p


def wan_i2v(start_image, prompt_text, width=832, height=480, length=81, seed=0, interp=3, filename_prefix="stickman/clip"):
    """Wan 2.2 I2V A14B GGUF + lightx2v 4-step LoRAs (2 high-noise + 2 low-noise steps, cfg 1).
    length must be 4n+1 frames at Wan's native 16 fps; RIFE then multiplies the frame rate by `interp` (16 -> 48)."""
    return {
        "img": {"class_type": "LoadImage", "inputs": {"image": start_image}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": WAN_TE, "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": WAN_NEGATIVE, "clip": ["clip", 0]}},
        "high": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": WAN_HIGH}},
        "low": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": WAN_LOW}},
        "high_lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": WAN_LORA_HIGH, "strength_model": 1.0, "model": ["high", 0]}},
        "low_lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": WAN_LORA_LOW, "strength_model": 1.0, "model": ["low", 0]}},
        "high_shift": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 5.0, "model": ["high_lora", 0]}},
        "low_shift": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 5.0, "model": ["low_lora", 0]}},
        "i2v": {"class_type": "WanImageToVideo", "inputs": {"width": width, "height": height, "length": length, "batch_size": 1,
                                                             "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "start_image": ["img", 0]}},
        "ks_high": {"class_type": "KSamplerAdvanced", "inputs": {"add_noise": "enable", "noise_seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                                                                 "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable",
                                                                 "model": ["high_shift", 0], "positive": ["i2v", 0], "negative": ["i2v", 1], "latent_image": ["i2v", 2]}},
        "ks_low": {"class_type": "KSamplerAdvanced", "inputs": {"add_noise": "disable", "noise_seed": 0, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                                                                "start_at_step": 2, "end_at_step": 4, "return_with_leftover_noise": "disable",
                                                                "model": ["low_shift", 0], "positive": ["i2v", 0], "negative": ["i2v", 1], "latent_image": ["ks_high", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["ks_low", 0], "vae": ["vae", 0]}},
        "rife": {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": RIFE_MODEL}},
        "interp": {"class_type": "FrameInterpolate", "inputs": {"interp_model": ["rife", 0], "images": ["decode", 0], "multiplier": interp}},
        "video": {"class_type": "CreateVideo", "inputs": {"fps": 16 * interp, "bit_depth": 8, "color_space": "sRGB", "codec": "none", "images": ["interp", 0]}},
        "save": {"class_type": "SaveVideo", "inputs": {"filename_prefix": filename_prefix, "format": "auto", "format.codec": "auto", "codec": "auto", "video": ["video", 0]}},
    }


def upscale(image, scale=0.5, model=UPSCALE_MODEL, filename_prefix="up"):
    """Anime upscaler (4x) then resize by `scale`: 0.5 gives a clean 2x of a flat cartoon image."""
    return {
        "load": {"class_type": "LoadImage", "inputs": {"image": image}},
        "model": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model}},
        "up": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["model", 0], "image": ["load", 0]}},
        "down": {"class_type": "ImageScaleBy", "inputs": {"upscale_method": "lanczos", "scale_by": scale, "image": ["up", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["down", 0]}},
    }


def ace_music(tags, seconds=120.0, bpm=100, key="C major", seed=0, filename_prefix="music"):
    """ACE-Step 1.5 turbo, as in the official audio_ace_step_1_5_split template (8 steps, cfg 1, shift 3).
    Instrumental: the lyrics are just "[Instrumental]"."""
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": ACE_UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": ACE_TE[0], "clip_name2": ACE_TE[1], "type": "ace", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": ACE_VAE}},
        "pos": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
            "clip": ["clip", 0], "tags": tags, "lyrics": "[Instrumental]", "seed": seed, "bpm": bpm, "duration": seconds,
            "timesignature": "4", "language": "en", "keyscale": key, "generate_audio_codes": True, "cfg_scale": 2.0,
            "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "latent": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": seconds, "batch_size": 1}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"shift": 3.0, "model": ["unet", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["shift", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                                                       "scheduler": "simple", "positive": ["pos", 0], "negative": ["neg", 0],
                                                       "latent_image": ["latent", 0], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveAudio", "inputs": {"audio": ["decode", 0], "filename_prefix": filename_prefix}},
    }


# ---------------------------------------------------------------- shot animation (Wan 2.1 I2V 480p + InfiniteTalk)
# One base model for every shot, so it stays loaded: plain image-to-video for action, InfiniteTalk for lip-synced dialogue.
# native fp8 weights, not GGUF: ComfyUI can stream them from disk, so 16 GB of RAM never spills into the pagefile
WAN21_UNET = "wan2.1_i2v_480p_14B_fp8_scaled.safetensors"
WAN21_LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
WAN21_VAE = "wan_2.1_vae.safetensors"
TALK_PATCH = "wan2.1_infiniteTalk_multi_fp16.safetensors"
TALK_AUDIO_ENCODER = "wav2vec2-chinese-base_fp16.safetensors"
TALK_FPS, TALK_CHUNK, TALK_MOTION = 25, 81, 9


def _wan21_base(prompt_text):
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": WAN21_UNET, "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": WAN21_LORA, "strength_model": 1.0, "model": ["unet", 0]}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": WAN_TE, "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": WAN21_VAE}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["clip", 0]}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
    }


def wan21_i2v(start_image, prompt_text, width=848, height=480, length=81, seed=0, interp=3, filename_prefix="cartoon/clip"):
    """Wan 2.1 I2V 480p (Q4 GGUF) + lightx2v 4-step distill LoRA, then RIFE to 16 * interp fps."""
    p = _wan21_base(prompt_text)
    p["img"] = {"class_type": "LoadImage", "inputs": {"image": start_image}}
    p.update({
        "shift": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 5.0, "model": ["lora", 0]}},
        "i2v": {"class_type": "WanImageToVideo", "inputs": {"width": width, "height": height, "length": length, "batch_size": 1,
                                                             "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "start_image": ["img", 0]}},
        "ks": {"class_type": "KSampler", "inputs": {"model": ["shift", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler",
                                                   "scheduler": "simple", "positive": ["i2v", 0], "negative": ["i2v", 1], "latent_image": ["i2v", 2], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "rife": {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": RIFE_MODEL}},
        "interp": {"class_type": "FrameInterpolate", "inputs": {"interp_model": ["rife", 0], "images": ["decode", 0], "multiplier": interp}},
        "video": {"class_type": "CreateVideo", "inputs": {"fps": 16 * interp, "bit_depth": 8, "color_space": "sRGB", "codec": "none", "images": ["interp", 0]}},
        "save": {"class_type": "SaveVideo", "inputs": {"filename_prefix": filename_prefix, "format": "auto", "format.codec": "auto", "codec": "auto", "video": ["video", 0]}},
    })
    return p


def talk_chunks(frames):
    """InfiniteTalk chunk lengths for a clip of `frames`: up to 81 frames each (4n+1), every chunk after the first re-using
    the last 9 frames of the one before, and the last chunk only as long as it needs to be."""
    sizes, made = [], 0
    while made < frames:
        need = frames - made + (TALK_MOTION if sizes else 0)
        sizes.append(min(TALK_CHUNK, 4 * -(-max(need, 17) // 4) + 1))
        made += sizes[-1] - (TALK_MOTION if len(sizes) > 1 else 0)
    return sizes


def infinitetalk(start_image, audio, prompt_text, frames, width=848, height=480, seed=0, masks=None, silence=None, filename_prefix="cartoon/talk"):
    """InfiniteTalk lip-sync at 25 fps, as in the official video_wan2_1_infinitetalk template (6 steps, cfg 1), chained in
    81-frame chunks that each continue from the last 9 frames until `frames` are made.
    masks = (speaker mask, others mask) input-image names with `silence` (a short silent clip) as the second voice, so only
    the speaker's mouth moves when other characters are in the frame."""
    p = _wan21_base(prompt_text)
    p["img"] = {"class_type": "LoadImage", "inputs": {"image": start_image}}
    p.update({
        "patch": {"class_type": "ModelPatchLoader", "inputs": {"name": TALK_PATCH}},
        "aenc": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": TALK_AUDIO_ENCODER}},
        "audio": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "a1": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["aenc", 0], "audio": ["audio", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    })
    mode = {"mode": "single_speaker"}
    if masks:
        p["silence"] = {"class_type": "LoadAudio", "inputs": {"audio": silence}}
        p["a2"] = {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["aenc", 0], "audio": ["silence", 0]}}
        p["m1"] = {"class_type": "LoadImageMask", "inputs": {"image": masks[0], "channel": "red"}}
        p["m2"] = {"class_type": "LoadImageMask", "inputs": {"image": masks[1], "channel": "red"}}
        mode = {"mode": "two_speakers", "mode.audio_encoder_output_2": ["a2", 0], "mode.mask_1": ["m1", 0], "mode.mask_2": ["m2", 0]}
    batch = None
    for i, length in enumerate(talk_chunks(frames)):
        it = {"model": ["lora", 0], "model_patch": ["patch", 0], "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
              "width": width, "height": height, "length": length, "start_image": ["img", 0], "audio_encoder_output_1": ["a1", 0],
              "motion_frame_count": TALK_MOTION, "audio_scale": 1.0, **mode}
        if batch:
            it["previous_frames"] = batch
        p[f"it{i}"] = {"class_type": "WanInfiniteTalkToVideo", "inputs": it}
        p[f"noise{i}"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + i}}
        p[f"guider{i}"] = {"class_type": "CFGGuider", "inputs": {"cfg": 1.0, "model": [f"it{i}", 0], "positive": [f"it{i}", 1], "negative": [f"it{i}", 2]}}
        p[f"sched{i}"] = {"class_type": "BasicScheduler", "inputs": {"scheduler": "normal", "steps": 6, "denoise": 1.0, "model": [f"it{i}", 0]}}
        p[f"sample{i}"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [f"noise{i}", 0], "guider": [f"guider{i}", 0], "sampler": ["sampler", 0],
                                                                            "sigmas": [f"sched{i}", 0], "latent_image": [f"it{i}", 3]}}
        p[f"decode{i}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"sample{i}", 0], "vae": ["vae", 0]}}
        if batch is None:
            batch = [f"decode{i}", 0]
        else:  # drop the frames that overlap the previous chunk, then append
            p[f"trim{i}"] = {"class_type": "ImageFromBatch", "inputs": {"image": [f"decode{i}", 0], "batch_index": [f"it{i}", 4], "length": 4096}}
            p[f"join{i}"] = {"class_type": "ImageBatch", "inputs": {"image1": batch, "image2": [f"trim{i}", 0]}}
            batch = [f"join{i}", 0]
    p["video"] = {"class_type": "CreateVideo", "inputs": {"fps": TALK_FPS, "bit_depth": 8, "color_space": "sRGB", "codec": "none", "images": batch}}
    p["save"] = {"class_type": "SaveVideo", "inputs": {"filename_prefix": filename_prefix, "format": "auto", "format.codec": "auto", "codec": "auto", "video": ["video", 0]}}
    return p
