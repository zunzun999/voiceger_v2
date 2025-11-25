#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voiceger Minimal API Implementation
Provides TTS and VC functionality via FastAPI interface
"""
import os
import sys
import uuid
import time
from typing import Optional, List, Tuple
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import soundfile as sf

# ----------------------------
# Path Configuration (read from local project, no download)
# ----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBUI_DIR = os.path.dirname(CURRENT_DIR)
SOVITS_DIR = os.path.join(WEBUI_DIR, "GPT-SoVITS")

# RVC directory: located under GPT-SoVITS
RVC_DIR = os.path.join(SOVITS_DIR, "Retrieval-based-Voice-Conversion-WebUI")

# Default model weight paths
GPT_MODEL_PATH = os.path.join(SOVITS_DIR, 'GPT_weights_v2', 'zudamon_style_1-e15.ckpt')
SOVITS_MODEL_PATH = os.path.join(SOVITS_DIR, 'SoVITS_weights_v2', 'zudamon_style_1_e8_s96.pth')

# Set environment variables (before importing modules, use absolute paths)
os.environ["gpt_path"] = os.path.abspath(GPT_MODEL_PATH)
os.environ["sovits_path"] = os.path.abspath(SOVITS_MODEL_PATH)

# Pretrained model paths - in GPT_SoVITS subdirectory
PRETRAINED_DIR = os.path.join(SOVITS_DIR, "GPT_SoVITS", "pretrained_models")
CNHUBERT_DIR = os.path.join(PRETRAINED_DIR, "chinese-hubert-base")
BERT_DIR = os.path.join(PRETRAINED_DIR, "chinese-roberta-wwm-ext-large")

# Set bert and cnhubert paths (must use absolute paths)
# If local directory exists, use local path; otherwise use HuggingFace repo ID
if os.path.exists(BERT_DIR):
    os.environ["bert_path"] = os.path.abspath(BERT_DIR)
    print(f"✓ BERT model path set: {BERT_DIR}")
else:
    raise FileNotFoundError(f"BERT model directory not found: {BERT_DIR}")

if os.path.exists(CNHUBERT_DIR):
    os.environ["cnhubert_base_path"] = os.path.abspath(CNHUBERT_DIR)
else:
    raise FileNotFoundError(f"CNHubert model directory not found: {CNHUBERT_DIR}")

os.environ["is_half"] = "False"

# Set g2pw model path (for Chinese pinyin conversion)
G2PW_MODEL_DIR = os.path.join(SOVITS_DIR, "GPT_SoVITS", "text", "G2PWModel")
if os.path.exists(G2PW_MODEL_DIR):
    os.environ["G2PW_MODEL_DIR"] = os.path.abspath(G2PW_MODEL_DIR)
    print(f"✓ G2PW model path set: {G2PW_MODEL_DIR}")
else:
    raise FileNotFoundError(f"G2PW model directory not found: {G2PW_MODEL_DIR}")

# Set rmvpe model path (for RVC pitch extraction)
RMVPE_ROOT = os.path.join(RVC_DIR, "assets", "rmvpe")
if os.path.exists(RMVPE_ROOT):
    os.environ["rmvpe_root"] = os.path.abspath(RMVPE_ROOT)
    print(f"✓ RMVPE model path set: {RMVPE_ROOT}")
else:
    raise FileNotFoundError(f"RMVPE model directory not found: {RMVPE_ROOT}")

# Add ffmpeg path to PATH (in GPT-SoVITS directory)
ffmpeg_dir = SOVITS_DIR
if os.path.exists(os.path.join(ffmpeg_dir, "ffmpeg.exe")):
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"✓ FFmpeg path added: {ffmpeg_dir}")
else:
    raise FileNotFoundError(f"FFmpeg directory not found: {ffmpeg_dir}")
# Inject sys.path (following zundamon_webui.py approach)
sys.path.insert(0, WEBUI_DIR)
sys.path.insert(0, SOVITS_DIR)
sys.path.append(os.path.join(SOVITS_DIR, 'GPT_SoVITS'))
sys.path.insert(0, RVC_DIR)

# Save current directory and switch to SOVITS_DIR (for loading relative path models)
# chinese2.py's g2pw uses relative path "GPT_SoVITS/text/G2PWModel", needs to start from SOVITS_DIR
_original_cwd = os.getcwd()
os.chdir(SOVITS_DIR)

# Import GPT-SoVITS modules
from AR.modules.activation import MhaPatched, disable_mha_patch
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

# Restore original working directory
os.chdir(_original_cwd)

app = FastAPI(title="Voiceger Minimal API", version="0.1.0")

# ----------------------------
# Work Directory Configuration
# ----------------------------
def get_work_directory():
    """Get or create work directory"""
    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        work_dir = os.path.join(local_appdata, "voiceger")
    else:
        work_dir = os.path.join(os.getenv("TEMP", "."), "voiceger_temp")
    
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return work_dir

WORK_DIR = get_work_directory()
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
REFERENCE_DIR = os.path.join(WEBUI_DIR, "reference")
DEFAULT_REF_WAV = os.path.join(REFERENCE_DIR, "reference.wav")
DEFAULT_REF_TEXT = os.path.join(REFERENCE_DIR, "ref_text.txt")

def sanitize_filename(name: str, suffix: str = ".wav") -> str:
    """Clean invalid characters from filename"""
    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join(c for c in name if c not in invalid_chars)
    cleaned = cleaned.strip()[:64] or "output"
    return f"{cleaned}_{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"

# ----------------------------
# RVC Utility Functions
# ----------------------------
@contextmanager
def pushd(new_dir):
    """Temporarily switch working directory"""
    prev = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev)

def ensure_rvc_env():
    """Set RVC environment variables"""
    os.environ["weight_root"] = os.path.join(RVC_DIR, "assets", "weights")
    os.environ["index_root"] = os.path.join(RVC_DIR, "assets", "indices")
    os.environ["outside_index_root"] = os.path.join(RVC_DIR, "assets", "indices")

def ensure_hubert_asset():
    """Ensure HuBERT model exists"""
    hubert_path = os.path.join(RVC_DIR, "assets", "hubert", "hubert_base.pt")
    if not os.path.exists(hubert_path):
        raise FileNotFoundError(f"HuBERT model not found: {hubert_path}")

_vc_model = None  # Lazy initialization

def get_vc_model():
    """Get or initialize VC model (lazy loading)"""
    global _vc_model
    if _vc_model is None:
        ensure_rvc_env()
        ensure_hubert_asset()
        with pushd(RVC_DIR):
            from configs.config import Config
            from infer.modules.vc.modules import VC
            cfg = Config()
            _vc_model = VC(cfg)
    return _vc_model

def pick_default_sid():
    """Select default RVC model"""
    weights_dir = os.path.join(RVC_DIR, "assets", "weights")
    preferred = os.path.join(weights_dir, "train-0814-2.pth")
    if os.path.exists(preferred):
        return os.path.basename(preferred)
    
    if os.path.exists(weights_dir):
        files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
        if files:
            return sorted(files)[0]
    
    raise FileNotFoundError("No RVC model file found, please place model in assets/weights/ directory")

# ----------------------------
# Request Data Models
# ----------------------------
class TtsRequest(BaseModel):
    """TTS request model"""
    text: str
    text_language: str
    ref_wav_path: Optional[str] = None

class VcSingleRequest(BaseModel):
    """VC single file conversion request model"""
    input_audio_path: str
    f0_method: Optional[str] = "rmvpe"
    output_dir: Optional[str] = None
    sid: Optional[str] = None

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/")
def root():
    """API root path"""
    return {
        "name": "Voiceger API",
        "version": "0.1.0",
        "endpoints": ["/tts", "/vc/single"]
    }

@app.post("/tts")
def tts(req: TtsRequest):
    """
    TTS text-to-speech endpoint
    
    Args:
        text: Text to convert
        text_language: Text language (Chinese, English, Japanese, Korean, etc.)
        ref_wav_path: Optional reference audio path
    """
    # Check model files
    if not os.path.exists(GPT_MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"GPT model not found: {GPT_MODEL_PATH}")
    if not os.path.exists(SOVITS_MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"SoVITS model not found: {SOVITS_MODEL_PATH}")

    # Reference audio and text
    ref_wav = os.path.abspath(req.ref_wav_path) if req.ref_wav_path else DEFAULT_REF_WAV
    if not os.path.exists(ref_wav):
        raise HTTPException(status_code=400, detail=f"Reference audio not found: {ref_wav}")
    
    if not os.path.exists(DEFAULT_REF_TEXT):
        raise HTTPException(status_code=500, detail=f"Reference text not found: {DEFAULT_REF_TEXT}")
    
    with open(DEFAULT_REF_TEXT, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    prompt_language = "Japanese"  # Zundamon fine-tuned model uses Japanese reference

    try:
        # Switch to SOVITS_DIR to ensure g2pw can find local model
        with pushd(SOVITS_DIR):
            with MhaPatched():
                change_gpt_weights(gpt_path=GPT_MODEL_PATH)
                change_sovits_weights(sovits_path=SOVITS_MODEL_PATH)
                result_list = list(
                    get_tts_wav(
                        ref_wav_path=ref_wav,
                        prompt_text=prompt_text,
                        prompt_language=prompt_language,
                        text=req.text,
                        text_language=req.text_language,
                        top_p=1, 
                        temperature=1
                    )
                )
        
        if not result_list:
            raise RuntimeError("TTS returned empty result")

        sr, audio = result_list[-1]
        outfile = os.path.join(OUTPUT_DIR, sanitize_filename(req.text, ".wav"))
        sf.write(outfile, audio, sr)
        
        return JSONResponse(status_code=200, content={
            "message": "success",
            "file_path": outfile,
            "sampling_rate": sr
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

@app.post("/vc/single")
def vc_single(req: VcSingleRequest):
    """
    VC voice conversion endpoint
    
    Args:
        input_audio_path: Input audio file path
        f0_method: Pitch extraction method (rmvpe or crepe)
        output_dir: Optional output directory
        sid: Optional model name
    """
    input_path = os.path.abspath(req.input_audio_path)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"Input audio not found: {input_path}")

    try:
        # Disable MHA patch (following zundamon_webui.py)
        disable_mha_patch()
        
        # Set environment variables and ensure resources exist
        ensure_rvc_env()
        ensure_hubert_asset()
        
        with pushd(RVC_DIR):
            from infer.modules.vc.utils import get_index_path_from_model

            sid = req.sid or pick_default_sid()
            vc_model = get_vc_model()
            
            # Load model
            vc_model.get_vc(sid)

            # Get index file path
            try:
                file_index = get_index_path_from_model(sid)
                if file_index is None or not os.path.exists(str(file_index)):
                    file_index = ""
                    print(f"Warning: Index file not found, will not use retrieval index")
            except Exception as e:
                print(f"Warning: Failed to get index path: {e}, will not use retrieval index")
                file_index = ""
            
            # Call vc_single (following zundamon_webui.py)
            ret = vc_model.vc_single(
                sid=0,  # Use speaker ID=0
                input_audio_path=input_path,
                f0_up_key=0,
                f0_file=None,
                f0_method=req.f0_method,
                file_index=file_index,
                file_index2=file_index,
                index_rate=0.75,
                filter_radius=3,
                resample_sr=0,
                rms_mix_rate=0.25,
                protect=0.33,
            )
            
            # Compatible with 2/3 return values (following zundamon_webui.py)
            if isinstance(ret, tuple):
                if len(ret) == 3:
                    info, opt, download_path = ret
                elif len(ret) == 2:
                    info, opt = ret
                    download_path = None
                else:
                    info, opt, download_path = "", None, None
            else:
                info, opt, download_path = str(ret), None, None
            
            # Parse audio data
            if isinstance(opt, tuple):
                sr, audio = opt
            else:
                raise RuntimeError(f"VC failed: Invalid return format, info={info}")
            
            if sr is None or audio is None:
                raise RuntimeError(f"VC failed: {info}")

            out_dir = os.path.abspath(req.output_dir) if req.output_dir else OUTPUT_DIR
            os.makedirs(out_dir, exist_ok=True)
            
            input_basename = os.path.splitext(os.path.basename(input_path))[0]
            outfile = os.path.join(out_dir, sanitize_filename(input_basename, ".wav"))
            sf.write(outfile, audio, sr)
            
            return JSONResponse(status_code=200, content={
                "message": "success",
                "info": info,
                "file_path": outfile,
                "sampling_rate": sr,
                "sid": sid,
                "file_index": file_index
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VC failed: {e}")

# ----------------------------
# Startup Instructions
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting Voiceger API server...")
    print(f"Work directory: {WORK_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"GPT model: {GPT_MODEL_PATH}")
    print(f"SoVITS model: {SOVITS_MODEL_PATH}")
    print(f"RVC directory: {RVC_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=8000)