# Module top-level
# Polyfill: ensure builtins.help exists across the whole process (even in frozen/non-interactive environments)
import builtins  # noqa: F401

try:
    getattr(builtins, "help")  # if not injected by site/pydoc, this raises AttributeError
except Exception:
    try:
        from pydoc import help as _pydoc_help  # noqa: F401
        builtins.help = _pydoc_help  # make it globally available to all modules
    except Exception:
        def _noop_help(*args, **kwargs):
            return None
        builtins.help = _noop_help

# ===== MeCab/EUNJEON DLL 路径配置（用于韩语支持）=====
# 必须在任何可能导入 eunjeon 的代码之前执行
import sys
import os

def _configure_mecab_dll():
    """配置 MeCab DLL 搜索路径，确保 eunjeon 可以找到 libmecab.dll"""
    try:
        if getattr(sys, 'frozen', False):
            # PyInstaller 打包环境
            base_dir = sys._MEIPASS
            mecab_data_dir = os.path.join(base_dir, 'eunjeon', 'data')
            
            if os.path.exists(mecab_data_dir):
                # 添加 DLL 搜索路径（Python 3.8+）
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(mecab_data_dir)
                
                # 添加到 PATH（兼容性）
                os.environ['PATH'] = mecab_data_dir + os.pathsep + os.environ.get('PATH', '')
                
                # 确保 eunjeon.data 是一个包（需要 __init__.py）
                init_file = os.path.join(mecab_data_dir, '__init__.py')
                if not os.path.exists(init_file):
                    try:
                        with open(init_file, 'w', encoding='utf-8') as f:
                            f.write('# Auto-generated for eunjeon.data package\n')
                    except:
                        pass
                
                print(f"[MeCab] DLL path configured: {mecab_data_dir}")
                return True
        return False
    except Exception as e:
        print(f"[MeCab] Configuration error: {e}")
        return False

_configure_mecab_dll()
# ===== MeCab 配置结束 =====

import streamlit as st
import tempfile
import os
import soundfile as sf
import tkinter as tk
from tkinter import filedialog
import threading
from pathlib import Path

import time
import uuid
import shutil
from contextlib import contextmanager
from datetime import datetime

import sys
import re
import base64
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# 设置环境变量和编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 重新配置标准输出编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)
sys.path.insert(0, current_dir)

sys.path.append(os.path.join(current_dir, 'GPT_SoVITS'))

from AR.modules.activation import MhaPatched, disable_mha_patch

# Import your inference functions and required packages (adjust import paths as needed)
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

# ====== RVC Voice Conversion Related Imports ======
# Add RVC path
rvc_dir = os.path.join(current_dir, 'Retrieval-based-Voice-Conversion-WebUI')
sys.path.insert(0, rvc_dir)

# Try to import RVC related modules
RVC_AVAILABLE = False
vc_model = None
RVC_ERROR_MESSAGE = ""
RVC_MODEL_PATH = None
RVC_INDEX_PATH = None
RVC_INDEX_SELECTION_MESSAGE = ""
MISSING_DEPENDENCIES = []

# 从 session_state 恢复 RVC 状态，避免重跑后变量回到默认值
if "RVC_AVAILABLE" in st.session_state:
    RVC_AVAILABLE = st.session_state.RVC_AVAILABLE
    vc_model = st.session_state.get("vc_model")
    RVC_ERROR_MESSAGE = st.session_state.get("RVC_ERROR_MESSAGE", "")
    RVC_MODEL_PATH = st.session_state.get("RVC_MODEL_PATH")
    RVC_INDEX_PATH = st.session_state.get("RVC_INDEX_PATH", "")
    RVC_INDEX_SELECTION_MESSAGE = st.session_state.get("RVC_INDEX_SELECTION_MESSAGE", "")

def ensure_rvc_environment():
    # 加载子模块内的 .env（若可用）
    if load_dotenv:
        try:
            load_dotenv(os.path.join(rvc_dir, ".env"))
            load_dotenv(os.path.join(rvc_dir, "sha256.env"))
        except Exception:
            pass

    # 强制设置到子模块内的 assets 路径
    os.environ["rmvpe_root"] = os.path.join(rvc_dir, "assets", "rmvpe")
    os.environ["weight_root"] = os.path.join(rvc_dir, "assets", "weights")
    os.environ["index_root"] = os.path.join(rvc_dir, "assets", "indices")
    os.environ["outside_index_root"] = os.path.join(rvc_dir, "assets", "indices")

    # 可选调试输出
    try:
        import streamlit as st
        st.debug(f"RVC dir = {rvc_dir}")
        st.debug(f"rmvpe_root = {os.environ.get('rmvpe_root')}")
        st.debug(f"weight_root = {os.environ.get('weight_root')}")
        st.debug(f"index_root = {os.environ.get('index_root')}")
        st.debug(f"outside_index_root = {os.environ.get('outside_index_root')}")
    except Exception:
        pass

    return rvc_dir

_ = ensure_rvc_environment()

# Ensure HuBERT model asset exists (prefer submodule path; copy from external project if necessary)
def ensure_hubert_asset():
    hubert_path = os.path.join(rvc_dir, "assets", "hubert", "hubert_base.pt")
    if not os.path.exists(hubert_path):
        # Try copying from external projects root RVC project
        external_rvc_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'Retrieval-based-Voice-Conversion-WebUI')
        external_hubert = os.path.join(external_rvc_dir, "assets", "hubert", "hubert_base.pt")
        if os.path.exists(external_hubert):
            os.makedirs(os.path.dirname(hubert_path), exist_ok=True)
            shutil.copy2(external_hubert, hubert_path)
            print(f"[DEBUG] Copied HuBERT model from external: {external_hubert} -> {hubert_path}")
        else:
            # If external also does not exist, prompt user to download or place the file
            raise FileNotFoundError(f"HuBERT model not found: {hubert_path}. Please place hubert_base.pt under {os.path.join(rvc_dir, 'assets', 'hubert')}.")

# A safe directory switch context manager to ensure RVC's relative paths are usable
@contextmanager
def pushd(new_dir):
    prev = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev)

# Ensure HuBERT file is in place before proceeding
try:
    ensure_hubert_asset()
except Exception as e:
    # 改为不在导入阶段中断程序启动：禁用 VC，并在页面中显示详细错误信息
    RVC_ERROR_MESSAGE = f"HuBERT asset missing: {e}"
    RVC_AVAILABLE = False
    st.session_state.RVC_AVAILABLE = False
    st.session_state.RVC_ERROR_MESSAGE = RVC_ERROR_MESSAGE


if "rvc_inited" not in st.session_state:
    try:
        # Check if RVC directory exists
        if not os.path.exists(rvc_dir):
            raise ImportError(f"RVC directory does not exist: {rvc_dir}")
        
        # Check if necessary files exist
        config_path = os.path.join(rvc_dir, 'configs', 'config.py')
        if not os.path.exists(config_path):
            raise ImportError("RVC config file does not exist")
        
        
        # Check critical dependencies with detailed error reporting
        # ✅ 修改：移除 pip 子进程调用，改用纯 importlib 检查，避免弹终端窗口
        import importlib.util

        def check_package_installation(package_name, import_name=None):
            """Check if a package is importable without spawning pip subprocess"""
            if import_name is None:
                import_name = package_name
            try:
                return importlib.util.find_spec(import_name) is not None
            except Exception:
                return False

        MISSING_DEPENDENCIES.clear()
        dependencies = [
            ('faiss-cpu', 'faiss'),
            ('librosa', 'librosa'),
            ('scikit-learn', 'sklearn'),
            ('pyworld', 'pyworld')
        ]
        
        for package_name, import_name in dependencies:
            if not check_package_installation(package_name, import_name):
                MISSING_DEPENDENCIES.append(package_name)
        
        if MISSING_DEPENDENCIES:
            error_msg = f"Missing required dependencies: {', '.join(MISSING_DEPENDENCIES)}"
            # ✅ 修改：不再调用 'pip list' 子进程；直接抛出错误，在页面中指导用户安装
            raise ImportError(error_msg)
        
        # Initialize RVC model core (lightweight; weights are loaded lazily on button click)
        original_cwd = os.getcwd()
        try:
            os.chdir(rvc_dir)
            from configs.config import Config
            from infer.modules.vc.modules import VC
            rvc_config = Config()
            vc_model = VC(rvc_config)
        finally:
            os.chdir(original_cwd)
        
        import numpy as np
        
        weights_dir = os.path.join(rvc_dir, 'assets', 'weights')
        indices_dir = os.path.join(rvc_dir, 'assets', 'indices')
        
        available_models = []
        available_indices = []
        
        if os.path.exists(weights_dir):
            for file in os.listdir(weights_dir):
                if file.endswith('.pth'):
                    available_models.append(file)
        
        if os.path.exists(indices_dir):
            for file in os.listdir(indices_dir):
                if file.endswith('.index'):
                    available_indices.append(file)
        
        preferred_model_name = 'train-0814-2.pth'
        preferred_model_path = os.path.join(weights_dir, preferred_model_name)

        if os.path.exists(preferred_model_path):
            RVC_MODEL_PATH = preferred_model_path
        else:
            if not available_models:
                raise ImportError("No RVC model files (.pth) found. Please place trained model files in assets/weights/ directory.")
            RVC_MODEL_PATH = os.path.join(weights_dir, available_models[0])

        DESIRED_V2_INDEX_NAME = 'train-0814-2_IVF256_Flat_nprobe_1_train-0814-2_v2.index'
        if DESIRED_V2_INDEX_NAME in available_indices:
            RVC_INDEX_PATH = os.path.join(indices_dir, DESIRED_V2_INDEX_NAME)
            RVC_INDEX_SELECTION_MESSAGE = f"Using v2 index: {RVC_INDEX_PATH}"
        else:
            if available_indices:
                fallback_name = available_indices[0]
                RVC_INDEX_PATH = os.path.join(indices_dir, fallback_name)
                RVC_INDEX_SELECTION_MESSAGE = (
                    f"v2 index '{DESIRED_V2_INDEX_NAME}' not found. Falling back to: {RVC_INDEX_PATH}"
                )
            else:
                RVC_INDEX_PATH = ""
                RVC_INDEX_SELECTION_MESSAGE = (
                    "No index files found. Conversion will run without retrieval index."
                )
        
        RVC_AVAILABLE = True

        # 同步到 session_state，供后续页面重跑恢复使用
        st.session_state.RVC_AVAILABLE = RVC_AVAILABLE
        st.session_state.vc_model = vc_model
        st.session_state.RVC_ERROR_MESSAGE = RVC_ERROR_MESSAGE
        st.session_state.RVC_MODEL_PATH = RVC_MODEL_PATH
        st.session_state.RVC_INDEX_PATH = RVC_INDEX_PATH
        st.session_state.RVC_INDEX_SELECTION_MESSAGE = RVC_INDEX_SELECTION_MESSAGE

        st.session_state.rvc_inited = True
    
    except ImportError as e:
        RVC_AVAILABLE = False
        vc_model = None
        RVC_ERROR_MESSAGE = str(e)
        # 同步到 session_state
        st.session_state.RVC_AVAILABLE = False
        st.session_state.vc_model = None
        st.session_state.RVC_ERROR_MESSAGE = RVC_ERROR_MESSAGE
    except Exception as e:
        RVC_AVAILABLE = False
        vc_model = None
        RVC_ERROR_MESSAGE = f"RVC initialization failed: {str(e)}"
        # 同步到 session_state
        st.session_state.RVC_AVAILABLE = False
        st.session_state.vc_model = None
        st.session_state.RVC_ERROR_MESSAGE = RVC_ERROR_MESSAGE
# ====== 背景图片设置 ======
def set_background_image():
    """设置背景图片"""
    background_image_path = os.path.join(current_dir, 'imgs', 'zundamon_img.png')
    
    if os.path.exists(background_image_path):
        try:
            # 读取背景图片并转换为base64
            with open(background_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            
            # 创建CSS样式
            background_css = f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: 400px auto;
                background-repeat: no-repeat;
                background-position: center center;
                background-attachment: fixed;
                background-color: rgba(255, 255, 255, 0.6);
                background-blend-mode: overlay;
            }}
            
            /* 确保内容区域有足够的不透明度 */
            .main .block-container {{
                background-color: rgba(255, 255, 255, 0.6);
                border-radius: 10px;
                padding: 2rem;
                margin-top: 1rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            
            /* 调整侧边栏背景 */
            .sidebar .sidebar-content {{
                background-color: rgba(255, 255, 255, 0.6);
            }}
            
            /* 确保标题区域可读性 */
            h1, h2, h3, h4, h5, h6 {{
                background-color: rgba(255, 255, 255, 0.75);
                padding: 0.5rem;
                border-radius: 5px;
                margin-bottom: 1rem;
            }}
            
            /* 调整输入框和按钮的可读性 */
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea,
            .stSelectbox > div > div > select {{
                background-color: rgba(255, 255, 255, 0.75) !important;
            }}
            
            /* 信息框样式调整 */
            .stSuccess, .stInfo, .stWarning, .stError {{
                background-color: rgba(255, 255, 255, 0.6) !important;
            }}
            </style>
            """
            
            st.markdown(background_css, unsafe_allow_html=True)
            print(f"Background image loaded successfully from: {background_image_path}")
            
        except Exception as e:
            print(f"Error loading background image: {e}")
            # 如果图片加载失败，使用纯色背景
            fallback_css = """
            <style>
            .stApp {
                background-color: #f0f8ff;
            }
            </style>
            """
            st.markdown(fallback_css, unsafe_allow_html=True)
    else:
        print(f"Background image not found at: {background_image_path}")
        # 使用默认的浅色背景
        fallback_css = """
        <style>
        .stApp {
            background-color: #f0f8ff;
        }
        </style>
        """
        st.markdown(fallback_css, unsafe_allow_html=True)

# ====== 作业文件夹设置 ======
def get_work_directory():
    """获取或创建用户作业文件夹"""
    # 使用 AppData\Local\voiceger 作为作业文件夹
    work_dir = os.path.join(os.getenv('LOCALAPPDATA'), 'voiceger')
    
    # 创建必要的子目录
    subdirs = ['temp', 'output', 'cache']
    
    try:
        # 创建主目录
        os.makedirs(work_dir, exist_ok=True)
        print(f"Work directory created/verified: {work_dir}")
        
        # 创建子目录
        for subdir in subdirs:
            subdir_path = os.path.join(work_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            print(f"Subdirectory created/verified: {subdir_path}")
            
        return work_dir
    except Exception as e:
        print(f"Error creating work directory: {e}")
        # 如果创建失败，使用系统临时目录作为备选
        fallback_dir = os.path.join(tempfile.gettempdir(), 'voiceger')
        os.makedirs(fallback_dir, exist_ok=True)
        print(f"Using fallback directory: {fallback_dir}")
        return fallback_dir

# 初始化作业目录
WORK_DIR = get_work_directory()
TEMP_DIR = os.path.join(WORK_DIR, 'temp')
OUTPUT_DIR = os.path.join(WORK_DIR, 'output')
CACHE_DIR = os.path.join(WORK_DIR, 'cache')

# 模型文件路径保持从安装目录读取（只读）
GPT_MODEL_PATH = os.path.join(current_dir, 'GPT_weights_v2', 'zudamon_style_1-e15.ckpt')
SOVITS_MODEL_PATH = os.path.join(current_dir, 'SoVITS_weights_v2', 'zudamon_style_1_e8_s96.pth')

@contextmanager
def safe_temp_file(suffix="", content="", encoding='utf-8'):
    """安全的临时文件管理器 - 使用作业目录"""
    temp_path = None
    try:
        # 在作业目录的temp子目录中创建临时文件
        fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=TEMP_DIR)
        os.close(fd)  # 关闭文件描述符
        
        # 写入内容
        if content:
            with open(temp_path, 'w', encoding=encoding) as f:
                f.write(content)
        
        yield temp_path
        
    finally:
        # 确保文件被删除
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Cleaned temp file: {temp_path}")
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_path}: {e}")

@contextmanager
def safe_temp_dir():
    """安全的临时目录管理器 - 使用作业目录"""
    temp_dir = None
    try:
        # 在作业目录的temp子目录中创建临时目录
        temp_dir = tempfile.mkdtemp(prefix='voiceger_', dir=TEMP_DIR)
        yield temp_dir
    finally:
        # 确保目录被清理
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned temp directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not remove temp dir {temp_dir}: {e}")

def clean_old_files():
    """清理旧的临时文件和输出文件"""
    try:
        current_time = time.time()
        # 清理超过1小时的临时文件
        for root, dirs, files in os.walk(TEMP_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getmtime(file_path) < current_time - 3600:  # 1小时
                    try:
                        os.remove(file_path)
                        print(f"Cleaned old temp file: {file_path}")
                    except Exception as e:
                        print(f"Could not remove old temp file {file_path}: {e}")
        
        # 清理超过24小时的输出文件
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getmtime(file_path) < current_time - 86400:  # 24小时
                    try:
                        os.remove(file_path)
                        print(f"Cleaned old output file: {file_path}")
                    except Exception as e:
                        print(f"Could not remove old output file {file_path}: {e}")
                        
    except Exception as e:
        print(f"Error during cleanup: {e}")

def sanitize_filename(text, max_length=50):
    """
    テキストをファイル名として使用可能な形式に変換
    Args:
        text: 入力テキスト
        max_length: 最大文字数（デフォルト50文字）
    Returns:
        サニタイズされたファイル名
    """
    # 改行、タブなどの制御文字を削除
    text = re.sub(r'[\r\n\t]', '', text)
    
    # Windowsファイル名で使用できない文字を削除
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    
    # 前後の空白を削除
    text = text.strip()
    
    # 長すぎる場合は切り詰める
    if len(text) > max_length:
        text = text[:max_length]
    
    # 空文字の場合はデフォルト名を使用
    if not text:
        text = "generated_audio"
    
    return text

def generate_filename(input_text):
    """
    入力テキストから yyyymmdd_hhmmss_入力されたことば.wav 形式のファイル名を生成
    Args:
        input_text: 入力されたテキスト
    Returns:
        生成されたファイル名
    """
    # 現在の日時を取得
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # テキストをファイル名として使用可能な形式に変換
    sanitized_text = sanitize_filename(input_text)
    
    # ファイル名を生成
    filename = f"{timestamp}_{sanitized_text}.wav"
    
    return filename

# Define the inference function
def synthesize(GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text_path, ref_language, target_text_path, target_language, output_path):
    # Read reference text
    with open(ref_text_path, 'r', encoding='utf-8') as file:
        ref_text = file.read()

    # Read target text
    with open(target_text_path, 'r', encoding='utf-8') as file:
        target_text = file.read()

    # Change model weights & generate audio under patched MHA
    with MhaPatched():
        change_gpt_weights(gpt_path=GPT_model_path)
        change_sovits_weights(sovits_path=SOVITS_MODEL_PATH)
        result_list = list(get_tts_wav(
            ref_wav_path=ref_audio_path, 
            prompt_text=ref_text, 
            prompt_language=ref_language, 
            text=target_text, 
            text_language=target_language, 
            top_p=1, temperature=1
        ))

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        
        # 创建唯一的输出文件名
        unique_id = str(uuid.uuid4())[:8]
        timestamp = int(time.time())
        output_filename = f"output_{unique_id}_{timestamp}.wav"
        output_wav_path = os.path.join(output_path, output_filename)
        
        # 使用上下文管理器确保文件正确关闭
        try:
            sf.write(output_wav_path, last_audio_data, last_sampling_rate)
            print(f"Audio saved to {output_wav_path}")
            
            # 验证文件已正确写入
            if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
                return output_wav_path
            else:
                raise Exception("Audio file was not properly saved")
                
        except Exception as e:
            print(f"Error saving audio: {e}")
            raise

def save_audio_to_file(audio_bytes, input_text, default_filename=None):

    if default_filename is None:
        default_filename = generate_filename(input_text)
    def file_dialog():
        try:
            # 创建隐藏的根窗口
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            # 获取用户的下载文件夹作为默认路径
            downloads_path = str(Path.home() / "Downloads")
            
            # 打开文件保存对话框
            file_path = filedialog.asksaveasfilename(
                title="Save Audio File",
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                initialdir=downloads_path,
                initialfile=default_filename
            )
            
            if file_path:
                # 保存文件
                with open(file_path, 'wb') as f:
                    f.write(audio_bytes)
                st.session_state.save_result = f"File saved to: {file_path}"
                st.session_state.save_success = True
            else:
                st.session_state.save_result = "Save cancelled"
                st.session_state.save_success = False
                
            root.destroy()
            
        except Exception as e:
            st.session_state.save_result = f"Save failed: {str(e)}"
            st.session_state.save_success = False
    
    # 在新线程中运行文件对话框
    thread = threading.Thread(target=file_dialog)
    thread.daemon = True
    thread.start()

# # 启动时清理旧文件
# clean_old_files()

# Page configuration
st.set_page_config(page_title="Voiceger", layout="wide")

st.markdown(
    """
    <style>
    /* 隐藏Deploy按钮的多种方法 */
    div[data-testid="stAppDeployButton"],
    .stAppDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

set_background_image()

# 初始化 session_state 变量
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "last_generated_text" not in st.session_state:
    st.session_state.last_generated_text = ""
if "last_selected_emotion" not in st.session_state:
    st.session_state.last_selected_emotion = ""
if "last_target_language" not in st.session_state:
    st.session_state.last_target_language = ""
if "generation_completed" not in st.session_state:
    st.session_state.generation_completed = False
if "show_success_message" not in st.session_state:
    st.session_state.show_success_message = False
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "save_result" not in st.session_state:
    st.session_state.save_result = ""
if "save_success" not in st.session_state:
    st.session_state.save_success = False

# Top tabs section, reserved for future features
tabs = st.tabs(["TTS", "Voice Conversion"])
with tabs[0]:
    # TTS Module
    st.markdown("<h1 style='text-align: center;'>Voiceger</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # # 显示工作目录信息
    # with st.expander("📁 Work Directory Information", expanded=False):
    #     st.info(f"**Work Directory:** `{WORK_DIR}`")
    #     st.info(f"**Temp Directory:** `{TEMP_DIR}`")
    #     st.info(f"**Output Directory:** `{OUTPUT_DIR}`")
    #     st.text("The application uses a separate work directory to avoid permission issues.")
        
    #     # 添加清理按钮
    #     if st.button("🗑️ Clean Old Files"):
    #         clean_old_files()
    #         st.success("Old files cleaned successfully!")
    
    st.markdown("To generate speech, enter text, select the corresponding languages, and click the **Generate Speech** button.")

    # First row: Reference Audio File and Target Text (side by side)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Step1 Select Emotion")

        PRESET_REF_AUDIOS = {
            "Neutral": "reference_audios/01_ref_emoNormal026.wav",
            "Sweet": "reference_audios/02_ref_emoAma026.wav",
            "Snippy": "reference_audios/03_ref_emoTsun026.wav",
            "Sexy": "reference_audios/04_ref_emoSexy026.wav",
            "Whispering": "reference_audios/05_ref_emoSasa026.wav",
            "Murmuring": "reference_audios/06_ref_emoMurmur026.wav",
            "Exhausted": "reference_audios/07_ref_emoHero026.wav",
            "Sobbing": "reference_audios/08_ref_emoSobbing026.wav"
        }

        # 如果已经生成过音频，使用之前的选择作为默认值
        default_emotion_index = 0
        if st.session_state.generation_completed and st.session_state.last_selected_emotion:
            try:
                default_emotion_index = list(PRESET_REF_AUDIOS.keys()).index(st.session_state.last_selected_emotion)
            except ValueError:
                default_emotion_index = 0

        selected_emotion = st.selectbox("Select an Emotion Voice", 
                                        list(PRESET_REF_AUDIOS.keys()),
                                        index=default_emotion_index)
        
        # 确保参考音频路径是相对于安装目录的
        ref_audio_path = os.path.join(current_dir, PRESET_REF_AUDIOS[selected_emotion])

        # Reference Text
        ref_text_input = "私はいつもミネラルウォーターを持ち歩いています。"
        st.markdown("#### Reference Text")
        st.text_input("Reference Text", value=ref_text_input, disabled=True, label_visibility="collapsed")

        if os.path.exists(ref_audio_path):
            with open(ref_audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
        else:
            st.warning(f"Audio file not found: {ref_audio_path}")

    with col2:
        st.markdown("### Step2: Generate Voice")

        # 选择语言
        language_options = [
            "Chinese", "English", "Japanese", "Cantonese", "Korean",
            "Chinese-English Mixed", "Japanese-English Mixed", 
            "Cantonese-English Mixed", "Korean-English Mixed"
        ]
        
        default_language_index = 0
        if st.session_state.generation_completed and st.session_state.last_target_language:
            try:
                default_language_index = language_options.index(st.session_state.last_target_language)
            except ValueError:
                default_language_index = 0

        target_language = st.selectbox("Target Language", 
                                       language_options,
                                       index=default_language_index)

        # 目标文本输入
        default_text = "Please enter the text content to generate speech"
        if st.session_state.generation_completed and st.session_state.last_generated_text:
            default_text = st.session_state.last_generated_text

        target_text_input = st.text_area("Enter the text you want the voice to say:",
                                        value=default_text,
                                        height=118)

        # 生成按钮
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
        generate_button = st.button("Generate Speech")
        st.markdown("</div>", unsafe_allow_html=True)

    # 处理生成按钮点击
    if generate_button:
        # 重置状态
        st.session_state.show_success_message = False
        st.session_state.is_generating = True
        st.session_state.save_result = ""
        
        try:
            # 固定的参考文本
            ref_text_input = "私はいつもミネラルウォーターを持ち歩いています。"
            ref_language = "Japanese"

            # 使用作业目录创建临时文件
            with safe_temp_file(suffix=".txt", content=ref_text_input) as tmp_ref_text_path, \
                 safe_temp_file(suffix=".txt", content=target_text_input) as tmp_target_text_path, \
                 safe_temp_dir() as tmp_output_dir:
                
                # 推理调用
                output_wav_path = synthesize(
                    GPT_MODEL_PATH, SOVITS_MODEL_PATH, ref_audio_path, 
                    tmp_ref_text_path, ref_language,
                    tmp_target_text_path, target_language, 
                    tmp_output_dir
                )
                
                # 检查输出并读取到内存
                if output_wav_path and os.path.exists(output_wav_path):
                    with open(output_wav_path, "rb") as f:
                        st.session_state.audio_bytes = f.read()
                    
                    # 保存状态
                    st.session_state.last_generated_text = target_text_input
                    st.session_state.last_selected_emotion = selected_emotion
                    st.session_state.last_target_language = target_language
                    st.session_state.generation_completed = True
                    st.session_state.show_success_message = True
                    st.session_state.is_generating = False
                    
                    print("Audio generation completed successfully")
                else:
                    st.session_state.is_generating = False
                    st.error("Failed to generate audio. Please check model configuration or logs.")

        except Exception as e:
            st.session_state.is_generating = False
            st.error(f"An error occurred during inference: {e}")
            import traceback
            st.error("Error! See logs")
            print(traceback.format_exc())

    # 显示状态信息
    if st.session_state.is_generating:
        st.info("Generating speech, please wait...")
    elif st.session_state.show_success_message:
        st.success("Speech generated successfully!")

    # 显示保存结果
    if st.session_state.save_result:
        if st.session_state.save_success:
            st.success(st.session_state.save_result)
        else:
            st.warning(st.session_state.save_result)

    # 如果音频已经生成，显示预览和下载选项
    if st.session_state.audio_bytes is not None:
        st.markdown("---")
        st.markdown("### Generated Audio Preview")
        
        # 显示生成信息
        if st.session_state.generation_completed:
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.info(f"**Emotion:** {st.session_state.last_selected_emotion}")
            with info_col2:
                st.info(f"**Language:** {st.session_state.last_target_language}")
            with info_col3:
                st.info(f"**Text Length:** {len(st.session_state.last_generated_text)} chars")
        
        # 音频播放器
        st.audio(st.session_state.audio_bytes, format="audio/wav")
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            # 方法1：使用系统文件对话框保存（推荐用于打包后的应用）
            if st.button("Save to File", help="Use file dialog to choose save location"):
                save_audio_to_file(st.session_state.audio_bytes, st.session_state.last_generated_text)

with tabs[1]:
    # Voice Conversion Module
    st.markdown("<h1 style='text-align: center;'>Voiceger</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if not RVC_AVAILABLE:
        st.error("🚫 Voice Conversion feature is not available")
        
        # Show detailed error information
        with st.expander("📋 Detailed Error Information", expanded=True):
            st.markdown(f"**Error Reason:** {RVC_ERROR_MESSAGE}")
            
            # Check if it's a dependency issue
            if MISSING_DEPENDENCIES:
                st.markdown("### 🔧 Install Missing Dependencies:")
                st.code(f"pip install {' '.join(MISSING_DEPENDENCIES)}", language="bash")
                
                st.markdown("**Or install complete RVC dependencies:**")
                rvc_requirements_path = os.path.join(rvc_dir, "requirements", "main.txt")
                if os.path.exists(rvc_requirements_path):
                    st.code(f"pip install -r \"{rvc_requirements_path}\"", language="bash")
                else:
                    st.code("pip install faiss-cpu librosa scikit-learn pyworld torchcrepe", language="bash")
                
                st.warning("⚠️ Please restart the application after installing dependencies.")
            
            st.markdown("### 🔧 Complete Solution:")
            st.markdown("""
            1. **Install Python Dependencies**:
               ```bash
               pip install faiss-cpu librosa scikit-learn pyworld torchcrepe
               ```
            
            2. **Check RVC Directory**: Ensure `Retrieval-based-Voice-Conversion-WebUI` directory exists
            
            3. **Install RVC Models**: 
               - Download trained RVC model files (`.pth` format)
               - Place model files in `Retrieval-based-Voice-Conversion-WebUI/assets/weights/` directory
               - (Optional) Place corresponding index files (`.index` format) in `Retrieval-based-Voice-Conversion-WebUI/assets/indices/` directory
            
            4. **Restart Application**: Restart the Voiceger application
            """)
            
            # Show directory status
            st.markdown("### 📁 Directory Status Check:")
            rvc_exists = os.path.exists(rvc_dir)
            weights_dir = os.path.join(rvc_dir, 'assets', 'weights')
            indices_dir = os.path.join(rvc_dir, 'assets', 'indices')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                status = "✅" if rvc_exists else "❌"
                st.markdown(f"{status} RVC Directory: {'Exists' if rvc_exists else 'Not Found'}")
            
            with col2:
                weights_exists = os.path.exists(weights_dir)
                status = "✅" if weights_exists else "❌"
                st.markdown(f"{status} Weights Directory: {'Exists' if weights_exists else 'Not Found'}")
            
            with col3:
                indices_exists = os.path.exists(indices_dir)
                status = "✅" if indices_exists else "❌"
                st.markdown(f"{status} Indices Directory: {'Exists' if indices_exists else 'Not Found'}")
            
            # Check model files
            if weights_exists:
                model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
                if model_files:
                    st.success(f"✅ Found {len(model_files)} model files: {', '.join(model_files)}")
                else:
                    st.warning("⚠️ No .pth model files found in weights directory")
            
            # Show dependency status
            st.markdown("### 📦 Dependency Status Check:")
            dependencies_to_check = ["faiss", "librosa", "sklearn", "pyworld", "torchcrepe"]
            
            for dep in dependencies_to_check:
                try:
                    __import__(dep)
                    st.success(f"✅ {dep}: Installed")
                except ImportError:
                    st.error(f"❌ {dep}: Not Installed")
    else:
        # 初始化VC相关的session state
        if "vc_audio_bytes" not in st.session_state:
            st.session_state.vc_audio_bytes = None
        if "vc_processing" not in st.session_state:
            st.session_state.vc_processing = False
        if "vc_result_message" not in st.session_state:
            st.session_state.vc_result_message = ""
        
        st.markdown("Upload an audio file and convert it to Zundamon's voice using Voice Conversion technology.")
        
        # 音频文件上传
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Step 1: Upload Audio")
            uploaded_file = st.file_uploader(
                "Choose an audio file (Filenames must not contain Chinese, Japanese, or Korean characters.)",
                type=['wav', 'mp3', 'flac', 'm4a'],
                help="Upload the audio file you want to convert to Zundamon's voice"
            )
            st.markdown('<span style="color:red">Note: English file name only.</span>', unsafe_allow_html=True)
            
            if uploaded_file is not None:
                # Block filenames containing Chinese/Japanese/Korean characters
                cjk_pattern = r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]'
                if re.search(cjk_pattern, uploaded_file.name):
                    st.error("Cannot upload files with filenames containing CJK characters. Please rename and try again.")
                    st.session_state.vc_invalid_filename = True
                else:
                    st.session_state.vc_invalid_filename = False
                    # 显示上传的音频
                    st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")
                    
                    # 保存上传的文件到受控的临时位置（WORK_DIR\\temp）
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}", dir=TEMP_DIR) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        input_audio_path = tmp_file.name
        
        with col2:
            st.markdown("### Step 2: Conversion Settings")
            
            # F0方法选择
            f0_method = st.selectbox(
                "Pitch Extraction Algorithm",
                ["rmvpe", "crepe"],
                index=0,
                help="Algorithm for extracting pitch information. RMVPE is faster, Crepe is more accurate."
            )
            
            # 移除 Transpose（音调调整），默认值设为 0
            f0_up_key = 0
            
            # 重采样率
            resample_sr = st.selectbox(
                "Resample Rate (Hz)",
                [0, 16000, 22050, 44100, 48000],
                index=0,
                help="Output sample rate. 0 means use original sample rate."
            )
            
            # RMS混合率
            rms_mix_rate = st.slider(
                "RMS Mix Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Controls volume envelope mixing. Higher values preserve more of the original volume dynamics."
            )
            
            # 保护无声辅音
            protect = st.slider(
                "Protect Voiceless Consonants",
                min_value=0.0,
                max_value=0.5,
                value=0.33,
                step=0.01,
                help="Protect voiceless consonants and breath sounds. Higher values provide more protection."
            )
            
            # 中值滤波半径
            filter_radius = st.slider(
                "Median Filtering Radius",
                min_value=0,
                max_value=7,
                value=3,
                help="Median filter radius for smoothing F0 curve. Higher values provide more smoothing."
            )
            
            # 特征搜索比例
            index_rate = st.slider(
                "Feature Searching Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                help="Ratio for feature searching in the index. Higher values use more index features."
            )
            
            # 显示当前索引选择状态（优先 v2，回退时给出警告）
            try:
                if RVC_INDEX_SELECTION_MESSAGE:
                    if "not found" in RVC_INDEX_SELECTION_MESSAGE or "No index files" in RVC_INDEX_SELECTION_MESSAGE:
                        st.warning(RVC_INDEX_SELECTION_MESSAGE)
                    else:
                        # 不显示“Using v2 index: ...”的信息提示
                        pass
            except Exception:
                pass

            # 模型就绪提示（移除自动加载，改为在点击转换时惰性加载）
            sid_name = os.path.basename(RVC_MODEL_PATH) if RVC_MODEL_PATH else ""
            if not sid_name:
                st.error("No RVC model found in assets/weights. Please add a .pth model and restart.")
            else:
                st.caption("The above parameters are recommended to be used with default values.")
        
        # 转换按钮
        st.markdown("---")
        col_convert, col_status = st.columns([1, 2])
        
        with col_convert:
            if st.button("🎵 Convert Voice", disabled=st.session_state.vc_processing):
                if uploaded_file is None:
                    st.error("Please upload an audio file first.")
                elif st.session_state.get("vc_invalid_filename"):
                    st.error("Cannot upload files with filenames containing CJK characters. Please rename and try again.")
                else:
                    st.session_state.vc_processing = True
                    st.session_state.vc_result_message = ""
                
                    try:
                        with st.spinner("Converting voice... This may take a few minutes."):
                            sid_name = os.path.basename(RVC_MODEL_PATH) if RVC_MODEL_PATH else ""
                            if getattr(vc_model, "net_g", None) is None:
                                if not sid_name:
                                    st.error("No RVC model found in assets/weights. Please add a .pth model and restart.")
                                    raise RuntimeError("RVC model not loaded")
                                # 在加载 RVC 前禁用猴子补丁
                                disable_mha_patch()
                                ensure_hubert_asset()
                                os.environ["weight_root"] = os.path.join(rvc_dir, "assets", "weights")
                                os.environ["index_root"] = os.path.join(rvc_dir, "assets", "indices")
                                os.environ["outside_index_root"] = os.path.join(rvc_dir, "assets", "indices")
                                with pushd(rvc_dir):
                                    vc_model.get_vc(sid_name)

                        # 根据原始 web.py 默认行为，使用说话人ID=0
                        sid_index = 0

                        # 在执行 RVC 推理前再次确保禁用补丁
                        disable_mha_patch()
                        with pushd(rvc_dir):
                            ret = vc_model.vc_single(
                                sid=sid_index,
                                input_audio_path=input_audio_path,
                                f0_up_key=f0_up_key,
                                f0_file=None,
                                f0_method=f0_method,
                                file_index=RVC_INDEX_PATH,
                                file_index2=RVC_INDEX_PATH,
                                index_rate=index_rate,
                                filter_radius=filter_radius,
                                resample_sr=resample_sr,
                                rms_mix_rate=rms_mix_rate,
                                protect=protect,
                            )

                        # 兼容 2/3 返回值
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

                        # 使用受控的临时文件路径，并在上下文退出时自动清理
                        if isinstance(opt, tuple):
                            sr, audio_opt = opt
                            if not download_path:
                                # 模型未返回文件路径：用 safe_temp_file 生成一次性临时文件，读入内存后自动清理
                                with safe_temp_file(suffix=".wav") as tmp_wav_path:
                                    sf.write(tmp_wav_path, audio_opt, sr)
                                    with open(tmp_wav_path, "rb") as f:
                                        audio_bytes = f.read()
                            else:
                                # 模型返回了 download_path：仅读取到内存，不保留额外副本
                                with open(download_path, "rb") as f:
                                    audio_bytes = f.read()

                            # 持久化到 session_state，避免交互后页面重跑导致预览消失
                            st.session_state.vc_audio_bytes = audio_bytes
                            simple_msg = "Voice conversion successful!"
                            st.session_state.vc_result_message = f"✅ {simple_msg}"
                        else:
                            st.error(f"Voice conversion failed: {info}")
                            st.session_state.vc_result_message = f"❌ {info}"
                    finally:
                        st.session_state.vc_processing = False
                        # 清理上传文件的临时副本
                        try:
                            if 'input_audio_path' in locals() and input_audio_path and os.path.exists(input_audio_path):
                                os.remove(input_audio_path)
                                print(f"Cleaned uploaded temp audio: {input_audio_path}")
                        except Exception as e:
                            print(f"Warning: could not remove uploaded temp audio {input_audio_path}: {e}")
        
        with col_status:
            if st.session_state.vc_processing:
                st.info("🔄 Processing voice conversion...")
            elif st.session_state.vc_result_message:
                if "✅" in st.session_state.vc_result_message:
                    st.success(st.session_state.vc_result_message)
                else:
                    st.error(st.session_state.vc_result_message)
        
        # 显示转换结果
        if st.session_state.vc_audio_bytes is not None:
            st.markdown("---")
            st.markdown("### Converted Audio Preview")
            
            # 音频播放器
            st.audio(st.session_state.vc_audio_bytes, format="audio/wav")
            
            # 保存/下载按钮
            col_save1, col_save2 = st.columns(2)
            
            with col_save1:
                # 方法1：使用系统文件对话框保存（与TTS一致，适配Windows打包）
                if st.button("Save Converted Audio", help="Save converted audio via file dialog"):
                    # 默认文件名为 时间戳_输入文件名（去扩展名）
                    input_name = "converted_audio"
                    try:
                        if uploaded_file is not None:
                            input_name = os.path.splitext(uploaded_file.name)[0]
                    except Exception:
                        pass
                    save_audio_to_file(st.session_state.vc_audio_bytes, input_name)

# 英文小字利用規約（页面最下面）
st.markdown("---")
st.markdown(
    "<p style='font-size:0.75rem; color:gray;'>"
    "<strong>Terms of Use:</strong> If you publish audio content using Zundamon from this software, please display the credit 'VOICEGER:Zundamon' near the content."
    "</p>",
    unsafe_allow_html=True
)

