import os
import re
import shutil

DEFAULT_TTS_TRAIN_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SUPPORTED_TTS_TRAIN_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
]


def get_project_root():
    '''Detect the project root directory.
    In Docker, it's usually /workspace.
    Otherwise, it's the parent directory of this src file.
    '''
    if os.path.exists('/.dockerenv') or os.environ.get('IS_DOCKER'):
        return '/workspace'
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path):
    '''Normalize a path to be absolute within the project root if it is relative.'''
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(get_project_root(), path))


def configure_visible_cuda_device(gpu):
    '''Limit CUDA visibility for single-device CLI jobs and return the remapped runtime device.'''
    gpu = str(gpu or "cpu").strip()
    if gpu == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if gpu == "cuda":
        return "cuda:0"
    if gpu.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu.split(":", 1)[1]
        return "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return "cuda:0"


def normalize_resume_checkpoint_arg(value):
    '''Normalize CLI resume values into latest, None, or an absolute checkpoint path.'''
    if value is None:
        return "latest"
    value = str(value).strip()
    if not value:
        return "latest"
    if value.lower() in {"none", "false", "no", "off", "0"}:
        return None
    if value.lower() == "latest":
        return "latest"
    return resolve_path(value)


def get_model_local_dir(model_id):
    root = get_project_root()
    return os.path.join(root, 'models', model_id)



def is_model_dir_ready(path):
    if not path or not os.path.isdir(path):
        return False

    payload_markers = (
        'model.safetensors',
        'model.safetensors.index.json',
        'pytorch_model.bin',
        'pytorch_model.bin.index.json',
        'tf_model.h5',
        'model.ckpt.index',
        'flax_model.msgpack',
        'tokenizer.json',
        'tokenizer.model',
    )

    if any(os.path.exists(os.path.join(path, marker)) for marker in payload_markers):
        return True

    try:
        entries = os.listdir(path)
    except OSError:
        return False

    return any(entry.endswith(('.safetensors', '.bin', '.model')) for entry in entries)



def _cleanup_empty_dir(path):
    if not os.path.isdir(path) or os.path.islink(path):
        return
    try:
        if not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass



def _ensure_shared_model_dir(model_id, downloaded_path):
    local_dir = get_model_local_dir(model_id)
    resolved_downloaded_path = os.path.realpath(downloaded_path) if downloaded_path else downloaded_path
    resolved_local_dir = os.path.realpath(local_dir)

    if is_model_dir_ready(local_dir):
        return local_dir

    if not downloaded_path or not is_model_dir_ready(downloaded_path):
        return downloaded_path

    if resolved_downloaded_path == resolved_local_dir:
        return local_dir

    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    _cleanup_empty_dir(local_dir)

    if not os.path.exists(local_dir):
        try:
            os.symlink(downloaded_path, local_dir, target_is_directory=True)
            print(f'Linked shared model directory {local_dir} -> {downloaded_path}')
            return local_dir
        except OSError:
            pass

    if not is_model_dir_ready(local_dir):
        shutil.copytree(downloaded_path, local_dir, dirs_exist_ok=True)
        print(f'Copied model into shared directory {local_dir}')

    return local_dir if is_model_dir_ready(local_dir) else downloaded_path



def is_model_downloaded(model_id):
    local_dir = get_model_local_dir(model_id)
    return is_model_dir_ready(local_dir)



def get_model_path(model_id, use_hf=False):
    resolved_input = resolve_path(model_id)
    if os.path.exists(resolved_input):
        return resolved_input
    output_candidate = resolve_path(os.path.join('output', model_id))
    if os.path.exists(output_candidate):
        print(f'Found local checkpoint at {output_candidate}')
        return output_candidate
    local_dir = get_model_local_dir(model_id)
    if is_model_downloaded(model_id):
        print(f'Found local model at {local_dir}, skipping download!')
        return local_dir
    print(f'Downloading model {model_id} into {local_dir}...')
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    try:
        if use_hf:
            from huggingface_hub import snapshot_download
            downloaded_path = snapshot_download(repo_id=model_id, local_dir=local_dir)
        else:
            from modelscope import snapshot_download
            downloaded_path = snapshot_download(model_id, cache_dir=os.path.join(get_project_root(), 'models'))
        return _ensure_shared_model_dir(model_id, downloaded_path)
    except Exception as e:
        print(f'Warning: Download failed, falling back to id: {e}')
        return model_id


def resolve_embed_base_model(model_id):
    model_id = model_id.strip() if isinstance(model_id, str) else ""
    if not model_id:
        return DEFAULT_TTS_TRAIN_MODEL
    if model_id.endswith("-Base"):
        return model_id
    if "0.6B" in model_id:
        return "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    if "1.7B" in model_id:
        return "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    return DEFAULT_TTS_TRAIN_MODEL


def is_custom_voice_model(model_id):
    return isinstance(model_id, str) and model_id.endswith("-CustomVoice")


def missing_speaker_embeddings(speaker_names):
    missing = []
    if not speaker_names:
        return missing
    for speaker_name in speaker_names:
        speaker_name = str(speaker_name).strip()
        if not speaker_name:
            continue
        emb_path = resolve_path(os.path.join("final-dataset", speaker_name, "speaker_emb.safetensors"))
        if not os.path.exists(emb_path):
            missing.append(speaker_name)
    return missing


def speaker_key(value):
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())

def resolve_speaker_choice(speaker, supported_speakers):
    if not speaker or not supported_speakers:
        return speaker
    if speaker in supported_speakers:
        return speaker
    lower_map = {str(s).lower(): s for s in supported_speakers}
    lowered = str(speaker).lower()
    if lowered in lower_map:
        return lower_map[lowered]
    normalized = speaker_key(speaker)
    normalized_map = {}
    for s in supported_speakers:
        normalized_map.setdefault(speaker_key(s), s)
    return normalized_map.get(normalized, speaker)
