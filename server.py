import json
import uuid
import base64
import subprocess
from pathlib import Path

import torch
import runpod
from pydub import AudioSegment
from huggingface_hub import snapshot_download

WORKDIR = Path("/workspace")
INF_DIR = WORKDIR / "InfiniteTalk"
WEIGHTS = WORKDIR / "weights"
INPUTS = WORKDIR / "inputs"
OUTPUTS = WORKDIR / "outputs"

WAN_DIR = WEIGHTS / "Wan2.1-I2V-14B-480P"
W2V_DIR = WEIGHTS / "chinese-wav2vec2-base"
IT_DIR = WEIGHTS / "InfiniteTalk" / "single"


def ensure_weights():
    # Wan I2V (оставим минимальный набор файлов)
    if not WAN_DIR.exists() or not any(WAN_DIR.iterdir()):
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir=WAN_DIR.as_posix(),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "*.txt",
                "*.model",
                "*.bin"
            ]
        )
    # wav2vec
    if not W2V_DIR.exists() or not any(W2V_DIR.iterdir()):
        snapshot_download(
            repo_id="TencentGameMate/chinese-wav2vec2-base",
            local_dir=W2V_DIR.as_posix(),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.safetensors",
                "*.bin",
                "*.json",
                "*.txt",
                "*.model"
            ]
        )
    # InfiniteTalk single (нужен файл infinitetalk.safetensors)
    it_single = IT_DIR / "infinitetalk.safetensors"
    if not it_single.exists():
        snapshot_download(
            repo_id="MeiGen-AI/InfiniteTalk",
            local_dir=(WEIGHTS / "InfiniteTalk").as_posix(),
            local_dir_use_symlinks=False,
            allow_patterns=["**/single/infinitetalk.safetensors"]
        )


def _b64_to_bytes(data_b64: str) -> bytes:
    return base64.b64decode(data_b64)


def _save_image(image_b64: str, path: Path):
    raw = _b64_to_bytes(image_b64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(raw)


def _save_audio(audio_b64: str, path_wav: Path):
    """Принимаем wav/mp3/m4a; приводим к WAV 16-bit PCM."""
    raw = _b64_to_bytes(audio_b64)
    tmp = path_wav.with_suffix(".tmpbin")
    with open(tmp, "wb") as f:
        f.write(raw)
    audio = AudioSegment.from_file(tmp)
    path_wav.parent.mkdir(parents=True, exist_ok=True)
    audio.export(path_wav, format="wav")
    tmp.unlink(missing_ok=True)


def _build_input_json(image_path: Path, audio_path: Path, dest_json: Path):
    # Список примеров; дублируем ключи для совместимости с пайплайном
    payload = [{
        "image": str(image_path),
        "img_path": str(image_path),
        "audio": str(audio_path),
        "audio_path": str(audio_path)
    }]
    with open(dest_json, "w") as f:
        json.dump(payload, f)


def _run_inference(input_json: Path, request_id: str, params: dict) -> str:
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        raise RuntimeError("No GPUs detected!")

    save_name = params.get("save_file") or f"infinitetalk_{request_id}"
    output_stem = (OUTPUTS / save_name).as_posix()

    cmd = [
        "torchrun", f"--nproc_per_node={gpu_num}", "--standalone",
        "generate_infinitetalk.py",
        "--ckpt_dir", (WEIGHTS / "Wan2.1-I2V-14B-480P").as_posix(),
        "--wav2vec_dir", (WEIGHTS / "chinese-wav2vec2-base").as_posix(),
        "--infinitetalk_dir", (WEIGHTS / "InfiniteTalk" / "single" / "infinitetalk.safetensors").as_posix(),
        "--dit_fsdp", "--t5_fsdp",
        "--ulysses_size", str(gpu_num),
        "--input_json", input_json.as_posix(),
        "--size", params.get("size", "infinitetalk-720"),
        "--sample_steps", str(params.get("sample_steps", 40)),
        "--mode", params.get("mode", "streaming"),
        "--motion_frame", str(params.get("motion_frame", 9)),
        "--save_file", output_stem
    ]

    if "audio_cfg" in params:
        cmd += ["--sample_audio_guide_scale", str(params["audio_cfg"])]
    if "text_cfg" in params:
        cmd += ["--sample_text_guide_scale", str(params["text_cfg"])]
    if "max_frame_num" in params:
        cmd += ["--max_frame_num", str(params["max_frame_num"])]

    proc = subprocess.run(
        cmd,
        cwd=INF_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)

    return output_stem + ".mp4"


def _file_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(event):
    """
    event["input"]:
    {
      "image_b64": "<...>",
      "audio_b64": "<...>",
      "params": {
        "size": "infinitetalk-720",
        "sample_steps": 40,
        "mode": "streaming",
        "motion_frame": 9,
        "return_base64": true   # если true — добавим output.video (base64)
      }
    }
    """
    try:
        data = event.get("input", {})
        image_b64 = data["image_b64"]
        audio_b64 = data["audio_b64"]
        params = data.get("params", {}) or {}
        return_b64 = bool(params.pop("return_base64", True))

        req_id = str(uuid.uuid4())[:8]
        sess = INPUTS / req_id
        sess.mkdir(parents=True, exist_ok=True)

        img_path = sess / "image.png"
        wav_path = sess / "audio.wav"
        input_json = sess / "input.json"

        _save_image(image_b64, img_path)
        _save_audio(audio_b64, wav_path)
        _build_input_json(img_path, wav_path, input_json)

        out_mp4 = _run_inference(input_json, req_id, params)

        output = {
            "task_id": req_id,
            "video_path": out_mp4
        }
        if return_b64:
            output["video"] = _file_to_base64(Path(out_mp4))

        return {"status": "COMPLETED", "output": output}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


if __name__ == '__main__':
    ensure_weights()
    runpod.serverless.start({'handler': handler})
