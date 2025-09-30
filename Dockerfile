FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    WORKDIR=/workspace

WORKDIR ${WORKDIR}

# build deps + системные пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3.10-dev \
    git git-lfs curl ca-certificates \
    ffmpeg libsndfile1 \
    build-essential ninja-build cmake \
 && rm -rf /var/lib/apt/lists/* && git lfs install

# pip и "безопасные" версии инструментов
RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.10
# setuptools<70 и numpy<2 критичны для сборки flash-attn
RUN python3.10 -m pip install --no-cache-dir --upgrade "pip<24.1" "setuptools<70" wheel "numpy==1.26.4"

# PyTorch стек (cu121)
RUN python3.10 -m pip install --no-cache-dir \
  "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cu121
RUN python3.10 -m pip install --no-cache-dir \
  "xformers==0.0.28" --index-url https://download.pytorch.org/whl/cu121

# ВАЖНО: сначала инструменты, потом flash-attn
RUN python3.10 -m pip install --no-cache-dir packaging psutil huggingface_hub[cli] librosa soundfile pillow scipy pydub runpod

# Теперь flash-attn — уже при наличии dev-окружения и правильных версий
RUN python3.10 -m pip install --no-cache-dir flash_attn==2.7.4.post1

RUN mkdir -p /workspace/weights/Wan2.1-I2V-14B-480P \
             /workspace/weights/chinese-wav2vec2-base \
             /workspace/weights/InfiniteTalk/single \
             /workspace/inputs /workspace/outputs

RUN huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /workspace/weights/Wan2.1-I2V-14B-480P
RUN huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir /workspace/weights/chinese-wav2vec2-base
RUN huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir /workspace/weights/InfiniteTalk

COPY server.py /workspace/server.py
ENV GPU_NUM=8
CMD ["python3.10", "/workspace/server.py"]

