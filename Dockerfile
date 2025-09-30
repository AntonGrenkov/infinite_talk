FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    WORKDIR=/workspace

WORKDIR ${WORKDIR}

# build deps + system
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3.10-dev \
    git git-lfs curl ca-certificates \
    ffmpeg libsndfile1 \
    build-essential ninja-build cmake \
 && rm -rf /var/lib/apt/lists/* && git lfs install

# pip toolchain (пины важны для flash-attn)
RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --no-cache-dir --upgrade "pip<24.1" "setuptools<70" wheel "numpy==1.26.4"

# PyTorch cu121
RUN python3.10 -m pip install --no-cache-dir \
  "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cu121
RUN python3.10 -m pip install --no-cache-dir \
  "xformers==0.0.28" --index-url https://download.pytorch.org/whl/cu121

# прочие зависимости
RUN python3.10 -m pip install --no-cache-dir packaging psutil huggingface_hub[cli] librosa soundfile pillow scipy pydub runpod

# flash-attn (если будет долго/падает — см. заметки ниже)
RUN python3.10 -m pip install --no-cache-dir flash_attn==2.7.4.post1

# код
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /workspace/InfiniteTalk

# пути под веса и данные
RUN mkdir -p /workspace/weights/Wan2.1-I2V-14B-480P \
             /workspace/weights/chinese-wav2vec2-base \
             /workspace/weights/InfiniteTalk/single \
             /workspace/inputs /workspace/outputs

# НЕ СКАЧИВАЕМ ВЕСА ВО ВРЕМЯ СБОРКИ!

COPY server.py /workspace/server.py

# можно задать токен на рантайме (если репо приватные/гейт)
# ENV HF_TOKEN=...

CMD ["python3.10", "/workspace/server.py"]
