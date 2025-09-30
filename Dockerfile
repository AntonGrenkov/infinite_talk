FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    WORKDIR=/workspace

WORKDIR ${WORKDIR}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils \
    git git-lfs curl ca-certificates \
    ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/* && git lfs install

RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /workspace/InfiniteTalk

# CUDA 12.1 stack
RUN python3.10 -m pip install --no-cache-dir \
    "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cu121
RUN python3.10 -m pip install --no-cache-dir \
    "xformers==0.0.28" --index-url https://download.pytorch.org/whl/cu121
RUN python3.10 -m pip install --no-cache-dir \
    flash_attn==2.7.4.post1 \
    runpod \
    huggingface_hub[cli] \
    numpy scipy pillow psutil packaging librosa soundfile pydub

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

