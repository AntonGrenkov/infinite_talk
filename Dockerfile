# База: CUDA 12.1 + Python 3.10 (подходит под torch 2.4.1/cu121 и xformers 0.0.28 из README)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    WORKDIR=/workspace

WORKDIR ${WORKDIR}

# Системные пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3.10-venv \
    git git-lfs wget curl ca-certificates \
    ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

# pip + uv
RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Клонируем InfiniteTalk (официальный репозиторий)
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git ${WORKDIR}/InfiniteTalk

# Python-зависимости
# ВАЖНО: закрепляем версии из их README (torch 2.4.1/cu121, xformers 0.0.28, flash_attn 2.7.4.post1)
RUN python3.10 -m pip install --no-cache-dir \
    "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cu121

RUN python3.10 -m pip install --no-cache-dir \
    "xformers==0.0.28" --index-url https://download.pytorch.org/whl/cu121

# Доп. зависимости из README/requirements
RUN python3.10 -m pip install --no-cache-dir \
    misaki[en] ninja psutil packaging flash_attn==2.7.4.post1 \
    huggingface_hub[cli] librosa soundfile numpy scipy pillow pydantic "fastapi>=0.115" uvicorn pydub

# Устанавливаем зависимости проекта
COPY ./requirements.txt ${WORKDIR}/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r ${WORKDIR}/requirements.txt || true

# Директории весов
RUN mkdir -p ${WORKDIR}/weights/Wan2.1-I2V-14B-480P \
             ${WORKDIR}/weights/chinese-wav2vec2-base \
             ${WORKDIR}/weights/InfiniteTalk/single \
             ${WORKDIR}/inputs ${WORKDIR}/outputs

# Скачивание весов через huggingface-cli (только I2V — без V2V-спец. весов)
# Пропусти, если будешь монтировать свои веса в /workspace/weights
RUN huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ${WORKDIR}/weights/Wan2.1-I2V-14B-480P
RUN huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ${WORKDIR}/weights/chinese-wav2vec2-base
# Некоторые сборки InfiniteTalk рекомендуют доброскачать model.safetensors из PR ветки
RUN huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ${WORKDIR}/weights/chinese-wav2vec2-base || true
RUN huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ${WORKDIR}/weights/InfiniteTalk

# API-сервер
COPY ./server.py ${WORKDIR}/server.py
COPY ./runner.sh ${WORKDIR}/runner.sh
RUN chmod +x ${WORKDIR}/runner.sh

# Экспонируем HTTP для Runpod Serverless HTTP
EXPOSE 8000

# По умолчанию запускаем FastAPI-сервер
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
