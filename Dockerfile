# CUDA 12.1 devel (есть nvcc для сборки flash-attn)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    WORKDIR=/workspace

WORKDIR ${WORKDIR}
SHELL ["/bin/bash", "-lc"]

# ===== system =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git git-lfs \
    build-essential ninja-build cmake \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/* && git lfs install

# ===== Miniconda =====
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -o /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm -f /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# =========================================================
# 1) Create conda env and install PyTorch + xformers (cu121)
# =========================================================
RUN conda create -y -n multitalk python=3.10 && conda clean -afy
ENV CONDA_DEFAULT_ENV=multitalk
ENV PATH=/opt/conda/envs/multitalk/bin:/opt/conda/bin:$PATH
ENV PIP_NO_CACHE_DIR=1

# PyTorch/cu121
RUN pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

# xformers/cu121
RUN pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# ===============================
# 2) Flash-attn installation (order)
# ===============================
# (ровно как в инструкции: misaki[en] → ninja → psutil → packaging → wheel → flash_attn)
RUN pip install "misaki[en]"
RUN pip install ninja
RUN pip install psutil
RUN pip install packaging
RUN pip install wheel
RUN pip install flash_attn==2.7.4.post1

# =======================
# 3) Other dependencies
# =======================
RUN pip install easydict einops pyyaml tqdm
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
# librosa через conda-forge
RUN conda install -y -n multitalk -c conda-forge librosa && conda clean -afy

# ===================
# 4) FFmpeg via conda
# ===================
RUN conda install -y -n multitalk -c conda-forge ffmpeg && conda clean -afy

# ===== extra tools for runtime model download =====
RUN pip install "huggingface_hub[cli]" hf-transfer pydub runpod

# ===== project code =====
# (при необходимости: RUN git clone .../InfiniteTalk.git /workspace/InfiniteTalk)
# если репо уже смонтируешь — эту строку можно убрать
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /workspace/InfiniteTalk

COPY server.py /workspace/server.py

# ускорение загрузок HF (опционально)
ENV HF_HUB_ENABLE_HF_TRANSFER=1
# передай токен при необходимости (секретом в Runpod), если какие-то репы гейтятся
# ENV HF_TOKEN=...

# не тянем модели на этапе build!
# директории под веса и данные
RUN mkdir -p /workspace/weights/Wan2.1-I2V-14B-480P \
             /workspace/weights/chinese-wav2vec2-base \
             /workspace/weights/InfiniteTalk/single \
             /workspace/inputs /workspace/outputs

# unbuffered вывод логов
CMD ["python", "-u", "/workspace/server.py"]
