# Worker RunPod Serverless : VGGT (poses COLMAP) + GenRecon (mesh GLB)
# Build linux/amd64 uniquement (GitHub Actions), cible A100 80GB (sm_80).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.0" \
    CUDA_HOME=/usr/local/cuda \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget build-essential \
    libgl1 libglib2.0-0 libgomp1 libx11-6 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python de base (liste setup.sh --basic, sans gradio/pillow-simd/
# tensorboard/lpips qui ne sont pas importés par le chemin d'inférence)
RUN pip install \
    numpy==1.26.4 \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh \
    transformers==4.57.3 pandas zstandard kornia timm einops safetensors \
    "huggingface_hub[cli]" hf_transfer \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# flash-attn : le setup.py télécharge la wheel précompilée correspondant à
# torch/CUDA/ABI (pas de compilation)
RUN pip install flash-attn==2.7.3 --no-build-isolation

# Extensions CUDA (compilées pour sm_80, pas besoin de GPU au build)
RUN mkdir -p /tmp/ext \
    && git clone -b v0.4.0 --depth 1 https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast \
    && pip install /tmp/ext/nvdiffrast --no-build-isolation \
    && rm -rf /tmp/ext

RUN mkdir -p /tmp/ext \
    && git clone -b renderutils --depth 1 https://github.com/JeffreyXiang/nvdiffrec.git /tmp/ext/nvdiffrec \
    && pip install /tmp/ext/nvdiffrec --no-build-isolation \
    && rm -rf /tmp/ext

RUN mkdir -p /tmp/ext \
    && git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh \
    && pip install /tmp/ext/CuMesh --no-build-isolation \
    && rm -rf /tmp/ext

RUN mkdir -p /tmp/ext \
    && git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM \
    && pip install /tmp/ext/FlexGEMM --no-build-isolation \
    && rm -rf /tmp/ext

# GenRecon (commit épinglé) + son extension o-voxel (submodule eigen)
RUN git clone https://github.com/kasothaphie/GenRecon.git /opt/GenRecon \
    && cd /opt/GenRecon \
    && git checkout eaf1468118d20469d17079a4a19737297d2ef87b \
    && git submodule update --init --recursive \
    && pip install /opt/GenRecon/o-voxel --no-build-isolation

# VGGT (commit épinglé) — deps déjà présentes, on évite son pin torch==2.3.1
RUN git clone https://github.com/facebookresearch/vggt.git /opt/vggt \
    && cd /opt/vggt \
    && git checkout a288dd0f14786c93483e45524328726ab7b1b4ce \
    && pip install --no-deps -e /opt/vggt

# Outils du handler : COLMAP I/O, nettoyage de points, storage, serverless
RUN pip install pycolmap open3d runpod supabase

COPY worker/ /opt/worker/

ENV PYTHONPATH=/opt/GenRecon \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    ATTN_BACKEND=flash_attn \
    SPARSE_CONV_BACKEND=flex_gemm

WORKDIR /opt/GenRecon
CMD ["python", "-u", "/opt/worker/handler.py"]
