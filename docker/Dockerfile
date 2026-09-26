# syntax=docker/dockerfile:1.7
# API image: FastAPI + retrieval/reranking on the GPU (CUDA torch). The LLM runs in the
# separate `ollama` service; Qdrant and Phoenix are their own services (docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # models are mounted from the host (D:) and never downloaded in the container
    HF_HOME=/models/huggingface \
    LLAMA_INDEX_CACHE_DIR=/models/llama_index \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PARENT_STORE_READONLY=1

WORKDIR /app

# CUDA torch first, from the PyTorch index (the wheel bundles the CUDA runtime; the host driver
# is exposed by the NVIDIA container runtime). A constraint keeps later installs from swapping it
# for the CPU build from PyPI.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126 \
 && echo "torch==2.14.0+cu126" > /tmp/constraints.txt

COPY docker/requirements-api.txt /tmp/requirements-api.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-api.txt -c /tmp/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu126

COPY src/ src/
COPY configs/ configs/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8).status == 200 else 1)"

# Single worker: one GPU, pipeline runs are serialised inside the app.
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
