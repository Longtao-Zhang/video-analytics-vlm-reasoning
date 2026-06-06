# Warehouse Safety Video Intelligence — inference service image.
#
#   Build:  docker build -t safety-video .
#   Run:    docker run --rm -p 8000:8000 safety-video
#           (add `-e DASHSCOPE_API_KEY=...` to enable --vlm-mode api)
#
# A slim Python base keeps the image small. (ultralytics + torch are still large.)
FROM python:3.11-slim

# OpenCV (pulled in by ultralytics as opencv-python) needs a few shared libraries
# at runtime that the slim image lacks. Missing ones surface as
# "ImportError: libXXX.so.1: cannot open shared object file" the moment cv2 is
# imported. Install the minimal set and clean the apt cache to keep the layer small.
# (libgomp1 is for torch's OpenMP runtime.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependency layer (build-cached) ----------------------------------------
# Copy ONLY requirements.txt first. Docker caches layers, so as long as this file
# is unchanged the (slow) pip install is reused across rebuilds. If you copied all
# the source before installing, every code edit would reinstall torch from scratch.
COPY requirements.txt .

# Optional PyPI mirror for restricted networks. Override at build time, e.g.:
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t safety-video .
ARG PIP_INDEX_URL=https://pypi.org/simple
# --timeout/--retries make large wheel downloads (torch is ~200MB) resilient to
# slow or flaky links (e.g. when traffic is routed through a VPN/Tailscale exit).
RUN pip install --no-cache-dir --timeout 120 --retries 5 --index-url "$PIP_INDEX_URL" -r requirements.txt

# --- Application layer -------------------------------------------------------
# Now copy the rest of the project (src/, app/, configs/, yolov8n.pt weights).
# .dockerignore keeps out .venv, .env, big example videos, and generated outputs.
COPY . .

# The API listens on 8000 inside the container; map it with `-p 8000:8000`.
EXPOSE 8000

# TODO(you): start the API server with uvicorn against `app.main:app`.
#   Critical gotcha: bind to host 0.0.0.0 (NOT 127.0.0.1) or the port is not
#   reachable from outside the container.
#   Use the exec form (a JSON array), e.g.:  CMD ["uvicorn", "...", "--host", ...]
# CMD [...]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
