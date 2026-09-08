# Dev-container for the fake-NUMA testbed (docs/numa-testbed.md).
#
# Toolchain only — the source tree is bind-mounted at /src and built inside, so a
# one-file ggml.c change rebuilds in seconds via the shared ccache volume instead of
# rebuilding the image. Models live in the ik-models named volume mounted at /models
# (bind-mounting GGUFs from the Windows drive is gRPC-FUSE slow).
#
# Build:  docker build -f docker/numa-testbed.Containerfile -t ik-numa-testbed .
# Run:    scripts/testbed/run-testbed.ps1 (computes --cpuset-cpus, wires mounts + env)
#
# ARG BASE exists so a CUDA variant (BASE=nvidia/cuda:12.x-devel-ubuntu24.04 plus
# -DGGML_CUDA=ON and --gpus all at run time) can slot in later; not wired yet.
ARG BASE=ubuntu:24.04

FROM docker.io/${BASE}

ENV LC_ALL=C.utf8 \
    DEBIAN_FRONTEND=noninteractive \
    CCACHE_DIR=/ccache \
    CCACHE_MAXSIZE=5G \
    CCACHE_COMPRESS=1

RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        ca-certificates build-essential cmake ccache git \
        curl libcurl4-openssl-dev libgomp1 \
        python3 python3-pip \
        numactl util-linux time && \
    rm -rf /var/lib/apt/lists/*

# huggingface_hub for scripts/testbed/download-model.sh
RUN pip install --break-system-packages --no-cache-dir "huggingface_hub[cli]"

WORKDIR /src

CMD ["bash"]
