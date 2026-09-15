ARG CUDA_VERSION=13.0.2
ARG PYTHON_VERSION=3.12.10
ARG MSHIP_VARIANT=cuda
ARG UID=1000
ARG GID=1000

# =============================================================================
# base — runtime OS + uv + non-root user + env vars.
# =============================================================================
FROM ubuntu:24.04 AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG MSHIP_VARIANT
ARG UID
ARG GID

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        espeak-ng \
        gcc \
        g++ \
        gnupg \
        gosu \
        libc6-dev \
        libgomp1 \
        libnuma1 \
        ninja-build && \
    rm -rf /var/lib/apt/lists/*

# nvcc/cuobjdump stay in the runtime image because torch, triton and flashinfer
# JIT-compile kernels at model-load time. cuda-cudart is separate from torch's
# bundled copy: vLLM's C extensions hard-reference it by RPATH.
RUN if [ "$MSHIP_VARIANT" = "cuda" ]; then \
    CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
        | gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
        > /etc/apt/sources.list.d/cuda.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH} && \
    apt-get purge -y --auto-remove gnupg && \
    rm -f /etc/apt/sources.list.d/cuda.list /usr/share/keyrings/cuda-keyring.gpg && \
    rm -rf /var/lib/apt/lists/*; \
    fi

RUN if ! getent group $GID >/dev/null; then groupadd -g $GID modelship; fi && \
    if ! getent passwd $UID >/dev/null; then useradd -m -u $UID -g $GID modelship; \
    else existing=$(getent passwd $UID | cut -d: -f1) && usermod -l modelship -d /home/modelship -m "$existing"; fi

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

WORKDIR /modelship

# Absolute, never ~: a different runtime UID would move it off the environment
# bootstrap built.
ENV MSHIP_HOME=/opt/mship
ENV UV_TOOL_DIR=/opt/uv/tools
ENV UV_TOOL_BIN_DIR=/opt/uv/bin
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_CACHE_DIR=/opt/uv/cache

ENV MSHIP_CACHE_DIR=/.cache
ENV MSHIP_NODE_CACHE_DIR=/opt/mship/node-cache
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV MSHIP_USE_EXISTING_RAY_CLUSTER=false
ENV MSHIP_METRICS=true
ENV RAY_METRICS_EXPORT_PORT=8079
ENV MSHIP_LOG_LEVEL=INFO
ENV MSHIP_LOG_FORMAT=text

# On PATH, not just in the entrypoint: KubeRay injects its own `ray start`.
ENV PATH="${MSHIP_HOME}/envs/${MSHIP_VARIANT}/.venv/bin:${UV_TOOL_BIN_DIR}:$PATH"

RUN mkdir -p /.cache /opt/mship/node-cache /opt/uv && \
    chown -R $UID:$GID /modelship /.cache /opt/mship /opt/uv

# =============================================================================
# builder — the native install, against local wheels instead of PyPI. Build
# toolchain for wheels compiled from source; not inherited by prod.
# =============================================================================
FROM base AS builder

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG MSHIP_VARIANT
ARG MSHIP_VERSION
ARG UID
ARG GID

# First, so a missing arg fails in seconds rather than as an opaque `mship==`
# resolution error after the apt layers.
RUN [ -n "${MSHIP_VERSION}" ] || { \
    echo "error: this target needs --build-arg MSHIP_VERSION=<version> and"; \
    echo "       --build-context wheels=<dir>. See docs/development.md."; \
    exit 1; }

RUN if [ "$MSHIP_VARIANT" = "cuda" ]; then \
    apt-get update -y && \
    apt-get install -y --no-install-recommends gnupg && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
        | gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
        > /etc/apt/sources.list.d/cuda.list && \
    rm -rf /var/lib/apt/lists/*; \
    fi

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git && \
    if [ "$MSHIP_VARIANT" = "cuda" ]; then \
    CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH}; \
    fi && \
    rm -rf /var/lib/apt/lists/*

USER modelship

# UV_FIND_LINKS resolves the wheels locally without changing the requirement the
# bootstrapper composes, so the pins fingerprint still records a version.
# --python pins the tool env to the engine's interpreter: one managed CPython.
RUN --mount=type=cache,target=/opt/uv/cache,uid=$UID,gid=$GID \
    --mount=from=wheels,target=/wheels \
    UV_FIND_LINKS=/wheels uv tool install --python ${PYTHON_VERSION} "mship==${MSHIP_VERSION}"

# Installs the engine env + llama.cpp and records the variant, so `mship deploy`
# takes no variant flag at runtime.
RUN --mount=type=cache,target=/opt/uv/cache,uid=$UID,gid=$GID \
    --mount=from=wheels,target=/wheels \
    UV_FIND_LINKS=/wheels mship bootstrap --${MSHIP_VARIANT}

# =============================================================================
# dev — deps from the lockfile into /.venv, project never installed. Branches
# off base, so it needs neither the release wheels nor a version.
# =============================================================================
FROM base AS dev

ARG PYTHON_VERSION
ARG MSHIP_VARIANT
ARG UID
ARG GID

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git && \
    rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV VIRTUAL_ENV=/.venv
ENV PATH="/.venv/bin:$PATH"

RUN mkdir -p /.venv && chown -R $UID:$GID /.venv

USER modelship

RUN --mount=type=cache,target=/opt/uv/cache,uid=$UID,gid=$GID \
    uv python install ${PYTHON_VERSION}
RUN --mount=type=cache,target=/opt/uv/cache,uid=$UID,gid=$GID \
    uv venv

ADD --chown=$UID:$GID ./pyproject.toml pyproject.toml
ADD --chown=$UID:$GID ./README.md README.md
ADD --chown=$UID:$GID ./uv.lock uv.lock
ADD --chown=$UID:$GID ./Makefile Makefile

RUN --mount=type=cache,target=/opt/uv/cache,uid=$UID,gid=$GID \
    if [ "$MSHIP_VARIANT" = "cpu" ]; then \
        uv sync --locked --no-install-project --extra dev --extra $MSHIP_VARIANT --extra vllm-cpu; \
    else \
        uv sync --locked --no-install-project --extra dev --extra $MSHIP_VARIANT; \
    fi

USER root

ENTRYPOINT ["/modelship/scripts/entrypoint.sh"]

# =============================================================================
# prod — the bootstrapped environment, nothing else. No source tree: modelship
# is installed from the release wheel like any other package.
# =============================================================================
FROM base AS prod

ARG UID
ARG GID

COPY --from=builder --chown=$UID:$GID /opt/uv /opt/uv
COPY --from=builder --chown=$UID:$GID /opt/mship /opt/mship

ADD --chown=$UID:$GID ./scripts scripts
# Examples only: config/models.yaml is gitignored, and mounted over at runtime.
ADD --chown=$UID:$GID ./config/examples config/examples

USER root

# Prepends the command, so `docker run <image> deploy --config …` reaches the
# same CLI as a native install.
ENTRYPOINT ["/modelship/scripts/entrypoint.sh", "mship"]
