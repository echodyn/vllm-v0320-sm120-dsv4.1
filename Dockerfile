ARG BASE_IMAGE=vllm/vllm-openai:v0.32.0
FROM ${BASE_IMAGE}

COPY patch_sm120.py /opt/vllm-sm120-backports/
COPY patches/ /opt/vllm-sm120-backports/patches/
COPY dependencies/lock.json /opt/vllm-sm120-backports/dependencies/lock.json
COPY scripts/install_dependencies.py scripts/download_wheels.py scripts/check_runtime.py /opt/vllm-sm120-backports/scripts/

ARG PYPI_MIRROR=official
ARG PYPI_INDEX_URL=""
ARG GITHUB_PROXY_URL=""
ARG KERNEL_WHEEL_BASE_URL=""

# Validate the bundled sources before downloading. No host wheel directory is needed.
# The cache survives failed builds for resuming, is serialized across builds, and
# is excluded from the resulting image layer. Every reused wheel is rehashed.
RUN --mount=type=cache,id=vllm-v0320-sm120-wheels,target=/opt/vllm-sm120-backports/dependencies/wheels,sharing=locked \
    python3 /opt/vllm-sm120-backports/patch_sm120.py --check \
    && python3 /opt/vllm-sm120-backports/scripts/download_wheels.py --runtime-target \
       --pypi-mirror "${PYPI_MIRROR}" \
       --pypi-index-url "${PYPI_INDEX_URL}" \
       --github-proxy-url "${GITHUB_PROXY_URL}" \
       --kernel-wheel-base-url "${KERNEL_WHEEL_BASE_URL}" \
    && cp /opt/vllm-sm120-backports/dependencies/wheels/manifest.json \
          /opt/vllm-sm120-backports/dependency-downloads.json \
    && python3 /opt/vllm-sm120-backports/scripts/install_dependencies.py --install \
       --wheel-dir /opt/vllm-sm120-backports/dependencies/wheels \
    && python3 /opt/vllm-sm120-backports/patch_sm120.py \
    && python3 /opt/vllm-sm120-backports/patch_sm120.py --verify \
    && python3 /opt/vllm-sm120-backports/scripts/install_dependencies.py --verify

LABEL org.opencontainers.image.title="vLLM 0.32.0 SM120 DeepSeek V4.1 Flash backports (online build)" \
      org.opencontainers.image.source="https://github.com/echodyn/vllm-v0320-sm120-dsv4.1" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vllm.backports.baseline="5df63a7203cdb9328d824a27b682e77c7b9928be" \
      ai.vllm.backports.reviewed="2026-10-28"
