# A minimal application image for SandboxFusion service
# - Based on the prebuilt server runtime image (languages, toolchains preinstalled)
# - Copies the current repo code in, installs Python deps with poetry
# - Defaults to lite isolation (SANDBOX_CONFIG=ci). For hosts without working cgroup v1
#   you can set SANDBOX_LITE_NO_CGROUP=1 at runtime to skip cgroup while keeping overlay+netns.

FROM vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # default to lite isolation
    SANDBOX_CONFIG=ci \
    HOST=0.0.0.0 \
    PORT=8080

# Optional small utilities; image already contains most toolchains
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git tmux zsh htop net-tools iproute2 traceroute && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root/SandboxFusion

# Install Python deps (use layers for caching)
COPY pyproject.toml poetry.lock ./
RUN python3 -m pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    # some base images expect this file to exist when using conda python
    (touch /root/miniconda3/pyvenv.cfg || true) && \
    poetry install --no-interaction --no-ansi

# Add the rest of the source
COPY . .

# Avoid static-mount error if docs aren't built in this image
RUN mkdir -p docs/build

EXPOSE 8080

# Start the FastAPI server; honors HOST/PORT envs
CMD ["bash", "/root/SandboxFusion/run.sh"]

