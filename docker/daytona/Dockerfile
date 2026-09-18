# The Daytona GPU host for pi/ arms — an Ollama box with the weights already in it.
#
# This is the stereOS idea applied to Daytona: bake the setup into the image so a
# sandbox is usable the second it boots. `scripts/daytona_host.py snapshot` builds
# it server-side (no local Docker) and appends one `ollama pull` per requested tag,
# so the pull — the only slow step, ~11 min for 65 GB — happens once at build time
# instead of on every `up`. The snapshot carries the GPU reservation; the image
# itself holds no ROM and no credentials.
#
# The `ollama` runtime that install.sh drops here bundles its CUDA userland; the
# driver comes from the host at sandbox launch, the same way the ollama/ollama
# image works. Weights live in /models (OLLAMA_MODELS) rather than a home
# directory because the sandbox user is not necessarily the build user.
FROM --platform=linux/amd64 ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    OLLAMA_MODELS=/models \
    OLLAMA_HOST=0.0.0.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && mkdir -p /models && chmod -R a+rwX /models \
    && ollama --version

LABEL org.opencontainers.image.source="https://github.com/pcc-labs/tetris" \
      com.papercompute.role="tetris-ollama-host"

# The sandbox is a box to exec into, not a service: `up` starts `ollama serve`
# itself so its log is where the driver can read the GPU line back.
CMD ["sleep", "infinity"]
