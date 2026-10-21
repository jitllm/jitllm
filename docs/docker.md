# Docker

The image runs `jitllmd`, the daemon, with the `jitllm` CLI beside it. Models
are not in it: they live in a volume mounted at `/models`.

| | |
|---|---|
| image | `ghcr.io/jitllm/jitllm:<version>` and `:latest`, linux/amd64 and linux/arm64 |
| entrypoint | `jitllmd serve -addr :4242`; arguments to `docker run` are appended as more `serve` flags |
| port | 4242: the OpenAI API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`), the Anthropic API (`/v1/messages`), the ConnectRPC control plane and `/healthz` |
| volume | `/models`, scanned by the server and written by `jitllm convert` (`JITLLM_MODELS=/models`) |
| user | `jitllm`, uid and gid 10001, not root |
| health | `jitllmd models -loaded`, the daemon's own client asking the server over the control plane |
| base | `debian:trixie-slim` with `ca-certificates` and the Vulkan loader (`libvulkan1`) |

Why Debian and not a distroless static base: the binaries are pure Go
(`CGO_ENABLED=0`) but not static. CUDA and Vulkan are opened at run time
through goffi (`jit_old/gpu/ffi`), which links the binary against glibc's dynamic
loader, so the image needs glibc, and it needs `ldconfig` for the NVIDIA
Container Toolkit to register the driver libraries it mounts.

## Build

    docker build -t jitllm .

Both platforms, with buildx (the Go compile is a cross-build on the build
host; only the arm64 image's `apt-get` step runs under emulation, so
`docker run --privileged --rm tonistiigi/binfmt --install arm64` first on an
amd64 host without it):

    docker buildx build --platform linux/amd64,linux/arm64 -t jitllm .

A build with a docker-container builder keeps the result in the build cache
unless it is given `--push` or, for one platform, `--load`. `VERSION` sets
what `jitllm version` reports: `--build-arg VERSION=v0.1.0`.

## Models

A model is converted once into a `.jlm` container in the volume. By name from
the library (`jitllm library` lists the names), from a Hugging Face URL, or
from a GGUF already in the volume:

    docker volume create jitllm-models
    docker run --rm -v jitllm-models:/models --entrypoint jitllm \
      ghcr.io/jitllm/jitllm convert qwen3-30b-a3b

A gated Hub repository needs a token: `-e HF_TOKEN=...`. `HF_ENDPOINT` points
the download at a mirror.

A named volume is the simple choice: Docker gives a new one the image's
ownership of `/models`, so uid 10001 can write it. A bind mount of a host
directory has to be writable by uid 10001 (`chown 10001:10001 DIR`), or the
container runs as your own user (`--user "$(id -u):$(id -g)"`, rootful
Docker). Under rootless Docker container uids map to subordinate ids on the
host, so a bind mount wants a named volume or `--user 0:0` (which is you).
On Docker Desktop a bind mount crosses the VM's file sharing, which may refuse
`O_DIRECT`; the pager then reads through the page cache and stays correct, but
a named volume is faster.

## Run on the CPU

    docker run -d --name jitllm -p 4242:4242 -v jitllm-models:/models \
      ghcr.io/jitllm/jitllm -load Qwen3-30B-A3B-Q4_K_M.jlm -id qwen3

    curl localhost:4242/v1/models
    curl localhost:4242/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen3","messages":[{"role":"user","content":"What is the capital of France?"}]}'

`-load` is optional: a model can be loaded into a running server with the
same image's client, `docker exec jitllm jitllmd models -load NAME.jlm`, and
`jitllmd models` lists what is on disk and what is loaded. A base model with
no chat template answers `/v1/completions`, not `/v1/chat/completions`.

The engine sizes itself from what the container is given:

- `--cpuset-cpus` is the core set. The worker pool is built from the
  process's affinity, so on a hybrid CPU pin the performance cores
  (`--cpuset-cpus 0,2,4,6,8,10` on a six-P-core Intel part): decode is
  limited by memory bandwidth, and E-cores and hyperthreads add stragglers,
  not bandwidth.
- `--memory` is the page budget. With no `-maxmem`, the budget is eight tenths
  of the cgroup's limit (or of `MemAvailable` when that is lower), so a model
  larger than the limit pages in and out of the container file rather than
  being killed.

The container listens on every interface (`-addr :4242`), as a container must,
so `serve` prints its no-key warning. Publish the port on loopback
(`-p localhost:4242:4242`) or set a key with `-e JITLLM_API_KEY=...`; the
health check then sends the same key.

Change the published port with `-p 9000:4242` rather than `-addr`: the health
check asks for port 4242.

Every `serve` flag is available (`docker run --rm ghcr.io/jitllm/jitllm -h`).
The whole CLI is there too, for example a one-off decode:

    docker run --rm -v jitllm-models:/models --entrypoint jitllm \
      ghcr.io/jitllm/jitllm run -chat Qwen3-30B-A3B-Q4_K_M.jlm "What is the capital of France?"

## Run on an NVIDIA GPU

The image carries no CUDA library; the driver's are mounted when the
container starts. The host needs the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
registered with Docker (`sudo nvidia-ctk runtime configure --runtime=docker`,
then restart Docker). Then add `--gpus all`:

    docker run -d --name jitllm --gpus all -p 4242:4242 -v jitllm-models:/models \
      ghcr.io/jitllm/jitllm -load Qwen3-30B-A3B-Q4_K_M.jlm -id qwen3 -devices cuda

    docker run --rm --gpus all --entrypoint jitllm ghcr.io/jitllm/jitllm hardware

`hardware` should list the card. What the toolkit mounts is chosen by
`NVIDIA_DRIVER_CAPABILITIES`, which the image sets to
`compute,utility,graphics`:

- `compute`: `libcuda.so.1`, which jitllm opens, and the driver's PTX
  compiler, which turns the PTX jitllm generates into machine code. No CUDA
  toolkit, runtime or NVRTC is needed.
- `utility`: `nvidia-smi`.
- `graphics`: NVIDIA's Vulkan driver and its ICD file, so `-devices vulkan`
  reaches the same card through the loader the image ships. The card is one
  device under two backends (matched by UUID), and the default picks CUDA.
  `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility` leaves CUDA alone.

`--gpus '"device=1"'` (or `NVIDIA_VISIBLE_DEVICES=1`) exposes one card;
Podman and Kubernetes with CDI use `--device nvidia.com/gpu=all` instead of
`--gpus`.

## Vulkan on AMD and Intel GPUs

The image ships the Vulkan loader but no Mesa driver: NVIDIA's comes with the
toolkit, and Mesa's would add its LLVM to every image. For an AMD or Intel GPU,
extend the image with Mesa's drivers and pass the render node:

    FROM ghcr.io/jitllm/jitllm
    USER root
    RUN apt-get update \
     && apt-get install -y --no-install-recommends mesa-vulkan-drivers \
     && rm -rf /var/lib/apt/lists/*
    USER 10001:10001

    docker run -d --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
      -p 4242:4242 -v jitllm-models:/models my-jitllm -load NAME.jlm -devices vulkan

Mesa's package also carries lavapipe, a Vulkan device that runs on the CPU. It
exercises the Vulkan path on a machine with no GPU and is slower than the
engine's own CPU kernels, so it is not a way to serve.

On macOS, Docker runs Linux in a virtual machine with no access to Metal: the
container is a CPU deployment there. Run the native binary for Metal.

## Environment

| variable | in the image | meaning |
|---|---|---|
| `JITLLM_MODELS` | `/models` | the model directory: `serve -models`' default and where `jitllm convert` writes |
| `HF_TOKEN` | unset | Hugging Face token for gated repositories (`HUGGING_FACE_HUB_TOKEN` also read) |
| `HF_ENDPOINT` | unset | a Hub mirror |
| `NVIDIA_VISIBLE_DEVICES` | `all` | which GPUs the NVIDIA toolkit exposes (read only by the toolkit) |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,graphics` | which driver libraries the toolkit mounts |

Every other `JITLLM_*` knob the CLI reads is passed the same way, with `-e`.

## Release and CI

`.goreleaser.yaml`'s `dockers_v2` builds the image from this Dockerfile with
`BIN=dist`, copying the binaries the release archives carry instead of
compiling again, and pushes `ghcr.io/jitllm/jitllm:<version>` for both
platforms under one manifest. The release workflow moves `latest` to it only
after the archives pass their smoke runs, and never to a prerelease. CI's
`docker` job builds both platforms without pushing, then runs
`.github/scripts/docker-smoke.sh` against the amd64 image: convert the
committed stories260K inside the container, serve it, and check the text of a
greedy completion. The same script runs locally:

    docker build -t jitllm:local . && .github/scripts/docker-smoke.sh jitllm:local
