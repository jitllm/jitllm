# Memory budgets and device placement

Hold a model to a memory budget, see what the budget costs, and put blocks on
the devices you choose.

**Prerequisites:** `jitllm` and `jitllmd` on your `PATH`, `curl`. The GPU steps
need a GPU; everything else runs on any machine.

A model does not have to fit in RAM or VRAM. Its weights are pages of the
container: what fits stays resident, the rest is read from disk when a token
needs it. A budget below the model gives the same answer, more slowly.

## 1. What this machine has

```sh
jitllm hardware
```
<!-- test: expect memory -->

It lists the CPU cores jitllm decodes on, the memory, the code generation tier
and every GPU each backend sees (`cuda:0`, `vulkan:1`, `metal`, ...). Those ids
are what `-devices` takes.

<!-- test: instead cp "$JITLLM_REPO/testdata/models/stories260K.gguf" . -->
```sh
curl -fLo stories260K.gguf https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
```

```sh
mkdir -p models
jitllm convert stories260K.gguf models/stories260K.jlm
```
<!-- test: expect models/stories260K.jlm -->

## 2. A host budget below the model

`-maxmem` is the host weight budget. This one holds two of the model's five
blocks, so every token reads the rest from disk, and the answer is unchanged:

```sh
jitllm run -devices cpu -maxmem 400K -n 16 -temp 0 models/stories260K.jlm "Once upon a time"
```
<!-- test: expect there was a little girl named Lily -->
<!-- test: expect eviction(s) -->

```text
memory   budget 0.00 GiB   weights 0.00 GiB   ...   ★ OVER BUDGET by 0.00 GiB
pages    0 of 5 host block(s) resident at once ... -- 5 page-in(s) per token
Once upon a time, there was a little girl named Lily. She loved to play
pages    1 frame(s), 85 page-in(s), 84 eviction(s), ...
```

The `pages` lines are the cost: page-ins per token and the bytes read. On a
real model, a budget that holds the blocks a token touches is the difference
between memory speed and disk speed.

## 3. The same on a server, changed while it runs

<!-- test: background -->
```sh
jitllmd serve -addr 127.0.0.1:4242 -models models -load stories260K.jlm -id stories -devices cpu -maxmem 1M
```

```sh
for i in $(seq 60); do curl -sf http://127.0.0.1:4242/healthz && break; sleep 1; done
jitllmd devices -addr 127.0.0.1:4242
```
<!-- test: expect spendable -->

`jitllmd devices` is the memory that is actually spendable: the host budget plus
every device whose memory is not host memory (an integrated GPU's and Apple
silicon's are host memory, counted once).

Read the model's pager, then lower its budget without a restart:

```sh
jitllmd place -addr 127.0.0.1:4242 -model stories
jitllmd place -addr 127.0.0.1:4242 -model stories -maxmem 400K
curl -s http://127.0.0.1:4242/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"stories","prompt":"Once upon a time","max_tokens":16,"temperature":0}'
jitllmd stats -addr 127.0.0.1:4242
```
<!-- test: expect budget -->
<!-- test: expect there was a little girl named Lily -->
<!-- test: expect eviction(s) -->

## 4. Devices

`-devices` picks where blocks run; the host runs whatever is not placed.

<!-- test: skip needs a GPU -->
```sh
jitllm run -devices all models/model.jlm "Hello"                          # every GPU, fastest first
jitllm run -devices cuda:0=3G,vulkan:1=8G models/model.jlm "Hello"        # a budget per device
jitllm run -devices cuda:0 -gpu-layers 10 models/model.jlm "Hello"        # at most ten blocks on the card
jitllm run -devices cuda:0=2G -placement '*=cuda:0~' models/model.jlm "Hello"   # stream every block through the card
```

`jitllmd serve` and `jitllmd models -load` take the same `-devices`,
`-gpu-layers` and `-maxmem`. On a running server, `jitllmd place -session ID`
reports a session's seam and `-move FIRST:LAST -to host|DEVICE` relocates blocks
mid-conversation, with their KV history.

Every request's reply says where it ran:
`"jitllm":{"device_blocks":0,"cpu_blocks":5,...}`.

## When it does not work

| You see | Cause | Do this |
|---|---|---|
| `★ OVER BUDGET` and a slow token | the budget holds fewer blocks than a token reads | raise `-maxmem`, or accept disk speed |
| `device_blocks` is 0 with a GPU present | `-devices cpu`, a device that measured slower, a model too big for its free memory, or a software Vulkan device (declined) | `jitllm hardware` lists what was found; name the device with `-devices` and give it a budget |
| the GPU's free memory is tiny | another process holds the card | `jitllmd devices` shows free, not total; close the other process or lower `-vram` |
| out-of-memory kills the process | the cgroup's limit is below what jitllm was told | lower `-maxmem` to the cgroup's limit less the KV cache |
| a 200 long-context request is slow | KV pages grow with the context and may push blocks home (`-relocate`) | see `jitllmd stats` for page-ins and evictions |
