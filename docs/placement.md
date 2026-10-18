# Memory and placement

[Back to the README](../README.md)

## Memory and placement

A model's full size does not have to fit in RAM or VRAM. jitllm keeps what is
needed resident and re-reads what it evicted from the container.

```sh
# Every GPU, with the CPU running the blocks that are not placed.
./jitllm run -devices all -chat models/smollm2.jlm "Hello"

# A 4 GiB host weight budget.
./jitllm run -devices cpu -maxmem 4G models/model.jlm "Hello"

# A budget per device; jitllm hardware lists the device ids.
./jitllm run -devices cuda:0=3G,vulkan:1=8G models/model.jlm "Hello"

# Stream every block (*) through cuda:0's slots (~); ! would pin it instead.
./jitllm run -devices cuda:0=2G -placement '*=cuda:0~' models/model.jlm "Hello"

# Two devices, each holding a range of blocks. Name both in -devices; a block
# (its attention, KV history and FFN) runs on one device. The terminal chat
# (Settings, "Block placement") and the desktop app (Devices, "Block
# placement") take the same map.
./jitllm run -devices cuda:0,vulkan:0 -placement 0-15=cuda:0,16-31=vulkan:0 models/model.jlm "Hello"

# Reuse the KV cache of earlier prompts that share a prefix.
./jitllm run -chat -kv-cache models/kv-cache -n 128 models/smollm2.jlm "Explain a solar panel."
```

| Flag | Meaning |
|---|---|
| `-maxmem BYTES` | host weight residency budget (default: what the cgroup or the machine has, less headroom) |
| `-vram BYTES`, `-devices dev=BYTES` | device weight budgets (default: what each device reports free, less headroom) |
| `-placement MAP` | where each block and the head run, per block or range |
| `-gpu-layers N` | cap the blocks offered to the devices |
| `-gpu-grow` | start on the CPU and move blocks onto the device while serving |
| `-relocate` | when the device cannot grow the context, hand its last block to the host |
| `-tune-seam` | measure whether fewer device blocks are faster, and move the seam |
| `-kv-cache DIR`, `-kv-cache-max BYTES` | the prefix cache and its size (default 8 GiB) |

Shared CPU/GPU memory (an integrated GPU, Apple silicon) is counted once.
Paging trades memory for reads, so a fast SSD helps when weights exceed RAM.
