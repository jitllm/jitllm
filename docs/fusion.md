# Kernel fusion and block bridging

A decode token is a few hundred small operations; on a small model the cost of
starting each one is a large share of the token. jitllm removes it two ways:
**fusion** joins operations into fewer kernels or dispatches, and **bridging**
runs consecutive blocks on one device as one unit. The website page
(`website/src/content/docs/docs/concepts/fusion.mdx`) is the interactive
version.

## The unit is a block

A block is the unit of placement and relocation: it moves host <-> device,
between devices, and in and out of memory through the pager. So fusion never
crosses a block's device boundary, and bridging follows placement.

```
block     0  1  2   3  4  5  6  7   8  9 10 11   head
device    CPU CPU   GPU1 .........  GPU2 ......   CPU
runs      [-----]   [-----------]   [---------]   [--]   head joins a host run only if adjacent

evict 5   [-----]   [----] 5 [--]   [---------]   [--]   5 is read first: the GPU1 run splits
```

## Why: dispatch cost

| where | cost | host |
|---|---|---|
| CPU | ~2.7 us per pool region (`TestFusionCeiling`) plus the slowest-worker barrier | Linux x86-64 |
| CPU | regions per decode token: Llama-3.2-1B 218, OLMoE-1B-7B 234, Qwen3.5-0.8B 354, gemma-3-1b 353 (Q4_K_M) | counted |
| Metal | ~316 barriers among ~370 dispatches per decode token | Ministral-3B, Apple M4 |
| CUDA | ~0.6 us per launch inside a captured graph, so fusion is not the main decode lever | V100, T4 |

## Fusion

```
per op:  [door|work..|wait][door|work....|wait][door|work.|wait] ...   218 regions
fused:   [door] work.. work.... work. ...                              1 region, 218 phases
```

1. Pairwise kernels: the norm writes the matmul's quantized input (shipped);
   activation times gate writes the down projection's input (shipped); on CUDA
   q and k rotation, the k page write and the v copy as one launch (shipped;
   `TestRoPEQKVPagedMatchesItsParts`, `TestFusedDecodeMatchesPerOp`,
   `jitllm verify -ab fuse`).
2. Fused host regions (`sched.Pool.Fuse`, `engine/fuse.go`): workers join
   once, each op is a phase published by one store, completion counted in
   chunks; same kernels, chunks and order, so the same logit bits (in progress,
   not yet on main).
3. IR collectives: subgroup reduce, broadcast and shuffle lowered to PTX,
   SPIR-V, MSL and AMDGPU, and tensor-core tiles on `mma.sync`, so fused
   kernels pass data between threads in registers (shipped in the IR; the
   fused kernels that use them are to come).

## Bridging

- Host: one fused region per token across the resident host run plus the head
  (in progress). Nothing is compiled for a run, so after a relocation, eviction
  or seam move the next token follows the new placement.
- CUDA: the device's resident range is one submission captured as a graph,
  re-captured when the range changes (shipped).
- A block on another device, or a block that must be paged in, ends a run.

## What it keeps

- Per-op kernels are the reference. On main, the CUDA writer is gated
  kernel-to-parts (`TestRoPEQKVPagedMatchesItsParts`) and whole-decode against
  per op, every block placed and on a partial seam
  (`TestFusedDecodeMatchesPerOp`). On the host branch, `TestFusedBlocksMatchPerOp`
  (same logit bits, every principle fixture, resident and paged), run against
  violations: `TestFusedPhaseOrder`, and workers that drop chunks fail the
  model gate at token 0.
- The five principles' gates (AGENTS.md) run with fusion on. A crew shared with
  another session, a NUMA pool, prefill and the batched step are not fused, by
  choice.
- `jitllm verify -ab fuse` times CUDA's fused launches against their parts;
  on the host branch, `JITLLM_FUSE=0` runs decode one region per op.

Throughput from fusion is being measured; no speedup is claimed here yet.
