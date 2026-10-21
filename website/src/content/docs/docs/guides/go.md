---
title: Embed it in Go
description: Use the engine as a Go library, with the same model format and placement control as the CLI and server.
---

The CLI, the server and the desktop app are all thin layers over the same packages. Embedding jitllm means importing them: the engine runs inside your process, and its only dependency is [`goffi`](https://github.com/go-webgpu/goffi).

```sh
go get github.com/jitllm/jitllm
```

## Generate text

```go
package main

import (
	"fmt"
	"log"

	"github.com/jitllm/jitllm/engine/model"
)

func main() {
	m, err := model.Open("models/qwen3.jlm")
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()

	ids, err := m.ChatIDs([]model.ChatMessage{
		{Role: "user", Content: "Why is the sky blue?"},
	}, true)
	if err != nil {
		log.Fatal(err)
	}

	state := m.NewState(8192) // KV capacity, in positions
	defer state.Close()

	logits, err := state.Prefill(ids)
	for err == nil && state.Pos() < 8192 {
		id := model.Greedy(logits)
		if m.Vocab.IsEOG(id) {
			break
		}
		fmt.Print(m.Vocab.Decode([]int32{id}))
		logits, err = state.Forward(id)
	}
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}
```

A `Model` is the weights, shared and read-only. A `State` is one conversation: its KV cache, its position, and where its blocks run. Open a model once and give each conversation its own state.

## Sample

`model.Sampler` has llama.cpp's knobs:

```go
s := &model.Sampler{Temp: 0.7, TopP: 0.9, MinP: 0.05, RepeatPen: 1.1, Seed: 42}
for _, t := range ids {
	s.Observe(t) // the repeat penalty sees only observed tokens
}
id := s.Sample(logits)
s.Observe(id)
```

A zero `Sampler` (or `model.Greedy`) is greedy. Call `Observe` for each prompt and generated token when you use `RepeatPen`.

## Chat templates and tools

`ChatIDs` renders messages through the model's own template and tokenizes the result; `ChatPrompt` returns the rendered text instead. The `...Tools` variants take tool definitions as JSON and hand them to the template unchanged, and `model.ParseToolCalls` reads the calls back out of the reply.

## Put it on a GPU

Open the devices with the device tier and hand them to a state. The spec is the same [device grammar](/docs/guides/devices/#the-device-grammar) the CLI takes.

```go
import "github.com/jitllm/jitllm/jit_old/gpu/tier"

g, err := tier.OpenWith(tier.WithDevices("auto"))
if err != nil {
	log.Printf("no GPU (%v); running on the CPU", err)
} else {
	defer g.Close()
	state.SetDevice(g) // place as many blocks as fit
	// or: state.SetDeviceLayers(g, 20)
}
```

Placement can change between tokens without losing the conversation:

```go
state.SetGPULayers(12) // move the seam; KV and recurrent state follow the blocks
```

To say where particular blocks go, give the model a [placement map](/docs/guides/devices/#placement-maps) when you open it. It applies whenever a state attaches a device:

```go
p, err := model.ParsePlacement("0-7=cuda:0!,head=cuda:0,*=host", false)
if err != nil {
	log.Fatal(err)
}
m, err := model.Open("models/m.jlm", model.WithPlacement(p))
```

## Control memory

```go
m, err := model.Open("models/big.jlm", model.WithPageBudget(8<<30))

// Later, between tokens:
m.SetPageBudget(4 << 30)
```

Every decision the engine takes on its own (budgets, placement, tuning, cores, KV width) has a `With...` option on `model.Open`. The libraries read no environment variables; the `jitllm` command reads a few and turns them into options.

## Several conversations at once

- `model.Step(states, tokens)` advances several states by one token each and returns each one's logits. When they share a model and a device that holds every block and the head, it runs them as rows of one step, so each block's weights are read once for all of them; states of one model wholly on the CPU go as one host pass over the weights the same way. Otherwise (a state split between the CPU and a device, or one `State.HostRefusal` names) it steps them one after another, with the same answer. On a V100, eight Llama-3.1-8B sessions decode 4.6× faster together than in turn. `jitllm speed -sessions N` measures it.
- `State.ForwardBatch` and `ForwardBatchGreedy` step many sequences of one state at once, which is what `jitllm batch` measures; `model.Scheduler` admits and retires sequences on batch rows.

`model.StepRuns` is the general form, a token for some sessions and a prompt chunk for others in one step; it is what `jitllmd` batches requests with. See [Serve an API](/docs/guides/serve/#batching).

## Prefix reuse and long context

- `State.SetKVStore` and `SetCacheKey` give a state a backing store for its KV pages, and `State.PrefillCached` then skips whatever prefix of a prompt the store already holds. This is what `-kv-cache` uses.
- `State.SetKVBudget(bytes)` caps the history held in memory. Older pages spill to the store and come back when attention reads them, so a context longer than memory still runs. It needs a store first; without one the cap would drop history instead of spilling it, so it is refused.
- `State.ShareKV(store)` with a `model.NewSharedStore()` makes sessions that share a prompt prefix hold its pages once, copy-on-write. Batches, recurrent models, DeepSeek V4 and a session with blocks on a device refuse it by name.
- `State.Park` brings a session's blocks, KV pages and recurrent state home and evicts its sealed pages to a store; `State.Resume` faults them back, byte for byte. `jitllmd`'s engine parks idle sessions this way under a model's KV budget (`server.Engine.SetKVBudget`).
- `model.WithKVType(model.KVQ8_0)` (or `KVF16`, `KVF32`) forces the KV cache's format; q8_0 runs on every host tier and is carried to a device as float32. MLA, the lightning indexer, MSA and DeepSeek V4 refuse q8_0 by name. The `jitllm` command reads it from `JITLLM_KV_TYPE`.
- `State.PrefillMixed` takes spans of tokens and image embeddings, for a vision model; `m.Tower()` encodes the image.

## Embeddings

```go
e, err := m.NewEmbedder()
if err != nil {
	log.Fatal(err)
}
defer e.Close()
vec, err := e.EmbedText("a sentence to embed") // pooled and L2-normalised
```

## Run a server in process

The server package is a separate module, `github.com/jitllm/jitllm/server`. `server.New` returns an engine, `LoadModel` loads into it, and `Handler()` is an `http.Handler` serving the OpenAI, Anthropic and Connect APIs, to mount on your own mux.
