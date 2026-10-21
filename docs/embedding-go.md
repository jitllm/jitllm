# Go library and desktop app

[Back to the README](../README.md)

## Embed it in Go

The CLI, server and desktop app use the same engine. The core module's one
dependency is `goffi`.

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
    state := m.NewState(8192)
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

| API | What it controls |
|---|---|
| `model.Open(path, opts...)` | open a container; `With...` options set every tuner and budget |
| `Model.SetPageBudget` | host weight residency |
| `State.SetGPULayers`, `State.SetDevice` | move execution between devices, with the history |
| `State.SetKVStore`, `State.PrefillCached` | KV spill and prefix reuse |
| `State.ShareKV`, `model.NewSharedStore` | one copy of a prompt prefix's pages across sessions |
| `State.Park`, `State.Resume` | preemption: a session's history home and out to a store, then back |
| `model.WithKVType` | the KV cache's format: f32, f16 or q8_0 |
| `Model.NewEmbedder` | embeddings |
| `model.Scheduler`, `model.Step`, `model.StepRuns` | several sequences or sessions in one batched step, on a device or the host |

See the [model API](../engine/model/) and the [device tier](../jit_old/gpu/tier/).

The desktop app (model downloads, conversion, text and image chat, memory and
device views) is a separate module in [ui/](../ui/). Install it from a release
with `scripts/install.sh desktop` ([README](../README.md#install)), or build it:

```sh
(cd ui && CGO_ENABLED=0 go build -o ../jitllm-desktop .)
./jitllm-desktop
```
