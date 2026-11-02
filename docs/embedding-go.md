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
    // An instance is the model's kernels composed for one placement; it
    // runs one conversation of its own here.
    in := m.NewInstance(8192)
    defer in.Close()

    logits, err := in.Prefill(ids)
    for err == nil && in.Pos() < 8192 {
        id := model.Greedy(logits)
        if m.Vocab.IsEOG(id) {
            break
        }
        fmt.Print(m.Vocab.Decode([]int32{id}))
        logits, err = in.Forward(id)
    }
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println()
}
```

Serving many conversations is the same model, an instance served to them
and a session each: a session is a conversation's history, opening one
composes nothing, and it leases an instance for each phase of a turn.

```go
package main

import (
    "context"
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

    // The model's kernels, composed once for this placement (SetDevice to
    // move it onto a card), served to every session.
    in := m.NewInstance(8192)
    defer in.Close()
    if err := m.Serve(in, 0); err != nil {
        log.Fatal(err)
    }

    s, err := m.NewSession(8192)
    if err != nil {
        log.Fatal(err)
    }
    defer s.Close()
    ids := m.Vocab.Encode("The sky is blue because", true)
    ctx := context.Background()
    var logits []float32
    if err := s.Run(ctx, model.PhasePrefill, func(in *model.Instance) error {
        logits, err = in.Prefill(ids)
        return err
    }); err != nil {
        log.Fatal(err)
    }
    for n := 0; n < 64; n++ {
        id := model.Greedy(logits)
        if m.Vocab.IsEOG(id) {
            break
        }
        fmt.Print(m.Vocab.Decode([]int32{id}))
        if err := s.Run(ctx, model.PhaseGenerate, func(in *model.Instance) error {
            logits, err = in.Forward(id)
            return err
        }); err != nil {
            log.Fatal(err)
        }
    }
    fmt.Println()
}
```

| API | What it controls |
|---|---|
| `model.Open(path, opts...)` | open a container; `With...` options set every tuner and budget |
| `Model.SetPageBudget` | host weight residency |
| `Model.NewInstance`, `Instance.SetGPULayers`, `Instance.SetDevice` | compose an instance and move its execution between devices, every session's history with it |
| `Model.NewSession`, `Session.Run`, `Session.Acquire`, `Session.Release` | a conversation's history, leased an instance per phase |
| `model.Scheduler`, `model.Pool`, `model.WithScheduler` | which instance a session leases for a phase: the default pool, or your own |
| `Session.SetKVStore`, `Instance.PrefillCached` | KV spill and prefix reuse |
| `Instance.ShareKV`, `model.NewSharedStore` | one copy of a prompt prefix's pages across sessions |
| `Instance.Park`, `Instance.Resume` | preemption: a session's history home and out to a store, then back |
| `model.WithKVType` | the KV cache's format: f32, f16 or q8_0 |
| `Model.NewEmbedder` | embeddings |
| `model.Batcher`, `model.StepRuns` | continuous batching: many sessions' rows in one step of an instance, on a device or the host |

See the [model API](../engine/model/) and the [device tier](../jit/gpu/tier/).

The desktop app (model downloads, conversion, text and image chat, memory and
device views) is a separate module in [ui/](../ui/). Install it from a release
with `scripts/install.sh desktop` ([README](../README.md#install)), or build it:

```sh
(cd ui && CGO_ENABLED=0 go build -o ../jitllm-desktop .)
./jitllm-desktop
```
