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

    "github.com/jitllm/jitllm/engine"
    "github.com/jitllm/jitllm/model"
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
    in, err := engine.NewInstance(m, 8192)
    if err != nil {
        log.Fatal(err)
    }
    defer in.Close()

    // Sample with no sampler is the greedy token, drawn by the CPU
    // device's composed head.
    logits, err := in.Prefill(ids)
    for err == nil && in.Pos() < 8192 {
        var id int32
        if id, err = in.Sample(nil, logits); err != nil || m.Vocab.IsEOG(id) {
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

    "github.com/jitllm/jitllm/engine"
    "github.com/jitllm/jitllm/model"
)

func main() {
    m, err := model.Open("models/qwen3.jlm")
    if err != nil {
        log.Fatal(err)
    }
    defer m.Close()

    // The model's kernels, composed once for this placement (SetDevice to
    // move it onto a card), served to every session by the pool, which
    // closes them.
    in, err := engine.NewInstance(m, 8192)
    if err != nil {
        log.Fatal(err)
    }
    pool := engine.NewPool()
    defer pool.Close()
    if err := pool.Serve(in, 0); err != nil {
        log.Fatal(err)
    }

    s, err := pool.NewSession(8192)
    if err != nil {
        log.Fatal(err)
    }
    defer s.Close()
    ids := m.Vocab.Encode("The sky is blue because", true)
    ctx := context.Background()
    // Each phase draws its next token on the instance it leased.
    var id int32
    if err := s.Run(ctx, engine.PhasePrefill, func(in *engine.Instance) error {
        logits, err := in.Prefill(ids)
        if err != nil {
            return err
        }
        id, err = in.Sample(nil, logits)
        return err
    }); err != nil {
        log.Fatal(err)
    }
    for n := 0; n < 64 && !m.Vocab.IsEOG(id); n++ {
        fmt.Print(m.Vocab.Decode([]int32{id}))
        if err := s.Run(ctx, engine.PhaseGenerate, func(in *engine.Instance) error {
            logits, err := in.Forward(id)
            if err != nil {
                return err
            }
            id, err = in.Sample(nil, logits)
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
| `engine.WithPageBudget`, `Instance.SetPageBudget` | an instance's page quota: the weight pages its devices keep for it |
| `engine.NewInstance`, `Instance.SetGPULayers`, `Instance.SetDevice` | compose an instance and move its execution between devices, every session's history with it |
| `engine.NewPool`, `Pool.Serve`, `Pool.NewSession`, `Session.Run`, `Session.Acquire`, `Session.Release` | a conversation's history, leased an instance per phase |
| `engine.Scheduler`, `engine.Pool`, `engine.WithScheduler` | which instance a session leases for a phase: the default pool, or your own |
| `Session.SetKVStore`, `Instance.PrefillCached` | KV spill and prefix reuse |
| `Instance.ShareKV`, `engine.NewSharedStore` | one copy of a prompt prefix's pages across sessions |
| `Instance.Park`, `Instance.Resume` | preemption: a session's history home and out to a store, then back |
| `engine.WithKVType` | the KV cache's format: f32 or f16 |
| `engine.NewEmbedder` | embeddings |
| `engine.Batcher`, `engine.StepRuns` | continuous batching: many sessions' rows in one step of an instance, on a card or the CPU device |

See the [model](../model/), the [engine](../engine/) and the [card tier](../engine/device/gpu/).

The desktop app (model downloads, conversion, text and image chat, memory and
device views) is a separate module in [ui/](../ui/). Install it from a release
with `scripts/install.sh desktop` ([README](../README.md#install)), or build it:

```sh
(cd ui && CGO_ENABLED=0 go build -o ../jitllm-desktop .)
./jitllm-desktop
```
