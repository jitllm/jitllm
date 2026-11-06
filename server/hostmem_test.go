package server

import (
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/engine"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// scratchBytes is the host scratch the model's instance holds and the
// scratch the model holds for routed steps and the next prompt: the engine's
// own byte counts, not the allocator's RSS. Sessions are histories and hold
// none.
func scratchBytes(e *Engine, lm *LoadedModel) (states, model uint64, n int) {
	lm.inst.With(func(in *engine.Instance) { states = in.ScratchBytes() })
	return states, lm.stepScratchBytes(), 1
}

// TestHostBatchScratchDoesNotGrowWithRows: a burst of sixteen concurrent
// generates on a host model, at the model's whole context, leaves the
// model's instance holding no more step scratch than one request alone left
// it, and the model one step's scratch for the next routed step: memory follows
// the rows in flight, not the widest burst the instance ever ran. Counted on
// the engine's own bytes (Instance.ScratchBytes, Instance.StepScratchBytes).
func TestHostBatchScratchDoesNotGrowWithRows(t *testing.T) {
	e, lm, c := hostBatchEngine(t, "Llama-3.2-1B-Instruct-Q4_K_M.jlm", Config{DefaultMaxSeq: -1, NoMemCache: true}) // -1: the model's whole context
	base := strings.Repeat("The little dog ran to the park and played with a red ball. ", 10)
	one := func(i int) completion {
		return complete(c, &v1.GenerateRequest{ModelId: "dev", Prompt: text(fmt.Sprintf("%d %s", i, base)),
			MaxTokens: 32, IgnoreEos: true})
	}
	if r := one(0); r.err != nil {
		t.Fatal(r.err)
	}
	idleStates, idleModel, n0 := scratchBytes(e, lm)
	const burst = 16
	var wg sync.WaitGroup
	got := make([]completion, burst)
	for i := range burst {
		wg.Add(1)
		go func() { defer wg.Done(); got[i] = one(i + 1) }()
	}
	wg.Wait()
	for i, g := range got {
		if g.err != nil || !g.started.GetBatched() {
			t.Fatalf("request %d: %v, batched %v", i, g.err, g.started.GetBatched())
		}
	}
	if lm.loop.stats.maxSessions.Load() < 2 {
		t.Fatal("no step carried two sessions: the burst never batched")
	}
	states, model, n := scratchBytes(e, lm)
	t.Logf("after one request: %d State(s) hold %.1f MiB, the model %.1f MiB; after a %d-row burst: %d State(s) hold %.1f MiB, the model %.1f MiB",
		n0, mib(idleStates), mib(idleModel), burst, n, mib(states), mib(model))
	// Each instance may hold what one request alone left its instance holding.
	if per := idleStates / uint64(max(n0, 1)); states > per*uint64(n)+(1<<20) {
		t.Fatalf("the pooled States hold %.1f MiB of scratch after the burst, against %.1f MiB after one request: "+
			"the scratch grew with the rows", mib(states), mib(idleStates))
	}
}

func mib(b uint64) float64 { return float64(b) / (1 << 20) }
