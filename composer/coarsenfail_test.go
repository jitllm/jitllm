package composer_test

import (
	"runtime"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/graph"
)

// TestCoarsenRefusesWithoutBuildingTheChain forms the sampler's candidates
// over a vocabulary of 151552 on a GPU. The chain of its reductions is
// refused (an integer sum cannot be exchanged between threads), and the
// refusal is found at the chain's first reduction: the nodes of the rest of
// a chain a thread unrolls over a vocabulary -- 29 million, 9 GiB of them,
// before the refusal was acted on -- are never built. The bound is the
// allocation of the whole form.
func TestCoarsenRefusesWithoutBuildingTheChain(t *testing.T) {
	g := (&graph.SampleTopK{NVocab: 151552, K: 64}).Build()
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: 30720, Lanes: 32, Units: 20, Shared: 102400, ExactDiv: true, ExactArith: true}
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	_, codes, err := composer.Default().Form(composer.Request{Graph: g, Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: dev})
	runtime.ReadMemStats(&after)
	if err != nil {
		t.Fatalf("forming: %v", err)
	}
	if len(codes) == 0 {
		t.Fatal("the form has no kernels")
	}
	const bound = 1 << 30
	if got := after.TotalAlloc - before.TotalAlloc; got > bound {
		t.Fatalf("forming the sampler's candidates allocated %d MiB, over the %d MiB a refused chain costs", got>>20, bound>>20)
	}
}
