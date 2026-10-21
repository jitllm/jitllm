package cpu

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestActWindowFollowsTheBlock: a model holding any 32-wide single-scale format
// (Q4_0, Q5_0, Q8_0, MXFP4) quantizes its activations per 32, whatever
// k-quants sit beside it.
func TestActWindowFollowsTheBlock(t *testing.T) {
	if !nativeHoistsActScale() {
		t.Skip("no k-quant kernel here hoists the scale, so every model gets 32")
	}
	narrow := 0
	for _, g := range quant.PackedTypes {
		q, _ := kernels.QuantOf(g)
		want := 256
		if kernels.NarrowScales(q) {
			want, narrow = Q8Block, narrow+1
		}
		if got := WideActWindow([]quant.Type{quant.Q4_K, g}); got != want {
			t.Errorf("%s beside Q4_K: window %d, want %d", g, got, want)
		}
	}
	if narrow < 5 {
		t.Fatalf("only %d narrow formats swept; Q4_0, Q5_0, Q5_1, Q8_0 and MXFP4 all are", narrow)
	}
}
