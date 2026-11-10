package engine

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestGemma3GlobalLayersAreLinearlyScaled: gemma-3-4b ships rope.scaling.type
// linear with factor 8, and llama.cpp divides the global layers' rotary angles
// by it while leaving the sliding ones alone (rope_freq_scale_train_swa = 1).
//
// The oracle is llama-eval-callback's rotated K for "The capital" (ids
// 2 818 5279), summed over the three positions -- position 0 is an identity
// rotation, so only a multi-token prompt can see a rotary scale at all:
//
//	llama-eval-callback -m gemma-3-4b-it-Q4_K_M.gguf -p "The capital" -n 1 -ngl 0 -c 64
//
// The rotated K is the history the blocks write. Its sum cancels to a few
// units over 3072 elements, and the engines' quantization moves it by
// percents at every layer (the local layer 4, which no scale touches, reads
// 3.9% from llama.cpp), so the gate compares two runs, with the global
// layers' scale and without it: the local layer's history is the same bit
// for bit, the global layer's moves, and the run with the scale is the one
// nearer llama.cpp (the old engine read -18.774827 with it and -21.585079
// without).
func TestGemma3GlobalLayersAreLinearlyScaled(t *testing.T) {
	path, ok := existingModel(testmodels.Path("gemma-3-4b-it-Q4_K_M.gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("gemma-3-4b-it-Q4_K_M.gguf")+" (set JITLLM_MODELS to the model directory) -- RULE 11, fetch it; this gate proves nothing without it")
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Cfg.RopeLinear != 8 {
		t.Fatalf("RopeLinear %v, want the file's 8", m.Cfg.RopeLinear)
	}
	ids := m.Vocab.Encode("The capital", true)
	// The rotated K is the history each block wrote: every position's keys,
	// gathered back as floats from an f32 history. The run without the scale
	// clears the global rotary's divisors, which the sliding layers' rotary
	// does not share.
	keys := func(unscaled bool) map[int][]float32 {
		t.Helper()
		if unscaled {
			was := m.Rope.Freqs
			m.Rope.Freqs = nil
			defer func() { m.Rope.Freqs = was }()
		}
		st := testInstance(t, m, 8, WithKVF16(false))
		defer st.Close()
		for _, id := range ids {
			if _, err := st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		out := map[int][]float32{}
		for _, li := range []int{4, 5} {
			l := st.kvlAt(li)
			k, v := make([]float32, len(ids)*l.kvDim()), make([]float32, len(ids)*l.kvDim())
			if !st.kv.MigrateKV(li, k, v, len(ids), false, st.kvConv) {
				t.Fatalf("block %d's history was not gathered", li)
			}
			out[li] = k
		}
		return out
	}
	sum := func(k []float32) float64 {
		var s float64
		for _, x := range k {
			s += float64(x)
		}
		return s
	}
	scaled, plain := keys(false), keys(true)
	const want5, want4 = -18.902901, -45.731182
	s5, p5 := sum(scaled[5]), sum(plain[5])
	t.Logf("ids %v; global layer 5 %.6f, unscaled %.6f (llama.cpp %.6f); local layer 4 %.6f, unscaled %.6f (llama.cpp %.6f)",
		ids, s5, p5, want5, sum(scaled[4]), sum(plain[4]), want4)
	switch {
	case !slices.Equal(scaled[4], plain[4]):
		t.Errorf("the local layer 4's history moved with the global layers' scale")
	case slices.Equal(scaled[5], plain[5]):
		t.Errorf("the global layer 5's history is the same with the scale and without it: no scale was applied")
	case !(math.Abs(s5-want5) < math.Abs(p5-want5)):
		t.Errorf("the global layer 5's rotated K sums to %.6f, and %.6f unscaled: the unscaled run is nearer llama.cpp's %.6f",
			s5, p5, want5)
	}
}
