package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/cpu"
)

// kvq8Models are the models the q8 cache gate runs: the smallest real llama
// (stories15M), a real Llama 3 (GQA, 64-wide heads), a real hybrid whose
// attention layers sit between gated delta-rule layers (Qwen3.5-0.8B), a short
// convolution hybrid (LFM2), and Gemma 4's two head geometries (GeomSplit:
// the sliding layers attend at a width of their own, and its E-model shares
// KV between layers).
var kvq8Models = []string{
	"stories15M-q8_0.gguf",
	"Llama-3.2-1B-Instruct-Q4_K_M.gguf",
	"Qwen3.5-0.8B-f16.gguf",
	"lfm2/LFM2-350M-Q4_K_M.gguf",
	"synth-gemma4.gguf",
	"synth-gemma4-e.gguf",
}

// kvTeacherForce runs ids through a fresh State at t and returns every
// position's logits. setup, when not nil, sees the State before its first
// token.
func kvTeacherForce(t *testing.T, m *Model, kt KVType, ids []int32, setup func(*State)) (*State, [][]float32) {
	t.Helper()
	if err := m.SetKVType(kt); err != nil {
		t.Fatal(err)
	}
	defer m.ClearKVF16()
	s := m.NewState(len(ids) + 1)
	if setup != nil {
		setup(s)
	}
	var outs [][]float32
	for _, id := range ids {
		out, err := s.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		outs = append(outs, append([]float32(nil), out...))
	}
	return s, outs
}

// q8Twin makes s's f32 cache store every row rounded through q8_0
// (kvLayout.q8Twin): the control arm a q8 cache must equal bit for bit.
func q8Twin(s *State) {
	c := s.c
	hd := max(c.HeadDim, c.HeadDimSWA)
	s.kvl.jit = s.jit
	s.jit.ReserveKVQuant(hd)
	s.kvl.q8Twin = make([]float32, (max(c.NKVHead, c.NKVHeadSWA)+1)*cpu.KVRowBytes(cpu.KVQ8, hd)/4)
}

// argmaxF is the index of the largest logit.
func argmaxF(x []float32) int {
	j := 0
	for i := range x {
		if x[i] > x[j] {
			j = i
		}
	}
	return j
}

// TestKVQ8RefusesByName: the caches a q8 row cannot express are refused at
// Open, naming the reason, and through SetKVType on an open model.
func TestKVQ8RefusesByName(t *testing.T) {
	for _, c := range []struct{ name, why string }{
		{"synth-deepseek-lite", "MLA"}, // the directory: jlmOf converts it fresh, a .jlm named here is opened as it stands
		{"synth-deepseek32.gguf", "lightning indexer"},
		{"synth-minimaxm3.gguf", "MiniMax Sparse Attention"},
		{"synth-deepseek4.gguf", "compressor"},
	} {
		t.Run(c.name, func(t *testing.T) {
			src, ok := existingModel(testmodels.Path(c.name))
			if !ok {
				testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(c.name))
			}
			p := jlmOf(t, src)
			if _, err := Open(p, WithKVType(KVQ8_0)); err == nil || !strings.Contains(err.Error(), c.why) {
				t.Fatalf("Open with a q8_0 cache: %v, want a refusal naming %q", err, c.why)
			}
			m, err := Open(p)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if err := m.SetKVType(KVQ8_0); err == nil || !strings.Contains(err.Error(), c.why) {
				t.Fatalf("SetKVType(q8_0): %v, want a refusal naming %q", err, c.why)
			}
		})
	}
}
