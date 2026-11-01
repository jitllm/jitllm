package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
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
