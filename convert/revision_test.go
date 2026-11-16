package convert

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestConverterStatesTheSlidingLayers holds what the library's revision
// statements state of a model's sliding layers: OLMo 3's (olmo2 with a
// window) rotate at their base unscaled, a picture's rows see each other on
// Gemma 4's 26B and 31B's sliding layers alone and on every layer of the
// E-models, ModernBERT's windowed period starts at its global layer, and a
// model with none of them states nothing but its revision. BitNet states
// its projections' int8 activations.
func TestConverterStatesTheSlidingLayers(t *testing.T) {
	for _, c := range []struct {
		name, arch  string
		keys        map[string]meta.Value
		win         uint32
		ple         uint32
		scale       float32
		bidir       []bool
		quant       jlm.ActQuant
		globalFirst bool
	}{
		{name: "OLMo 3", arch: "olmo2", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(512)}, win: 512, scale: 1},
		{name: "OLMo 2", arch: "olmo2"},
		{name: "Gemma 4 31B", arch: "gemma4", win: 512, bidir: []bool{true, true, true, true, true, false}},
		{name: "Gemma 4 E4B", arch: "gemma4", keys: map[string]meta.Value{"embedding_length_per_layer_input": meta.MakeUint(256)},
			win: 512, ple: 256},
		{name: "Gemma 4 unwindowed", arch: "gemma4"},
		{name: "ModernBERT", arch: "modern-bert", win: 128, globalFirst: true},
		{name: "ModernBERT unwindowed", arch: "modern-bert"},
		{name: "BitNet", arch: "bitnet-b1.58", quant: jlm.ActQuantRowInt8},
		{name: "Llama", arch: "llama", win: 512},
	} {
		kv := map[string]meta.Value{"general.architecture": meta.MakeString(c.arch)}
		for k, v := range c.keys {
			kv[c.arch+"."+k] = v
		}
		cfg := &Config{Config: &jlm.Config{}}
		if err := stateContainer(meta.New(c.arch+".gguf", kv, nil), cfg); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		cfg.NLayer, cfg.SWAWindow, cfg.SWAPeriod, cfg.PLEDim = 6, c.win, 6, c.ple
		StateRevision(cfg)
		if cfg.Revision != jlm.StatementRevision || cfg.RopeScaleSWA != c.scale || !slices.Equal(cfg.BidirLayers, c.bidir) {
			t.Errorf("%s: revision %d, sliding scale %v, picture layers %v; want %d, %v, %v", c.name,
				cfg.Revision, cfg.RopeScaleSWA, cfg.BidirLayers, jlm.StatementRevision, c.scale, c.bidir)
		}
		if cfg.ActQuant != c.quant || cfg.Flags2.Has(jlm.Flag2SWAGlobalFirst) != c.globalFirst {
			t.Errorf("%s: activation quantization %d, global first %v; want %d, %v", c.name,
				cfg.ActQuant, cfg.Flags2.Has(jlm.Flag2SWAGlobalFirst), c.quant, c.globalFirst)
		}
	}
}
