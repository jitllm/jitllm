package convert

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestConverterStatesTheSlidingLayers holds what the converter states of a
// model's sliding layers: OLMo 3's rotate at their base unscaled, a picture's
// rows see each other on Gemma 4's 26B and 31B's sliding layers alone and on
// every layer of the E-models, and a model with neither states nothing but
// its revision. BitNet states its projections' int8 activations.
func TestConverterStatesTheSlidingLayers(t *testing.T) {
	cfg := func(a Arch, ple uint32) *Config {
		return &Config{Arch: a, Config: &jlm.Config{NLayer: 6, SWAWindow: 512, SWAPeriod: 6, PLEDim: ple}}
	}
	for _, c := range []struct {
		name  string
		c     *Config
		scale float32
		bidir []bool
	}{
		{"OLMo 3", cfg(ArchOLMo3, 0), 1, nil},
		{"Gemma 4 31B", cfg(ArchGemma4, 0), 0, []bool{true, true, true, true, true, false}},
		{"Gemma 4 E4B", cfg(ArchGemma4, 256), 0, nil},
		{"Llama", cfg(ArchLlama, 0), 0, nil},
	} {
		StateRevision(c.c)
		if c.c.ActQuant != jlm.ActQuantNone {
			t.Errorf("%s: activation quantization %d", c.name, c.c.ActQuant)
		}
		if c.c.Revision != jlm.StatementRevision || c.c.RopeScaleSWA != c.scale || !slices.Equal(c.c.BidirLayers, c.bidir) {
			t.Errorf("%s: revision %d, sliding scale %v, picture layers %v; want %d, %v, %v", c.name,
				c.c.Revision, c.c.RopeScaleSWA, c.c.BidirLayers, jlm.StatementRevision, c.scale, c.bidir)
		}
	}
	b := cfg(ArchBitNet, 0)
	StateRevision(b)
	if b.ActQuant != jlm.ActQuantRowInt8 {
		t.Errorf("BitNet: activation quantization %d, want int8 per row", b.ActQuant)
	}
}
