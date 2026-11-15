package convert

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestLongRopeLongMagnitude: a LongRoPE's long_mscale, PhimoeRotaryEmbedding's
// magnitude past the original context, is the one magnitude the container
// states or the file is refused: equal to short_mscale (every published
// Phi-3.5-MoE) it converts, another is refused by name.
func TestLongRopeLongMagnitude(t *testing.T) {
	short, same, other := 1.243, 1.243, 1.3
	for _, r := range []struct {
		name string
		long *float64
		err  string
	}{
		{"absent", nil, ""},
		{"equal to short_mscale", &same, ""},
		{"another", &other, "long_mscale 1.3"},
	} {
		lr := &hfLongRope{Short: []float64{1, 1}, Long: []float64{2, 2}, ShortScale: &short, LongScale: r.long, OrigCtx: 64}
		cfg := &Config{Config: &jlm.Config{NRot: 4, NCtx: 256, AttnFactor: 1}}
		_, err := lr.state("config.json", cfg)
		switch {
		case r.err == "" && err != nil:
			t.Errorf("%s: %v", r.name, err)
		case r.err != "" && (err == nil || !strings.Contains(err.Error(), r.err)):
			t.Errorf("%s: %v, want an error saying %q", r.name, err, r.err)
		case r.err == "" && cfg.AttnFactor != float32(short):
			t.Errorf("%s: magnitude %g, want short_mscale's %g", r.name, cfg.AttnFactor, short)
		}
	}
}
