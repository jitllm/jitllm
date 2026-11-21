package convert

import (
	"errors"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestLongRopeLongMagnitude: a LongRoPE's long_mscale, PhimoeRotaryEmbedding's
// magnitude past the original context, is the one magnitude the container
// states or the file is refused: equal to short_mscale (every published
// Phi-3.5-MoE) it converts, another is refused by name as not implemented.
func TestLongRopeLongMagnitude(t *testing.T) {
	short, same, other := 1.243, 1.243, 1.3
	for _, r := range []struct {
		name string
		long *float64
		err  string
	}{
		{"absent", nil, ""},
		{"equal to short_mscale", &same, ""},
		{"another", &other, "long_mscale"},
	} {
		obj := map[string]any{"rope_type": "longrope", "short_factor": []any{1.0, 1.0}, "long_factor": []any{2.0, 2.0},
			"short_mscale": short, "original_max_position_embeddings": 64.0}
		if r.long != nil {
			obj["long_mscale"] = *r.long
		}
		cfg := &Config{Config: &jlm.Config{NRot: 4, NCtx: 256, AttnFactor: 1}}
		rd, err := rotaryScaling(cfg, obj, func(string) any { return nil }, nil)
		if err != nil {
			t.Fatalf("%s: %v", r.name, err)
		}
		var ts []jlm.Tensor
		ok, err := rd.holds(cfg)
		if err == nil && ok {
			ts, err = rd.settle(cfg)
		}
		switch {
		case r.err == "" && err != nil:
			t.Errorf("%s: %v", r.name, err)
		case r.err != "" && (err == nil || !strings.Contains(err.Error(), r.err) || !errors.Is(err, ErrNotImplemented)):
			t.Errorf("%s: %v, want a refusal saying %q", r.name, err, r.err)
		case r.err == "" && (cfg.AttnFactor != float32(short) || cfg.YarnOrigCtx != 64 || len(ts) != 2 ||
			ts[0].Role != jlm.RoleRopeFactorsShort || ts[1].Role != jlm.RoleRopeFactorsLong):
			t.Errorf("%s: magnitude %g over %d, %d tensors; want short_mscale's %g over 64, the short and long factors",
				r.name, cfg.AttnFactor, cfg.YarnOrigCtx, len(ts), short)
		}
	}
}
