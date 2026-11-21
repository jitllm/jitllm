package convert

import (
	"maps"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestHFYarnReadsAsTransformers holds rotary.json's YaRN reading to
// _compute_yarn_parameters: truncate false takes the correction range as
// computed, true or absent rounds it; a beta of zero is the default, as
// `get("beta_fast") or 32` reads it. The truncation must reach the
// divisors: the exact range turns them otherwise.
func TestHFYarnReadsAsTransformers(t *testing.T) {
	var divs [][]float32
	for _, r := range []struct {
		name       string
		obj        map[string]any
		truncate   bool
		fast, slow float64
	}{
		{"truncate false", map[string]any{"truncate": false}, false, 32, 1},
		{"truncate true", map[string]any{"truncate": true}, true, 32, 1},
		{"truncate absent", nil, true, 32, 1},
		{"zero betas", map[string]any{"beta_fast": 0.0, "beta_slow": 0.0}, true, 32, 1},
		{"stated betas", map[string]any{"beta_fast": 16.0, "beta_slow": 16.0}, true, 16, 16},
	} {
		obj := map[string]any{"rope_type": "yarn", "factor": 40.0, "original_max_position_embeddings": 64.0}
		maps.Copy(obj, r.obj)
		cfg := &Config{Config: &jlm.Config{NCtx: 4096, AttnFactor: 1, NRot: 64, RopeBase: 10000}}
		rd, err := rotaryScaling(cfg, obj, func(string) any { return nil }, nil)
		if err != nil || rd == nil {
			t.Fatalf("%s: %v", r.name, err)
		}
		p := rd.params
		if p["truncate"] != r.truncate || p["beta_fast"] != r.fast || p["beta_slow"] != r.slow {
			t.Errorf("%s: truncate %v, betas %v/%v; want %v, %v/%v", r.name, p["truncate"], p["beta_fast"],
				p["beta_slow"], r.truncate, r.fast, r.slow)
		}
		if ok, err := rd.holds(cfg); err != nil || !ok {
			t.Fatalf("%s: states nothing: %v", r.name, err)
		}
		d, err := rd.divisors(cfg)
		if err != nil || len(d) != 32 {
			t.Fatalf("%s: %d divisors, %v", r.name, len(d), err)
		}
		divs = append(divs, d)
	}
	if slices.Equal(divs[0], divs[1]) {
		t.Error("the exact correction range turns the pairs as the rounded one does")
	}
	if !slices.Equal(divs[1], divs[2]) || !slices.Equal(divs[1], divs[3]) {
		t.Error("an absent truncation or zero betas turn the pairs otherwise than the defaults")
	}
}
