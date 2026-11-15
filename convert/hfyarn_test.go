package convert

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestHFYarnReadsAsTransformers holds a safetensors YaRN to
// _compute_yarn_parameters: truncate false takes the correction range as
// computed (FlagYarnExact), true or absent rounds it; a beta of zero is the
// default, as `get("beta_fast") or 32` reads it.
func TestHFYarnReadsAsTransformers(t *testing.T) {
	f, z, b := 40.0, 0.0, 16.0
	no, yes := false, true
	for _, r := range []struct {
		name       string
		y          hfYarn
		exact      bool
		fast, slow float32
	}{
		{"truncate false", hfYarn{Factor: f, OrigCtx: 64, Truncate: &no}, true, 32, 1},
		{"truncate true", hfYarn{Factor: f, OrigCtx: 64, Truncate: &yes}, false, 32, 1},
		{"truncate absent", hfYarn{Factor: f, OrigCtx: 64}, false, 32, 1},
		{"zero betas", hfYarn{Factor: f, OrigCtx: 64, BetaFast: &z, BetaSlow: &z}, false, 32, 1},
		{"stated betas", hfYarn{Factor: f, OrigCtx: 64, BetaFast: &b, BetaSlow: &b}, false, 16, 16},
	} {
		cfg := &Config{Config: &jlm.Config{NCtx: 4096, AttnFactor: 1}}
		hfYarnApply(&r.y, cfg)
		if cfg.Flags.Has(jlm.FlagYarnExact) != r.exact || cfg.YarnBetaFast != r.fast || cfg.YarnBetaSlow != r.slow {
			t.Errorf("%s: exact %v, betas %v/%v; want %v, %v/%v", r.name, cfg.Flags.Has(jlm.FlagYarnExact),
				cfg.YarnBetaFast, cfg.YarnBetaSlow, r.exact, r.fast, r.slow)
		}
	}
}
