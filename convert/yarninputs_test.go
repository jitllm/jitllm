package convert

import (
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestYarnAgreesAcrossInputs holds YaRN's container fields -- the ramp, the
// magnitude on cos and sin (AttnFactor) and the score's correction
// (YarnLogMul) -- equal between a safetensors directory and the GGUF
// llama.cpp's converter wrote from the same config.json. The fixtures are
// scripts/c6gold.py's, one per family and per way of stating the magnitude;
// DeepSeek-V2-Lite (mscale == mscale_all_dim == 0.707, factor 40) is the real
// checkpoint, run where it is present.
//
// The two readers are independent witnesses: the GGUF path takes llama.cpp's
// keys (rope.scaling.yarn_attn_factor, yarn_log_multiplier), the safetensors
// path transformers' rope_parameters, and both must arrive at the reference's
// arithmetic.
func TestYarnAgreesAcrossInputs(t *testing.T) {
	pairs := [][2]string{
		{"synth-olmo3-af-hf", "synth-olmo3-af.gguf"},
		{"synth-mistral3-af-hf", "synth-mistral3-af.gguf"},
		{"synth-deepseek-yarn-hf", "synth-deepseek-yarn.gguf"},
		{"synth-deepseek-yarn-af-hf", "synth-deepseek-yarn-af.gguf"},
		{"DeepSeek-V2-Lite", "DeepSeek-V2-Lite.Q4_K_M.gguf"},
	}
	ran := 0
	for _, p := range pairs {
		t.Run(p[0], func(t *testing.T) {
			dir, gp := testmodels.Path(p[0]), testmodels.Path(p[1])
			for _, q := range []string{dir, gp} {
				if _, err := os.Stat(q); err != nil {
					testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS; scripts/c6gold.py builds the fixtures) -- "+
						"this gate proved nothing", q)
				}
			}
			a, b := hfSourceLazy(t, dir).Config, ggufSource(t, gp).Config
			if a.YarnFactor <= 1 {
				t.Fatalf("the safetensors config carries no YaRN (factor %v)", a.YarnFactor)
			}
			for _, f := range []struct {
				name string
				x, y any
			}{
				{"YarnFactor", a.YarnFactor, b.YarnFactor},
				{"YarnOrigCtx", a.YarnOrigCtx, b.YarnOrigCtx},
				{"YarnBetaFast", a.YarnBetaFast, b.YarnBetaFast},
				{"YarnBetaSlow", a.YarnBetaSlow, b.YarnBetaSlow},
				{"AttnFactor", a.AttnFactor, b.AttnFactor},
				{"YarnLogMul", a.YarnLogMul, b.YarnLogMul},
			} {
				if fmt.Sprint(f.x) != fmt.Sprint(f.y) {
					t.Errorf("%s: safetensors %v, gguf %v", f.name, f.x, f.y)
				}
			}
			t.Logf("factor %v, AttnFactor %v, YarnLogMul %v on both", a.YarnFactor, a.AttnFactor, a.YarnLogMul)
			ran++
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "%s", "no pair was present; this gate proved nothing (RULE 10)")
	}
}

// hfSourceLazy is hfSource with every tensor left unread: the config is the
// subject, and DeepSeek-V2-Lite's 31 GB need not be read to answer it.
func hfSourceLazy(t testing.TB, dir string) *jlm.Source {
	t.Helper()
	_, shards, err := hfShards(dir)
	if err != nil {
		t.Fatal(err)
	}
	var open []*safetensors.File
	for _, p := range shards {
		f, err := safetensors.Open(p)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { f.Close() })
		open = append(open, f)
	}
	s, err := hfSourceOf(dirFiles(dir), open)
	if err != nil {
		t.Fatal(err)
	}
	return s
}
