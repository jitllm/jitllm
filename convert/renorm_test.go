package convert

import (
	"maps"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestRoutersThatNeverRenormaliseRefuseTheKey: Llama 4's, PhiMoE's and
// Jamba's routers never renormalise, so a file stating expert_weights_norm
// true is refused rather than obeyed by the generic reading, and one stating
// false or nothing keeps the router as it is.
func TestRoutersThatNeverRenormaliseRefuseTheKey(t *testing.T) {
	for _, name := range []string{"synth-llama4.gguf", "synth-phimoe.gguf", "synth-jamba.gguf"} {
		t.Run(name, func(t *testing.T) {
			path := testmodels.Path(name)
			if _, err := os.Stat(path); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v (see docs/testing.md) -- this gate proved nothing", err)
			}
			f, err := gguf.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			for _, norm := range []string{"absent", "false", "true"} {
				kv := maps.Clone(f.KV)
				delete(kv, f.Arch()+".expert_weights_norm")
				if norm != "absent" {
					kv[f.Arch()+".expert_weights_norm"] = meta.MakeBool(norm == "true")
				}
				c, err := configOf(meta.New(f.Path, kv, f.Tensors))
				switch {
				case norm == "true":
					if err == nil || !strings.Contains(err.Error(), "does not renormalise") {
						t.Errorf("expert_weights_norm true: %v, want the refusal", err)
					}
				case err != nil:
					t.Errorf("expert_weights_norm %s: %v", norm, err)
				case !c.Flags.Has(jlm.FlagNoExpertNorm):
					t.Errorf("expert_weights_norm %s: the router renormalises", norm)
				}
			}
		})
	}
}
