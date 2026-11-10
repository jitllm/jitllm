package convert

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestDbrxCarriesItsClamp is the converter half of the DBRX gate: the clamp
// reaches the container, a DBRX without one is refused, a clamp on another
// architecture is refused rather than dropped, and DBRX's attn_output_norm
// lands on the FFN norm where BERT's keeps its post-norm role. The model half
// is model.TestC6MatchesTransformers and TestC6FeaturesAreLoadBearing.
func TestDbrxCarriesItsClamp(t *testing.T) {
	k := baseKeys(3)
	k["expert_count"] = meta.MakeUint(4)
	k["expert_used_count"] = meta.MakeUint(2)
	k["attention.layer_norm_epsilon"] = meta.MakeFloat(1e-5)
	k["attention.clamp_kqv"] = meta.MakeFloat(8)
	c, err := configOf(headerOnly("dbrx", k))
	if err != nil {
		t.Fatal(err)
	}
	if c.Arch != ArchDBRX || c.ClampKQV != 8 || !c.Flags.Has(jlm.FlagLayerNorm) ||
		!c.Flags.Has(jlm.FlagRopeNeox) || c.Flags.Has(jlm.FlagParallel) || c.NExpert != 4 {
		t.Fatalf("dbrx config %+v", c)
	}
	delete(k, "attention.clamp_kqv")
	if _, err := configOf(headerOnly("dbrx", k)); err == nil {
		t.Error("a dbrx with no clamp converted")
	}
	s := baseKeys(3)
	s["attention.clamp_kqv"] = meta.MakeFloat(8)
	if _, err := configOf(headerOnly("llama", s)); !errors.Is(err, ErrNotImplemented) {
		t.Errorf("a llama carrying a clamp: %v, want ErrNotImplemented", err)
	}
	if r := retarget(ArchDBRX, jlm.RoleAttnOutNorm); r != jlm.RoleFFNNorm {
		t.Errorf("dbrx's attn_output_norm is %v, want the FFN norm", r)
	}
	if r := retarget(ArchBERT, jlm.RoleAttnOutNorm); r != jlm.RoleAttnOutNorm {
		t.Errorf("bert's attn_output_norm became %v", r)
	}
	if !fusesQKV(ArchDBRX) {
		t.Error("dbrx's attn_qkv is not split at conversion")
	}
}
