package convert

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestDbrxCarriesItsClamp is the converter half of the DBRX gate: the clamp
// reaches the container, a DBRX without one is refused, a clamp on another
// architecture reaches its container too and a negative one is refused,
// and DBRX's attn_output_norm
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
	if c.Name != "dbrx" || c.ClampKQV != 8 || !c.Flags.Has(jlm.FlagLayerNorm) ||
		!c.Flags.Has(jlm.FlagRopeNeox) || c.Flags.Has(jlm.FlagParallel) || c.NExpert != 4 {
		t.Fatalf("dbrx config %+v", c)
	}
	delete(k, "attention.clamp_kqv")
	if _, err := configOf(headerOnly("dbrx", k)); err == nil {
		t.Error("a dbrx with no clamp converted")
	}
	s := baseKeys(3)
	s["attention.clamp_kqv"] = meta.MakeFloat(8)
	if c, err := configOf(headerOnly("llama", s)); err != nil || c.ClampKQV != 8 {
		t.Errorf("a llama carrying a clamp: %v, want it stated", err)
	}
	s["attention.clamp_kqv"] = meta.MakeFloat(-8)
	if _, err := configOf(headerOnly("llama", s)); err == nil || !strings.Contains(err.Error(), "clamp_kqv -8") {
		t.Errorf("a negative clamp: %v, want it refused", err)
	}
	dbrx, err := retargets("dbrx", nil)
	if err != nil {
		t.Fatal(err)
	}
	if r := dbrx[jlm.RoleAttnOutNorm]; r != jlm.RoleFFNNorm {
		t.Errorf("dbrx's attn_output_norm is %v, want the FFN norm", r)
	}
	bert, err := retargets("bert", nil)
	if err != nil {
		t.Fatal(err)
	}
	if r, ok := bert[jlm.RoleAttnOutNorm]; ok {
		t.Errorf("bert's attn_output_norm became %v", r)
	}
}
