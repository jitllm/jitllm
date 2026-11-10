package convert

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestApertusFoldsItsActivation is the converter half of Apertus: the four
// xielu.* keys (bare, as llama.cpp's writer stores them; one per layer)
// become each block's RoleXIELU vector with the softplus folded in, and a file
// without them is refused rather than run with a default activation.
func TestApertusFoldsItsActivation(t *testing.T) {
	k := baseKeys(2)
	f := headerOnly("apertus", k)
	kv := f.KV
	kv["xielu.alpha_p"] = meta.MakeFloat32s([]float32{0.5, -1})
	kv["xielu.alpha_n"] = meta.MakeFloat32s([]float32{0.25, 30})
	kv["xielu.beta"] = meta.MakeFloat32s([]float32{0.5, 0.4})
	kv["xielu.eps"] = meta.MakeFloat32s([]float32{-1e-6, -0.1})
	f = meta.New(f.Path, kv, f.Tensors)
	c, err := configOf(f)
	if err != nil {
		t.Fatal(err)
	}
	if c.Arch != ArchApertus || !c.Flags.Has(jlm.FlagQKNorm) || !c.Flags.Has(jlm.FlagRopeNeox) {
		t.Fatalf("arch %v flags %v", c.Arch, c.Flags)
	}
	s := &jlm.Source{Config: c.Config}
	if err := apertusXIELU(c, f, s); err != nil {
		t.Fatal(err)
	}
	if len(s.Tensors) != 2 {
		t.Fatalf("%d tensors, want one per block", len(s.Tensors))
	}
	sp := func(x float64) float64 { return math.Log1p(math.Exp(x)) }
	want := [][4]float64{
		{sp(0.5), 0.5 + sp(0.25), 0.5, -1e-6},
		{sp(-1), float64(float32(0.4)) + 30, float64(float32(0.4)), float64(float32(-0.1))},
	}
	for b, tn := range s.Tensors {
		got := f32s(tn.Data)
		if tn.Role != jlm.RoleXIELU || tn.Block != int32(b) || len(got) != 4 {
			t.Fatalf("tensor %d: %v block %d, %d values", b, tn.Role, tn.Block, len(got))
		}
		for i := range got {
			if math.Abs(float64(got[i])-want[b][i]) > 1e-6*math.Abs(want[b][i]) {
				t.Errorf("block %d value %d: %g, want %g", b, i, got[i], want[b][i])
			}
		}
	}
	delete(kv, "xielu.eps")
	if _, err := configOf(meta.New(f.Path, kv, f.Tensors)); err == nil {
		t.Fatal("an Apertus with no xielu.eps converted")
	}
}
