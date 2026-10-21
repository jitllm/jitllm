package jlm

import (
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestLayoutMatchesThePacker holds Layout -- the on-disk specification a reader
// is promised needs no second document -- to the packer that writes the bytes,
// field by field, for every type code the format defines.
//
// Nothing else reads Layout, so a field that drifts from the packer would
// fail nothing without this.
func TestLayoutMatchesThePacker(t *testing.T) {
	packed := 0
	for c := TypeNone + 1; c.Valid(); c++ {
		src, ok := SourceType(c)
		if !ok {
			t.Errorf("%s: a valid code with no source type", c)
			continue
		}
		if back, ok := TypeOf(src); !ok || back != c {
			t.Errorf("%s: SourceType %s maps back to %s", c, src, back)
		}
		l, hasLayout := Layout(c)
		q, isPacked := packerOf(c)
		if hasLayout != isPacked {
			t.Errorf("%s: Layout says packed=%v and the writer says %v", c, hasLayout, isPacked)
			continue
		}
		if !isPacked {
			continue
		}
		packed++
		sub, bits, biasK, biasArray := kernels.Layout(q)
		perSuper, scOff := kernels.ScaleLayout(q)
		want := TypeLayout{
			Sub: sub, Planes: [2]int{bits, kernels.HiPlane(q)}, SuperSub: perSuper,
			ScaleOff: scOff, HasMin: biasArray, Signed: kernels.SignedPayload(q),
			Codes: kernels.Codes(q), D32: kernels.D32(q),
		}
		if !biasArray {
			want.Bias = biasK
		}
		if (l.Codes == nil) != (want.Codes == nil) || (l.Codes != nil && *l.Codes != *want.Codes) {
			t.Errorf("%s: code table %v on disk, %v in the packer", c, l.Codes, want.Codes)
		}
		l.Codes, want.Codes = nil, nil
		if l != want {
			t.Errorf("%s: Layout %+v, the packer %+v", c, l, want)
		}
	}
	if packed == 0 {
		t.Fatal("no packed type was compared -- this gate proved nothing")
	}
	t.Logf("%d packed type codes match the packer", packed)
}
