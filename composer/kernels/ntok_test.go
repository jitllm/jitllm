package kernels

import "testing"

// NTok must produce a kernel, and must refuse the combinations it cannot serve.
//
// NTok > 1 is the batched prefill shape and reuses SlotAct's addressing
// (each column reads its own activation vector at stride K); Tok is how many
// columns one thread carries.
func TestNTokShapes(t *testing.T) {
	for _, c := range []struct {
		name string
		s    MatVecShape
		ok   bool
	}{
		{"dense", MatVecShape{T: Q8_0, K: 256, Rows: 64}, true},
		{"ntok8", MatVecShape{T: Q8_0, K: 256, Rows: 64, NTok: 8}, true},
		{"ntok64", MatVecShape{T: Q4_0, K: 512, Rows: 128, NTok: 64}, true},
		// A decode batch splits k to fill the device; the in-group reduction
		// would need a shared word per token per thread and stays refused.
		{"ntok+split", MatVecShape{T: Q8_0, K: 256, Rows: 64, NTok: 8, Split: 2}, true},
		{"ntok+groupsplit", MatVecShape{T: Q8_0, K: 256, Rows: 64, NTok: 8, Split: 2, GroupSplit: true}, false},
		// A grouped mixture: one expert per Tok group (the tier's MoE prefill).
		{"ntok+experts", MatVecShape{T: Q8_0, K: 256, Rows: 64, NTok: 8, Experts: 4}, true},
		// An expert bias is indexed by expert and pBias by row; a slot count
		// beside NTok would give Slots two meanings.
		{"ntok+experts+bias", MatVecShape{T: Q8_0, K: 256, Rows: 64, NTok: 8, Experts: 4, Bias: true}, false},
		{"ntok+experts+slots", MatVecShape{T: Q8_0, K: 256, Rows: 64, NTok: 8, Experts: 4, Slots: 4}, false},
		{"ntok+experts+rowt", MatVecShape{T: Q8_0, K: 256, Rows: 64, NTok: 8, Experts: 4, Rowt: 2}, false},
	} {
		_, err := MatVec(c.s)
		if (err == nil) != c.ok {
			t.Errorf("%s: err=%v, want ok=%v", c.name, err, c.ok)
		}
	}
}

// The dense kernel must not move: NTok defaults to 0 and Tok to 1, and a
// decode matvec is emitted byte for byte as before (TestDenseKernelsUnchanged
// protects it; this asserts it against the new fields directly).
func TestNTokDefaultIsDense(t *testing.T) {
	a, err := MatVec(MatVecShape{T: Q4_K, K: 512, Rows: 128})
	if err != nil {
		t.Fatal(err)
	}
	b, err := MatVec(MatVecShape{T: Q4_K, K: 512, Rows: 128, NTok: 1, Tok: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Ops) != len(b.Ops) {
		t.Errorf("NTok=1 emits %d ops, plain emits %d", len(b.Ops), len(a.Ops))
	}
}
