// Package paritytest holds the old compiler (oldkernels: main's pre-made
// kernel generators) and the tests that hold the composer's instructions to
// it. Nothing outside this directory imports it; srcgate fails the build's
// gate if anything does.
package paritytest

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestMatVecEmitsMainsInstructions holds the composed matvec to main's at
// the schedules main picks: decode, an in-group split, a chunk at one token a
// thread, at every token a thread, at four tokens and two rows a thread, and a
// row tile, over a k-quant.
func TestMatVecEmitsMainsInstructions(t *testing.T) {
	for _, c := range []struct {
		rows, k, ntok, tok, split, rowt int
		grp                             bool
	}{
		{64, 256, 1, 1, 1, 1, false}, {64, 256, 1, 1, 2, 1, true}, {64, 256, 32, 1, 2, 1, false},
		{64, 256, 32, 32, 2, 1, false}, {64, 256, 32, 4, 2, 2, false}, {128, 1024, 1, 1, 1, 2, false},
	} {
		s := oldkernels.MatVecShape{T: kernels.Q4_K, K: c.k, Rows: c.rows, NTok: c.ntok, Tok: c.tok, Split: c.split,
			Rowt: c.rowt, GroupSplit: c.grp}
		mk, err := oldkernels.MatVec(s)
		if err != nil {
			t.Fatalf("%+v: main: %v", c, err)
		}
		ck, err := (&compose.MatVec{T: kernels.Q4_K, K: c.k, Rows: c.rows, NTok: c.ntok, Tok: c.tok, Split: c.split,
			Rowt: c.rowt, Group: c.grp}).Emit()
		if err != nil {
			t.Fatalf("%+v: composer: %v", c, err)
		}
		if a, b := recorder.Text(mk), recorder.Text(ck); recorder.Hash(mk) != recorder.Hash(ck) {
			t.Errorf("%+v: the composer's %d ops are not main's %d:\n%s", c, len(ck.Ops), len(mk.Ops), firstDiff(a, b))
		}
	}
}
