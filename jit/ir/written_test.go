package ir_test

import (
	"testing"

	"github.com/jitllm/jitllm/jit/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestWrittenParamsNamesTheOutputs holds the mask a concurrent Metal encoder
// orders dispatches by: GemmTile stores through scalar, vector and tile ops
// into shared memory, and into its output alone among its parameters; a
// kernel storing into two parameters reports both.
func TestWrittenParamsNamesTheOutputs(t *testing.T) {
	k, err := kernels.GemmTile(kernels.MatVecShape{T: kernels.Q4_K, K: 512, Rows: 64, NTok: 32},
		kernels.TileGemm{MT: 4, NT: 2, WM: 2, WN: 2, KB: 1})
	if err != nil {
		t.Fatal(err)
	}
	out := -1
	for i, p := range k.Params {
		if p.Name == "pOut" {
			out = i
		}
	}
	if got, want := k.WrittenParams(), uint64(1)<<uint(out); out < 0 || got != want {
		t.Fatalf("GemmTile writes %b, want only pOut (%b)", got, want)
	}

	b := ir.New("two", [3]int{32, 1, 1})
	a := b.Param("a", ir.U32)
	x := b.Param("x", ir.U32)
	y := b.Param("y", ir.U32)
	v := b.Load(ir.U32, a, b.TID(), 0)
	b.Store(x, b.TID(), v, 0)
	b.Store(y, b.TID(), v, 0)
	if got := mustBuild(t, b).WrittenParams(); got != 0b110 {
		t.Fatalf("a kernel storing into x and y reports %b, want 110", got)
	}
}
