package ir_test

import (
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
)

// TestWrittenParamsNamesTheOutputs holds the mask a concurrent Metal encoder
// orders dispatches by: a kernel storing into shared memory and into one
// parameter reports that parameter alone, and one storing into two reports
// both.
func TestWrittenParamsNamesTheOutputs(t *testing.T) {
	s := ir.New("shared", [3]int{32, 1, 1})
	in, out := s.Param("in", ir.F32), s.Param("out", ir.F32)
	sh := s.Shared("sh", ir.F32, 32)
	s.Store(sh, s.TID(), s.Load(ir.F32, in, s.TID(), 0), 0)
	s.Barrier()
	s.Store(out, s.TID(), s.Load(ir.F32, sh, s.TID(), 0), 0)
	if got := mustBuild(t, s).WrittenParams(); got != 0b10 {
		t.Fatalf("a kernel storing into shared memory and out reports %b, want 10", got)
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
