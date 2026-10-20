package compose

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// TestBarrierInALoop stages a value a trip, waits, reads a neighbour's: the
// barrier and both accesses stay inside the loop, in token order.
func TestBarrierInALoop(t *testing.T) {
	g := Row{Width: 64, Rows: 2}
	x, n, out := g.Buf(BufRow), g.Buf(BufU32), g.Buf(BufOut)
	sh := g.Shared(64, false)
	tid := g.Column()
	c := g.LoopN(g.LoadAt(n, g.UConst(0)))
	it := g.Iter(c)
	acc := g.Fold(c, g.Const(0))
	top := g.Barrier(it)
	w := g.Write(sh, tid, g.LoadAt(x, g.UBin(OpAdd, g.UBin(OpMul, it, g.UConst(64)), tid)), top)
	bar := g.Barrier(w)
	nb := g.UBin(OpRem, g.UBin(OpAdd, tid, g.UConst(1)), g.UConst(64))
	g.SetFold(acc, g.Bin(OpAdd, acc, g.LoadAt(sh, g.After(nb, bar))))
	g.Store(out, acc)
	ps, err := g.Plan(Facts{Target: Target{API: APIPTX, ISA: ir.ISA{Family: ir.FamilyPTX, Version: 80}, Lanes: 32}})
	if err != nil {
		t.Fatal(err)
	}
	k, err := ps[0].G.Emit(ps[0].S)
	if err != nil {
		t.Fatal(err)
	}
	depth, bars, inLoop, order := 0, 0, 0, []ir.Kind{}
	for _, o := range k.Ops {
		switch o.Kind {
		case ir.OpLoop:
			depth++
		case ir.OpEndLoop:
			depth--
		case ir.OpBarrier:
			bars++
			if depth > 0 {
				inLoop++
			}
		}
		if depth > 0 && (o.Kind == ir.OpBarrier || o.Kind == ir.OpStore || o.Kind == ir.OpLoad && o.Args[0] == 4) {
			order = append(order, o.Kind)
		}
	}
	want := []ir.Kind{ir.OpBarrier, ir.OpStore, ir.OpBarrier, ir.OpLoad}
	if bars != 2 || inLoop != 2 || len(order) != len(want) {
		t.Fatalf("%d barriers, %d in the loop, body order %v; want 2, 2, %v", bars, inLoop, order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("body order %v, want %v", order, want)
		}
	}
}
