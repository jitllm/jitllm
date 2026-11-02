package composer

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// TestCPUMapLoadsAtASelectedIndex runs on the CPU a map whose load index is
// a select over its lane's coordinate: a NEOX rotary's partner element,
// x[r, b, d < P ? d+P : d-P], each element of a [R, NB, D] map, alone, with
// a table read at the pair's index, and as the whole turn. Between two maps
// of the form (its input and its output both the form's own), InPlace gave
// the output the input's storage although the map reads the input at
// another element than it writes: half of every row read a partner already
// turned. The pooled selection's turn is this map.
func TestCPUMapLoadsAtASelectedIndex(t *testing.T) {
	if !HostCPULowers() {
		t.Skip("no CPU lowering on this host")
	}
	for _, form := range []string{"partner", "tab0", "table", "turn", "made", "between"} {
		for _, sh := range [][3]int{{1, 1, 16}, {2, 3, 16}, {3, 2, 32}, {2, 2, 8}, {2, 5, 12}, {1, 18, 8}, {1, 19, 8}, {2, 18, 8}} {
			R, NB, D := sh[0], sh[1], sh[2]
			t.Run(fmt.Sprintf("%s/%dx%dx%d", form, R, NB, D), func(t *testing.T) {
				selIndexCase(t, form, R, NB, D)
			})
		}
	}
}

// selIndexCase is one of TestCPUMapLoadsAtASelectedIndex's maps: the
// partner alone, the element times its pair's table entry (the table read
// at a selected index), or the whole NEOX turn.
func selIndexCase(t *testing.T, form string, R, NB, D int) {
	P := D / 2
	U, F := ir.U32, ir.F32
	var x exprs
	r, b, d := x.coord(0), x.coord(1), x.coord(2)
	low := x.op(OpLess, ir.Pred, d, x.u(uint32(P)))
	partner := x.sel(U, low, x.op(OpAdd, U, d, x.u(uint32(P))), x.op(OpSub, U, d, x.u(uint32(P))))
	pair := x.sel(U, low, d, x.op(OpSub, U, d, x.u(uint32(P))))
	xp := func() int { return x.at(0, []int{R, NB, D}, r, b, partner) }
	co := func() int { return x.at(1, []int{NB, P, 2}, b, pair, x.u(0)) }
	switch form {
	case "partner":
		xp()
	case "table":
		x.op(OpMul, F, x.at(0, []int{R, NB, D}, r, b, d), co())
	default:
		sign := x.sel(F, low, x.f(-1), x.f(1))
		c := co()
		si := x.at(1, []int{NB, P, 2}, b, pair, x.u(1))
		x.op(OpAdd, F, x.op(OpMul, F, x.at(0, []int{R, NB, D}, r, b, d), c),
			x.op(OpMul, F, x.op(OpMul, F, xp(), si), sign))
	}
	p := Prim{Kind: PrimMap, Shape: []int{R, NB, D},
		In: []Operand{{Kind: OperandValue, Value: 0, Name: "x", Elem: ir.F32},
			{Kind: OperandScratch, Name: "tab", Elem: ir.F32}},
		Out: Operand{Kind: OperandValue, Value: 1, Name: "y", Elem: ir.F32, Out: true}, Expr: x.e}
	if form == "partner" {
		p.In = p.In[:1]
	}
	prims := []Prim{p}
	if form == "made" || form == "between" {
		// The turn reads a value an earlier map of the form makes, and
		// (between) makes one a later map copies out: its input and its
		// output both the form's own, the output may take the input's
		// storage only where nothing reads the input at another element.
		var m exprs
		m.op(OpMul, F, m.at(0, []int{R, NB, D}, m.coord(0), m.coord(1), m.coord(2)), m.f(1))
		mid := Operand{Kind: OperandScratch, Name: "mid", Elem: ir.F32, Temp: true}
		prims = []Prim{{Kind: PrimMap, Shape: []int{R, NB, D}, In: p.In[:1], Out: mid, Expr: m.e}, p}
		prims[1].In = []Operand{mid, p.In[1]}
		if form == "between" {
			turned := Operand{Kind: OperandScratch, Name: "turned", Elem: ir.F32, Temp: true}
			prims[1].Out = turned
			prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, NB, D}, In: []Operand{turned}, Out: p.Out, Expr: m.e})
		}
	}
	var ks []*Kernel
	for _, p := range prims {
		k, err := generate(p)
		if err != nil {
			t.Fatal(err)
		}
		k.Apart = true
		ks = append(ks, k)
	}
	dev := Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
	r0 := Default()
	ks, err := r0.Profiles[DeviceCPU].on(dev).run(ks, dev)
	if err != nil {
		t.Fatal(err)
	}
	pcs, err := r0.Lower(ks, dev)
	if err != nil {
		t.Fatal(err)
	}
	codes := make([]Code, len(pcs))
	for i, c := range pcs {
		codes[i] = c.Code
	}
	n := R * NB * D
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(i + 1)
	}
	tab := make([]float32, NB*P*2)
	for i := range tab {
		tab[i] = float32(i%7) + 0.5
	}
	got, err := runCPUBound(codes, in, n, [3][]uint32{}, nil, func(o Operand) []uint32 {
		if o.Name != "tab" {
			return nil
		}
		return unsafe.Slice((*uint32)(unsafe.Pointer(&tab[0])), len(tab))
	})
	if err != nil {
		t.Fatal(err)
	}
	bad := 0
	for i := range n {
		dd := i % D
		j := i - dd + (dd+P)%D
		bi := i / D % NB
		e := (bi*P + dd%P) * 2
		sg := float32(1)
		if dd < P {
			sg = -1
		}
		want := in[i]*tab[e] + in[j]*tab[e+1]*sg
		switch form {
		case "partner":
			want = in[j]
		case "table":
			want = in[i] * tab[e]
		}
		if got[i] != want {
			if bad < 4 {
				t.Errorf("element %d (lane %d of its row) is %v, want %v", i, dd, got[i], want)
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d elements read another element than their partner", bad, n)
	}
}
