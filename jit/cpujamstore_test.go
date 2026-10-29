package jit

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/jit/ir"
)

// TestCPUJamStoresAtAFlatIndex executes on the CPU sums over a column-major
// matrix's rows stored at a flat
// index r*S, the rows jammed into the sum loop's lanes several vectors a
// trip: each vector stores at its own rows, and at S > 1 each lane at its
// own place.
func TestCPUJamStoresAtAFlatIndex(t *testing.T) {
	U, F := ir.U32, ir.F32
	const R, K = 64, 32
	for _, S := range []int{1, 3} {
		name := fmt.Sprintf("%d rows of %d stored every %d", R, K, S)
		in := Operand{Kind: OperandValue, Value: 0, Elem: F}
		out := Operand{Kind: OperandScratch, Name: "jam.out", Elem: F}
		var x exprs
		r, k := x.coord(0), x.coord(1)
		x.load(0, x.op(OpAdd, U, x.op(OpMul, U, k, x.u(R)), r))
		var ix exprs
		ix.op(OpMul, U, ix.coord(0), ix.u(uint32(S)))
		p := Prim{Kind: PrimReduce, Shape: []int{R, K}, In: []Operand{in}, Out: out, Combine: OpAdd, Type: F,
			Expr: x.e, Index: ix.e, Facts: Facts{Extents: map[int]int{0: R * K}, OutExtent: R * S}}
		dev := cpuDevice()
		reg := Default()
		kn, err := generate(p)
		if err != nil {
			t.Fatal(err)
		}
		ks, err := reg.Profiles[DeviceCPU].on(dev).run([]*Kernel{kn}, dev)
		if err != nil {
			t.Fatal(err)
		}
		jammed := false
		for _, k := range ks {
			jammed = jammed || slices.Contains(k.Passes, "jam")
		}
		if !jammed {
			for _, k := range ks {
				t.Logf("%s passes %v", k.Name, k.Passes)
			}
			t.Fatalf("%s: no loop jammed: the form under test was not selected", name)
		}
		pcs, err := reg.Lower(ks, dev)
		if err != nil {
			t.Fatal(err)
		}
		var codes []Code
		for _, c := range pcs {
			codes = append(codes, c.Code)
		}
		rng := rand.New(rand.NewSource(int64(S)))
		xs := make([]float32, R*K)
		for i := range xs {
			xs[i] = float32(rng.Intn(17) - 8)
		}
		poison := math.Float32bits(float32(math.NaN()))
		got := make([]uint32, R*S+16)
		for i := range got {
			got[i] = poison
		}
		if _, err := runCPUBound(codes, xs, 1, [3][]uint32{}, nil, func(o Operand) []uint32 {
			switch {
			case o.Name == out.Name:
				return got
			case o.Kind == OperandValue && o.Value == 0:
				return unsafe.Slice((*uint32)(unsafe.Pointer(&xs[0])), len(xs))
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		bad := 0
		for i := range R * S {
			want := poison
			if i%S == 0 {
				s := float32(0)
				for j := range K {
					s += xs[j*R+i/S]
				}
				want = math.Float32bits(s)
			}
			if got[i] != want {
				if bad == 0 {
					t.Errorf("%s: element %d is %v, want %v", name, i, math.Float32frombits(got[i]), math.Float32frombits(want))
				}
				bad++
			}
		}
		if bad > 0 {
			t.Errorf("%s: %d of %d elements part", name, bad, R*S)
		}
	}
}

// TestCPUJamLooksUpByAWeightWord executes on the CPU a sum over k of a
// table's entry at a weight word's low bits times x[k][t], the tokens t
// jammed into lanes: the weight word, the same in every lane, is an address
// and stays in a general register.
func TestCPUJamLooksUpByAWeightWord(t *testing.T) {
	U, F := ir.U32, ir.F32
	const T, K = 32, 64
	in := Operand{Kind: OperandValue, Value: 0, Elem: F}
	w := Operand{Kind: OperandWeight, Region: 0, Elem: U}
	tab := Operand{Kind: OperandScratch, Name: "jam.table", Elem: F}
	out := Operand{Kind: OperandScratch, Name: "jam.out", Elem: F}
	var x exprs
	tk, k := x.coord(0), x.coord(1)
	e := x.load(2, x.op(OpAnd, U, x.load(1, k), x.u(15)))
	x.op(OpMul, F, e, x.load(0, x.op(OpAdd, U, x.op(OpMul, U, k, x.u(T)), tk)))
	table := make([]uint32, 16)
	for i := range table {
		table[i] = math.Float32bits(float32(i) - 7.5)
	}
	p := Prim{Kind: PrimReduce, Shape: []int{T, K}, In: []Operand{in, w, tab}, Out: out, Combine: OpAdd, Type: F,
		Expr: x.e, Facts: Facts{Extents: map[int]int{0: T * K, 1: K, 2: 16}}}
	dev := cpuDevice()
	reg := Default()
	kn, err := generate(p)
	if err != nil {
		t.Fatal(err)
	}
	ks, err := reg.Profiles[DeviceCPU].on(dev).run([]*Kernel{kn}, dev)
	if err != nil {
		t.Fatal(err)
	}
	jammed := false
	for _, k := range ks {
		jammed = jammed || slices.Contains(k.Passes, "jam")
	}
	if !jammed {
		for _, k := range ks {
			t.Logf("%s passes %v", k.Name, k.Passes)
		}
		t.Fatal("no loop jammed: the form under test was not selected")
	}
	pcs, err := reg.Lower(ks, dev)
	if err != nil {
		t.Fatal(err)
	}
	var codes []Code
	for _, c := range pcs {
		codes = append(codes, c.Code)
	}
	rng := rand.New(rand.NewSource(7))
	xs := make([]float32, T*K)
	for i := range xs {
		xs[i] = float32(rng.Intn(9) - 4)
	}
	ws := make([]uint32, K)
	for i := range ws {
		ws[i] = rng.Uint32()
	}
	poison := math.Float32bits(float32(math.NaN()))
	got := make([]uint32, T+16)
	for i := range got {
		got[i] = poison
	}
	if _, err := runCPUBound(codes, xs, 1, [3][]uint32{ws}, nil, func(o Operand) []uint32 {
		switch {
		case o.Name == tab.Name:
			return table
		case o.Name == out.Name:
			return got
		case o.Kind == OperandValue && o.Value == 0:
			return unsafe.Slice((*uint32)(unsafe.Pointer(&xs[0])), len(xs))
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	bad := 0
	for i := range T {
		s := float32(0)
		for j := range K {
			s += math.Float32frombits(table[ws[j]&15]) * xs[j*T+i]
		}
		if got[i] != math.Float32bits(s) {
			if bad == 0 {
				t.Errorf("token %d sums %v, want %v", i, math.Float32frombits(got[i]), s)
			}
			bad++
		}
	}
	for _, g := range got[T:] {
		if g != poison {
			t.Fatal("written past the tokens")
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d tokens part", bad, T)
	}
}
