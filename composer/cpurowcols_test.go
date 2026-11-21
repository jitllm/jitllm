package composer

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/composer/ir"
)

// TestCPUSumsEachRowsOwnColumns executes on the CPU sums over each row's
// own first ((5r mod K) + 1) columns: a range whose end reads the row. Each
// row counts its own columns, so its count stays a general register's (an
// induction of the rows is vector data) and the rows are not jammed into
// lanes that would share one count. The control is a range of every row's
// K columns at 296 rows, its rows jammed into lanes with a tail after the
// whole trips.
func TestCPUSumsEachRowsOwnColumns(t *testing.T) { gateCPUSumsEachRowsOwnColumns(t, hostArm()) }
func gateCPUSumsEachRowsOwnColumns(t *testing.T, cpuOn cpuArm) {
	for _, tc := range []struct {
		rows  int
		byRow bool
	}{{256, true}, {296, false}, {296, true}} {
		t.Run(fmt.Sprintf("%d_rows_by_row_%v", tc.rows, tc.byRow), func(t *testing.T) { jamTailSums(t, cpuOn, tc.rows, tc.byRow) })
	}
}

func jamTailSums(t *testing.T, cpuOn cpuArm, R int, byRow bool) {
	U, F := ir.U32, ir.F32
	const K = 32
	cols := func(r int) int {
		if byRow {
			return (5*r)&(K-1) + 1
		}
		return K
	}
	in := Operand{Kind: OperandValue, Value: 0, Elem: F}
	out := Operand{Kind: OperandScratch, Name: "tail.out", Elem: F}
	var x exprs
	x.load(0, x.op(OpAdd, U, x.op(OpMul, U, x.coord(1), x.u(uint32(R))), x.coord(0)))
	var lo, hi exprs
	lo.u(0)
	if byRow {
		hi.op(OpAdd, U, hi.op(OpAnd, U, hi.op(OpMul, U, hi.coord(0), hi.u(5)), hi.u(K-1)), hi.u(1))
	} else {
		hi.u(K)
	}
	var ix exprs
	ix.coord(0)
	p := Prim{Kind: PrimReduce, Shape: []int{R, K}, In: []Operand{in}, Out: out, Combine: OpAdd, Type: F,
		Expr: x.e, Index: ix.e, Facts: Facts{Range: &Range{Axis: 1, First: lo.e, End: hi.e}, Extents: map[int]int{0: R * K}, OutExtent: R}}
	dev := cpuDevice(cpuOn)
	reg := Default()
	kn, err := generate(p)
	if err != nil {
		t.Fatal(err)
	}
	ks, err := reg.Profiles[DeviceCPU].on(dev).run([]*Kernel{kn}, dev)
	if err != nil {
		t.Fatal(err)
	}
	tails := 0
	for _, k := range ks {
		t.Logf("%s passes %v, %d loops", k.Name, k.Passes, len(k.Loops))
		for _, l := range k.Loops {
			if l.Of > 0 {
				tails++
			}
		}
	}
	if !byRow && tails == 0 {
		t.Fatalf("no tail loop at %d rows of K columns: the control's jammed form was not selected", R)
	}
	pcs, err := reg.Lower(ks, dev)
	if err != nil {
		t.Fatal(err)
	}
	var codes []Code
	for _, c := range pcs {
		codes = append(codes, c.Code)
	}
	xs := make([]float32, R*K)
	for i := range xs {
		xs[i] = float32(i%13 - 6)
	}
	poison := math.Float32bits(float32(math.NaN()))
	got := make([]uint32, R+16)
	for i := range got {
		got[i] = poison
	}
	if _, err := runCPUBound(cpuOn, codes, xs, 1, [3][]uint32{}, nil, func(o Operand) []uint32 {
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
	for i := range R + 16 {
		want := poison
		if i < R {
			s := float32(0)
			for j := range cols(i) {
				s += xs[j*R+i]
			}
			want = math.Float32bits(s)
		}
		if got[i] != want {
			if bad == 0 {
				t.Errorf("row %d is %v, want %v", i, math.Float32frombits(got[i]), math.Float32frombits(want))
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d rows part", bad, R)
	}
	t.Logf("%d rows, %d tail loop(s)", R, tails)
}
