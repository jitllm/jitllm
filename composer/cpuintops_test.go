package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// TestCPUIndexTakesEveryIntegerForm executes row maps reading a float at an
// index an integer op makes from the row's words, the index a general
// register's: an arithmetic shift by a constant and by a word, a signed
// division and remainder by a word -- zero, -1 and the least value among
// them -- and by a constant, and an unsigned division and remainder by 0, 1
// and a constant past 2^31. Each row is held to the float its index picks.
func TestCPUIndexTakesEveryIntegerForm(t *testing.T) { gateCPUIndexTakesEveryIntegerForm(t, hostArm()) }
func gateCPUIndexTakesEveryIntegerForm(t *testing.T, cpuOn cpuArm) {
	const R, N = 8, 64
	U, I := ir.U32, ir.I32
	idiv := func(x, y int32, rem bool) int32 {
		switch {
		case y == 0 && rem:
			return x
		case y == 0:
			return -1
		case y == -1 && rem:
			return 0
		case y == -1:
			return -x
		case rem:
			return x % y
		}
		return x / y
	}
	udiv := func(x, y uint32, rem bool) uint32 {
		switch {
		case y == 0 && rem:
			return x
		case y == 0:
			return ^uint32(0)
		case rem:
			return x % y
		}
		return x / y
	}
	words := []uint32{0, 1, 7, 100, 0xFFFFFFF9, 0x80000000, 0x7FFFFFFF, 0xFFFFFF9C}
	divs := []uint32{3, 0, 0xFFFFFFFF, 7, 0xFFFFFFFD, 0xFFFFFFFF, 2, 0x80000000}
	for _, c := range []struct {
		name  string
		op    Op
		typ   ir.Type
		imm   uint32
		byReg bool
		want  func(w, b uint32) uint32
	}{
		{"sar by 3", OpSar, I, 3, false, func(w, b uint32) uint32 { return uint32(int32(w) >> 3) }},
		{"sar by a word", OpSar, I, 0, true, func(w, b uint32) uint32 { return uint32(int32(w) >> (b & 31)) }},
		{"signed div by a word", OpDiv, I, 0, true, func(w, b uint32) uint32 { return uint32(idiv(int32(w), int32(b), false)) }},
		{"signed rem by a word", OpRem, I, 0, true, func(w, b uint32) uint32 { return uint32(idiv(int32(w), int32(b), true)) }},
		{"signed div by -7", OpDiv, I, uint32(0xFFFFFFF9), false, func(w, b uint32) uint32 { return uint32(idiv(int32(w), -7, false)) }},
		{"signed rem by 5", OpRem, I, 5, false, func(w, b uint32) uint32 { return uint32(idiv(int32(w), 5, true)) }},
		{"unsigned div by 0", OpDiv, U, 0, false, func(w, b uint32) uint32 { return udiv(w, 0, false) }},
		{"unsigned rem by 0", OpRem, U, 0, false, func(w, b uint32) uint32 { return udiv(w, 0, true) }},
		{"unsigned div by 1", OpDiv, U, 1, false, func(w, b uint32) uint32 { return w }},
		{"unsigned rem by 1", OpRem, U, 1, false, func(w, b uint32) uint32 { return 0 }},
		{"unsigned div by 2^31+3", OpDiv, U, 0x80000003, false, func(w, b uint32) uint32 { return w / 0x80000003 }},
		{"unsigned rem by 2^31+3", OpRem, U, 0x80000003, false, func(w, b uint32) uint32 { return w % 0x80000003 }},
	} {
		t.Run(c.name, func(t *testing.T) {
			var x exprs
			r := x.coord(0)
			w := x.at(0, []int{R, 1}, r, x.u(0))
			b := x.u(c.imm)
			if c.byReg {
				b = x.at(1, []int{R, 1}, r, x.u(0))
			}
			ix := x.op(OpAnd, U, x.op(c.op, c.typ, w, b), x.u(N-1))
			v := x.un(OpBitcast, U, x.load(2, ix))
			out := Operand{Kind: OperandValue, Value: 3, Name: "o", Elem: U, Out: true}
			p := Prim{Kind: PrimMap, Shape: []int{R, 1}, Expr: x.e, Facts: Facts{Extents: map[int]int{2: N}},
				In: []Operand{{Kind: OperandValue, Value: 0, Name: "w", Elem: U}, {Kind: OperandValue, Value: 1, Name: "b", Elem: U},
					{Kind: OperandValue, Value: 2, Name: "f", Elem: ir.F32}},
				Stores: []PrimStore{{Out: out, Ax: []int{r, x.u(0)}, Ext: []int{R, 1}, Value: v}}}
			g := &graph.Graph{Values: []graph.Value{{Name: "w", Width: 1, Elem: graph.ElemU32}, {Name: "b", Width: 1, Elem: graph.ElemU32},
				{Name: "f", Width: 1}, {Name: "o", Width: 1, Elem: graph.ElemU32}}}
			fs := make([]float32, N)
			for i := range fs {
				fs[i] = 0.5 + float32(i)
			}
			got := make([]uint32, R)
			runCPUPrim(t, cpuOn, p, g, map[string][]uint32{"w": words, "b": divs, "f": f32Words(fs), "o": got})
			for i, wi := range words {
				want := math.Float32bits(fs[c.want(wi, divs[i])&(N-1)])
				if got[i] != want {
					t.Fatalf("row %d (word %#x, divisor %#x) is %#x, want %#x", i, wi, divs[i], got[i], want)
				}
			}
		})
	}
}
