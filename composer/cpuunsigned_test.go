package composer

import (
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// unsignedPairs are words on both sides of 2^31, where a signed compare
// orders them the other way, and the all-ones word a table states "none" by.
func unsignedPairs() [][2]uint32 {
	return [][2]uint32{
		{11, 0xFFFFFFFF}, {0xFFFFFFFF, 11}, {0x80000000, 1}, {1, 0x80000000},
		{0x7FFFFFFF, 0x80000000}, {0x80000000, 0x7FFFFFFF}, {5, 5}, {0xFFFFFFFE, 0xFFFFFFFF},
		{0xFFFFFFFF, 0xFFFFFFFF}, {0, 0xFFFFFFFF}, {0x90000000, 0xA0000000}, {0xA0000000, 0x90000000},
		{3, 4}, {4, 3}, {0x80000001, 0x80000000}, {0, 0},
	}
}

// unsignedWant is the three compares' bits unsignedLess states: a < b, then
// a < 0xFFFFFFFF, then 0x80000000 < a.
func unsignedWant(a, b uint32) uint32 {
	var w uint32
	if a < b {
		w |= 1
	}
	if a < 0xFFFFFFFF {
		w |= 2
	}
	if 0x80000000 < a {
		w |= 4
	}
	return w
}

// unsignedLess is the expression of x's words a and b: a < b, a against the
// all-ones word and the top bit against a, each a bit of the word stored.
func unsignedLess(x *exprs, a, b int) int {
	U := ir.U32
	bit := func(c, v int) int { return x.sel(U, c, x.u(uint32(v)), x.u(0)) }
	return x.op(OpOr, U, x.op(OpOr, U, bit(x.op(OpLess, ir.Pred, a, b), 1),
		bit(x.op(OpLess, ir.Pred, a, x.u(0xFFFFFFFF)), 2)),
		bit(x.op(OpLess, ir.Pred, x.u(0x80000000), a), 4))
}

// TestCPUCompareIsUnsigned holds the CPU device's compare of two u32 words
// to the order a u32 has, as the cards' lowerings compare them: in a row
// map, where a row's words sit in general registers, and in a map over a
// row's elements, where they are vector lanes, each against another word,
// the all-ones word and the top bit. A signed compare orders every pair
// across 2^31 the other way.
func TestCPUCompareIsUnsigned(t *testing.T) {
	pairs := unsignedPairs()
	n := len(pairs)
	as, bs := make([]uint32, n), make([]uint32, n)
	for i, p := range pairs {
		as[i], bs[i] = p[0], p[1]
	}
	U := ir.U32
	g := &graph.Graph{Values: []graph.Value{{Name: "a", Width: n, Elem: graph.ElemU32}, {Name: "b", Width: n, Elem: graph.ElemU32}, {Name: "o", Width: n, Elem: graph.ElemU32}}}
	in := []Operand{{Kind: OperandValue, Value: 0, Name: "a", Elem: U}, {Kind: OperandValue, Value: 1, Name: "b", Elem: U}}
	out := Operand{Kind: OperandValue, Value: 2, Name: "o", Elem: U, Out: true}
	for _, c := range []struct {
		name  string
		shape []int
	}{
		{"a word a row", []int{n, 1}},
		{"a row's lanes", []int{1, n}},
	} {
		t.Run(c.name, func(t *testing.T) {
			var x exprs
			r, e := x.coord(0), x.coord(1)
			a, b := x.at(0, c.shape, r, e), x.at(1, c.shape, r, e)
			v := unsignedLess(&x, a, b)
			p := Prim{Kind: PrimMap, Shape: c.shape, Expr: x.e, In: in,
				Stores: []PrimStore{{Out: out, Ax: []int{r, e}, Ext: c.shape, Value: v}}}
			got := make([]uint32, n)
			for i := range got {
				got[i] = 0xdeadbeef
			}
			runCPUPrim(t, hostArm(), p, g, map[string][]uint32{"a": as, "b": bs, "o": got})
			bad := 0
			for i, pr := range pairs {
				if want := unsignedWant(pr[0], pr[1]); got[i] != want {
					bad++
					t.Errorf("%#x against %#x: bits %03b, want %03b", pr[0], pr[1], got[i], want)
				}
			}
			if bad == 0 {
				t.Logf("%d pairs across 2^31 ordered as u32", n)
			}
		})
	}
}
