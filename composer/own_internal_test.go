package composer

import (
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
)

// TestOwnIsWhereNoTwoPointsWrite states a scratch its calls' own where each
// point of the space writes a place no other point does: at its own
// coordinates, or at a flat index whose strides keep the points apart. A
// stride under the inner axis's extent, the coordinates swapped, or an axis
// the index does not move must not be stated own.
func TestOwnIsWhereNoTwoPointsWrite(t *testing.T) {
	o := Operand{Kind: OperandScratch, Name: "s"}
	flat := func(stride, inner int, swap bool) *Kernel {
		k := &Kernel{Space: []int{4, 8}}
		c0, c1 := k.add(Node{Op: OpCoord, A: 0, Type: ir.U32}), k.add(Node{Op: OpCoord, A: 1, Type: ir.U32})
		if swap {
			c0, c1 = c1, c0
		}
		ix := k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, c0, k.u32(stride)), k.bin(OpMul, ir.U32, c1, k.u32(inner)))
		b := k.buf(Buf{Kind: BufOut, Operand: o, Extent: 64})
		k.Stores = append(k.Stores, Store{Buf: b, Ix: ix, Val: c0, At: -1})
		return k
	}
	at := func(swap bool) *Kernel {
		k := &Kernel{Space: []int{4, 8}}
		c0, c1 := k.add(Node{Op: OpCoord, A: 0, Type: ir.U32}), k.add(Node{Op: OpCoord, A: 1, Type: ir.U32})
		ax := []int{c0, c1}
		sh := []int{4, 8}
		if swap {
			ax, sh = []int{c1, c0}, []int{8, 4}
		}
		b := k.buf(Buf{Kind: BufOut, Operand: o, Shape: sh})
		k.Stores = append(k.Stores, Store{Buf: b, Ax: ax, Val: c0, At: -1})
		return k
	}
	for _, c := range []struct {
		name string
		k    *Kernel
		own  bool
	}{
		{"rows of 8", flat(8, 1, false), true},
		{"rows of 10, a gap after each", flat(10, 1, false), true},
		{"strided columns", flat(1, 4, false), true},
		{"rows of 4 over 8 columns", flat(4, 1, false), false},
		{"column-major", flat(8, 1, true), true},
		{"columns of 2 over 4 rows", flat(2, 1, true), false},
		{"a column the index does not move", flat(8, 0, false), false},
		{"at its own coordinates", at(false), true},
		{"at its coordinates swapped", at(true), false},
	} {
		if got := len(c.k.ownWrites()) == 1; got != c.own {
			t.Errorf("%s: own %v, want %v", c.name, got, c.own)
		}
	}
}
