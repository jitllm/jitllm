package compose

import "github.com/jitllm/jitllm/jit/gpu/kernels"

// The op emitters: each appends one op's fragment -- its loads, its math, its
// stores -- to a group, as a function of the element it is handed. The kernel
// former (Row.Emit) stitches what a group holds into one kernel; nothing here
// is a kernel.

// Buf adds a buffer of kind k to the group and returns its index.
func (g *Row) Buf(k BufKind) int {
	g.Bufs = append(g.Bufs, Buf{Kind: k})
	return len(g.Bufs) - 1
}

// Add appends node n and returns its index.
func (g *Row) Add(n Node) int {
	g.Nodes = append(g.Nodes, n)
	return len(g.Nodes) - 1
}

// Load is buffer b at the element.
func (g *Row) Load(b int) int { return g.Add(Node{Kind: NLoad, Buf: b}) }

// LoadAt is buffer b at index node ix.
func (g *Row) LoadAt(b, ix int) int { return g.Add(Node{Kind: NLoad, Buf: b, Indexed: true, Ix: ix}) }

// Const is the float f.
func (g *Row) Const(f float32) int { return g.Add(Node{Kind: NConst, F: f}) }

// Bin is op of floats a and b.
func (g *Row) Bin(op Op, a, b int) int { return g.Add(Node{Kind: NBinary, Op: op, A: a, B: b}) }

// UConst is the index n.
func (g *Row) UConst(n int) int { return g.Add(Node{Kind: NUConst, At: n}) }

// UBin is op of indices a and b.
func (g *Row) UBin(op Op, a, b int) int { return g.Add(Node{Kind: NUBinary, Op: op, A: a, B: b}) }

// Less is a < b of indices, a condition.
func (g *Row) Less(a, b int) int { return g.Add(Node{Kind: NLess, A: a, B: b}) }

// Select is cond ? a : b.
func (g *Row) Select(cond, a, b int) int { return g.Add(Node{Kind: NSelect, A: a, B: b, C: cond}) }

// Column is the element's place in its row; RowIdx the row's.
func (g *Row) Column() int { return g.Add(Node{Kind: NColumn}) }

// RowIdx is the row's index.
func (g *Row) RowIdx() int { return g.Add(Node{Kind: NRowIdx}) }

// Reduce is red of a over the row.
func (g *Row) Reduce(red Red, a int) int { return g.Add(Node{Kind: NReduce, Red: red, A: a}) }

// Act is activation k of a.
func (g *Row) Act(k kernels.ActKind, a int) int {
	return g.Add(Node{Kind: NUnary, Op: OpAct, Act: k, A: a})
}

// Store writes v at the element into buffer b; StoreAt at index node ix.
func (g *Row) Store(b, v int) { g.Stores = append(g.Stores, Store{Buf: b, Val: v}) }

// StoreAt writes v into buffer b at index node ix.
func (g *Row) StoreAt(b, v, ix int) {
	g.Stores = append(g.Stores, Store{Buf: b, Val: v, Indexed: true, Ix: ix})
}

// RMSScale is 1/sqrt(mean(x*x) + eps) over a row of width w.
func (g *Row) RMSScale(x, w int, eps float32) int {
	mean := g.Bin(OpMul, g.Reduce(RedSum, g.Bin(OpMul, x, x)), g.Const(1/float32(w)))
	return g.Add(Node{Kind: NUnary, Op: OpRsqrt, A: g.Bin(OpAdd, mean, g.Const(eps))})
}

// LayerNormed is (x - mean)*rsqrt(var + eps)*w + b over a row of width w.
func (g *Row) LayerNormed(x, wt, bias, w int, eps float32) int {
	c := g.Bin(OpSub, x, g.Bin(OpMul, g.Reduce(RedSum, x), g.Const(1/float32(w))))
	return g.Bin(OpAdd, g.Bin(OpMul, g.Bin(OpMul, c, g.RMSScale(c, w, eps)), wt), bias)
}

// Rotary is the rotary of a row of heads of hd, its first nrot dimensions
// turned by the table cs (one row of nrot floats, cos then sin a pair, for
// each of the group's rows) in NORM pairs (2p, 2p+1) or NEOX (p, p+nrot/2),
// the rest of each head as it is: the element's value, read from src (a
// row of the group's width), as RoPERows computes it.
func (g *Row) Rotary(src, cs, hd, nrot int, neox bool) int {
	col := g.Column()
	j := g.UBin(OpRem, col, g.UConst(hd)) // the place in the head
	half := nrot / 2
	var p, first, partner int
	if neox {
		p = g.UBin(OpRem, j, g.UConst(half))
		first = g.Less(j, g.UConst(half))
		partner = g.Select(first, g.UBin(OpAdd, col, g.UConst(half)), g.UBin(OpSub, col, g.UConst(half)))
	} else {
		p = g.UBin(OpDiv, j, g.UConst(2))
		first = g.Less(g.UBin(OpRem, j, g.UConst(2)), g.UConst(1))
		partner = g.Select(first, g.UBin(OpAdd, col, g.UConst(1)), g.UBin(OpSub, col, g.UConst(1)))
	}
	// A tail element's pair index is clamped into the table's row: its
	// value is the element as it is, but its loads must stay in bounds.
	p = g.UBin(OpMin, p, g.UConst(half-1))
	tab := g.UBin(OpAdd, g.UBin(OpMul, g.RowIdx(), g.UConst(nrot)), g.UBin(OpMul, p, g.UConst(2)))
	c := g.LoadAt(cs, tab)
	s := g.LoadAt(cs, g.UBin(OpAdd, tab, g.UConst(1)))
	row := g.UBin(OpMul, g.RowIdx(), g.UConst(g.Width))
	x := g.Load(src)
	other := g.LoadAt(src, g.UBin(OpAdd, row, g.UBin(OpMin, partner, g.UConst(g.Width-1))))
	// first: x*c - other*s; second (other is x0): other*s + x*c.
	turned := g.Select(first, g.Bin(OpSub, g.Bin(OpMul, x, c), g.Bin(OpMul, other, s)),
		g.Bin(OpAdd, g.Bin(OpMul, other, s), g.Bin(OpMul, x, c)))
	return g.Select(g.Less(j, g.UConst(nrot)), turned, x)
}
