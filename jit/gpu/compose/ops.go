package compose

import (
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

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
// row of the group's width), as RoPERows computes it; split is XD-RoPE's two
// tables a row (RoPERowsSplit), NEOX pairs whose halves turn by their own.
func (g *Row) Rotary(src, cs, hd, nrot int, neox, split bool) int {
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
	// A row's table is nrot floats, (cos, sin) a pair; split (XD-RoPE) is two
	// tables a row, a pair's second element turning by the second's.
	tabW := nrot
	if split {
		tabW = 2 * nrot
	}
	tab := g.UBin(OpAdd, g.UBin(OpMul, g.RowIdx(), g.UConst(tabW)), g.UBin(OpMul, p, g.UConst(2)))
	if split {
		tab = g.Select(first, tab, g.UBin(OpAdd, tab, g.UConst(nrot)))
	}
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

// RotaryPairs is the rotary of a row of heads of hd, a group element a
// rotated pair (the group's Width heads*nrot/2): the pair's two elements of
// src -- NORM (2p, 2p+1) or NEOX (p, p+nrot/2) of head h -- turned by the
// row's (cos, sin) at 2p of cs (nrot a row; split, XD-RoPE's two tables a
// row, the second element turning by the second table's), both stored at
// the row's offset off (word row of BufU32 off) plus the element's place in
// out. The dimensions past nrot are not written.
func (g *Row) RotaryPairs(src, cs, off, out, heads, hd, nrot int, neox, split bool) {
	u, ub := g.UConst, g.UBin
	pairs := nrot / 2
	e, row := g.Column(), g.RowIdx()
	p, h := ub(OpRem, e, u(pairs)), ub(OpDiv, e, u(pairs))
	head := ub(OpMul, h, u(hd))
	stride := 1
	var base int
	if neox {
		base, stride = ub(OpAdd, head, p), pairs
	} else {
		base = ub(OpAdd, head, ub(OpMul, p, u(2)))
	}
	tabW := nrot
	if split {
		tabW = 2 * nrot
	}
	at := ub(OpAdd, ub(OpMul, p, u(2)), ub(OpMul, row, u(tabW)))
	x := ub(OpAdd, base, ub(OpMul, row, u(heads*hd)))
	c, s := g.LoadAt(cs, at), g.LoadAt(cs, ub(OpAdd, at, u(1)))
	c2, s2 := c, s
	if split {
		c2, s2 = g.LoadAt(cs, ub(OpAdd, at, u(nrot))), g.LoadAt(cs, ub(OpAdd, at, u(nrot+1)))
	}
	x0, x1 := g.LoadAt(src, x), g.LoadAt(src, ub(OpAdd, x, u(stride)))
	o := ub(OpAdd, g.Load(off), base)
	g.StoreAt(out, g.Bin(OpSub, g.Bin(OpMul, x0, c), g.Bin(OpMul, x1, s)), o)
	g.StoreAt(out, g.Bin(OpAdd, g.Bin(OpMul, x0, s2), g.Bin(OpMul, x1, c2)), ub(OpAdd, o, u(stride)))
}

// PagedPos is where the group's row lands in a paged history (the ABI of
// kernels/paged.go): the page id and the offset in the page of the position
// its descriptor names for the write, the descriptors at desc (PRowWords
// words a row) and the page ids at tab, page positions a page. Two loads of
// the descriptor, one of the table, and their arithmetic.
func (g *Row) PagedPos(desc, tab, page int) (pid, off int) {
	base := g.UBin(OpMul, g.RowIdx(), g.UConst(kernels.PRowWords))
	first := g.LoadAt(desc, g.UBin(OpAdd, base, g.UConst(kernels.PRowTab)))
	pos := g.LoadAt(desc, g.UBin(OpAdd, base, g.UConst(kernels.PRowWrite)))
	pid = g.LoadAt(tab, g.UBin(OpAdd, first, g.UBin(OpDiv, pos, g.UConst(page))))
	return pid, g.UBin(OpRem, pos, g.UConst(page))
}

// PagedK is element column of the group's row in a paged K pool, which is
// transposed: [width][page] a page, so a position's elements are page apart.
func (g *Row) PagedK(pid, off, width, page int) int {
	return g.UBin(OpAdd, g.UBin(OpMul, pid, g.UConst(page*width)),
		g.UBin(OpAdd, g.UBin(OpMul, g.Column(), g.UConst(page)), off))
}

// PagedRow is the place of the group's row in a paged V pool, [page][width]
// a page: (pid*page + off)*width + col, col its first column there.
func (g *Row) PagedRow(pid, off, page, width, col int) int {
	at := g.UBin(OpMul, g.UBin(OpAdd, g.UBin(OpMul, pid, g.UConst(page)), off), g.UConst(width))
	if col != 0 {
		at = g.UBin(OpAdd, at, g.UConst(col))
	}
	return at
}

// Loop opens a fold loop of trip trips and returns its id; Iter is its
// iterator, Fold a value it carries from start, stepped by SetFold.
func (g *Row) Loop(trip int) int {
	g.Loops = append(g.Loops, trip)
	g.Trip = append(g.Trip, -1)
	return len(g.Loops) - 1
}

// Iter is loop c's iterator: 0, 1, ... up to its trip count.
func (g *Row) Iter(c int) int { return g.Add(Node{Kind: NIter, At: c}) }

// Fold is a value loop c carries, start before its first trip; its step
// is set by SetFold once the body that reads it is built.
func (g *Row) Fold(c, start int) int { return g.Add(Node{Kind: NFold, At: c, B: start, A: -1}) }

// SetFold makes next the value fold f carries into the next trip.
func (g *Row) SetFold(f, next int) { g.Nodes[f].A = next }

// Flag is 1 where cond holds and 0 where it does not, an index.
func (g *Row) Flag(cond int) int { return g.Select(cond, g.UConst(1), g.UConst(0)) }

// Rank is element e's place among n keys in descending order: how many
// keys exceed its own (mine), and how many equal it at a lower index --
// ties to the lower index, so the ranks are a permutation. key(f) is the
// key of index node f, built inside the counting loop. Equality is "neither
// less nor greater" and the conjunction a product, of 0/1 indices.
func (g *Row) Rank(n, e, mine int, key func(f int) int) int {
	c := g.Loop(n)
	f := g.Iter(c)
	rank := g.Fold(c, g.UConst(0))
	kf := key(f)
	gt := g.Flag(g.Less(mine, kf))
	lt := g.Flag(g.Less(kf, mine))
	eq := g.UBin(OpSub, g.UConst(1), g.UBin(OpAdd, gt, lt))
	lower := g.Flag(g.Less(f, e))
	g.SetFold(rank, g.UBin(OpAdd, rank, g.UBin(OpAdd, gt, g.UBin(OpMul, eq, lower))))
	return rank
}

// FoldMax is the maximum of x(f) over f < n, from x(0), in index order.
func (g *Row) FoldMax(n int, x func(f int) int) int {
	c := g.Loop(n)
	m := g.Fold(c, x(g.UConst(0)))
	g.SetFold(m, g.Bin(OpMax, m, x(g.Iter(c))))
	return m
}

// FoldSum is the sum of x(f) over f < n, from zero, in index order.
func (g *Row) FoldSum(n int, x func(f int) int) int {
	c := g.Loop(n)
	s := g.Fold(c, g.Const(0))
	g.SetFold(s, g.Bin(OpAdd, s, x(g.Iter(c))))
	return s
}

// LoopN is a loop whose trip count is index node n's value, known at run
// time: it never runs unrolled.
func (g *Row) LoopN(n int) int {
	c := g.Loop(-1)
	g.Trip[c] = n
	return c
}

// tripOf is the node a run-time loop c runs for, -1 for a static one.
func (g *Row) tripOf(c int) int {
	if c < len(g.Trip) {
		return g.Trip[c]
	}
	return -1
}

// storeLoop is the loop whose body a store runs in: the one whose iterator
// its index or value reads (not through a fold's result), -1 for none.
func (g *Row) storeLoop(st Store) int {
	memo := map[int]int{}
	var in func(n int) int
	in = func(n int) int {
		if c, ok := memo[n]; ok {
			return c
		}
		memo[n] = -1
		c := -1
		if nd := g.Nodes[n]; nd.Kind == NIter {
			c = nd.At
		}
		for _, o := range g.operands(n) {
			if x := in(o); x > c {
				c = x
			}
		}
		memo[n] = c
		return c
	}
	c := in(st.Val)
	if st.Indexed {
		c = max(c, in(st.Ix))
	}
	return c
}

// PartFold is the group element's part of a row's fold: element p of parts
// consecutive runs of the row's k elements at x (BufRow, read by index),
// step(element, acc) over its run in element order from zero.
func (g *Row) PartFold(x, k, parts int, step func(v, acc int) int) int {
	u, ub := g.UConst, g.UBin
	per := k / parts
	start := ub(OpMul, g.Column(), u(per))
	c := g.Loop(per)
	acc := g.Fold(c, g.Const(0))
	g.SetFold(acc, step(g.LoadAt(x, ub(OpAdd, start, g.Iter(c))), acc))
	return acc
}

// SquareSumParts is a row's sum of squares in parts (PartFold); NormApplyParts
// reads them back.
func (g *Row) SquareSumParts(x, k, parts int) int {
	return g.PartFold(x, k, parts, func(v, acc int) int { return g.fma(v, v, acc) })
}

// SumParts is a row's sum in parts.
func (g *Row) SumParts(x, k, parts int) int {
	return g.PartFold(x, k, parts, func(v, acc int) int { return g.Bin(OpAdd, acc, v) })
}

// partsMean is the mean of a row of k from its parts partials of sums at
// sum, the row's first at base.
func (g *Row) partsMean(sum, base, k, parts int) int {
	s := g.Const(0)
	for p := 0; p < parts; p++ {
		s = g.Bin(OpAdd, s, g.LoadAt(sum, g.UBin(OpAdd, base, g.UConst(p))))
	}
	return g.Bin(OpMul, s, g.Const(1/float32(k)))
}

// VarParts is a row's sum of squared deviations from its mean in parts, the
// mean from the parts of sums at sum.
func (g *Row) VarParts(x, sum, k, parts int) int {
	base := g.UBin(OpMul, g.UBin(OpDiv, g.Column(), g.UConst(parts)), g.UConst(parts))
	mean := g.partsMean(sum, base, k, parts)
	return g.PartFold(x, k, parts, func(v, acc int) int {
		d := g.Bin(OpSub, v, mean)
		return g.fma(d, d, acc)
	})
}

// LayerNormApplyParts is LayerNorm's element from a row's parts of sums and
// of squared deviations: (x - mean) / sqrt(var + eps) times the weight w
// (BufVec), plus the bias bias (BufVec) where bias >= 0.
func (g *Row) LayerNormApplyParts(x, w, bias, sum, vr, k, parts int, eps float32) int {
	base := g.UBin(OpMul, g.RowIdx(), g.UConst(parts))
	mean := g.partsMean(sum, base, k, parts)
	vs := g.Const(0)
	for p := 0; p < parts; p++ {
		vs = g.Bin(OpAdd, vs, g.LoadAt(vr, g.UBin(OpAdd, base, g.UConst(p))))
	}
	varr := g.Bin(OpAdd, g.Bin(OpMul, vs, g.Const(1/float32(k))), g.Const(eps))
	scale := g.Bin(OpDiv, g.Const(1), g.Add(Node{Kind: NUnary, Op: OpSqrt, A: varr}))
	v := g.Bin(OpMul, g.Bin(OpMul, g.Bin(OpSub, g.Load(x), mean), scale), g.Load(w))
	if bias >= 0 {
		v = g.Bin(OpAdd, v, g.Load(bias))
	}
	return v
}

func (g *Row) fma(a, b, c int) int { return g.Add(Node{Kind: NBinary, Op: OpFma, A: a, B: b, C: c}) }

// NormApplyParts is RMSNorm's element from a row's sums of squares in parts
// partials at part (BufRow, read by index): the element of x times
// 1/sqrt(sum/k + eps) times the weight w (BufVec), one more where addOne.
func (g *Row) NormApplyParts(x, w, part, k, parts int, eps float32, addOne bool) int {
	u, ub := g.UConst, g.UBin
	base := ub(OpMul, g.RowIdx(), u(parts))
	sum := g.Const(0)
	for p := 0; p < parts; p++ {
		sum = g.Bin(OpAdd, sum, g.LoadAt(part, ub(OpAdd, base, u(p))))
	}
	mean := g.Bin(OpAdd, g.Bin(OpMul, sum, g.Const(1/float32(k))), g.Const(eps))
	scale := g.Bin(OpDiv, g.Const(1), g.Add(Node{Kind: NUnary, Op: OpSqrt, A: mean}))
	wt := g.Load(w)
	if addOne {
		wt = g.Bin(OpAdd, wt, g.Const(1))
	}
	return g.Bin(OpMul, g.Bin(OpMul, g.Load(x), scale), wt)
}

// Softcap is c*tanh(x/c), as c*(1 - 2/(exp(2x/c) + 1)), where x is not a
// masked logit (below -1e30), which passes as it is.
func (g *Row) Softcap(x int, c float32) int {
	e := g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpMul, g.Const(2/c), x)})
	t := g.Bin(OpSub, g.Const(1), g.Bin(OpDiv, g.Const(2), g.Bin(OpAdd, e, g.Const(1))))
	return g.Select(g.Less(x, g.Const(-1e30)), x, g.Bin(OpMul, g.Const(c), t))
}

// Argmax stores at out (BufOutU32) each row's argmax of x: the index of its
// greatest element, the lowest where several are.
func (g *Row) Argmax(x, out int) {
	r := g.Add(Node{Kind: NReduce, Red: RedArgmax, A: x})
	g.Stages = append(g.Stages, Stage{Width: 1, Stores: []Store{{Buf: out, Val: r, Indexed: true, Ix: g.RowIdx()}}})
}

// RouterLogits stores each group row's logit: weight row r (BufRow, rows of
// k, read by index) against token t's row of x (BufRow, read by index),
// the group's rows token-major over the weight's rows (row t*rows + r).
func (g *Row) RouterLogits(w, x, out, rows, k int) {
	u, ub := g.UConst, g.UBin
	row, e := g.RowIdx(), g.Column()
	wv := g.LoadAt(w, ub(OpAdd, ub(OpMul, ub(OpRem, row, u(rows)), u(k)), e))
	xv := g.LoadAt(x, ub(OpAdd, ub(OpMul, ub(OpDiv, row, u(rows)), u(k)), e))
	r := g.Add(Node{Kind: NReduce, Red: RedSum, A: g.Bin(OpMul, wv, xv)})
	g.Stages = append(g.Stages, Stage{Width: 1, Stores: []Store{{Buf: out, Val: r, Indexed: true, Ix: row}}})
}
