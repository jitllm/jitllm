package compose

import "github.com/jitllm/jitllm/jit_old/gpu/kernels"

// Scan is a recurrent block's update over a chunk's tokens, run on lane rows
// (Schedule.Lanes): the group's rows are the state's VHeads*VDim rows of K
// elements, a row's lanes (the group's Width) holding K/Width elements each,
// element lane+i*Width in register i, carried through the tokens.
type Scan struct {
	VHeads, VDim, K, KHeads int
	// KeyTiled pairs value head vh with key head vh % KHeads, not vh / (VHeads/KHeads).
	KeyTiled bool
	// Slot is a sequence's floats in the state.
	Slot int
	// Chans is the convolution's row, q | k | v, where the inputs are one.
	Chans int
	// Delta subtracts the row's projection on the key, beta-gated, and
	// q and k are L2-normed (Eps) and scaled (k by Scale, q by its square);
	// otherwise the update writes x*dt along the key and adds D*x.
	Delta bool
	// ChanDecay is a decay per key channel, else one a value head.
	ChanDecay bool
	// BARep is the value heads a group of the beta|alpha interleave.
	BARep      int
	Eps, Scale float32
	// Bound, nonzero, is exp(Bound*sigmoid(-A*(alpha+dt))) for the per-channel decay.
	Bound float32
	// Separate is x, B and C each a buffer of their own, x through its bias
	// and SiLU, a decay row of A per value head.
	Separate bool
}

// ScanBufs are a scan's buffers, each a node buffer of the group (-1 for
// one its form does not read).
type ScanBufs struct {
	S, SOut, Out, Desc int
	Conv, X, B, C      int
	BA, Alpha, Beta    int
	DtRaw, Dt, A       int
	ConvBias, D        int
}

// ScanBuffers declares a scan's buffers in its form's parameter order.
func (g *Row) ScanBuffers(sc Scan) ScanBufs {
	f := ScanBufs{Conv: -1, X: -1, B: -1, C: -1, BA: -1, Alpha: -1, Beta: -1, DtRaw: -1, ConvBias: -1, D: -1}
	row := func() int { return g.Buf(BufRow) }
	f.S = row()
	switch {
	case sc.Separate:
		f.X, f.B, f.C, f.DtRaw = row(), row(), row(), row()
	case !sc.Delta:
		f.Conv, f.DtRaw = row(), row()
	case sc.ChanDecay:
		f.Conv, f.Alpha, f.Beta = row(), row(), row()
	default:
		f.Conv, f.BA = row(), row()
	}
	f.Dt, f.A = row(), row()
	if !sc.Delta {
		if !sc.Separate {
			f.ConvBias = row()
		}
		f.D = row()
	}
	f.Out, f.SOut, f.Desc = g.Buf(BufOut), g.Buf(BufOut), g.Buf(BufU32)
	return f
}

// Scan steps the state through the chunk's tokens: each token's output to
// Out at [token][row], the state read at the sequence's in slot (word 1 of
// Desc) and written at its out slot (word 2), word 0 the tokens.
func (g *Row) Scan(sc Scan, f ScanBufs) {
	u, ub := g.UConst, g.UBin
	add := func(x, off int) int {
		if off == 0 {
			return x
		}
		return ub(OpAdd, x, u(off))
	}
	ld := func(b, x, off int) int { return g.LoadAt(b, add(x, off)) }
	L, K := g.Width, sc.K
	nr := K / L
	rows := sc.VHeads * sc.VDim
	qk := sc.KHeads * K
	lane, row := g.Column(), g.RowIdx()
	vh := ub(OpDiv, row, u(sc.VDim))
	kh := ub(OpDiv, vh, u(sc.VHeads/sc.KHeads))
	if sc.KeyTiled {
		kh = ub(OpRem, vh, u(sc.KHeads))
	}
	sBase := ub(OpAdd, ub(OpMul, row, u(K)), lane)
	inAt := ub(OpAdd, sBase, ub(OpMul, g.LoadAt(f.Desc, u(1)), u(sc.Slot)))
	outAt := ub(OpAdd, sBase, ub(OpMul, g.LoadAt(f.Desc, u(2)), u(sc.Slot)))
	kBase := ub(OpAdd, ub(OpMul, kh, u(K)), lane)
	chBase := ub(OpAdd, ub(OpMul, vh, u(K)), lane)
	silu := func(x int) int { return g.Act(kernels.ActSiLU, x) }

	c := g.LoopN(g.LoadAt(f.Desc, u(0)))
	st := make([]int, nr)
	for i := range st {
		st[i] = g.Fold(c, ld(f.S, inAt, i*L))
	}
	t := g.Iter(c)
	th := ub(OpAdd, ub(OpMul, t, u(sc.VHeads)), vh)
	vIdx := ub(OpAdd, ub(OpMul, t, u(rows)), row)
	key, query, decay := make([]int, nr), make([]int, nr), make([]int, nr)
	var value, beta, dt int
	switch {
	case sc.Separate:
		bc := ub(OpAdd, ub(OpMul, t, u(K)), lane)
		for i := 0; i < nr; i++ {
			key[i], query[i] = ld(f.B, bc, i*L), ld(f.C, bc, i*L)
		}
		value = g.LoadAt(f.X, vIdx)
		dt = g.Act(kernels.ActSoftplus, g.Bin(OpAdd, g.LoadAt(f.DtRaw, th), g.LoadAt(f.Dt, vh)))
		for i := range decay {
			decay[i] = g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpMul, ld(f.A, chBase, i*L), dt)})
		}
	default:
		cb := ub(OpAdd, ub(OpMul, t, u(sc.Chans)), kBase)
		cv := ub(OpAdd, ub(OpMul, t, u(sc.Chans)), row)
		in := func(off, bias int) int {
			x := ld(f.Conv, cb, off)
			if f.ConvBias >= 0 {
				x = g.Bin(OpAdd, x, ld(f.ConvBias, kBase, bias))
			}
			return silu(x)
		}
		for i := 0; i < nr; i++ {
			query[i], key[i] = in(i*L, i*L), in(qk+i*L, qk+i*L)
		}
		v := ld(f.Conv, cv, 2*qk)
		if f.ConvBias >= 0 {
			v = g.Bin(OpAdd, v, ld(f.ConvBias, row, 2*qk))
		}
		value = silu(v)
	}
	if !sc.Delta && !sc.Separate {
		dt = g.Act(kernels.ActSoftplus, g.Bin(OpAdd, g.LoadAt(f.DtRaw, th), g.LoadAt(f.Dt, vh)))
		dec := g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpMul, g.LoadAt(f.A, vh), dt)})
		for i := range decay {
			decay[i] = dec
		}
	}
	if sc.Delta {
		norm := func(x []int) int {
			acc := g.Const(0)
			for _, v := range x {
				acc = g.fma(v, v, acc)
			}
			sum := g.Add(Node{Kind: NReduce, Red: RedSum, A: acc, At: 1})
			mean := g.Bin(OpMul, sum, g.Const(1/float32(K)))
			return g.Add(Node{Kind: NUnary, Op: OpRsqrt, A: g.Bin(OpAdd, mean, g.Const(sc.Eps))})
		}
		iq, ik := norm(query), norm(key)
		s := g.Const(sc.Scale)
		for i := 0; i < nr; i++ {
			key[i] = g.Bin(OpMul, g.Bin(OpMul, key[i], ik), s)
			query[i] = g.Bin(OpMul, g.Bin(OpMul, g.Bin(OpMul, query[i], iq), s), s)
		}
		softDecay := func(alpha, d, a int) int {
			sp := g.Act(kernels.ActSoftplus, g.Bin(OpAdd, alpha, d))
			return g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpMul, a, sp)})
		}
		if sc.ChanDecay {
			ab := ub(OpAdd, ub(OpMul, t, u(sc.VHeads*K)), chBase)
			for i := range decay {
				alpha, d, a := ld(f.Alpha, ab, i*L), ld(f.Dt, chBase, i*L), ld(f.A, chBase, i*L)
				if sc.Bound != 0 {
					e := g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpMul, g.Bin(OpAdd, alpha, d), a)})
					decay[i] = g.Add(Node{Kind: NUnary, Op: OpExp, A: g.Bin(OpDiv, g.Const(sc.Bound), g.Bin(OpAdd, e, g.Const(1)))})
				} else {
					decay[i] = softDecay(alpha, d, a)
				}
			}
			beta = g.Act(kernels.ActSigmoid, g.LoadAt(f.Beta, th))
		} else {
			rep := u(sc.BARep)
			grp := ub(OpDiv, vh, rep)
			baOff := ub(OpAdd, ub(OpMul, grp, u(2*sc.BARep)), ub(OpSub, vh, ub(OpMul, grp, rep)))
			bo := ub(OpAdd, ub(OpMul, t, u(2*sc.VHeads)), baOff)
			dec := softDecay(ld(f.BA, bo, sc.BARep), g.LoadAt(f.Dt, vh), g.LoadAt(f.A, vh))
			for i := range decay {
				decay[i] = dec
			}
			beta = g.Act(kernels.ActSigmoid, g.LoadAt(f.BA, bo))
		}
	}

	dec := make([]int, nr)
	for i := range dec {
		dec[i] = g.Bin(OpMul, st[i], decay[i])
	}
	var w int
	if sc.Delta {
		sk := g.Const(0)
		for i := range dec {
			sk = g.fma(dec[i], key[i], sk)
		}
		w = g.Bin(OpMul, g.Bin(OpSub, value, g.Reduce(RedSum, sk)), beta)
	} else {
		w = g.Bin(OpMul, value, dt)
	}
	o := g.Const(0)
	for i := range st {
		next := g.fma(w, key[i], dec[i])
		o = g.fma(next, query[i], o)
		g.SetFold(st[i], next)
	}
	o = g.Reduce(RedSum, o)
	if !sc.Delta {
		o = g.fma(g.LoadAt(f.D, vh), value, o)
	}
	g.StoreAt(f.Out, o, vIdx)
	for i := range st {
		g.StoreAt(f.SOut, st[i], add(outAt, i*L))
	}
}
