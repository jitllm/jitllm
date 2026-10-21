package compose

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// Program is a run of a block's ops as the composer takes them: values of
// a step's Rows rows and the primitive ops over them, each a function of
// its operands and attributes alone. Lower forms each op as its own row
// group, nothing fused; the passes after it (Fuse) transform the groups.
type Program struct {
	Rows  int
	Vals  []Value
	Prims []Prim
	// Out is whether a value is read past the program (the block's
	// caller): what a pass may not drop.
	Out []bool
}

// ValElem is a value's element: a float, or an int8 operand (its codes
// and scales, two bindings).
type ValElem uint8

const (
	ValF32 ValElem = iota
	ValQ8
	// ValU32 is a table of words (row offsets, descriptors, pages).
	ValU32
)

// Value is a program value: Width elements a row, rows contiguous, in
// buffer Buf (values of one Buf share it; -1 a buffer of its own).
type Value struct {
	Elem  ValElem
	Width int
	Buf   int
}

// PrimKind is a primitive op.
type PrimKind uint8

const (
	// PrimAdd is In[0] + Alpha*In[1] (a bound scalar, Const, where Alpha
	// is neither 0 nor 1: one rounding); with one operand, In[0] plus the
	// prim's bound vector.
	PrimAdd PrimKind = iota
	PrimMul
	PrimCopy
	// PrimAct is Act of In[0], scaled by Alpha first where it is neither 0
	// nor 1.
	PrimAct
	// PrimActMul is the gated activation of In[0] times In[1], GLU's
	// bounds where it states them.
	PrimActMul
	// PrimClamp is In[0] held to [-Alpha, Alpha].
	PrimClamp
	// PrimScale is In[0] times the bound scalar Vec.
	PrimScale
	// PrimNorm is In[0] normed over its row: RMS by the bound vector Vec,
	// or LayerNorm with the bound bias VecB.
	PrimNorm
	// PrimQuant is In[0]'s int8 operand over windows of Win (32 where it
	// states none) clamped to the row.
	PrimQuant
	// PrimMatVec is the weight W's product with In[0], an int8 operand,
	// or with In[Float] where W is a float format; plus the weight's bias
	// where Bias. Facts are its planning's.
	PrimMatVec
	// PrimRope is In[0]'s rows of Rope.Heads heads of Rope.HD, each head's
	// first Rope.NRot dimensions turned by the table In[1] (NORM or NEOX
	// pairs; Split two tables), the rest passed through, each row landing in
	// Out[0] at the offset its row's word of In[2] names.
	PrimRope
	// PrimKVWrite is a row's K (In[0], KV.KW wide) and V (In[1], KV.VW) into
	// the paged history Out[0] (K) and Out[1] (V): each row's place from
	// its descriptor In[3] and the page table In[4]; K's first KV.NRot of
	// each of KV.Heads heads turned by the table In[2] where KV.Turn, the
	// rest as it is; V packed in binary16 pairs where KV.F16.
	PrimKVWrite
	// PrimSoftmax is each (row, head, split) item's softmax over the slots
	// of its split its row attends: In[0] the scores, Paged.Chunk slots an
	// item, In[1] the row descriptors; Out[0] the weights at the slots,
	// Out[1] the item's maximum and sum, 32 copies each past its padded
	// numerators (the merge's partial layout).
	PrimSoftmax
	// PrimScores is each (row, head, split, slot)'s score: the head's query
	// In[0] dot the slot's key in the paged pool In[1] (through the page
	// table In[2] and the row descriptors In[3]), scaled by Scale, capped
	// by Softcap where set, -inf where the row does not attend; into
	// Out[0], Paged.Chunk slots an item.
	PrimScores
	// PrimAcc is each (row, head, split, value element)'s sum over its
	// split's keys of the weight In[0] (Paged.Chunk slots an item) times the
	// key's value in the paged pool In[1], through the page table In[2] and
	// the row descriptors In[3]; into the partials Out[0] (padded numerators
	// a split, the merge's layout).
	PrimAcc
	// PrimMerge is each (row, head, value element)'s output from the splits'
	// partials In[0] (each split's numerators, then its maximum and sum):
	// the splits weighted by exp of their maximum less the greatest, the
	// weighted sums (and the sink In[1]'s exp, where present) the
	// normaliser.
	PrimMerge
	// PrimGatherAdd is In[0], rows of Out[0]'s width, each row plus the
	// row of the prim's bound bank (its vector) In[1]'s word for it names:
	// In[1] a word a row, Width of them a program row.
	PrimGatherAdd
	// PrimMap is Map of its operands at each element of Out[0]: In's
	// values (one narrower than Out[0] read at its element's place over
	// the ratio, a word one as a float), then the prim's bound vector where
	// Bound (every row's, at the column), then Alpha where they fall one
	// short.
	PrimMap
	// PrimExtern is an op the composer does not form here: its operands
	// and results, which no pass moves anything across.
	PrimExtern
	numPrimKinds
)

// Prim is a primitive op over program values.
type Prim struct {
	Kind    PrimKind
	In, Out []int
	Alpha   float32
	Act     kernels.ActKind
	GLU     jlm.GLU
	Eps     float32
	Layer   bool
	Win     int
	Paged   Paged
	Scale   float32
	Softcap float32
	Rope    RopeAttrs
	KV      KVAttrs
	W       MatVecWeight
	Bias    bool
	Map     MapFn
	Bound   bool
	// Alone is a hint: what reads the prim's result does not fuse into it.
	Alone bool
	// Epi is a hint: the consumer kinds that may fuse into the prim, a bit
	// each; 0 for any.
	Epi   uint32
	Float int
	Facts Facts
}

// RopeAttrs is a rotary's heads, their width and the turned dimensions.
type RopeAttrs struct {
	Heads, HD, NRot int
	Neox, Split     bool
}

// KVAttrs is a paged write's widths and page, the heads K turns and how.
type KVAttrs struct {
	KW, VW, Page, Heads, HD, NRot int
	Turn, Neox, Whole, F16        bool
}

// BindKind is what a group's buffer binds to.
type BindKind uint8

const (
	BindVal   BindKind = iota // value Index
	BindQ8                    // value Index's int8 codes
	BindQ8X                   // and its scales
	BindVec                   // prim Index's bound vector (or scalar)
	BindVecB                  // prim Index's bound bias
	BindConst                 // prim Index's bound constant (an add's Alpha)
	BindWQS                   // prim Index's weight: its payload,
	BindWD                    // its scales,
	BindWSC                   // its sub-block scales,
	BindWBias                 // its bias
)

// Bind is a group buffer's binding.
type Bind struct {
	Kind  BindKind
	Index int
}

// Group is a row group of the program: the group, what each of its
// buffers binds to, and the prims it computes.
type Group struct {
	Row   Row
	Binds []Bind
	Prims []int
	// Extern is an extern prim's place: no group.
	Extern bool
	// Segs, where set, are the group's members as one segmented launch
	// (Siblings): its Row and Binds are unused.
	Segs []Group
	// Sched, where set, is the schedule the group's shape states.
	Sched *Schedule
	// Contract says the group holds a contraction, planned under Facts;
	// any other is planned under the target alone.
	Contract bool
	Facts    Facts
}

// Lower is the program's ops each its own group, nothing fused: every
// operand loaded, every result stored.
func (p *Program) Lower() ([]Group, error) {
	var gs []Group
	for i := range p.Prims {
		b := &builder{p: p, binds: map[Bind]int{}}
		b.g.Prims = []int{i}
		o := p.Prims[i]
		if o.Kind == PrimExtern {
			b.g.Extern = true
			gs = append(gs, b.g)
			continue
		}
		if o.Kind == PrimQuant {
			gs = append(gs, b.quantAlone(o))
			continue
		}
		if o.Kind == PrimScores {
			b.scores(o)
			gs = append(gs, b.g)
			continue
		}
		if o.Kind == PrimAcc {
			b.acc(o)
			gs = append(gs, b.g)
			continue
		}
		if o.Kind == PrimMerge {
			b.g.Row = Row{Width: o.Paged.Rows * o.Paged.Heads * o.Paged.valueDim(), Rows: 1}
			sink := -1
			if len(o.In) > 1 {
				sink = b.buf(BufRow, Bind{BindVal, o.In[1]})
			}
			b.g.Row.PagedMerge(b.buf(BufRow, Bind{BindVal, o.In[0]}), b.buf(BufOut, Bind{BindVal, o.Out[0]}), sink, o.Paged)
			gs = append(gs, b.g)
			continue
		}
		if o.Kind == PrimGatherAdd {
			n, k := p.Vals[o.Out[0]].Width, p.Vals[o.In[1]].Width
			b.g.Row = Row{Width: p.Rows * k * n, Rows: 1}
			r := &b.g.Row
			in, sel := b.buf(BufRow, Bind{BindVal, o.In[0]}), b.buf(BufU32, Bind{BindVal, o.In[1]})
			bank := b.buf(BufRow, Bind{BindVec, i})
			r.StoreAt(b.buf(BufOut, Bind{BindVal, o.Out[0]}), r.IndexedBias(in, sel, bank, n), r.Column())
			gs = append(gs, b.g)
			continue
		}
		if o.Kind == PrimMap {
			if err := b.mapPrim(i); err != nil {
				return nil, err
			}
			gs = append(gs, b.g)
			continue
		}
		if o.Kind == PrimSoftmax {
			b.g.Row = Row{Width: 1, Rows: o.Paged.Rows * o.Paged.Heads * o.Paged.Splits}
			b.g.Facts = o.Facts
			b.softmax(o)
			gs = append(gs, b.g)
			continue
		}
		if o.Kind == PrimRope || o.Kind == PrimKVWrite {
			parts, err := p.lowerParts(i)
			if err != nil {
				return nil, err
			}
			gs = append(gs, parts...)
			continue
		}
		b.g.Row = Row{Width: p.Vals[o.Out[0]].Width, Rows: p.Rows}
		if o.Kind == PrimMatVec {
			b.g.Row.Width = o.W.Rows
		}
		if err := b.lowerPrim(i); err != nil {
			return nil, err
		}
		gs = append(gs, b.g)
	}
	return gs, nil
}

// builder is a group being lowered.
type builder struct {
	p     *Program
	g     Group
	binds map[Bind]int
}

func (b *builder) buf(k BufKind, bd Bind) int {
	key := Bind{Kind: bd.Kind | BindKind(k)<<4, Index: bd.Index}
	if i, ok := b.binds[key]; ok {
		return i
	}
	b.g.Row.Bufs = append(b.g.Row.Bufs, Buf{Kind: k})
	b.g.Binds = append(b.g.Binds, bd)
	b.binds[key] = len(b.g.Row.Bufs) - 1
	return len(b.g.Row.Bufs) - 1
}

// in is value v at the group's element.
func (b *builder) in(v int) int { return b.g.Row.Load(b.buf(BufRow, Bind{BindVal, v})) }

func (b *builder) c(f float32) int { return b.g.Row.Const(f) }

// quantWin is a quantize's window: 32 where it states none, ending at the
// row.
func quantWin(win, w int) int {
	if win <= 0 {
		win = 32
	}
	return min(win, w)
}

// quantAlone is a quantize of a value no op of its region computes: rows of
// the value's width padded to whole blocks of 32.
func (b *builder) quantAlone(q Prim) Group {
	iw := b.p.Vals[q.In[0]].Width
	kp := (iw + 31) / 32 * 32
	b.g.Row = Row{Width: b.p.Rows * kp / 4, Rows: 1}
	x := b.buf(BufRow, Bind{BindVal, q.In[0]})
	a, ax := b.buf(BufOutU32, Bind{BindQ8, q.Out[0]}), b.buf(BufOut, Bind{BindQ8X, q.Out[0]})
	win := quantWin(q.Win, kp)
	if kp != iw {
		b.g.Row.Quantize8Padded(x, a, ax, win, iw, kp)
	} else {
		b.g.Row.Quantize8(x, a, ax, win)
	}
	return b.g
}

// lowerPrim adds prim i's nodes and the store of its output.
func (b *builder) lowerPrim(i int) error {
	o := b.p.Prims[i]
	r := &b.g.Row
	var y int
	switch o.Kind {
	case PrimAdd:
		x := b.in(o.In[0])
		var z int
		if len(o.In) == 1 {
			// The prim's bound vector, every row's.
			z = r.Load(b.buf(BufVec, Bind{BindVec, i}))
		} else {
			z = b.in(o.In[1])
		}
		if o.Alpha != 0 && o.Alpha != 1 {
			al := r.Load(b.buf(BufScalar, Bind{BindConst, i}))
			y = r.Add(Node{Kind: NBinary, Op: OpFma, A: al, B: z, C: x})
		} else {
			y = r.Bin(OpAdd, x, z)
		}
	case PrimMul:
		y = r.Bin(OpMul, b.in(o.In[0]), b.in(o.In[1]))
	case PrimCopy:
		y = b.in(o.In[0])
	case PrimAct:
		x := b.in(o.In[0])
		if o.Alpha != 0 && o.Alpha != 1 {
			x = r.Bin(OpMul, x, b.c(o.Alpha))
		}
		y = r.Add(Node{Kind: NUnary, Op: OpAct, Act: o.Act, A: x})
	case PrimActMul:
		y = b.gated(o, b.in(o.In[0]), b.in(o.In[1]))
	case PrimClamp:
		if o.Alpha <= 0 {
			return fmt.Errorf("compose: a clamp to %v", o.Alpha)
		}
		y = r.Bin(OpMin, r.Bin(OpMax, b.in(o.In[0]), b.c(-o.Alpha)), b.c(o.Alpha))
	case PrimScale:
		y = r.Bin(OpMul, b.in(o.In[0]), r.Load(b.buf(BufScalar, Bind{BindVec, i})))
	case PrimNorm:
		y = b.normed(i, o, b.in(o.In[0]), b.p.Vals[o.In[0]].Width)
	case PrimMatVec:
		w := o.W
		w.QS, w.D, w.SC = b.buf(BufU32, Bind{BindWQS, i}), -1, -1
		if !kernels.IsFloat(w.T) {
			w.D = b.buf(BufU32, Bind{BindWD, i})
		}
		if w.T.Info().PerSuper > 1 {
			w.SC = b.buf(BufU32, Bind{BindWSC, i})
		}
		a := MatVecOperand{Toks: b.p.Rows}
		if kernels.IsFloat(w.T) {
			a.X = b.buf(BufRow, Bind{BindVal, o.In[o.Float]})
		} else {
			a.A, a.AX = b.buf(BufU32, Bind{BindQ8, o.In[0]}), b.buf(BufRow, Bind{BindQ8X, o.In[0]})
		}
		v, err := r.MatVec(w, a)
		if err != nil {
			return err
		}
		if o.Bias {
			v = r.Bin(OpAdd, v, r.Load(b.buf(BufVec, Bind{BindWBias, i})))
		}
		b.g.Contract, b.g.Facts = true, o.Facts
		y = v
	default:
		return fmt.Errorf("compose: prim %d inside a region", o.Kind)
	}
	r.Stores = append(r.Stores, Store{Buf: b.buf(BufOut, Bind{BindVal, o.Out[0]}), Val: y})
	return nil
}

// normed is x normed over the row of w: RMS by the bound vector, or
// LayerNorm with its bound bias.
func (b *builder) normed(i int, o Prim, x, w int) int {
	r := &b.g.Row
	wt := r.Load(b.buf(BufVec, Bind{BindVec, i}))
	inv := b.c(1 / float32(w))
	rms := func(x int) int {
		ss := r.Add(Node{Kind: NReduce, Red: RedSum, A: r.Bin(OpMul, x, x)})
		return r.Add(Node{Kind: NUnary, Op: OpRsqrt, A: r.Bin(OpAdd, r.Bin(OpMul, ss, inv), b.c(o.Eps))})
	}
	if !o.Layer {
		return r.Bin(OpMul, r.Bin(OpMul, x, rms(x)), wt)
	}
	mean := r.Bin(OpMul, r.Add(Node{Kind: NReduce, Red: RedSum, A: x}), inv)
	c := r.Bin(OpSub, x, mean)
	bias := r.Load(b.buf(BufVec, Bind{BindVecB, i}))
	return r.Bin(OpAdd, r.Bin(OpMul, r.Bin(OpMul, c, rms(c)), wt), bias)
}

// gated is act(g)*up, GLU's bounds where stated, in the order of
// operations the kernels' own gated forms use.
func (b *builder) gated(o Prim, gt, up int) int {
	r := &b.g.Row
	u := o.GLU
	if !u.Bounded() {
		return r.Add(Node{Kind: NBinary, Op: OpActMul, Act: o.Act, A: gt, B: up})
	}
	exp := func(x int) int { return r.Add(Node{Kind: NUnary, Op: OpExp, A: x}) }
	bound := func(x int, bd float32) int {
		t := r.Add(Node{Kind: NUnary, Op: OpAct, Act: kernels.ActTanh, A: r.Bin(OpMul, b.c(1/bd), x)})
		return r.Bin(OpMul, b.c(bd), t)
	}
	g := gt
	if u.Clamp {
		g = r.Bin(OpMin, gt, b.c(u.GateMax))
	}
	z := g
	if u.SigScale != 0 && u.SigScale != 1 {
		z = r.Bin(OpMul, b.c(u.SigScale), g)
	}
	denom := r.Bin(OpAdd, b.c(1), exp(r.Bin(OpSub, b.c(0), z)))
	var a int
	if u.GateTanh != 0 {
		a = r.Bin(OpMul, bound(g, u.GateTanh), r.Bin(OpDiv, b.c(1), denom))
	} else {
		a = r.Bin(OpDiv, g, denom)
	}
	v := up
	if u.Clamp {
		v = r.Bin(OpMax, r.Bin(OpMin, up, b.c(u.UpMax)), b.c(u.UpMin))
	}
	if u.UpShift != 0 {
		v = r.Bin(OpAdd, v, b.c(u.UpShift))
	}
	if u.UpTanh != 0 {
		v = bound(v, u.UpTanh)
	}
	return r.Bin(OpMul, a, v)
}

// lowerParts is a prim lowered as several groups, each a launch: a rotary's
// pass-through copy and its turned pairs, a paged write's K and V.
func (p *Program) lowerParts(i int) ([]Group, error) {
	o := p.Prims[i]
	var gs []Group
	part := func(w int, body func(b *builder)) {
		b := &builder{p: p, binds: map[Bind]int{}}
		b.g.Prims = []int{i}
		b.g.Row = Row{Width: w, Rows: p.Rows}
		body(b)
		gs = append(gs, b.g)
	}
	row := func(b *builder, v int) int { return b.buf(BufRow, Bind{BindVal, v}) }
	u32 := func(b *builder, v int) int { return b.buf(BufU32, Bind{BindVal, v}) }
	switch o.Kind {
	case PrimRope:
		ra := o.Rope
		iw := ra.Heads * ra.HD
		if ra.HD <= 0 || ra.NRot <= 0 || ra.NRot%2 != 0 || ra.NRot > ra.HD {
			return nil, fmt.Errorf("compose: a rotary of %d of a head of %d", ra.NRot, ra.HD)
		}
		if ra.NRot < ra.HD {
			part(iw, func(b *builder) {
				r := &b.g.Row
				x, off := row(b, o.In[0]), u32(b, o.In[2])
				r.StoreAt(b.buf(BufOut, Bind{BindVal, o.Out[0]}), r.Load(x), r.UBin(OpAdd, r.Load(off), r.Column()))
			})
		}
		part(ra.Heads*ra.NRot/2, func(b *builder) {
			in, cs, off := row(b, o.In[0]), b.buf(BufVec, Bind{BindVal, o.In[1]}), u32(b, o.In[2])
			b.g.Row.RotaryPairs(in, cs, off, b.buf(BufOut, Bind{BindVal, o.Out[0]}), ra.Heads, ra.HD, ra.NRot, ra.Neox, ra.Split)
		})
	case PrimKVWrite:
		kv := o.KV
		if !kv.Turn || kv.NRot < kv.HD || kv.Whole {
			part(kv.KW, func(b *builder) {
				r := &b.g.Row
				in, desc, tab := row(b, o.In[0]), u32(b, o.In[3]), u32(b, o.In[4])
				pid, off := r.PagedPos(desc, tab, kv.Page)
				r.StoreAt(b.buf(BufOut, Bind{BindVal, o.Out[0]}), r.Load(in), r.PagedK(pid, off, kv.KW, kv.Page))
			})
		}
		if kv.Turn {
			part(kv.Heads*kv.NRot/2, func(b *builder) {
				in, cs := row(b, o.In[0]), b.buf(BufVec, Bind{BindVal, o.In[2]})
				desc, tab := u32(b, o.In[3]), u32(b, o.In[4])
				b.g.Row.PagedRotaryPairs(in, cs, desc, tab, b.buf(BufOut, Bind{BindVal, o.Out[0]}), kv.Heads, kv.HD, kv.NRot, kv.Neox, kv.Page, kv.KW)
			})
		}
		w := kv.VW
		if kv.F16 {
			w = kv.VW / 2
		}
		part(w, func(b *builder) {
			r := &b.g.Row
			vin, desc, tab := row(b, o.In[1]), u32(b, o.In[3]), u32(b, o.In[4])
			pid, off := r.PagedPos(desc, tab, kv.Page)
			if kv.F16 {
				base := r.UBin(OpMul, r.RowIdx(), r.UConst(kv.VW))
				e := r.UBin(OpAdd, base, r.UBin(OpMul, r.Column(), r.UConst(2)))
				pk := r.Bin(OpPackF16, r.LoadAt(vin, e), r.LoadAt(vin, r.UBin(OpAdd, e, r.UConst(1))))
				at := r.UBin(OpAdd, r.UBin(OpDiv, r.PagedRow(pid, off, kv.Page, kv.KW, 0), r.UConst(2)), r.Column())
				r.StoreAt(b.buf(BufOutU32, Bind{BindVal, o.Out[1]}), pk, at)
				return
			}
			r.StoreAt(b.buf(BufOut, Bind{BindVal, o.Out[1]}), r.Load(vin), r.UBin(OpAdd, r.PagedRow(pid, off, kv.Page, kv.KW, 0), r.Column()))
		})
	}
	return gs, nil
}

// softmax is a staged softmax's naive group: an item a row of one element,
// a loop over its split's keys for the maximum, one over the chunk for the
// weights and their sum, and the partials' 32 copies.
func (b *builder) softmax(o Prim) {
	r, p := &b.g.Row, o.Paged
	u, ub := r.UConst, r.UBin
	scores := b.buf(BufRow, Bind{BindVal, o.In[0]})
	desc := b.buf(BufU32, Bind{BindVal, o.In[1]})
	out, part := b.buf(BufOut, Bind{BindVal, o.Out[0]}), b.buf(BufOut, Bind{BindVal, o.Out[1]})
	item := r.RowIdx()
	sp := ub(OpRem, item, u(p.Splits))
	row := ub(OpDiv, item, u(p.Heads*p.Splits))
	ps := r.pagedSplitOf(desc, row, sp, p.Splits)
	base := ub(OpMul, item, u(p.Chunk))
	c1 := r.LoopN(ub(OpSub, ps.en, ps.st))
	m := r.Fold(c1, r.Const(-math.MaxFloat32))
	r.SetFold(m, r.Bin(OpMax, m, r.LoadAt(scores, ub(OpAdd, base, r.Iter(c1)))))
	c2 := r.Loop(p.Chunk)
	si := r.Iter(c2)
	at := ub(OpAdd, base, si)
	w := r.Select(r.valid(ps, si), r.Add(Node{Kind: NUnary, Op: OpExp, A: r.Bin(OpSub, r.LoadAt(scores, at), m)}), r.Const(0))
	r.StoreAt(out, w, at)
	l := r.Fold(c2, r.Const(0))
	r.SetFold(l, r.Bin(OpAdd, l, w))
	padded := (p.valueDim() + 31) / 32 * 32
	pb := ub(OpMul, item, u(padded+64))
	c3 := r.Loop(32)
	r.StoreAt(part, m, ub(OpAdd, ub(OpAdd, pb, r.Iter(c3)), u(padded)))
	r.StoreAt(part, l, ub(OpAdd, ub(OpAdd, pb, r.Iter(c3)), u(padded+32)))
}

// acc is a staged accumulation's naive group: an element a (row, head,
// split, value element), a loop over its split's keys, each key's page
// looked up and its value times its weight summed.
func (b *builder) acc(o Prim) {
	p := o.Paged
	vd := p.valueDim()
	S, C, P := max(1, p.Splits), p.Chunk, p.Page
	b.g.Row = Row{Width: p.Rows * p.Heads * S * vd, Rows: 1}
	r := &b.g.Row
	u, ub := r.UConst, r.UBin
	wts, v := b.buf(BufRow, Bind{BindVal, o.In[0]}), b.buf(BufRow, Bind{BindVal, o.In[1]})
	table, desc := b.buf(BufU32, Bind{BindVal, o.In[2]}), b.buf(BufU32, Bind{BindVal, o.In[3]})
	part := b.buf(BufOut, Bind{BindVal, o.Out[0]})
	kvRow, gqa := p.KVHeads*p.Dim, p.Heads/p.KVHeads
	dg := r.DigitsOf(r.Column(), vd, S, p.Heads)
	j, sp, h, row := dg[0], dg[1], dg[2], dg[3]
	ps := r.pagedSplitOf(desc, row, sp, S)
	col := j
	if p.MLA == 0 {
		col = ub(OpAdd, ub(OpMul, ub(OpDiv, h, u(gqa)), u(p.Dim)), j)
	}
	// The split's keys, cut in blocks of eight by the block pass: every
	// block masked from the row's first key, a value through its page's
	// remainder selected, the page from the block's first key held at the
	// last 32-aligned tile.
	st := r.Add(Node{Kind: NHint, Hint: HintMultiple, A: ps.st, At: 32})
	c, t := r.RangeLoopForm(st, st, ps.en, C, BlockForm{All: true, Base: ps.ks + 1, Paged: true, Align: 32})
	pid := r.pageOf(table, ps.tab, r.KeyAt(t, ps.live, ps.last), P)
	vv := r.LoadAt(v, ub(OpAdd, ub(OpMul, ub(OpAdd, ub(OpMul, pid, u(P)), r.pageRem(t, P)), u(kvRow)), col))
	item := ub(OpAdd, ub(OpMul, ub(OpAdd, ub(OpMul, row, u(p.Heads)), h), u(S)), sp)
	w := r.LoadAt(wts, ub(OpAdd, ub(OpMul, item, u(C)), ub(OpSub, t, st)))
	acc := r.Fold(c, r.Const(0))
	r.SetFold(acc, r.Add(Node{Kind: NBinary, Op: OpFma, A: w, B: vv, C: acc}))
	padded := (vd + 31) / 32 * 32
	r.StoreAt(part, acc, ub(OpAdd, ub(OpMul, item, u(padded+64)), j))
	// Main's schedule by the shape: a KV head's HeadGroup query heads a
	// thread, sharing each value read, their columns S*vd apart.
	b.g.Sched = &Schedule{Group: 128, Flat: true, TileCols: p.HeadGroup(), TileDigit: 3, Block: 8, Split: 1}
}

// scores is a staged scores' naive group: an element a (row, head, split,
// slot), its key's page from the slot's place held inside the row's keys,
// a loop over the head's dimensions folding q times k, scaled, capped and
// masked.
func (b *builder) scores(o Prim) {
	p := o.Paged
	S, C, P := max(1, p.Splits), p.Chunk, p.Page
	b.g.Row = Row{Width: p.Rows * p.Heads * S * C, Rows: 1}
	r := &b.g.Row
	u, ub := r.UConst, r.UBin
	q, k := b.buf(BufRow, Bind{BindVal, o.In[0]}), b.buf(BufRow, Bind{BindVal, o.In[1]})
	table, desc := b.buf(BufU32, Bind{BindVal, o.In[2]}), b.buf(BufU32, Bind{BindVal, o.In[3]})
	out := b.buf(BufOut, Bind{BindVal, o.Out[0]})
	kvRow, gqa := p.KVHeads*p.Dim, p.Heads/p.KVHeads
	dg := r.DigitsOf(r.Column(), C, S, p.Heads)
	slot, sp, h, row := dg[0], dg[1], dg[2], dg[3]
	ps := r.pagedSplitOf(desc, row, sp, S)
	t := r.Select(ps.live, ub(OpMin, ub(OpMax, ub(OpAdd, ps.st, slot), ps.ks), ps.last), u(0))
	pid := r.pageOf(table, ps.tab, t, P)
	rh := ub(OpAdd, ub(OpMul, row, u(p.Heads)), h)
	qb := ub(OpMul, rh, u(p.Dim))
	kb := ub(OpAdd, ub(OpAdd, ub(OpMul, pid, u(P*kvRow)), ub(OpMul, ub(OpMul, ub(OpDiv, h, u(gqa)), u(p.Dim)), u(P))), r.pageRem(t, P))
	step := P
	if p.MLA > 0 {
		kb, step = ub(OpMul, ub(OpAdd, ub(OpMul, pid, u(P)), r.pageRem(t, P)), u(kvRow)), 1
	}
	c := r.Loop(p.Dim)
	d := r.Iter(c)
	acc := r.Fold(c, r.Const(0))
	kv := r.LoadAt(k, ub(OpAdd, kb, ub(OpMul, d, u(step))))
	qv := r.LoadAt(q, ub(OpAdd, qb, d))
	r.SetFold(acc, r.Add(Node{Kind: NBinary, Op: OpFma, A: qv, B: kv, C: acc}))
	score := r.Bin(OpMul, acc, r.Const(o.Scale))
	if o.Softcap > 0 {
		e := r.Add(Node{Kind: NUnary, Op: OpExp, A: r.Bin(OpMul, r.Const(2/o.Softcap), score)})
		score = r.Bin(OpMul, r.Const(o.Softcap), r.Bin(OpSub, r.Const(1), r.Bin(OpDiv, r.Const(2), r.Bin(OpAdd, e, r.Const(1)))))
	}
	ob := ub(OpAdd, ub(OpMul, ub(OpAdd, ub(OpMul, rh, u(S)), sp), u(C)), slot)
	r.StoreAt(out, r.Select(r.valid(ps, slot), score, r.Const(-math.MaxFloat32)), ob)
	// Main's schedule by the shape: HeadGroup heads a thread sharing each
	// key read, the dimensions unrolled (in blocks of 32 past 512 products)
	// over 4/G chains.
	G := p.HeadGroup()
	b.g.Sched = &Schedule{Group: 128, Flat: true, TileCols: G, TileDigit: 3, Unroll: 32, UnrollOver: 512, Chains: max(1, 4/G)}
	if p.Lanes > 1 {
		// A slot over a subgroup: its dimensions dealt over the lanes, one
		// chain a head, the head group a tile of rows.
		b.g.Sched = &Schedule{Group: 128, Flat: true, Lanes: p.Lanes, LaneLoops: true, GroupRows: true, Tile: G, TileDigit: 3, Unroll: 1}
	}
}

// MapFn is an elementwise map's function.
type MapFn uint8

const (
	MapAdd MapFn = iota + 1
	MapSub
	MapMul
	MapDiv
	MapMax
	MapMin
	MapExp
	MapSigmoid
	MapSoftplus
	MapSqrt
	MapWhere // In[1] where In[0] > 0, else In[2]
	MapLess  // 1 where In[0] < In[1], else 0
	MapFma
)

// Arity is how many operands f takes.
func (f MapFn) Arity() int {
	switch f {
	case MapExp, MapSigmoid, MapSoftplus, MapSqrt:
		return 1
	case MapWhere, MapFma:
		return 3
	case MapAdd, MapSub, MapMul, MapDiv, MapMax, MapMin, MapLess:
		return 2
	}
	return 0
}

// mapPrim is map prim i's naive group: a row of Out[0]'s width a program
// row, each operand loaded at the element, the map, the store.
func (b *builder) mapPrim(i int) error {
	o := b.p.Prims[i]
	out := b.p.Vals[o.Out[0]].Width
	b.g.Row = Row{Width: out, Rows: b.p.Rows}
	r := &b.g.Row
	var ops []int
	for _, v := range o.In {
		n := b.p.Vals[v].Width
		if n <= 0 || out%n != 0 {
			return fmt.Errorf("compose: a map over %d of an operand of %d", out, n)
		}
		var x int
		switch {
		case b.p.Vals[v].Elem == ValU32:
			ix := r.Column()
			if n != out {
				ix = r.UBin(OpDiv, ix, r.UConst(out/n))
			}
			x = r.Add(Node{Kind: NUnary, Op: OpFloat, A: r.LoadAt(b.buf(BufU32, Bind{BindVal, v}), r.UBin(OpAdd, r.UBin(OpMul, r.RowIdx(), r.UConst(n)), ix))})
		case n == out:
			x = b.in(v)
		default:
			ix := r.UBin(OpAdd, r.UBin(OpMul, r.RowIdx(), r.UConst(n)), r.UBin(OpDiv, r.Column(), r.UConst(out/n)))
			x = r.LoadAt(b.buf(BufRow, Bind{BindVal, v}), ix)
		}
		ops = append(ops, x)
	}
	if o.Bound {
		ops = append(ops, r.Load(b.buf(BufVec, Bind{BindVec, i})))
	}
	if len(ops) == o.Map.Arity()-1 {
		ops = append(ops, b.c(o.Alpha))
	}
	if len(ops) != o.Map.Arity() {
		return fmt.Errorf("compose: a map of %d operands taking %d", len(ops), o.Map.Arity())
	}
	var y int
	switch o.Map {
	case MapAdd, MapSub, MapMul, MapDiv, MapMax, MapMin:
		y = r.Bin([...]Op{MapAdd: OpAdd, MapSub: OpSub, MapMul: OpMul, MapDiv: OpDiv, MapMax: OpMax, MapMin: OpMin}[o.Map], ops[0], ops[1])
	case MapExp:
		y = r.Add(Node{Kind: NUnary, Op: OpExp, A: ops[0]})
	case MapSqrt:
		y = r.Add(Node{Kind: NUnary, Op: OpSqrt, A: ops[0]})
	case MapSigmoid:
		y = r.Add(Node{Kind: NUnary, Op: OpAct, Act: kernels.ActSigmoid, A: ops[0]})
	case MapSoftplus:
		y = r.Add(Node{Kind: NUnary, Op: OpAct, Act: kernels.ActSoftplus, A: ops[0]})
	case MapWhere:
		y = r.Select(r.Less(r.Const(0), ops[0]), ops[1], ops[2])
	case MapLess:
		y = r.Select(r.Less(ops[0], ops[1]), r.Const(1), r.Const(0))
	case MapFma:
		y = r.Add(Node{Kind: NBinary, Op: OpFma, A: ops[0], B: ops[1], C: ops[2]})
	default:
		return fmt.Errorf("compose: a map of function %d", o.Map)
	}
	r.Store(b.buf(BufOut, Bind{BindVal, o.Out[0]}), y)
	return nil
}
