package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
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
	// is neither 0 nor 1: one rounding).
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
	W       MatVecWeight
	Bias    bool
	// Alone is a hint: what reads the prim's result does not fuse into it.
	Alone bool
	// Epi is a hint: the consumer kinds that may fuse into the prim, a bit
	// each; 0 for any.
	Epi   uint32
	Float int
	Facts Facts
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
		x, z := b.in(o.In[0]), b.in(o.In[1])
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
