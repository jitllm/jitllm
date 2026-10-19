package tier

import (
	"fmt"
	"math"
	"slices"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Code generation for the device program: each op's kernels are generated
// from the op itself -- its kind, its values' widths, its attributes and the
// variant's row count -- through the one shape-keyed cache (shapeKern), and
// recorded on the op as launches over generic arguments (its inputs and
// outputs, its bound vectors, the scratch's partial planes, a constant it
// owns). The executor runs an op's launches and nothing else: no kernel is
// taken from a scratch built for an architecture, none is chosen by the slot
// a value lands in. An op with no launches here still runs through the
// executor's own case; their count is the work list (TestGeneratedKinds).

// genArg names a generated launch's buffer.
type genArg uint8

const (
	gaIn0 genArg = iota
	gaIn1
	gaIn2
	gaIn3
	gaOut0
	gaOut1
	gaOut2
	gaV       // the op's bound vector
	gaVB      // its bias
	gaPart    // the scratch's partial plane (a norm's sums)
	gaPart2   // the second (a LayerNorm's variance)
	gaConst   // the op's own constant (a residual scale)
	gaOnes    // a ones row of the head width (a weightless norm's scale)
	gaCS      // the call's cos/sin table for this block (genEnv)
	gaQOff    // the rows' offsets into q's table rows
	gaKOff    // the rows' offsets into k's
	gaColPair // over rows, each sorted column's (token, slot) pair
	gaWQS     // the op's weight's payload, scale and scale-of-scale planes
	gaWD
	gaWSC
	gaOpA   // the int8 operand the op's matvec reads, values
	gaOpAX  // and scales
	gaMBias // the weight's own bias (mv.bias)
	gaF16   // the binary16 operand a GEMM reads (the device's f16 plane)
	gaQ8A   // an int8 output's values
	gaQ8X   // its scales and sums
	gaPerm  // the step's head-major order: each gathered row's source row (headPerm)
	gaSplit // a k split's partial plane (the tier's partBuf)
	gaKPool // the block's paged K pool
	gaVPool // its paged V pool
	gaTab   // its page table
	gaPRow  // the step's row descriptors (kernels.PRowWords a row)
	// An attention plan's (gen_attend.go, bound by runPlan): the scratch's
	// causal count, the layer's descriptor set (its window's), its sinks,
	// the plan's partials and the staged score and weight planes.
	gaN
	gaADesc
	gaSink
	gaAPart
	gaPlane
	gaPlane2
)

// genLaunch is one generated launch: threads for a flat launch, or groups of
// width where width is set.
type genLaunch struct {
	k             backend.Kernel
	threads       int
	groups, width int
	args          [8]genArg
	n             uint8
	// q8 is which output holds the op's int8 value (gaQ8A and gaQ8X).
	q8 uint8
	// former, where set, runs the launch through a kernel former's own
	// launcher (the executor's form) rather than one kernel: the matvec's.
	former uint8
	// pairs launches a group a (token, expert) pair of the step, not a
	// group a token.
	pairs bool
	// toF16 is a binary16 operand made from the op's float row: skipped
	// where the step's last conversion was this kernel over this row (q, k
	// and v share one normed row, as do gate and up).
	toF16 bool
	// whole launches the kernel's full grid at any rows: a tiled GEMM's
	// grid is tiles of rows by tiles of tokens, which no proportional cut
	// keeps. Rows past the step's compute what nothing reads. (A kernel
	// built for the step's own rows in the prologue is the faster form.)
	whole bool
	// plan runs the call's decode plan in its place: the launches the
	// prologue formed for the step's key count (pagedScratch.cur), bound to
	// the op's query and output and the layer's history (gen_attend.go).
	plan bool
}

// The formers whose launchers run an op: the matvec's (the op's weight's
// MatVecShape kernel) and the paged attention's (its DecodePlan's).
const (
	formMatVec = 1 + iota
	formAttend
	formRoute
	formScan
)

// kernAct is a graph activation's kernel form, false for one whose kernel
// needs more than its kind (a clamp's bound).
func kernAct(a graph.Act) (kernels.ActKind, bool) {
	switch a {
	case graph.ActSiLU:
		return kernels.ActSiLU, true
	case graph.ActGELUTanh:
		return kernels.ActGELU, true
	case graph.ActGELUErf:
		return kernels.ActGELUErf, true
	case graph.ActSigmoid:
		return kernels.ActSigmoid, true
	case graph.ActReLU2:
		return kernels.ActReLU2, true
	case graph.ActReLU:
		return kernels.ActReLU, true
	}
	return 0, false
}

// genProgram generates the launches of every op of prog it can, for a
// variant of rows rows over scratch bs. Callers hold g.mu.
func (g *devTier) genProgram(prog *devProgram, bs *blockScratch, rows int) {
	for i := range prog.ops {
		o := &prog.ops[i]
		if o.FIn == 0 && (o.Kind == dkMatVec || o.Kind == dkMatVecRes || o.Kind == dkMatVecGate) && len(o.In) < 4 {
			if f, ok := prog.floatSource(i, o.In[0]); ok {
				o.In = append(o.In, f)
				o.FIn = int8(len(o.In))
			}
		}
		if o.Act == 0 {
			o.Act = o.A.Act // an activation its emit site did not record
		}
		if o.Kind == dkAdd && o.Alpha != 0 && o.Alpha != 1 && o.GC == nil {
			o.GC = g.constBuf(float32(o.Alpha))
		}
		o.Gen = g.genOp(o, prog, bs, rows)
	}
}

// constBuf is a one-float device buffer holding v, uploaded once a device
// and value: an add's scale, read by its kernel rather than baked so the
// kernel's key carries no float. Callers hold g.mu, outside a submission.
func (g *devTier) constBuf(v float32) backend.Buf {
	bits := math.Float32bits(v)
	if b, ok := g.consts[bits]; ok {
		return b
	}
	if g.genInSub {
		return nil // inside a submission nothing may allocate: the executor's case runs it
	}
	b, err := g.dev.Alloc(4)
	if err == nil {
		err = b.Write(f32b([]float32{v}))
	}
	if err != nil {
		g.LastErr = fmt.Sprintf("a constant %g: %v", v, err)
		return nil
	}
	if g.consts == nil {
		g.consts = map[uint32]backend.Buf{}
	}
	g.consts[bits] = b
	return b
}

// rowBuild is a composed row group being built from one op: the group and,
// for each of its buffers, the launch argument that binds it.
type rowBuild struct {
	r    compose.Row
	args []genArg
}

func (rb *rowBuild) buf(k compose.BufKind, a genArg) int {
	rb.r.Bufs = append(rb.r.Bufs, compose.Buf{Kind: k})
	rb.args = append(rb.args, a)
	return len(rb.r.Bufs) - 1
}

func (rb *rowBuild) node(n compose.Node) int {
	rb.r.Nodes = append(rb.r.Nodes, n)
	return len(rb.r.Nodes) - 1
}

func (rb *rowBuild) load(k compose.BufKind, a genArg) int {
	return rb.node(compose.Node{Kind: compose.NLoad, Buf: rb.buf(k, a)})
}

func (rb *rowBuild) bin(op compose.Op, a, b int) int {
	return rb.node(compose.Node{Kind: compose.NBinary, Op: op, A: a, B: b})
}

func (rb *rowBuild) store(v int, a genArg) {
	rb.r.Stores = append(rb.r.Stores, compose.Store{Buf: rb.buf(compose.BufOut, a), Val: v})
}

// rmsOf is the RMS scale of node x over the row: 1/sqrt(mean(x*x) + eps).
func (rb *rowBuild) rmsOf(x int, width int, eps float32) int {
	ss := rb.node(compose.Node{Kind: compose.NReduce, Red: compose.RedSum, A: rb.bin(compose.OpMul, x, x)})
	mean := rb.bin(compose.OpMul, ss, rb.node(compose.Node{Kind: compose.NConst, F: 1 / float32(width)}))
	return rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpRsqrt,
		A: rb.bin(compose.OpAdd, mean, rb.node(compose.Node{Kind: compose.NConst, F: eps}))})
}

// normed is node x normed over the row: RMSNorm by the bound scale, or the
// plan's LayerNorm with its bias; -1 where the op binds neither.
func (rb *rowBuild) normed(o *devOp, x, w int, eps float32, layer bool) int {
	if o.V == nil {
		return -1
	}
	wt := rb.load(compose.BufVec, gaV)
	if !layer {
		return rb.bin(compose.OpMul, rb.bin(compose.OpMul, x, rb.rmsOf(x, w, eps)), wt)
	}
	if o.VB == nil {
		return -1
	}
	mean := rb.bin(compose.OpMul, rb.node(compose.Node{Kind: compose.NReduce, Red: compose.RedSum, A: x}),
		rb.node(compose.Node{Kind: compose.NConst, F: 1 / float32(w)}))
	c := rb.bin(compose.OpSub, x, mean)
	bias := rb.load(compose.BufVec, gaVB)
	return rb.bin(compose.OpAdd, rb.bin(compose.OpMul, rb.bin(compose.OpMul, c, rb.rmsOf(c, w, eps)), wt), bias)
}

// quantize gives the group the int8 epilogue of node v, a row of width w,
// padded as a matrix of the plan reads it.
func (rb *rowBuild) quantize(v, w int, p *nn.LayerPlan) {
	kp := pad32(w)
	if w == p.NEmbd {
		kp = ePad(p)
	}
	win := p.ActWin
	if win <= 0 {
		win = 32
	}
	rb.r.Q8 = &compose.Q8{Val: v, A: rb.buf(compose.BufQ8, gaQ8A), X: rb.buf(compose.BufQ8X, gaQ8X),
		Window: kernels.ClampWindow(win, kp), Pad: kp}
}

// rowSchedule is a row group's schedule: the target's group for its width,
// a reduction's widest where it has one. The tuner's choice replaces it.
func rowSchedule(r *compose.Row) compose.Schedule {
	reduces := false
	for _, n := range r.Nodes {
		reduces = reduces || n.Kind == compose.NReduce
	}
	top := 256
	if reduces || r.Q8 != nil {
		top = kernels.RMSNormGroup
	}
	g := 32
	for g < r.Width && g < top {
		g *= 2
	}
	return compose.Schedule{Group: g}
}

// genOp is op o's launches, or nil where it has no generator yet or the
// backend refused its kernel (the executor's case runs it). Every kind here
// is a composed row group of op fragments (jit/gpu/compose) except the
// rotaries and the query/gate split, whose index maps the row former does not
// state yet: they launch their template until it does.
func (g *devTier) genOp(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	// The formers take their ops whatever their values' layouts: a former
	// lays out its own rows (a bank's pairs, a history's pages).
	switch o.Kind {
	case dkMatVec, dkMatVecRes, dkMatVecGate:
		if rows == 1 {
			if l := g.genMatVec(o, prog, bs); l != nil {
				return l
			}
		} else if l := g.genGemm(o, prog, bs, rows); l != nil {
			return l
		}
		return []genLaunch{{former: formMatVec}}
	case dkExpertMatVec, dkExpertMatVecE, dkQKV, dkQKVGemm, dkRouterLogits, dkResIn, dkQKVSep:
		if o.Kind == dkExpertMatVec && rows == 1 {
			if l := g.genBankMatVec(o, prog, bs); l != nil {
				return l
			}
		}
		if o.Kind == dkRouterLogits {
			if l := g.genRoute(o, prog, rows); l != nil {
				return l
			}
		}
		// The matvec former: the weight's (or bank's, or the fused q/k/v
		// grid's) kernel, emitted from its shape.
		return []genLaunch{{former: formMatVec}}
	case dkAttend, dkLatentAttend, dkKVWrite, dkKVCopy, dkRopeQKV, dkSlice, dkView, dkConcatHeads, dkRopeTail, dkLatentWrite:
		switch o.Kind {
		case dkAttend:
			if l := g.genAttend(o, bs, rows); l != nil {
				return l
			}
			o.Why = whyHistory
		case dkKVWrite, dkKVCopy:
			if l := g.genPagedWrite(o, prog, bs, rows, o.Kind == dkKVWrite, gaIn0, gaIn1); l != nil {
				return l
			}
			o.Why = whyHistory
		case dkRopeQKV:
			if l := g.genRopeQKV(o, prog, bs, rows); l != nil {
				return l
			}
			o.Why = whyHistory
		case dkSlice, dkConcatHeads, dkRopeTail:
			if l := g.genHeadCopy(o, prog, bs, rows); l != nil {
				return l
			}
			o.Why = whyLayout
		case dkLatentWrite:
			if l := g.genLatentWrite(o, prog, bs, rows); l != nil {
				return l
			}
			o.Why = whyHistory
		}
		// The attention former: the paged plans emitted from the attention's
		// shape and the device's tuning, the history's row writes, and a
		// latent attention's slices of its rows.
		return []genLaunch{{former: formAttend}}
	case dkTopK, dkRouteWeights, dkCombine, dkGroupByExpert, dkGateLogit, dkGroupMask, dkScaleWeights, dkRouterBias,
		dkExpertBias, dkExpertFetch:
		if o.Kind == dkCombine && rows == 1 {
			if l := g.genCombine(o, prog, bs); l != nil {
				return l
			}
		}
		if l := g.genRoute(o, prog, rows); l != nil {
			return l
		}
		// The router former: the selection's kernels emitted from the route's
		// shape (MoERoute), the bank's fetch, and the combine.
		return []genLaunch{{former: formRoute}}
	case dkConv1d, dkScan, dkGatedNorm, dkSSMNorm, dkConvBias:
		// The scan former: a recurrence's kernels emitted from its plan's
		// shape (RecurrentPlan).
		return []genLaunch{{former: formScan}}
	}
	if l := g.genStreamsA(o, prog, bs, rows); l != nil {
		return l
	}
	if l := g.genPairs(o, prog, bs, rows); l != nil {
		return l
	}
	// A head-major value is rows like any other to a row group: the order
	// is what its gather wrote (headMajor).
	for _, v := range o.In {
		if l := prog.vals[v].Layout; l != layoutRow && l != layoutHeads {
			return nil
		}
	}
	for _, v := range o.Out {
		if l := prog.vals[v].Layout; l != layoutRow && l != layoutHeads {
			return nil
		}
	}
	w := 0
	if len(o.Out) > 0 {
		w = prog.vals[o.Out[0]].Width
	}
	p := &bs.p
	// A norm's epsilon is its graph op's (a latent norm states its own),
	// the model's where the op states none.
	eps := float32(p.RMSEps)
	if o.A.Eps != 0 {
		eps = float32(o.A.Eps)
	}
	rb := &rowBuild{r: compose.Row{Width: w, Rows: rows}}
	var q8 uint8 // which output holds an int8 value
	switch o.Kind {
	case dkAdd:
		a, b := rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1)
		if o.Alpha != 0 && o.Alpha != 1 {
			if o.GC == nil {
				return nil
			}
			// a + alpha*b, one rounding (AddScaled's).
			al := rb.load(compose.BufScalar, gaConst)
			rb.store(rb.node(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: al, B: b, C: a}), gaOut0)
			break
		}
		rb.store(rb.bin(compose.OpAdd, a, b), gaOut0)
	case dkCopy:
		rb.store(rb.load(compose.BufRow, gaIn0), gaOut0)
	case dkAct:
		a, ok := kernAct(o.Act)
		if !ok || len(o.In) != 1 {
			return nil
		}
		x := rb.load(compose.BufRow, gaIn0)
		if o.Alpha != 0 && o.Alpha != 1 {
			// An input scale the step states (a stream gate's 1/streams).
			x = rb.bin(compose.OpMul, x, rb.node(compose.Node{Kind: compose.NConst, F: float32(o.Alpha)}))
		}
		rb.store(rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpAct, Act: a, A: x}), gaOut0)
	case dkActMul:
		if len(o.In) != 2 {
			return nil
		}
		y, ok := rb.gated(o, rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1))
		if !ok {
			return nil
		}
		rb.store(y, gaOut0)
	case dkMul:
		rb.store(rb.bin(compose.OpMul, rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1)), gaOut0)
	case dkGateOut:
		// sigmoid(gate) * the row.
		sig := rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpAct, Act: kernels.ActSigmoid, A: rb.load(compose.BufRow, gaIn0)})
		rb.store(rb.bin(compose.OpMul, sig, rb.load(compose.BufRow, gaIn1)), gaOut0)
	case dkGatedAdd:
		// acc + sigmoid(the row's logit) * the branch.
		acc, br := rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1)
		sig := rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpAct, Act: kernels.ActSigmoid, A: rb.load(compose.BufRowScalar, gaIn2)})
		rb.store(rb.bin(compose.OpAdd, acc, rb.bin(compose.OpMul, sig, br)), gaOut0)
	case dkResidJoin3:
		// (x + y + z) / sqrt(2).
		x, y, z := rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1), rb.load(compose.BufRow, gaIn2)
		s := rb.bin(compose.OpAdd, rb.bin(compose.OpAdd, x, y), z)
		rb.store(rb.bin(compose.OpMul, s, rb.node(compose.Node{Kind: compose.NConst, F: float32(1 / math.Sqrt2)})), gaOut0)
	case dkClamp:
		b := float32(o.Alpha)
		if b <= 0 {
			return nil
		}
		x := rb.load(compose.BufRow, gaIn0)
		hi := rb.bin(compose.OpMin, x, rb.node(compose.Node{Kind: compose.NConst, F: b}))
		rb.store(rb.bin(compose.OpMax, hi, rb.node(compose.Node{Kind: compose.NConst, F: -b})), gaOut0)
	case dkClampAt:
		// A pair of the block's bounds table: min(max(x, lo), hi).
		if o.V == nil {
			return nil
		}
		x := rb.load(compose.BufRow, gaIn0)
		tb := rb.buf(compose.BufScalar, gaV)
		lo := rb.node(compose.Node{Kind: compose.NLoad, Buf: tb, At: 2 * o.N})
		hi := rb.node(compose.Node{Kind: compose.NLoad, Buf: tb, At: 2*o.N + 1})
		rb.store(rb.bin(compose.OpMin, rb.bin(compose.OpMax, x, lo), hi), gaOut0)
	case dkScale:
		if o.V == nil {
			return nil
		}
		rb.store(rb.bin(compose.OpMul, rb.load(compose.BufRow, gaIn0), rb.load(compose.BufScalar, gaV)), gaOut0)
	case dkNorm, dkPostNorm, dkNormQuant:
		// A row norm; with its int8 operand (norm_quant).
		if prog.vals[o.In[0]].Width != w {
			return nil
		}
		y := rb.normed(o, rb.load(compose.BufRow, gaIn0), w, eps, p.LayerNorm)
		if y < 0 {
			return nil
		}
		rb.store(y, gaOut0)
		if o.Kind == dkNormQuant {
			q8 = 1
			rb.quantize(y, w, p)
		}
	case dkAddNorm, dkAddNormQuant:
		// The residual's sum, normed; with its int8 operand.
		s := rb.bin(compose.OpAdd, rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1))
		rb.store(s, gaOut0)
		y := rb.normed(o, s, w, eps, p.LayerNorm)
		if y < 0 {
			return nil
		}
		rb.store(y, gaOut1)
		if o.Kind == dkAddNormQuant {
			q8 = 2
			rb.quantize(y, w, p)
		}
	case dkActQuant:
		// act(gate) * up, and its int8 operand.
		if len(o.In) != 2 {
			return nil
		}
		y, ok := rb.gated(o, rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1))
		if !ok {
			return nil
		}
		rb.store(y, gaOut0)
		q8 = 1
		rb.quantize(y, w, p)
	case dkGatherHeads:
		// A chunk's rows head-major: each head's slice of every row in turn,
		// a group row a (head, row), read through the step's order.
		iw := prog.vals[o.In[0]].Width
		if o.Heads <= 0 || iw%o.Heads != 0 {
			return nil
		}
		hw := iw / o.Heads
		rb.r.Width, rb.r.Rows = hw, rows*o.Heads
		src, perm := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufU32, gaPerm)
		at := rb.r.UBin(compose.OpAdd, rb.r.UBin(compose.OpMul, rb.r.Load(perm), rb.r.UConst(hw)), rb.r.Column())
		rb.store(rb.r.LoadAt(src, at), gaOut0)
	case dkHeadNorm:
		// Each head a row of the head's width: rows*heads rows.
		hd := o.HD
		iw := prog.vals[o.In[0]].Width
		if hd <= 0 || iw%hd != 0 {
			return nil
		}
		rb.r.Width, rb.r.Rows = hd, rows*(iw/hd)
		x := rb.load(compose.BufRow, gaIn0)
		y := rb.bin(compose.OpMul, x, rb.rmsOf(x, hd, eps))
		if o.V != nil {
			y = rb.bin(compose.OpMul, y, rb.load(compose.BufVec, gaV))
		}
		rb.store(y, gaOut0)
	case dkQuant:
		if o.OneRow && rows > 1 {
			return nil
		}
		iw := prog.vals[o.In[0]].Width
		kp := pad32(iw)
		if iw == p.NEmbd {
			kp = ePad(p)
		}
		win := p.ActWin
		if win <= 0 {
			win = 32
		}
		rb.r.Width = iw
		x := rb.load(compose.BufRow, gaIn0)
		rb.r.Q8 = &compose.Q8{Val: x, A: rb.buf(compose.BufQ8, gaQ8A), X: rb.buf(compose.BufQ8X, gaQ8X),
			Window: kernels.ClampWindow(win, kp), Pad: kp}
	case dkRopeQ, dkRopeKRow:
		// The rotary over each head's first NRot dimensions (NORM or NEOX
		// pairs; XD-RoPE's NEOX pairs on two tables), the rest of the head
		// passed through, each row landing at its offset in the call's
		// table: q's, or k's row of the history.
		ra := o.A
		iw := prog.vals[o.In[0]].Width
		if ra.HeadDim <= 0 || iw%ra.HeadDim != 0 || ra.NRot <= 0 || ra.NRot%2 != 0 || ra.NRot > ra.HeadDim {
			return nil
		}
		offArg := gaQOff
		if o.Kind == dkRopeKRow {
			offArg = gaKOff
		}
		xd := ra.Layout == graph.RopeXD
		rb.r.Width = iw
		in, cs, off := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufVec, gaCS), rb.buf(compose.BufU32, offArg)
		v := rb.r.Rotary(in, cs, ra.HeadDim, ra.NRot, ra.Layout == graph.RopeNeox || xd, xd)
		out := rb.buf(compose.BufOut, gaOut0)
		rb.r.StoreAt(out, v, rb.r.UBin(compose.OpAdd, rb.r.Load(off), rb.r.Column()))
	case dkSplitQG:
		// A gated query's interleaved heads apart: each head's q, then its
		// gate, of 2*hd a head in, hd a head out.
		// The heads are the attention's, the plan's head width as
		// SplitHeadGate took it: on a gated query the op's HeadDim attribute
		// is another width, and splitting by it parted from the host.
		hd := bs.p.HeadDim
		if hd <= 0 || w%hd != 0 || prog.vals[o.In[0]].Width != 2*w {
			return nil
		}
		src := rb.buf(compose.BufRow, gaIn0)
		col := rb.r.Column()
		head, j := rb.r.UBin(compose.OpDiv, col, rb.r.UConst(hd)), rb.r.UBin(compose.OpRem, col, rb.r.UConst(hd))
		at := rb.r.UBin(compose.OpAdd, rb.r.UBin(compose.OpAdd, rb.r.UBin(compose.OpMul, rb.r.RowIdx(), rb.r.UConst(2*w)),
			rb.r.UBin(compose.OpMul, head, rb.r.UConst(2*hd))), j)
		rb.store(rb.r.LoadAt(src, at), gaOut0)
		rb.store(rb.r.LoadAt(src, rb.r.UBin(compose.OpAdd, at, rb.r.UConst(hd))), gaOut1)
	default:
		return nil
	}
	k := g.rowKern(&rb.r)
	if k.k == nil {
		return nil
	}
	l := genLaunch{k: k.k, groups: rb.r.Rows, width: k.group, n: uint8(len(rb.args)), q8: q8}
	copy(l.args[:], rb.args)
	return []genLaunch{l}
}

// rowKernel is a composed row group's compiled kernel and its group size.
type rowKernel struct {
	k     backend.Kernel
	group int
}

// genEntry is a generated kernel in the device's one cache, with the
// encoding it was formed from: a hash that matches with a different encoding
// is a collision, not a hit.
type genEntry struct {
	enc   []uint32
	k     backend.Kernel
	width int
}

// genKern is the kernel encoding enc names -- a group's structure and its
// schedule, a former's shape and epilogue, a template's id and sizes --
// built by build once a device. The one cache every generated kernel goes
// through; its width is the workgroup the kernel declares. A build the
// backend refuses is cached as nil, the reason in LastErr. Callers hold g.mu.
func (g *devTier) genKern(enc []uint32, build func() (*ir.Kernel, error)) (backend.Kernel, int) {
	key := compose.KeyOf(enc)
	for _, e := range g.genK[key] {
		if slices.Equal(e.enc, enc) {
			return e.k, e.width
		}
	}
	ent := genEntry{enc: enc}
	ker, err := build()
	if err == nil {
		ent.width = ker.Group[0]
		ent.k, err = g.dev.Compile(ker)
	}
	if err != nil {
		g.LastErr = fmt.Sprintf("a generated kernel: %v", err)
		ent.k = nil
	}
	if g.genK == nil {
		g.genK = map[compose.Key][]genEntry{}
	}
	g.genK[key] = append(g.genK[key], ent)
	return ent.k, ent.width
}

// rowKern is row group r's kernel under the target's schedule.
func (g *devTier) rowKern(r *compose.Row) rowKernel {
	s := rowSchedule(r)
	k, _ := g.genKern(append([]uint32{encRow}, r.Encode(s)...), func() (*ir.Kernel, error) { return r.Emit(s) })
	return rowKernel{k: k, group: s.Group}
}

// The encodings' first words: what built the kernel.
const (
	encRow = 1 + iota
	encEpi
	encTemplate
	encMatVec
)

// runGen runs op o's generated launches for R rows, its buffers resolved by
// at. A launch's thread count was generated for the variant's rows; a step
// of fewer runs fewer, the kernels clamping as every row kernel does.
// genState is what a generated launch reads that is not a value of its
// program, resolved by the executor for the op about to run: the call's
// tables, the op's weight planes and operand, the step's pair count. Data
// only, no functions: it lives on the executor's stack and a warm step
// allocates nothing for it.
type genState struct {
	bs             *blockScratch
	slots          []devSlot
	cs, qOff, kOff backend.Buf
	// The op's weight: its payload, scale (the float row for a float
	// format) and scale-of-scale planes, its own bias, and the int8 operand
	// its matvec reads.
	qs, d, sc, mbias backend.Buf
	a, ax            backend.Buf
	// perm is the step's head-major order for a gather (headPerm).
	perm backend.Buf
	// split is the partial plane a k split's segments write (partBuf).
	split backend.Buf
	// The block's paged history: its K and V pools, page table, and the
	// step's row descriptors.
	kpool, vpool, tab, prow backend.Buf
	// The layer's attention reads: its descriptor set (the window's where it
	// slides) and its sinks.
	adesc, sink backend.Buf
	// The step's binary16 operand: its plane, and the conversion the step
	// last made, by kernel and source (shared with the executor, so a
	// conversion either one made is not made again).
	f16     backend.Buf
	memoK   backend.Kernel
	memoSrc backend.Buf
	// The step's (token, expert) pairs: the grouped run's sorted columns
	// over rows (moe), else the selection's.
	moe     bool
	cols    int
	colPair backend.Buf
}

func runGen(lc *launcher, o *devOp, R, rows int, buf *[numSlots]backend.Buf, st *genState) {
	bs := st.bs
	at := func(v devVal) backend.Buf { return buf[st.slots[v]] }
	var bufs [8]backend.Buf
	for i := range o.Gen {
		l := &o.Gen[i]
		if l.plan {
			pk := bs.pkv
			ab := attendBufs{q: at(o.In[0]), k: st.kpool, v: st.vpool, n: bs.n, out: at(o.Out[0]), tab: st.tab,
				desc: st.adesc, sink: st.sink, part: pk.part, sc: pk.sc, wt: pk.wt}
			runPlan(lc, pk.cur.gen, &ab)
			continue
		}
		for j := 0; j < int(l.n); j++ {
			switch a := l.args[j]; a {
			case gaIn0, gaIn1, gaIn2, gaIn3:
				bufs[j] = at(o.In[a-gaIn0])
			case gaOut0, gaOut1, gaOut2:
				bufs[j] = at(o.Out[a-gaOut0])
			case gaV:
				bufs[j] = o.V
			case gaVB:
				bufs[j] = o.VB
			case gaPart:
				bufs[j] = bs.part
			case gaPart2:
				bufs[j] = bs.part2
			case gaConst:
				bufs[j] = o.GC
			case gaOnes:
				bufs[j] = bs.hdOnes
			case gaCS:
				bufs[j] = st.cs
			case gaQOff:
				bufs[j] = st.qOff
			case gaKOff:
				bufs[j] = st.kOff
			case gaColPair:
				bufs[j] = st.colPair
			case gaWQS:
				bufs[j] = st.qs
			case gaWD:
				bufs[j] = st.d
			case gaWSC:
				bufs[j] = st.sc
			case gaOpA:
				// The int8 operand the op reads: its input's own pair where
				// the value has one, else the block's (int8Of).
				bufs[j], _ = int8Of(st, o.In[0])
			case gaOpAX:
				_, bufs[j] = int8Of(st, o.In[0])
			case gaMBias:
				bufs[j] = st.mbias
			case gaF16:
				bufs[j] = st.f16
			case gaPerm:
				bufs[j] = st.perm
			case gaSplit:
				bufs[j] = st.split
			case gaKPool:
				bufs[j] = st.kpool
			case gaVPool:
				bufs[j] = st.vpool
			case gaTab:
				bufs[j] = st.tab
			case gaPRow:
				bufs[j] = st.prow
			case gaQ8A:
				bufs[j], _ = int8Of(st, o.Out[l.q8])
			case gaQ8X:
				_, bufs[j] = int8Of(st, o.Out[l.q8])
			}
		}
		if l.toF16 {
			// Its source is the first argument: the stated float row.
			if st.memoK == l.k && st.memoSrc == bufs[0] {
				continue
			}
			st.memoK, st.memoSrc = l.k, bufs[0]
		}
		threads := l.threads
		if rows > 0 && R < rows && !l.whole {
			threads = threads / rows * R
		}
		if l.width != 0 {
			groups := l.groups
			switch {
			case l.pairs && st.moe:
				groups = st.cols
			case l.pairs:
				groups = bs.nUsed * R
			case rows > 0 && R < rows && !l.whole:
				groups = (groups*R + rows - 1) / rows
			}
			if groups > l.groups && *lc.err == nil {
				// The kernel clamps its group to its last row: a wider launch
				// would recompute that row and leave the rest unwritten.
				*lc.err = fmt.Errorf("tier: op %s launched %d groups over %d rows", o.Kind, groups, l.groups)
			}
			if *lc.err == nil {
				*lc.err = lc.launch(l.k, groups, l.width, bufs[:l.n]...)
			}
			continue
		}
		lc.la(l.k, threads, bufs[:l.n]...)
	}
}

// int8Of is an int8 value's two buffers: its slot's own pair, or the
// block's operand.
func int8Of(st *genState, v devVal) (backend.Buf, backend.Buf) {
	if a, ax, ok := q8Pair(st.bs, st.slots[v]); ok {
		return a, ax
	}
	return st.bs.a, st.bs.ax
}

// normParts is a row norm's partial count for width w over rows rows: the
// tier's choice (scratchinit's), halved until it divides w.
func (g *devTier) normParts(w, rows int) int {
	parts := 64
	if rows > 1 {
		parts = g.tiles.parts
	}
	for parts > 1 && w%parts != 0 {
		parts /= 2
	}
	return parts
}

// genPairs is an op over (token, expert) pairs as a composed row group, a
// row a pair: an activation, a gated activation (or one weighted at its
// input by the pair's routed weight, Llama 4's), or the int8 operand the down
// bank's slots read. Its rows are the step's pairs, counted at launch: the
// selection's at a token, the grouped run's sorted columns over rows. nil
// where the op is not over pairs.
func (g *devTier) genPairs(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	if len(o.Out) == 0 || len(o.In) == 0 || prog.vals[o.Out[0]].Layout != layoutPairs && prog.vals[o.In[0]].Layout != layoutPairs {
		return nil
	}
	p := &bs.p
	moe := rows > 1 && prog.experts
	w := prog.vals[o.In[0]].Width
	// The rows the pairs fill: a token's selection, or over rows the grouped
	// run's sorted columns, each expert's padded to the run's tile -- the
	// capacity its producer laid out, which a launch may fill to the end.
	cap := max(rows*p.NExpertUsed, 1)
	if moe {
		if bs.mg == nil {
			return nil
		}
		cap = bs.mg.p
	}
	rb := &rowBuild{r: compose.Row{Width: w, Rows: cap}}
	var q8 uint8
	switch o.Kind {
	case dkAct:
		a, ok := kernAct(o.Act)
		if !ok {
			return nil
		}
		rb.store(rb.r.Act(a, rb.load(compose.BufRow, gaIn0)), gaOut0)
	case dkActMul:
		gt, up := rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1)
		if len(o.In) > 2 {
			a, ok := kernAct(o.Act)
			if !ok || kernels.ActMulOK(a) != nil || o.A.GLU.Bounded() {
				return nil
			}
			// act(w*gate) * (w*up), w the pair's routed weight.
			wb := rb.buf(compose.BufVec, gaIn2)
			at := rb.r.RowIdx()
			if moe {
				at = rb.r.LoadAt(rb.buf(compose.BufU32, gaColPair), at)
			}
			wt := rb.r.LoadAt(wb, at)
			gw, uw := rb.bin(compose.OpMul, wt, gt), rb.bin(compose.OpMul, wt, up)
			rb.store(rb.bin(compose.OpMul, rb.r.Act(a, gw), uw), gaOut0)
			break
		}
		y, ok := rb.gated(o, gt, up)
		if !ok {
			return nil
		}
		rb.store(y, gaOut0)
	case dkQuant:
		if moe {
			return nil // over rows the grouped matvec quantizes what it reads
		}
		x := rb.load(compose.BufRow, gaIn0)
		rb.r.Q8 = &compose.Q8{Val: x, A: rb.buf(compose.BufQ8, gaQ8A), X: rb.buf(compose.BufQ8X, gaQ8X),
			Window: kernels.ClampWindow(max(p.ActWin, 32), pad32(w)), Pad: pad32(w)}
	default:
		return nil
	}
	k := g.rowKern(&rb.r)
	if k.k == nil {
		return nil
	}
	l := genLaunch{k: k.k, groups: rb.r.Rows, width: k.group, n: uint8(len(rb.args)), q8: q8, pairs: true}
	copy(l.args[:], rb.args)
	return []genLaunch{l}
}

// genMatVec is a decode matvec composed with its epilogue: the weight's
// kernel at the schedule its tuning chose (mv: split, rows per thread, the
// in-group reduction) and, applied to each output before the store, the
// fragments of the ops the program fused into it -- the weight's bias, a
// residual added (matvec_res), act(gate) times the output (matvec_gate).
// nil where the shape reduces across launches (a split into a partial plane),
// where the table split swaps the kernel, or where it reads an operand of its
// own: the former runs those.
func (g *devTier) genMatVec(o *devOp, prog *devProgram, bs *blockScratch) []genLaunch {
	l := g.curLayer
	if l == nil || o.W < 0 || o.W >= len(l.ms) || *l.rs[o.W] == nil {
		o.Why = whyNoWeight
		return nil
	}
	m := *l.ms[o.W]
	switch {
	case m.kern == nil:
		o.Why = whyNoKernel
	case m.experts > 1 || m.slots > 0:
		o.Why = whyBank
	case m.pad:
		o.Why = whyPadded
	case g.TableSplit && m.alt != nil:
		o.Why = whyTable
	}
	if o.Why != 0 {
		return nil
	}
	e, args := matEpi(o, m, m.rows)
	if e == nil {
		o.Why = whyEpilogue
		return nil
	}
	rowt := max(m.rowt, 1)
	s := kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, Split: m.split, Rowt: rowt, GroupSplit: m.group}
	if m.split > 1 && !m.group {
		return g.genSplitMatVec(o, s, e, args)
	}
	k, width := g.mvEpiKern(s, e)
	if k == nil {
		return nil
	}
	n := m.rows * max(m.split, 1)
	if rowt > 1 {
		n /= rowt
	}
	gl := genLaunch{k: k, groups: (n + width - 1) / width, width: width,
		args: [8]genArg{gaWQS, gaWD, gaWSC, gaOpA, gaOpAX, gaOut0}, n: 6}
	if kernels.IsFloat(m.q) {
		// A float weight reads the float row in the scale slot.
		if o.FIn == 0 {
			return nil
		}
		gl.args[1] = gaIn0 + genArg(o.FIn-1)
	}
	for _, a := range args {
		gl.args[gl.n] = a
		gl.n++
	}
	return []genLaunch{gl}
}

// mvEpiKern is the matvec of shape s with epilogue e, formed by the matvec
// former (compose.MatVec: the loop nest around the format's SubDot) once a
// device and structure, and its workgroup width. Callers hold g.mu.
func (g *devTier) mvEpiKern(s kernels.MatVecShape, e *compose.Epi) (backend.Kernel, int) {
	mv := &compose.MatVec{T: s.T, K: s.K, Rows: s.Rows, Split: s.Split, Group: s.GroupSplit, Rowt: s.Rowt,
		Center: s.Center, Epi: e}
	return g.genKern(append([]uint32{encMatVec}, mv.Encode()...), mv.Emit)
}

// epiKern is a former's kernel of shape s with an epilogue, its structure
// enc (the epilogue's and the former's schedule), built by build once a
// device and structure, and its workgroup width. Callers hold g.mu.
func (g *devTier) epiKern(s kernels.MatVecShape, enc []uint32, build func() (*ir.Kernel, error)) (backend.Kernel, int) {
	full := append([]uint32{encEpi}, enc...)
	full = append(full, uint32(s.T), uint32(s.K), uint32(s.Rows), uint32(s.Split), uint32(s.Rowt), b2u(s.GroupSplit),
		uint32(s.NTok), b2u(s.Center.NoCenter), uint32(s.Center.MinTok))
	return g.genKern(full, build)
}

func b2u(x bool) uint32 {
	if x {
		return 1
	}
	return 0
}

// matEpi is op o's epilogue over a matvec of rows outputs, and the launch
// arguments its buffers bind: the weight's bias, a residual added
// (matvec_res), act(gate) times the output (matvec_gate).
func matEpi(o *devOp, m mv, rows int) (*compose.Epi, []genArg) {
	ep := &compose.Row{Width: rows, Rows: 1}
	var args []genArg
	buf := func(k compose.BufKind, a genArg) int {
		args = append(args, a)
		return ep.Buf(k)
	}
	v := ep.Add(compose.Node{Kind: compose.NInput})
	if m.bias != nil {
		v = ep.Bin(compose.OpAdd, v, ep.Load(buf(compose.BufVec, gaMBias)))
	}
	switch o.Kind {
	case dkMatVecRes:
		v = ep.Bin(compose.OpAdd, v, ep.Load(buf(compose.BufRow, gaIn1)))
	case dkMatVecGate:
		a, ok := kernAct(o.Act)
		if !ok {
			return nil, nil
		}
		v = ep.Bin(compose.OpMul, ep.Act(a, ep.Load(buf(compose.BufRow, gaIn1))), v)
	}
	e := &compose.Epi{G: ep, Out: v}
	if e.Check() != nil {
		return nil, nil
	}
	return e, args
}

// genGemm is a prompt chunk's matvec as the binary16 GEMM its schedule chose
// (batchMV: tile and split for this card and these rows), composed with its
// epilogue, behind the conversion of its float row to the GEMM's operand.
// nil where the schedule is another (the former runs it) or splits into a
// partial plane.
func (g *devTier) genGemm(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	l := g.curLayer
	if l == nil || o.W < 0 || o.W >= len(l.ms) || *l.rs[o.W] == nil || prog.vals[o.In[0]].Layout != layoutRow {
		o.Why = whyNoWeight
		return nil
	}
	m := *l.ms[o.W]
	switch {
	case m.slots > 0:
		o.Why = whyBank
	case m.pad:
		o.Why = whyPadded
	case g.TableSplit && m.alt != nil:
		o.Why = whyTable
	}
	if o.Why != 0 {
		return nil
	}
	bm, ok := g.batchMV(m, rows, bs.tok)
	if ok && bm.form == mvFormDot4 {
		return g.genChunkMatVec(o, prog, bs, m, bm, rows)
	}
	switch {
	case !ok:
		o.Why = whyNoBatch
	case bm.act == nil || !bm.actT || bm.vt.MT == 0:
		o.Why = whyNotStaged
	case bm.tail != nil:
		o.Why = whyTail
	}
	if o.Why != 0 {
		return nil
	}
	e, args := matEpi(o, m, m.rows)
	if e == nil {
		o.Why = whyEpilogue
		return nil
	}
	// The GEMM former at the schedule batchMV chose (tile, instruction,
	// staging, split); a split's planes summed by the reduction former.
	split := max(bm.split, 1)
	gm := &compose.Gemm{T: m.q, K: m.k, Rows: m.rows, NTok: rows, Tile: bm.vt, Split: split}
	out := gaOut0
	if split > 1 {
		need := m.rows * rows * split
		if g.genInSub && g.partCap < need || !g.genInSub && !g.sizePart(need) {
			o.Why = whySplit
			return nil
		}
		out = gaSplit
	} else {
		gm.Epi = e
	}
	k, width := g.genKern(append([]uint32{encMatVec}, gm.Encode()...), gm.Emit)
	if k == nil {
		o.Why = whyNoKernel
		return nil
	}
	gl := genLaunch{k: k, threads: gm.Groups() * width, groups: gm.Groups(), width: width,
		args: [8]genArg{gaWQS, gaWD, gaWSC, gaF16, out}, n: 5, whole: true}
	var red *genLaunch
	if split > 1 {
		ss := &compose.SplitSum{Rows: m.rows * rows, Split: split, Width: m.rows, Epi: e}
		rk, rwidth := g.genKern(append([]uint32{encMatVec}, ss.Encode()...), ss.Emit)
		if rk == nil {
			o.Why = whyNoKernel
			return nil
		}
		red = &genLaunch{k: rk, groups: (m.rows*rows + rwidth - 1) / rwidth, width: rwidth, args: [8]genArg{gaSplit, gaOut0}, n: 2, whole: true}
	}
	tail := &gl
	if red != nil {
		tail = red
	}
	for _, a := range args {
		tail.args[tail.n] = a
		tail.n++
	}
	// The binary16 operand, converted here from the float row in this
	// GEMM's layout, which is a function of the token count it was built
	// for: an f16 value a writer made for another count is not it.
	if o.FIn == 0 {
		o.Why = whyNoFloat // no float row stated to convert
		return nil
	}
	// The operand: the GEMM's own group, a row a token (Gemm.Operand).
	ok2 := gm.Operand()
	ck := g.rowKern(ok2)
	if ck.k == nil {
		o.Why = whyNoKernel
		return nil
	}
	conv := genLaunch{k: ck.k, groups: rows, width: ck.group, args: [8]genArg{gaIn0 + genArg(o.FIn-1), gaF16}, n: 2,
		toF16: true, whole: true}
	if red != nil {
		return []genLaunch{conv, gl, *red}
	}
	return []genLaunch{conv, gl}
}

// headPerm is the head-major order of R rows of h heads: row h*R + r of the
// gathered rows reads row r*h + h of the chunk's. Built in the prologue,
// outside any submission, once a (heads, rows) pair; a step looks it up.
func (g *devTier) headPerm(h, R int) backend.Buf {
	k := [2]int{h, R}
	if b, ok := g.perms[k]; ok {
		return b
	}
	if g.genInSub {
		return nil // inside a submission nothing may allocate
	}
	t := make([]uint32, h*R)
	for i := range t {
		t[i] = uint32(i%R*h + i/R)
	}
	b, err := g.dev.Alloc(4 * len(t))
	if err == nil {
		err = b.Write(u32b(t))
	}
	if err != nil {
		g.LastErr = fmt.Sprintf("a head-major order of %d x %d: %v", h, R, err)
		return nil
	}
	if g.perms == nil {
		g.perms = map[[2]int]backend.Buf{}
	}
	g.perms[k] = b
	return b
}

// floatSource is the float row int8 value v (read by op j) was quantized
// from: a quantize's input, or the float output a fused norm or activation
// writes beside its int8 one. False where v is not an int8 value or its
// writer states no float row.
func (p *devProgram) floatSource(j int, v devVal) (devVal, bool) {
	if p.vals[v].Elem != elemQ8 {
		return 0, false
	}
	for i := j - 1; i >= 0; i-- {
		o := p.ops[i]
		for k, out := range o.Out {
			if out != v {
				continue
			}
			if o.Kind == dkQuant {
				return o.In[0], true
			}
			if k > 0 && p.vals[o.Out[k-1]].Elem == elemF32 && p.vals[o.Out[k-1]].Width == p.vals[v].Width {
				return o.Out[k-1], true
			}
			return 0, false
		}
	}
	return 0, false
}

// gated is act(gate) * up for op o, from its attributes alone: the gated
// form of its activation, or, where the op carries a gated unit's bounds
// (graph.Attrs.GLU, the container's numbers), the unit composed from them.
// The bounded unit is
//
//	g' = min(gate, GateMax)            (Clamp)
//	l  = GateTanh*tanh(g'/GateTanh)    (GateTanh, else g')
//	a  = l * sigmoid(g')               (GateTanh), l / (1 + exp(-SigScale*g')) otherwise
//	u  = clamp(up, UpMin, UpMax)       (Clamp) + UpShift, then bounded by UpTanh
//
// the order of operations the kernels' own gated forms use, so the two are
// the same bits. False where the op's activation has no gated form.
func (rb *rowBuild) gated(o *devOp, gt, up int) (int, bool) {
	u := o.A.GLU
	if !u.Bounded() {
		a, ok := kernAct(o.Act)
		if !ok || kernels.ActMulOK(a) != nil {
			return 0, false
		}
		return rb.node(compose.Node{Kind: compose.NBinary, Op: compose.OpActMul, Act: a, A: gt, B: up}), true
	}
	c := func(f float32) int { return rb.node(compose.Node{Kind: compose.NConst, F: f}) }
	exp := func(x int) int { return rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpExp, A: x}) }
	bound := func(x int, b float32) int {
		t := rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpAct, Act: kernels.ActTanh, A: rb.bin(compose.OpMul, c(1/b), x)})
		return rb.bin(compose.OpMul, c(b), t)
	}
	g := gt
	if u.Clamp {
		g = rb.bin(compose.OpMin, gt, c(u.GateMax))
	}
	z := g
	if u.SigScale != 0 && u.SigScale != 1 {
		z = rb.bin(compose.OpMul, c(u.SigScale), g)
	}
	denom := rb.bin(compose.OpAdd, c(1), exp(rb.bin(compose.OpSub, c(0), z)))
	var a int
	if u.GateTanh != 0 {
		a = rb.bin(compose.OpMul, bound(g, u.GateTanh), rb.bin(compose.OpDiv, c(1), denom))
	} else {
		a = rb.bin(compose.OpDiv, g, denom)
	}
	v := up
	if u.Clamp {
		v = rb.bin(compose.OpMax, rb.bin(compose.OpMin, up, c(u.UpMax)), c(u.UpMin))
	}
	if u.UpShift != 0 {
		v = rb.bin(compose.OpAdd, v, c(u.UpShift))
	}
	if u.UpTanh != 0 {
		v = bound(v, u.UpTanh)
	}
	return rb.bin(compose.OpMul, a, v), true
}

// genWhy is why a generator handed an op to a former: what the dump prints
// beside it, so a decline is a named work item rather than an absence.
type genWhy uint8

const (
	whyNone genWhy = iota
	whyNoWeight
	whyNoKernel
	whySplit
	whyBank
	whyPadded
	whyTable
	whyEpilogue
	whyNoBatch
	whyNotStaged
	whyTail
	whyNoFloat
	whyHistory
	whyLayout
	whyRoute
)

func (w genWhy) String() string {
	return [...]string{"", "no resident weight", "no decode kernel", "a k split", "a bank's sheets",
		"padded rows", "a table split", "an epilogue it cannot compose",
		"no batched form", "a prompt form other than the staged GEMM", "a tail launch",
		"no float row stated", "a history that is not paged, or a row wider than its heads",
		"rows whose head-major layout the program does not state",
		"a route its op's attributes do not state"}[w]
}

// genSplitMatVec is a decode matvec whose k is split across threads: the
// former's segments write the partial plane, and the reduction former sums
// them in segment order through the op's epilogue. Callers hold g.mu.
func (g *devTier) genSplitMatVec(o *devOp, s kernels.MatVecShape, e *compose.Epi, eargs []genArg) []genLaunch {
	// The plane is grown here, in the prologue; inside a submission nothing
	// may allocate, so there it must already hold the segments.
	need := s.Rows * s.Split
	if g.genInSub && g.partCap < need || !g.genInSub && !g.sizePart(need) {
		o.Why = whySplit
		return nil
	}
	mv := &compose.MatVec{T: s.T, K: s.K, Rows: s.Rows, Split: s.Split, Rowt: s.Rowt, Center: s.Center}
	k, width := g.genKern(append([]uint32{encMatVec}, mv.Encode()...), mv.Emit)
	ss := &compose.SplitSum{Rows: s.Rows, Split: s.Split, Epi: e}
	rk, rwidth := g.genKern(append([]uint32{encMatVec}, ss.Encode()...), ss.Emit)
	if k == nil || rk == nil {
		o.Why = whyNoKernel
		return nil
	}
	part := genLaunch{k: k, groups: (mv.Threads() + width - 1) / width, width: width,
		args: [8]genArg{gaWQS, gaWD, gaWSC, gaOpA, gaOpAX, gaSplit}, n: 6}
	if kernels.IsFloat(s.T) {
		if o.FIn == 0 {
			o.Why = whyNoFloat
			return nil
		}
		part.args[1] = gaIn0 + genArg(o.FIn-1)
	}
	red := genLaunch{k: rk, groups: (s.Rows + rwidth - 1) / rwidth, width: rwidth, args: [8]genArg{gaSplit, gaOut0}, n: 2}
	for _, a := range eargs {
		red.args[red.n] = a
		red.n++
	}
	return []genLaunch{part, red}
}

// genChunkMatVec is a prompt chunk's matvec in the batched dp4a form batchMV
// chose (dot4Split: tokens and rows a thread, a k split), formed by the matvec
// former over the chunk's int8 operand, and a split's segments summed by the
// reduction former through the op's epilogue. Launched whole: the grid is
// row tiles by token groups by segments, which no proportional cut keeps.
// Callers hold g.mu.
func (g *devTier) genChunkMatVec(o *devOp, prog *devProgram, bs *blockScratch, m, bm mv, rows int) []genLaunch {
	e, eargs := matEpi(o, m, m.rows)
	if e == nil {
		o.Why = whyEpilogue
		return nil
	}
	rowt, tok := m.rows/max(bm.rows, 1), rows/max(bm.groups, 1)
	split := max(bm.split, 1)
	mv := &compose.MatVec{T: m.q, K: m.k, Rows: m.rows, NTok: rows, Tok: tok, Rowt: rowt, Split: split, Center: g.center}
	out := gaOut0
	if split > 1 {
		need := m.rows * rows * split
		if g.genInSub && g.partCap < need || !g.genInSub && !g.sizePart(need) {
			o.Why = whySplit
			return nil
		}
		out = gaSplit
	} else {
		mv.Epi = e
	}
	k, width := g.genKern(append([]uint32{encMatVec}, mv.Encode()...), mv.Emit)
	if k == nil {
		o.Why = whyNoKernel
		return nil
	}
	gl := genLaunch{k: k, groups: (mv.Threads() + width - 1) / width, width: width,
		args: [8]genArg{gaWQS, gaWD, gaWSC, gaOpA, gaOpAX, out}, n: 6, whole: true}
	if kernels.IsFloat(m.q) {
		if o.FIn == 0 {
			o.Why = whyNoFloat
			return nil
		}
		gl.args[1] = gaIn0 + genArg(o.FIn-1)
	}
	if split == 1 {
		for _, a := range eargs {
			gl.args[gl.n] = a
			gl.n++
		}
		return []genLaunch{gl}
	}
	ss := &compose.SplitSum{Rows: m.rows * rows, Split: split, Width: m.rows, Epi: e}
	rk, rwidth := g.genKern(append([]uint32{encMatVec}, ss.Encode()...), ss.Emit)
	if rk == nil {
		o.Why = whyNoKernel
		return nil
	}
	red := genLaunch{k: rk, groups: (m.rows*rows + rwidth - 1) / rwidth, width: rwidth, args: [8]genArg{gaSplit, gaOut0}, n: 2, whole: true}
	for _, a := range eargs {
		red.args[red.n] = a
		red.n++
	}
	return []genLaunch{gl, red}
}

// genPagedWrite is a block's new K and V rows into its paged history: K
// (turned by the rotary on a KV write, its tail passed through) at its
// transposed place in its position's page, V at its row there, packed in
// binary16 pairs where the pool is. Each place from the row's descriptor and
// the page table (compose.PagedPos). nil where the history is not paged, a
// row is wider than the heads it holds (an indexer's key beside them), or a
// segment writes in phases.
func (g *devTier) genPagedWrite(o *devOp, prog *devProgram, bs *blockScratch, rows int, turn bool, kArg, vArg genArg) []genLaunch {
	p := &bs.p
	kw, vw := p.KVRow(), p.NKVHead*p.HeadDim
	ki, vi := int(kArg-gaIn0), int(vArg-gaIn0)
	if bs.pkv == nil || p.NonCausal || len(o.In) <= max(ki, vi) || prog.vals[o.In[ki]].Width != vw || kw != vw ||
		prog.vals[o.In[vi]].Width != vw {
		return nil
	}
	P := g.kvPage()
	if turn && (p.NRot <= 0 || p.NoPosEnc || p.RopeSplit || p.NRot%2 != 0 || p.NRot > p.HeadDim) {
		return nil
	}
	kr := &rowBuild{r: compose.Row{Width: kw, Rows: rows}}
	in := kr.buf(compose.BufRow, kArg)
	var cs int
	if turn {
		cs = kr.buf(compose.BufVec, gaCS)
	}
	desc, tab := kr.buf(compose.BufU32, gaPRow), kr.buf(compose.BufU32, gaTab)
	pid, off := kr.r.PagedPos(desc, tab, P)
	x := kr.r.Load(in)
	if turn {
		x = kr.r.Rotary(in, cs, p.HeadDim, p.NRot, p.RopeNeox, false)
	}
	kr.r.StoreAt(kr.buf(compose.BufOut, gaKPool), x, kr.r.PagedK(pid, off, kw, P))

	vr := &rowBuild{r: compose.Row{Width: vw, Rows: rows}}
	vin := vr.buf(compose.BufRow, vArg)
	vdesc, vtab := vr.buf(compose.BufU32, gaPRow), vr.buf(compose.BufU32, gaTab)
	vpid, voff := vr.r.PagedPos(vdesc, vtab, P)
	if g.KVF16 {
		vr.r.Width = vw / 2
		row := vr.r.UBin(compose.OpMul, vr.r.RowIdx(), vr.r.UConst(vw))
		e := vr.r.UBin(compose.OpAdd, row, vr.r.UBin(compose.OpMul, vr.r.Column(), vr.r.UConst(2)))
		w := vr.r.Bin(compose.OpPackF16, vr.r.LoadAt(vin, e), vr.r.LoadAt(vin, vr.r.UBin(compose.OpAdd, e, vr.r.UConst(1))))
		at := vr.r.UBin(compose.OpAdd, vr.r.UBin(compose.OpDiv, vr.r.PagedRow(vpid, voff, P, kw, 0), vr.r.UConst(2)), vr.r.Column())
		vr.r.StoreAt(vr.buf(compose.BufOutU32, gaVPool), w, at)
	} else {
		vr.r.StoreAt(vr.buf(compose.BufOut, gaVPool), vr.r.Load(vin),
			vr.r.UBin(compose.OpAdd, vr.r.PagedRow(vpid, voff, P, kw, 0), vr.r.Column()))
	}
	var out []genLaunch
	for _, rb := range []*rowBuild{kr, vr} {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return nil
		}
		l := genLaunch{k: k.k, groups: rows, width: k.group, n: uint8(len(rb.args))}
		copy(l.args[:], rb.args)
		out = append(out, l)
	}
	return out
}

// genHeadCopy is a row's per-head rearrangement as one composed index map: a
// slice (each head's Width columns from Offset of its HeadDim-wide stride in
// the source, or a row's Width from Offset), a concatenation of two values
// head by head, and the rotary over a value of rotary-wide heads. nil over a
// chunk of a program with latent attention, whose rows the executor keeps
// head-major where the program types them row-major.
func (g *devTier) genHeadCopy(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	if rows > 1 && prog.latent || len(o.In) == 0 || len(o.Out) == 0 {
		return nil
	}
	p := &bs.p
	iw, w := prog.vals[o.In[0]].Width, prog.vals[o.Out[0]].Width
	rb := &rowBuild{r: compose.Row{Width: w, Rows: rows}}
	r := &rb.r
	u := r.UConst
	switch o.Kind {
	case dkSlice:
		a := o.A
		src := rb.buf(compose.BufRow, gaIn0)
		col := r.Column()
		at := r.UBin(compose.OpAdd, col, u(a.Offset))
		if a.HeadDim > 0 {
			if a.Width <= 0 || iw%a.HeadDim != 0 || w != iw/a.HeadDim*a.Width || a.Offset+a.Width > a.HeadDim {
				return nil
			}
			at = r.UBin(compose.OpAdd, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.UBin(compose.OpDiv, col, u(a.Width)), u(a.HeadDim)),
				u(a.Offset)), r.UBin(compose.OpRem, col, u(a.Width)))
		} else if a.Offset+w > iw {
			return nil
		}
		rb.store(r.LoadAt(src, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(iw)), at)), gaOut0)
	case dkConcatHeads:
		nh := p.NHead
		if len(o.In) != 2 || nh <= 0 || iw%nh != 0 || prog.vals[o.In[1]].Width%nh != 0 || w != iw+prog.vals[o.In[1]].Width {
			return nil
		}
		pw := prog.vals[o.In[1]].Width
		la, lb := iw/nh, pw/nh
		a, b := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufRow, gaIn1)
		col := r.Column()
		h, c := r.UBin(compose.OpDiv, col, u(la+lb)), r.UBin(compose.OpRem, col, u(la+lb))
		first := r.Less(c, u(la))
		ia := r.UBin(compose.OpAdd, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(iw)), r.UBin(compose.OpMul, h, u(la))),
			r.UBin(compose.OpMin, c, u(la-1)))
		ib := r.UBin(compose.OpAdd, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(pw)), r.UBin(compose.OpMul, h, u(lb))),
			r.Select(first, u(0), r.UBin(compose.OpSub, c, u(la))))
		rb.store(r.Select(first, r.LoadAt(a, ia), r.LoadAt(b, ib)), gaOut0)
	case dkRopeTail:
		nh := p.NHead
		if nh <= 0 || w%nh != 0 || iw != w || (w/nh)%2 != 0 {
			return nil
		}
		rot := w / nh
		in, cs := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufVec, gaCS)
		rb.store(r.Rotary(in, cs, rot, rot, p.RopeNeox, false), gaOut0)
	default:
		return nil
	}
	k := g.rowKern(r)
	if k.k == nil {
		return nil
	}
	l := genLaunch{k: k.k, groups: rows, width: k.group, n: uint8(len(rb.args))}
	copy(l.args[:], rb.args)
	return []genLaunch{l}
}

// genBankMatVec is a decode step's matvec over a bank's selected sheets
// (an expert bank, or a latent attention's heads), formed by the matvec
// former at the bank's schedule: its slots, an operand vector a slot where
// the bank reads one, its k split in the group or into planes summed by the
// reduction former. The selection is the op's second input; a float bank
// reads the float rows, its third. Callers hold g.mu.
func (g *devTier) genBankMatVec(o *devOp, prog *devProgram, bs *blockScratch) []genLaunch {
	l := g.curLayer
	if l == nil || o.W < 0 || o.W >= len(l.ms) || *l.rs[o.W] == nil || len(o.In) < 2 {
		o.Why = whyNoWeight
		return nil
	}
	m := *l.ms[o.W]
	switch {
	case m.kern == nil:
		o.Why = whyNoKernel
	case m.experts < 2 || m.rowt > 1 || m.pad || g.TableSplit && m.alt != nil:
		o.Why = whyBank
	}
	if o.Why != 0 {
		return nil
	}
	split := max(m.split, 1)
	mv := &compose.MatVec{T: m.q, K: m.k, Rows: m.rows, Experts: m.experts, Slots: max(m.slots, 1), SlotAct: m.slotAct,
		Split: split, Group: m.group, Center: g.center}
	n := m.rows * mv.Slots
	out := gaOut0
	if split > 1 && !m.group {
		need := n * split
		if g.genInSub && g.partCap < need || !g.genInSub && !g.sizePart(need) {
			o.Why = whySplit
			return nil
		}
		out = gaSplit
	}
	k, width := g.genKern(append([]uint32{encMatVec}, mv.Encode()...), mv.Emit)
	if k == nil {
		o.Why = whyNoKernel
		return nil
	}
	gl := genLaunch{k: k, groups: (mv.Threads() + width - 1) / width, width: width,
		args: [8]genArg{gaWQS, gaWD, gaWSC, gaOpA, gaOpAX, out, gaIn1}, n: 7}
	if kernels.IsFloat(m.q) {
		if len(o.In) < 3 {
			o.Why = whyNoFloat
			return nil
		}
		gl.args[1] = gaIn2
	}
	if out == gaOut0 {
		return []genLaunch{gl}
	}
	ss := &compose.SplitSum{Rows: n, Split: split}
	rk, rwidth := g.genKern(append([]uint32{encMatVec}, ss.Encode()...), ss.Emit)
	if rk == nil {
		o.Why = whyNoKernel
		return nil
	}
	return []genLaunch{gl, {k: rk, groups: (n + rwidth - 1) / rwidth, width: rwidth, args: [8]genArg{gaSplit, gaOut0}, n: 2}}
}

// genCombine is a decode step's weighted sum of its selected experts' rows:
// each column the slots' fused multiply-adds in slot order from zero (the
// pairs' rows and the weights the op's two inputs). Callers hold g.mu.
func (g *devTier) genCombine(o *devOp, prog *devProgram, bs *blockScratch) []genLaunch {
	if len(o.In) != 2 || len(o.Out) != 1 {
		return nil
	}
	n, slots := prog.vals[o.Out[0]].Width, prog.vals[o.In[1]].Width
	if n < 1 || slots < 1 || prog.vals[o.In[0]].Width != n {
		return nil
	}
	rb := &rowBuild{r: compose.Row{Width: n, Rows: 1}}
	r := &rb.r
	e, w := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufRow, gaIn1)
	acc := r.Const(0)
	for j := 0; j < slots; j++ {
		x := r.LoadAt(e, r.UBin(compose.OpAdd, r.UConst(j*n), r.Column()))
		acc = r.Add(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: r.LoadAt(w, r.UConst(j)), B: x, C: acc})
	}
	rb.store(acc, gaOut0)
	k := g.rowKern(r)
	if k.k == nil {
		return nil
	}
	l := genLaunch{k: k.k, groups: 1, width: k.group, n: uint8(len(rb.args))}
	copy(l.args[:], rb.args)
	return []genLaunch{l}
}

// genRopeQKV is a step's q, k and v into place in one op: q turned by the
// rotary at its offset in the call's table, k turned into its paged K
// place and v copied to its row (genPagedWrite), q, k and v the op's three
// inputs. Callers hold g.mu.
func (g *devTier) genRopeQKV(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	p := &bs.p
	if len(o.In) != 3 || len(o.Out) < 1 || p.HeadDim <= 0 || p.NRot <= 0 || p.NRot%2 != 0 || p.NRot > p.HeadDim {
		return nil
	}
	qw := prog.vals[o.In[0]].Width
	if qw%p.HeadDim != 0 || prog.vals[o.Out[0]].Width != qw {
		return nil
	}
	kv := g.genPagedWrite(o, prog, bs, rows, true, gaIn1, gaIn2)
	if kv == nil {
		return nil
	}
	qb := &rowBuild{r: compose.Row{Width: qw, Rows: rows}}
	in, cs, off := qb.buf(compose.BufRow, gaIn0), qb.buf(compose.BufVec, gaCS), qb.buf(compose.BufU32, gaQOff)
	v := qb.r.Rotary(in, cs, p.HeadDim, p.NRot, p.RopeNeox, false)
	qb.r.StoreAt(qb.buf(compose.BufOut, gaOut0), v, qb.r.UBin(compose.OpAdd, qb.r.Load(off), qb.r.Column()))
	k := g.rowKern(&qb.r)
	if k.k == nil {
		return nil
	}
	l := genLaunch{k: k.k, groups: rows, width: k.group, n: uint8(len(qb.args))}
	copy(l.args[:], qb.args)
	return append([]genLaunch{l}, kv...)
}
