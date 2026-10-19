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
	gaFSrc  // the float row a binary16 operand is made from
	gaQ8A   // an int8 output's values
	gaQ8X   // its scales and sums
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
		o.Gen = g.genOp(o, prog, bs, rows)
	}
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
		// The matvec former: the weight's (or bank's, or the fused q/k/v
		// grid's) kernel, emitted from its shape.
		return []genLaunch{{former: formMatVec}}
	case dkAttend, dkLatentAttend, dkKVWrite, dkKVCopy, dkRopeQKV, dkSlice, dkView, dkConcatHeads, dkRopeTail, dkLatentWrite:
		// The attention former: the paged plans emitted from the attention's
		// shape and the device's tuning, the history's row writes, and a
		// latent attention's slices of its rows.
		return []genLaunch{{former: formAttend}}
	case dkTopK, dkRouteWeights, dkCombine, dkGroupByExpert, dkGateLogit, dkGroupMask, dkScaleWeights, dkRouterBias,
		dkExpertBias, dkExpertFetch:
		// The router former: the selection's kernels emitted from the route's
		// shape (MoERoute), the bank's fetch, and the combine.
		return []genLaunch{{former: formRoute}}
	case dkConv1d, dkScan, dkGatedNorm, dkSSMNorm, dkConvBias:
		// The scan former: a recurrence's kernels emitted from its plan's
		// shape (RecurrentPlan).
		return []genLaunch{{former: formScan}}
	}
	if l := g.genPairs(o, prog, bs, rows); l != nil {
		return l
	}
	for _, v := range o.In {
		if prog.vals[v].Layout != layoutRow {
			return nil
		}
	}
	for _, v := range o.Out {
		if prog.vals[v].Layout != layoutRow {
			return nil
		}
	}
	w := 0
	if len(o.Out) > 0 {
		w = prog.vals[o.Out[0]].Width
	}
	p := &bs.p
	eps := float32(p.RMSEps)
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
		rb.store(rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpAct, Act: a, A: rb.load(compose.BufRow, gaIn0)}), gaOut0)
	case dkActMul:
		a, ok := kernAct(o.Act)
		if !ok || len(o.In) != 2 || kernels.ActMulOK(a) != nil {
			return nil
		}
		gt, up := rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1)
		rb.store(rb.node(compose.Node{Kind: compose.NBinary, Op: compose.OpActMul, Act: a, A: gt, B: up}), gaOut0)
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
		a, ok := kernAct(o.Act)
		if !ok || len(o.In) != 2 || kernels.ActMulOK(a) != nil {
			return nil
		}
		y := rb.node(compose.Node{Kind: compose.NBinary, Op: compose.OpActMul, Act: a,
			A: rb.load(compose.BufRow, gaIn0), B: rb.load(compose.BufRow, gaIn1)})
		rb.store(y, gaOut0)
		q8 = 1
		rb.quantize(y, w, p)
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
	case dkRopeQ:
		// The rotary over each head's first NRot dimensions (NORM or NEOX
		// pairs), the rest of the head passed through: one composed group
		// where a partial rotary took a copy and a turn. Each row lands at
		// its offset in the call's table.
		ra := o.A
		iw := prog.vals[o.In[0]].Width
		if ra.HeadDim <= 0 || iw%ra.HeadDim != 0 || ra.NRot <= 0 || ra.NRot%2 != 0 || ra.Layout == graph.RopeXD {
			return g.genTemplate(o, prog, bs, rows)
		}
		rb.r.Width = iw
		in, cs, off := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufVec, gaCS), rb.buf(compose.BufU32, gaQOff)
		v := rb.r.Rotary(in, cs, ra.HeadDim, ra.NRot, ra.Layout == graph.RopeNeox)
		out := rb.buf(compose.BufOut, gaOut0)
		rb.r.StoreAt(out, v, rb.r.UBin(compose.OpAdd, rb.r.Load(off), rb.r.Column()))
	case dkRopeKRow, dkSplitQG:
		return g.genTemplate(o, prog, bs, rows)
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

// rowEntry is a cached row kernel and the structure it was formed from: a
// hash that matches with a different structure is a collision, not a hit.
type rowEntry struct {
	enc []uint32
	rk  rowKernel
}

// rowKern is row group r's kernel under the target's schedule, formed once
// per device and structure. Callers hold g.mu.
func (g *devTier) rowKern(r *compose.Row) rowKernel {
	s := rowSchedule(r)
	enc := r.Encode(s)
	key := compose.KeyOf(enc)
	for _, e := range g.rowK[key] {
		if slices.Equal(e.enc, enc) {
			return e.rk
		}
	}
	rk := rowKernel{group: s.Group}
	ker, err := r.Emit(s)
	if err == nil {
		rk.k, err = g.dev.Compile(ker)
	}
	if err != nil {
		g.LastErr = fmt.Sprintf("a composed row group: %v", err)
		rk.k = nil
	}
	if g.rowK == nil {
		g.rowK = map[compose.Key][]rowEntry{}
	}
	g.rowK[key] = append(g.rowK[key], rowEntry{enc: enc, rk: rk})
	return rk
}

// genTemplate is a rotary's or the query/gate split's launches from their
// templates, until the row former states their index maps.
func (g *devTier) genTemplate(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	var out []genLaunch
	failed := false
	add := func(key convKey, build func() (*ir.Kernel, error), threads int, args ...genArg) {
		k := g.shapeKern(key, build)
		if k == nil {
			failed = true
			return
		}
		l := genLaunch{k: k, threads: threads, n: uint8(len(args))}
		copy(l.args[:], args)
		out = append(out, l)
	}
	p := &bs.p
	w := prog.vals[o.Out[0]].Width
	switch o.Kind {
	case dkSplitQG:
		nh, hd := p.NHead, p.HeadDim
		if w != nh*hd || prog.vals[o.In[0]].Width != 2*w {
			return nil
		}
		add(convKey{id: kidSplitheadgate, a: [8]int{nh, hd, rows}}, func() (*ir.Kernel, error) {
			return kernels.SplitHeadGate(nh, hd, rows)
		}, rows*w, gaIn0, gaOut0, gaOut1)
	case dkRopeQ, dkRopeKRow:
		a := o.A
		iw := prog.vals[o.In[0]].Width
		if a.HeadDim <= 0 || iw%a.HeadDim != 0 || a.NRot <= 0 {
			return nil
		}
		heads, hd, nrot := iw/a.HeadDim, a.HeadDim, a.NRot
		off := gaQOff
		if o.Kind == dkRopeKRow {
			off = gaKOff
		}
		if o.Kind == dkRopeQ && nrot != hd {
			add(convKey{id: kidCopyat, a: [8]int{iw, rows}}, func() (*ir.Kernel, error) { return kernels.CopyAtRows(iw, rows) },
				rows*iw, gaIn0, gaOut0, off)
		}
		if a.Layout == graph.RopeXD {
			add(convKey{id: kidRopesplit, a: [8]int{heads, hd, nrot, rows}}, func() (*ir.Kernel, error) {
				return kernels.RoPERowsSplit(heads, hd, nrot, rows, 0)
			}, rows*heads*nrot/2, gaIn0, gaCS, off, gaOut0)
			break
		}
		if o.Kind == dkRopeKRow {
			return nil
		}
		neox := a.Layout == graph.RopeNeox
		nx := 0
		if neox {
			nx = 1
		}
		add(convKey{id: kidRope, a: [8]int{heads, hd, nrot, nx, rows}}, func() (*ir.Kernel, error) {
			return kernels.RoPERows(heads, hd, nrot, neox, rows)
		}, rows*heads*nrot/2, gaIn0, gaCS, off, gaOut0)
	}
	if failed {
		return nil
	}
	return out
}

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
	// The step's binary16 operand: its plane, the float row it is made
	// from, and the conversion the step last made (shared with the
	// executor, so a conversion either one made is not made again).
	f16, fsrc backend.Buf
	memoK     backend.Kernel
	memoSrc   backend.Buf
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
				bufs[j] = st.a
			case gaOpAX:
				bufs[j] = st.ax
			case gaMBias:
				bufs[j] = st.mbias
			case gaF16:
				bufs[j] = st.f16
			case gaFSrc:
				bufs[j] = st.fsrc
			case gaQ8A:
				bufs[j], _ = int8Of(st, o.Out[l.q8])
			case gaQ8X:
				_, bufs[j] = int8Of(st, o.Out[l.q8])
			}
		}
		if l.toF16 {
			if st.memoK == l.k && st.memoSrc == st.fsrc {
				continue
			}
			st.memoK, st.memoSrc = l.k, st.fsrc
		}
		threads := l.threads
		if rows > 0 && R < rows {
			threads = threads / rows * R
		}
		if l.width != 0 {
			groups := l.groups
			switch {
			case l.pairs && st.moe:
				groups = st.cols
			case l.pairs:
				groups = bs.nUsed * R
			case rows > 0 && R < rows:
				groups = (groups*R + rows - 1) / rows
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
	rb := &rowBuild{r: compose.Row{Width: w, Rows: max(rows*p.NExpertUsed, 1)}}
	var q8 uint8
	switch o.Kind {
	case dkAct:
		a, ok := kernAct(o.Act)
		if !ok {
			return nil
		}
		rb.store(rb.r.Act(a, rb.load(compose.BufRow, gaIn0)), gaOut0)
	case dkActMul:
		a, ok := kernAct(o.Act)
		if !ok || kernels.ActMulOK(a) != nil {
			return nil
		}
		gt, up := rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1)
		if len(o.In) > 2 {
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
		rb.store(rb.node(compose.Node{Kind: compose.NBinary, Op: compose.OpActMul, Act: a, A: gt, B: up}), gaOut0)
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
		return nil
	}
	m := *l.ms[o.W]
	if m.kern == nil || m.split > 1 && !m.group || m.experts > 1 || m.slots > 0 || m.pad || g.TableSplit && m.alt != nil {
		return nil
	}
	if _, _, own := q8Pair(bs, prog.slots[o.In[0]]); own {
		return nil
	}
	e, args := matEpi(o, m, m.rows)
	if e == nil {
		return nil
	}
	rowt := max(m.rowt, 1)
	s := kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, Split: m.split, Rowt: rowt, GroupSplit: m.group}
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
	for _, a := range args {
		gl.args[gl.n] = a
		gl.n++
	}
	return []genLaunch{gl}
}

// mvEpiEntry is a cached composed matvec and what it was formed from.
type mvEpiEntry struct {
	shape kernels.MatVecShape
	enc   []uint32
	k     backend.Kernel
	width int
}

// mvEpiKern is the matvec of shape s composed with epilogue e, formed once a
// device and structure, and its workgroup width. Callers hold g.mu.
func (g *devTier) mvEpiKern(s kernels.MatVecShape, e *compose.Epi) (backend.Kernel, int) {
	return g.epiKern(s, e.Encode(), func() (*ir.Kernel, error) { return kernels.MatVecEpi(s, e) })
}

// epiKern is a former's kernel of shape s with an epilogue, its structure
// enc (the epilogue's and the former's schedule), built by build once a
// device and structure, and its workgroup width. Callers hold g.mu.
func (g *devTier) epiKern(s kernels.MatVecShape, enc []uint32, build func() (*ir.Kernel, error)) (backend.Kernel, int) {
	key := compose.KeyOf(append(append([]uint32(nil), enc...), uint32(s.T), uint32(s.K), uint32(s.Rows),
		uint32(s.Split), uint32(s.Rowt), b2u(s.GroupSplit), uint32(s.NTok)))
	for _, x := range g.mvEpiK[key] {
		if x.shape == s && slices.Equal(x.enc, enc) {
			return x.k, x.width
		}
	}
	ent := mvEpiEntry{shape: s, enc: enc}
	ker, err := build()
	if err == nil {
		ent.width = ker.Group[0]
		ent.k, err = g.dev.Compile(ker)
	}
	if err != nil {
		g.LastErr = fmt.Sprintf("a composed matvec: %v", err)
		ent.k = nil
	}
	if g.mvEpiK == nil {
		g.mvEpiK = map[compose.Key][]mvEpiEntry{}
	}
	g.mvEpiK[key] = append(g.mvEpiK[key], ent)
	return ent.k, ent.width
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
		return nil
	}
	m := *l.ms[o.W]
	if m.slots > 0 || m.pad || g.TableSplit && m.alt != nil {
		return nil
	}
	bm, ok := g.batchMV(m, rows, bs.tok)
	if !ok || bm.act == nil || !bm.actT || bm.split > 1 || bm.tail != nil || bm.vt.MT == 0 {
		return nil
	}
	e, args := matEpi(o, m, m.rows)
	if e == nil {
		return nil
	}
	s := kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: rows}
	enc := append(e.Encode(), 7, uint32(bm.vt.MT), uint32(bm.vt.NT), uint32(bm.vt.WM), uint32(bm.vt.WN),
		uint32(bm.vt.KB), b2u(bm.vt.Single), uint32(bm.vt.F16K))
	k, _ := g.epiKern(s, enc, func() (*ir.Kernel, error) { return kernels.GemmVoltaEpi(s, bm.vt, e) })
	if k == nil {
		return nil
	}
	conv := genLaunch{k: bm.act, threads: bm.actN, args: [8]genArg{gaFSrc, gaF16}, n: 2, toF16: true}
	gl := genLaunch{k: k, threads: bm.threads, groups: bm.threads / bm.width, width: bm.width,
		args: [8]genArg{gaWQS, gaWD, gaWSC, gaF16, gaOut0}, n: 5}
	for _, a := range args {
		gl.args[gl.n] = a
		gl.n++
	}
	return []genLaunch{conv, gl}
}
