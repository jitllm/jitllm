package tier

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
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
	gaOpA       // the int8 operand the op's matvec reads, values
	gaOpAX      // and scales
	gaMBias     // the weight's own bias (mv.bias)
	gaF16       // the binary16 operand a GEMM reads (the device's f16 plane)
	gaQ8A       // an int8 output's values
	gaQ8X       // its scales and sums
	gaPerm      // the step's head-major order: each gathered row's source row (headPerm)
	gaSplit     // a k split's partial plane (the tier's partBuf)
	gaKPool     // the block's paged K pool
	gaVPool     // its paged V pool
	gaTab       // its page table
	gaPRow      // the step's row descriptors (kernels.PRowWords a row)
	gaPRowW     // the windowed descriptors, a sliding layer's
	gaScores    // the staged attention's score plane
	gaWts       // its weight plane
	gaParts     // the splits' partials the merge reads
	gaConvIn    // a recurrent block's convolution window, the half the step reads
	gaConvOut   // the half it writes
	gaConvW     // the convolution's weights
	gaRecDesc   // the step's recurrent descriptor
	gaStateIn   // a recurrent block's state, the half the step reads
	gaStateOut  // where it writes the next
	gaRecDt     // the block's dt bias
	gaRecA      // its decay rates
	gaRecD      // its skip
	gaRecCB     // its convolution's bias
	gaRecNorm   // its output norm's weight
	gaRecMid    // the scratch between its gated norm's two launches
	gaRouter    // a mixture's router matrix
	gaRouterB   // its bias on the logits
	gaSelB      // its selection bias
	gaRouteW    // the route's weights (a fused route writes them with the selection)
	gaSink      // the block's per-head attention sinks
	gaCount     // the step's per-row key counts (the decode's one, then each row's)
	gaTemp      // the attention temperature over each row's count
	gaRagMap    // a ragged call's run table: each slot's row
	gaRagBase   // each slot's partition base
	gaRagBaseW  // the windowed descriptors' bases
	gaPreDesc   // a chunk's staged prefill descriptors, padding rows emptied
	gaPreDescW  // and its windowed ones
	gaPLEIn     // the step's per-layer inputs, every block's a row
	gaHeadTiles // each token group's head, a chunk's rows head-major
	gaMgBC      // a mixture's grouping over rows: the block counts,
	gaMgOff     // the offsets,
	gaMgCnt     // the totals,
	gaMgEnds    // each run's end,
	gaMgColE    // each column's expert,
	gaMgPerm    // each column's row,
	gaMgPos     // each pair's column,
	gaMgTsel    // each tile's expert,
	gaMgHG      // the sorted rows,
	gaMgGA      // and their int8 operand
	gaMgGAX
	gaOwn0 // the op's own buffers (devOp.Own)
	gaOwn1
	gaOwn2
	gaOwn3
	gaOwn4
	gaOwn5
	gaOwn6
	gaOwn7
)

// genLaunch is one generated launch: threads for a flat launch, or groups of
// width where width is set.
type genLaunch struct {
	k             backend.Kernel
	threads       int
	groups, width int
	args          [12]genArg
	n             uint8
	// q8 is which output holds the op's int8 value (gaQ8A and gaQ8X).
	q8 uint8
	// former, where set, runs the launch through a kernel former's own
	// launcher (the executor's form) rather than one kernel: the matvec's.
	former uint8
	// pairs launches a group a (token, expert) pair of the step, not a
	// group a token.
	pairs bool
	// flat is a flat kernel's elements a row, 0 for a group a row.
	flat int
	// toF16 is a binary16 operand made from the op's float row: skipped
	// where the step's last conversion was this kernel over this row (q, k
	// and v share one normed row, as do gate and up).
	toF16 bool
	// whole launches the kernel's full grid at any rows: a tiled GEMM's
	// grid is tiles of rows by tiles of tokens, which no proportional cut
	// keeps. Rows past the step's compute what nothing reads. (A kernel
	// built for the step's own rows in the prologue is the faster form.)
	whole bool
	// colScale, where set, sizes the launch by the step's used columns of a
	// mixture's sorted rows: cols*colScale/colDiv threads, flat.
	colScale, colDiv int
	// readEnds sends what was launched, then reads the grouping's run ends
	// home: the used column count the next launches are sized by.
	readEnds bool
	// once, where set, skips the launch where its first buffer is the one
	// its kind last read this step (a gather, a quantize shared by gate and
	// up).
	once uint8
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
	prog.floatSources()
	for i := range prog.ops {
		o := &prog.ops[i]
		if o.Act == 0 {
			o.Act = o.A.Act // an activation its emit site did not record
		}
		if o.Kind == dkAdd && o.Alpha != 0 && o.Alpha != 1 && o.GC == nil {
			o.GC = g.constBuf(float32(o.Alpha))
		}
		o.Gen = g.genOp(o, prog, bs, rows)
	}
}

// floatSources states each matvec's float row as an input of its own (FIn):
// what a float-format weight's kernel reads beside the int8 operand. It runs
// before the values are coloured, so the colouring sees the read.
func (p *devProgram) floatSources() {
	for i := range p.ops {
		o := &p.ops[i]
		if o.FIn == 0 && (o.Kind == dkMatVec || o.Kind == dkMatVecRes || o.Kind == dkMatVecGate) && len(o.In) < 4 {
			if f, ok := p.floatSource(i, o.In[0]); ok {
				o.In = append(o.In, f)
				o.FIn = int8(len(o.In))
			}
		}
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

// genOp is op o's launches, or nil where the composer has no form for it
// yet or the backend refused its kernel, o.Why saying which. Nil fails the
// block: nothing else runs an op.
func (g *devTier) genOp(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	if ls, ok := g.genMoERows(o, prog, bs, rows); ok {
		if ls == nil {
			o.Why = whyNoForm
		}
		return ls
	}
	// The formers take their ops whatever their values' layouts: a former
	// lays out its own rows (a bank's pairs, a history's pages).
	switch o.Kind {
	case dkMap, dkReduce, dkSelectK, dkRank, dkTake, dkTile, dkConcat, dkRelayout:
		return g.genPrim(o, prog, rows)
	case dkMatVec, dkMatVecRes, dkMatVecGate, dkExpertMatVec:
		return g.genMatVecNodes(o, prog, bs, rows)
	case dkConvBias, dkSSMNorm:
		l := g.genSSMRow(o, prog, bs, rows)
		o.Why = whyNoForm
		if l != nil {
			o.Why = whyNone
		}
		return l
	case dkGatedNorm:
		l := g.genGatedNorm(o, prog, bs, rows)
		o.Why = whyNoForm
		if l != nil {
			o.Why = whyNone
		}
		return l
	case dkScan:
		l := g.genScan(o, prog, bs, rows)
		o.Why = whyNoForm
		if l != nil {
			o.Why = whyNone
		}
		return l
	case dkConv1d:
		l := g.genConv(o, prog, bs, rows)
		o.Why = whyNoForm
		if l != nil {
			o.Why = whyNone
		}
		return l
	case dkStreamNorm, dkStreamDot, dkGatedVal, dkNGramConv, dkNGramAdd:
		ls := g.genStreams(o, prog, bs, rows)
		if ls == nil {
			o.Why = whyNoForm
		}
		return ls
	case dkIdxAssemble, dkIdxScore, dkSelectBlocks, dkGatherKV, dkAttendGathered:
		ls := g.genSelect(o, prog, bs, rows)
		if ls == nil {
			o.Why = whyNoForm
		}
		return ls
	case dkAttend, dkLatentAttend:
		l := g.genAttend(o, prog, bs, rows)
		o.Why = whyNoForm
		if l != nil {
			o.Why = whyNone
		}
		return l
	case dkRouterLogits, dkTopK, dkRouteWeights, dkRouterBias, dkGroupByExpert:
		return g.genRoute(o, prog, bs, rows)
	case dkView:
		// A row's prefix read in place: at one row the value is its source's
		// own buffer (its slot is bound there), and nothing launches.
		if rows == 1 {
			o.Why = whyNothing
			return nil
		}
		o.Why = whyNoForm
		return nil
	case dkExpertMatVecE, dkQKV, dkQKVGemm,
		dkResIn, dkQKVSep, dkGateLogit,
		dkGroupMask, dkExpertBias, dkExpertFetch:
		// No form in the composer's primitives yet: the op fails to
		// generate, and its block does not place.
		o.Why = whyNoForm
		return nil
	case dkKVWrite, dkKVCopy, dkRopeQKV, dkSlice, dkConcatHeads, dkRopeTail, dkLatentWrite, dkCombine:
		var l []genLaunch
		switch o.Kind {
		case dkKVWrite, dkKVCopy:
			l, o.Why = g.genPagedWrite(o, prog, bs, rows, o.Kind == dkKVWrite, gaIn0, gaIn1), whyHistory
		case dkRopeQKV:
			l, o.Why = g.genRopeQKV(o, prog, bs, rows), whyHistory
		case dkSlice, dkConcatHeads, dkRopeTail:
			l, o.Why = g.genHeadCopy(o, prog, bs, rows), whyLayout
		case dkLatentWrite:
			l, o.Why = g.genLatentWrite(o, prog, bs, rows), whyHistory
		case dkCombine:
			if rows == 1 {
				l = g.genCombine(o, prog, bs)
			}
			o.Why = whyNoForm
		}
		if l != nil {
			o.Why = whyNone
		}
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
		if o.Act == graph.ActXIELU {
			if o.V == nil || len(o.In) != 1 {
				return nil
			}
			rb.store(rb.r.XIELU(rb.load(compose.BufRow, gaIn0), rb.buf(compose.BufScalar, gaV)), gaOut0)
			break
		}
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
	case dkGatedAdd:
		// acc + sigmoid(the row's logit) * the branch.
		acc, br := rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1)
		sig := rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpAct, Act: kernels.ActSigmoid, A: rb.load(compose.BufRowScalar, gaIn2)})
		rb.store(rb.bin(compose.OpAdd, acc, rb.bin(compose.OpMul, sig, br)), gaOut0)
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
	case dkPLESlice:
		// This block's slice of each row's per-layer inputs: the row's
		// inputs at the block's offset (its own word).
		if o.V == nil || bs.pleIn == nil || p.PLEWidth < w {
			return nil
		}
		src := rb.buf(compose.BufRow, gaPLEIn)
		off := rb.r.LoadAt(rb.buf(compose.BufU32, gaV), rb.r.UConst(0))
		at := rb.r.UBin(compose.OpAdd, rb.r.UBin(compose.OpAdd, rb.r.UBin(compose.OpMul, rb.r.RowIdx(), rb.r.UConst(p.PLEWidth)), off), rb.r.Column())
		rb.store(rb.r.LoadAt(src, at), gaOut0)
	case dkQTemp:
		// Each row's query by its temperature: the table at its causal count
		// (n's word 0 at one row, 1+r over rows), the count's position
		// offset and floored by the plan's two integers.
		tl := len(l4TempTab(p))
		if p.AttnTempScale == 0 || tl < 1 {
			return nil
		}
		ni := rb.r.UConst(0)
		if rows > 1 {
			ni = rb.r.UBin(compose.OpAdd, rb.r.RowIdx(), rb.r.UConst(1))
		}
		cnt := rb.r.LoadAt(rb.buf(compose.BufU32, gaCount), ni)
		pos := rb.r.UBin(compose.OpSub, rb.r.UBin(compose.OpMax, cnt, rb.r.UConst(1)), rb.r.UConst(1))
		k := rb.r.UBin(compose.OpMin, rb.r.UBin(compose.OpDiv, rb.r.UBin(compose.OpAdd, pos, rb.r.UConst(int(p.AttnTempOffset))),
			rb.r.UConst(int(p.AttnTempFloor))), rb.r.UConst(tl-1))
		x := rb.load(compose.BufRow, gaIn0)
		rb.store(rb.bin(compose.OpMul, x, rb.r.LoadAt(rb.buf(compose.BufRow, gaTemp), k)), gaOut0)
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
		if p.LayerNorm && o.Kind != dkNormQuant {
			return g.genLayerNorm(o, bs, rows, w, eps)
		}
		if !p.LayerNorm && (o.Kind == dkPostNorm || prog.scans && o.Role == jlm.RoleAttnNorm) {
			// A norm of a branch's output, or a recurrent block's pre-norm: two
			// passes, the int8 operand apart.
			return g.genNormParts(o, prog, bs, rows, w, eps, bs.parts)
		}
		if !p.LayerNorm && (o.Role == jlm.RoleAttnQANorm || o.Role == jlm.RoleAttnKVANorm) {
			// A latent's norm over rows: two passes at the latent's own parts.
			return g.genNormParts(o, prog, bs, rows, w, eps, partsFor(w))
		}
		y := rb.normed(o, rb.load(compose.BufRow, gaIn0), w, eps, p.LayerNorm)
		if y < 0 {
			return nil
		}
		rb.store(y, gaOut0)
		if o.Kind == dkNormQuant {
			q8 = 1
			rb.quantStage(y, w, p.ActWin)
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
			rb.quantStage(y, w, p.ActWin)
		}
	case dkActQuant:
		// act(gate) * up, and its int8 operand.
		if len(o.In) != 2 {
			return nil
		}
		// An elementwise row: its int8 operand formed over its words, the
		// row computed at each word's elements and stored there.
		if w%32 != 0 {
			o.Why = whyNoForm
			return nil
		}
		rb.r.Width, rb.r.Rows = rows*w/4, 1
		y, ok := rb.gated(o, rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1))
		if !ok {
			return nil
		}
		win := p.ActWin
		if win <= 0 {
			win = 32
		}
		q8 = 1
		rb.r.Quantize8Words(y, rb.buf(compose.BufOut, gaOut0), rb.buf(compose.BufOutU32, gaQ8A), rb.buf(compose.BufOut, gaQ8X), min(win, w))
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
		return g.genQuant(o, prog, rows, gaIn0, gaQ8A, gaQ8X, 0, p.ActWin)
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
		if o.N > 0 {
			// Rows landing N apart in a table of the op's own.
			if o.Own[0] = g.ownBuf(bs, o, 0, rows, rowTable(rows, o.N, 0)); o.Own[0] == nil {
				return nil
			}
			offArg = gaOwn0
		}
		return g.ropePairs(rows, iw, ra.HeadDim, ra.NRot, ra.Layout == graph.RopeNeox || ra.Layout == graph.RopeXD,
			ra.Layout == graph.RopeXD, gaIn0, offArg, gaOut0)
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
	l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args)), q8: q8}
	copy(l.args[:], rb.args)
	return []genLaunch{l}
}

// quantStage adds the int8 operand of node y (a row of width w) to the row
// group as a stage of it (compose.Row.Quantize8Stage), over the session's
// window (32 where it states none) clamped to the row.
func (rb *rowBuild) quantStage(y, w, win int) {
	if win <= 0 {
		win = 32
	}
	if win > w {
		win = w // the host's window ends at the row
	}
	rb.r.Quantize8Stage(y, rb.buf(compose.BufOutU32, gaQ8A), rb.buf(compose.BufOut, gaQ8X), win)
}

// genQuant is the int8 operand of the value at argument src as its own
// group (compose.Row.Quantize8): rows of the value's width, quantized over
// the session's window (32 where it states none) clamped to the row,
// the codes and scales into the output q8 names (0 for the op's own int8
// value). Callers hold g.mu.
func (g *devTier) genQuant(o *devOp, prog *devProgram, rows int, src, a, ax genArg, q8 uint8, win int) []genLaunch {
	var iw int
	switch {
	case src >= gaIn0 && src <= gaIn3:
		iw = prog.vals[o.In[src-gaIn0]].Width
	case src >= gaOut0 && src <= gaOut2:
		iw = prog.vals[o.Out[src-gaOut0]].Width
	default:
		return nil
	}
	kp := pad32(iw)
	if win <= 0 {
		win = 32
	}
	if win > kp {
		win = kp // the host's window ends at the row
	}
	rb := &rowBuild{r: compose.Row{Width: rows * kp / 4, Rows: 1}}
	if kp != iw {
		// A row padded to whole blocks, its columns past the width zero.
		rb.r.Quantize8Padded(rb.buf(compose.BufRow, src), rb.buf(compose.BufOutU32, a), rb.buf(compose.BufOut, ax), win, iw, kp)
	} else {
		rb.r.Quantize8(rb.buf(compose.BufRow, src), rb.buf(compose.BufOutU32, a), rb.buf(compose.BufOut, ax), win)
	}
	ps, err := rb.r.Plan(compose.Facts{Target: g.composeTarget()})
	if err != nil || len(ps) != 1 {
		o.Why = whyNoForm
		return nil
	}
	p := ps[0]
	s := p.S
	k, _ := g.genKern(append([]uint32{encRow}, p.G.Encode(s)...), func() (*ir.Kernel, error) { return p.G.Emit(s) })
	if k == nil {
		o.Why = whyNoKernel
		return nil
	}
	gl := genLaunch{k: k, groups: (p.Threads + s.Group - 1) / s.Group, width: s.Group, q8: q8}
	gl.n = uint8(copy(gl.args[:], rb.args))
	return []genLaunch{gl}
}

// rowKernel is a composed row group's compiled kernel and its group size.
type rowKernel struct {
	k     backend.Kernel
	group int
	// groups is the launch's workgroups: a row each, or the flat grid's.
	groups int
	flat   int
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
	var ker *ir.Kernel
	var err error
	if g.Config != nil && g.forms != nil {
		ker, err = g.forms.form(enc, build)
	} else if ker, err = build(); err == nil {
		ker.Comment = composedComment(ker.Comment)
	}
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

// rowKern is row group r's kernel under the schedule compose plans for it
// on this target: one kernel, a row group (a contraction goes through
// genMatVecNodes).
func (g *devTier) rowKern(r *compose.Row) rowKernel { return g.rowKernWith(r, compose.Facts{}) }

// rowKernWith is rowKern under the tier's pins in f.
func (g *devTier) rowKernWith(r *compose.Row, f compose.Facts) rowKernel {
	f.Target = g.composeTarget()
	ps, err := r.Plan(f)
	if err != nil || len(ps) != 1 {
		return rowKernel{}
	}
	s := ps[0].S
	k, _ := g.genKern(append([]uint32{encRow}, r.Encode(s)...), func() (*ir.Kernel, error) { return r.Emit(s) })
	groups, flat := r.Rows, 0
	if s.Flat {
		groups, flat = (ps[0].Threads+s.Group-1)/s.Group, r.Width
	}
	return rowKernel{k: k, group: s.Group, groups: groups, flat: flat}
}

// composeTarget is this device as the composer's passes ask of it, stated
// once a tier.
func (g *devTier) composeTarget() compose.Target {
	if g.ctarget.API != compose.APINone {
		return g.ctarget
	}
	t := compose.Target{API: compose.APIOf(g.dev.API()), ISA: backend.ArchOf(g.dev)}
	if ok, _ := backend.GuaranteedLanes(g.dev, ir.SubgroupLanes); ok {
		t.Lanes = ir.SubgroupLanes
	}
	g.ctarget = t
	return t
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
	// The windowed descriptors, and the staged attention's planes.
	prowW, scores, wts, parts backend.Buf
	// A recurrent block's window halves, its convolution weights, and the
	// step's recurrent descriptor.
	convIn, convOut, convW, recDesc backend.Buf
	// Its state's halves and its dt bias, decay rates, skip and convolution bias.
	stateIn, stateOut, recDt, recA, recD, recCB backend.Buf
	recNorm, recMid                             backend.Buf
	router, routerB, selB, routeW               backend.Buf
	sink                                        backend.Buf
	// once is what each once-kind's launch last read this step.
	once [3]backend.Buf
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
	var bufs [12]backend.Buf
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
			case gaPRowW:
				bufs[j] = st.prowW
			case gaScores:
				bufs[j] = st.scores
			case gaWts:
				bufs[j] = st.wts
			case gaParts:
				bufs[j] = st.parts
			case gaConvIn:
				bufs[j] = st.convIn
			case gaConvOut:
				bufs[j] = st.convOut
			case gaConvW:
				bufs[j] = st.convW
			case gaRecDesc:
				bufs[j] = st.recDesc
			case gaStateIn:
				bufs[j] = st.stateIn
			case gaStateOut:
				bufs[j] = st.stateOut
			case gaRecDt:
				bufs[j] = st.recDt
			case gaRecA:
				bufs[j] = st.recA
			case gaRecD:
				bufs[j] = st.recD
			case gaRecCB:
				bufs[j] = st.recCB
			case gaRecNorm:
				bufs[j] = st.recNorm
			case gaRecMid:
				bufs[j] = st.recMid
			case gaRouter:
				bufs[j] = st.router
			case gaRouterB:
				bufs[j] = st.routerB
			case gaSelB:
				bufs[j] = st.selB
			case gaRouteW:
				bufs[j] = st.routeW
			case gaSink:
				bufs[j] = st.sink
			case gaCount:
				bufs[j] = st.bs.n
			case gaTemp:
				bufs[j] = st.bs.tempTab
			case gaRagMap:
				bufs[j] = st.bs.pkv.rag.pMap
			case gaRagBase:
				bufs[j] = st.bs.pkv.rag.pBase[0]
			case gaRagBaseW:
				bufs[j] = st.bs.pkv.rag.pBase[1]
			case gaPreDesc:
				bufs[j] = st.bs.pkv.pre.pDesc[0]
			case gaPreDescW:
				bufs[j] = st.bs.pkv.pre.pDesc[1]
			case gaPLEIn:
				bufs[j] = st.bs.pleIn
			case gaHeadTiles:
				bufs[j] = st.bs.headTiles
			case gaOwn0, gaOwn1, gaOwn2, gaOwn3, gaOwn4, gaOwn5, gaOwn6, gaOwn7:
				bufs[j] = o.Own[a-gaOwn0]
			case gaMgBC, gaMgOff, gaMgCnt, gaMgEnds, gaMgColE, gaMgPerm, gaMgPos, gaMgTsel, gaMgHG, gaMgGA, gaMgGAX:
				m := bs.mg
				bufs[j] = [...]backend.Buf{m.bc, m.off, m.cnt, m.ends, m.colE, m.perm, m.pos, m.tsel, m.hg, m.ga, m.gax}[a-gaMgBC]
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
		if l.once != 0 {
			if st.once[l.once] == bufs[0] {
				continue
			}
			st.once[l.once] = bufs[0]
		}
		if l.readEnds {
			m := bs.mg
			if *lc.err == nil {
				*lc.err = lc.to.s.Sync()
			}
			if *lc.err == nil {
				*lc.err = lc.to.s.Read(m.ends, u32b(m.endsH))
			}
			st.cols = int(m.endsH[m.nExp-1])
			st.once = [3]backend.Buf{}
			continue
		}
		if l.colScale > 0 {
			n := st.cols * l.colScale / max(l.colDiv, 1)
			if n > 0 && *lc.err == nil {
				*lc.err = lc.launch(l.k, (n+l.width-1)/l.width, l.width, bufs[:l.n]...)
			}
			continue
		}
		threads := l.threads
		if rows > 0 && R < rows && !l.whole {
			threads = threads / rows * R
		}
		if l.width != 0 {
			groups := l.groups
			switch {
			case l.pairs:
				n := bs.nUsed * R
				if st.moe {
					n = st.cols
				}
				groups = n
				if l.flat > 0 {
					groups = (n*l.flat + l.width - 1) / l.width
				}
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
			if !ok || o.A.GLU.Bounded() {
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
			o.Why = whyNothing // over rows the grouped matvec quantizes what it reads
			return nil
		}
		// The slots' rows quantized as one flat row, a window never crossing
		// a slot.
		return g.genQuant(o, prog, cap, gaIn0, gaQ8A, gaQ8X, 0, p.ActWin)
	default:
		return nil
	}
	k := g.rowKern(&rb.r)
	if k.k == nil {
		return nil
	}
	l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args)), q8: q8, pairs: true, flat: k.flat}
	copy(l.args[:], rb.args)
	return []genLaunch{l}
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
		if !ok {
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

// genMatVecNodes is a matvec op's kernels: its product formed from primitive
// nodes (compose.Row.MatVec), the weight's bias, a residual add (matvec_res)
// or a gate (act(gate) times the product, matvec_gate) as nodes after it,
// and the kernels compose plans for the group on this target (Plan): its
// schedule and, under a k split, the rewrite into the segments' planes and
// their sum. Callers hold g.mu.
func (g *devTier) genMatVecNodes(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	l := g.curLayer
	if l == nil || o.W < 0 || o.W >= len(l.ms) || *l.rs[o.W] == nil {
		o.Why = whyNoWeight
		return nil
	}
	m := *l.ms[o.W]
	bank := o.Kind == dkExpertMatVec
	// Over rows a bank over heads reads its rows head-major, each head's
	// run of rows its own sheet: the grouped form, a token group a thread.
	heads := bank && rows > 1 && o.Heads > 0
	if bank != (m.experts > 1) || m.pad || bank && !heads && (rows != 1 || g.genRag || len(o.In) < 2) {
		o.Why = whyNoForm
		return nil
	}
	if heads && (m.experts != o.Heads || rows%groupTok != 0 || !g.headTiles(bs, o.Heads, rows)) {
		o.Why = whyNoForm
		return nil
	}
	if rows == 1 && m.group {
		// The in-group split (segments reduced in the workgroup) has no
		// schedule in the former yet.
		o.Why = whyNoForm
		return nil
	}
	// The product, into the output or a split's planes; a bank's slots are
	// the group's rows, each reading its expert's sheet.
	grows := rows
	switch {
	case heads:
		grows = o.Heads * rows
	case bank:
		grows = prog.vals[o.In[1]].Width
	}
	rb := &rowBuild{r: compose.Row{Width: m.rows, Rows: grows}}
	w := compose.MatVecWeight{T: m.q, K: m.k, Rows: m.rows, QS: rb.buf(compose.BufU32, gaWQS), D: -1, SC: -1}
	if !kernels.IsFloat(m.q) {
		w.D = rb.buf(compose.BufU32, gaWD)
	}
	if m.q.Info().PerSuper > 1 {
		w.SC = rb.buf(compose.BufU32, gaWSC)
	}
	a := compose.MatVecOperand{Toks: grows}
	switch {
	case heads:
		w.Experts, w.Sel, w.SelPer = m.experts, rb.buf(compose.BufU32, gaHeadTiles), groupTok
	case bank:
		w.Experts, w.Sel = m.experts, rb.buf(compose.BufU32, gaIn1)
		// The gate and up read the row, the down a slot's own activation.
		if prog.vals[o.In[2]].Layout != layoutPairs {
			a.One, a.Toks = true, 1
		}
	}
	if kernels.IsFloat(m.q) {
		// A float weight reads the float row the program states.
		if bank {
			o.FIn = 3
		}
		if o.FIn == 0 {
			o.Why = whyNoFloat
			return nil
		}
		a.X = rb.buf(compose.BufRow, gaIn0+genArg(o.FIn-1))
	} else {
		a.A, a.AX = rb.buf(compose.BufU32, gaOpA), rb.buf(compose.BufRow, gaOpAX)
	}
	v, err := rb.r.MatVec(w, a)
	if err != nil {
		g.LastErr = err.Error()
		o.Why = whyNoForm
		return nil
	}
	// The epilogue: the weight's bias, a residual, a gate.
	if m.bias != nil {
		v = rb.bin(compose.OpAdd, v, rb.load(compose.BufVec, gaMBias))
	}
	switch o.Kind {
	case dkMatVecRes:
		v = rb.bin(compose.OpAdd, v, rb.load(compose.BufRow, gaIn1))
	case dkMatVecGate:
		act, ok := kernAct(o.Act)
		if !ok {
			o.Why = whyEpilogue
			return nil
		}
		gate := rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpAct, Act: act, A: rb.load(compose.BufRow, gaIn1)})
		v = rb.bin(compose.OpMul, gate, v)
	}
	rb.store(v, gaOut0)
	// The plan: compose's schedule for this shape on this target, the tier
	// stating what it knows -- the step, its pins, the weight's tuned split.
	f := compose.Facts{Target: g.composeTarget(), Step: compose.StepChunk, Tok: bs.tok, RowTile: pinned(g.tiles.rowt),
		Split: g.kb.batch.Split, DecodeSplit: max(m.split, 1), DecodeRowt: max(m.rowt, 1),
		RagTok: g.kb.batch.RagTok, RagRowT: g.kb.batch.RagRowT, RagSplit: g.kb.batch.RagSplit,
		MMAMT: g.tiles.mmaMT, MMANT: g.tiles.mmaNT, NoMMA: g.NoMMA}
	switch {
	case g.genRag:
		f.Step = compose.StepRagged
	case rows == 1:
		f.Step = compose.StepDecode
	}
	ps, err := rb.r.Plan(f)
	if err != nil {
		g.LastErr = err.Error()
		o.Why = whyNoForm
		return nil
	}
	var out []genLaunch
	for _, p := range ps {
		if p.S.Split > 1 {
			need := m.rows * grows * p.S.Split
			if g.genInSub && g.partCap < need || !g.genInSub && !g.sizePart(need) {
				o.Why = whySplit
				return nil
			}
		}
		s := p.S
		k, _ := g.genKern(append([]uint32{encRow}, p.G.Encode(s)...), func() (*ir.Kernel, error) { return p.G.Emit(s) })
		if k == nil || len(p.Bufs) > 8 {
			o.Why = whyNoKernel
			return nil
		}
		gl := genLaunch{k: k, groups: (p.Threads + s.Group - 1) / s.Group, width: s.Group}
		for i, b := range p.Bufs {
			gl.args[i] = gaSplit
			if b >= 0 {
				gl.args[i] = rb.args[b]
			}
		}
		gl.n = uint8(len(p.Bufs))
		out = append(out, gl)
	}
	return out
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
	whyShape
	whyNoForm
	// whyNothing is an op formed with nothing to launch at this step (a
	// grouping a row does not need).
	whyNothing
)

func (w genWhy) String() string {
	return [...]string{"", "no resident weight", "no decode kernel", "a k split", "a bank's sheets",
		"padded rows", "a table split", "an epilogue it cannot compose",
		"no batched form", "a prompt form other than the staged GEMM", "a tail launch",
		"no float row stated", "a history that is not paged, or a row wider than its heads",
		"rows whose head-major layout the program does not state",
		"values whose widths the primitive does not take",
		"an op with no form in the composer's primitives yet", "nothing to launch at this step"}[w]
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
	if bs.pkv == nil || p.NonCausal || len(o.In) <= max(ki, vi) || prog.vals[o.In[ki]].Width != kw ||
		prog.vals[o.In[vi]].Width != vw {
		return nil
	}
	// A row wider than the heads carries a block selection's key after them:
	// turned with them, or (pooled, turned per block when scored) not.
	heads, whole := p.NKVHead, false
	if kw != vw {
		x := selectOfProg(prog)
		if x == nil || kw != vw+p.HeadDim {
			return nil
		}
		if x.Attr.Pooled {
			whole = true
		} else {
			heads++
		}
	}
	P := g.kvPage()
	if turn && (p.NRot <= 0 || p.NoPosEnc || p.RopeSplit || p.NRot%2 != 0 || p.NRot > p.HeadDim) {
		return nil
	}
	// K into the pool as it is where nothing turns it, or where the pairs
	// leave a tail or an unturned key; then the pairs turned into place, a
	// thread a pair.
	var ks []*rowBuild
	if !turn || p.NRot < p.HeadDim || whole {
		kr := &rowBuild{r: compose.Row{Width: kw, Rows: rows}}
		in := kr.buf(compose.BufRow, kArg)
		desc, tab := kr.buf(compose.BufU32, gaPRow), kr.buf(compose.BufU32, gaTab)
		pid, off := kr.r.PagedPos(desc, tab, P)
		kr.r.StoreAt(kr.buf(compose.BufOut, gaKPool), kr.r.Load(in), kr.r.PagedK(pid, off, kw, P))
		ks = append(ks, kr)
	}
	if turn {
		kp := &rowBuild{r: compose.Row{Width: heads * p.NRot / 2, Rows: rows}}
		in, cs := kp.buf(compose.BufRow, kArg), kp.buf(compose.BufVec, gaCS)
		desc, tab := kp.buf(compose.BufU32, gaPRow), kp.buf(compose.BufU32, gaTab)
		kp.r.PagedRotaryPairs(in, cs, desc, tab, kp.buf(compose.BufOut, gaKPool), heads, p.HeadDim, p.NRot, p.RopeNeox, P, kw)
		ks = append(ks, kp)
	}

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
	for _, rb := range append(ks, vr) {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return nil
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
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
	if o.Kind == dkSlice && len(o.In) == 0 && len(o.Out) == 1 && o.V != nil && o.A.HeadDim == 0 {
		// A slice of a bound vector: every row reads the same elements.
		w := prog.vals[o.Out[0]].Width
		rb := &rowBuild{r: compose.Row{Width: w, Rows: rows}}
		r := &rb.r
		x := r.LoadAt(rb.buf(compose.BufVec, gaV), r.UBin(compose.OpAdd, r.Column(), r.UConst(o.A.Offset)))
		r.StoreAt(rb.buf(compose.BufOut, gaOut0), x, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), r.UConst(prog.stride(o.Out[0]))), r.Column()))
		k := g.rowKern(r)
		if k.k == nil {
			return nil
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
		copy(l.args[:], rb.args)
		return []genLaunch{l}
	}
	if len(o.In) == 0 || len(o.Out) == 0 {
		return nil
	}
	if rows > 1 && prog.latent {
		for _, v := range append(append([]devVal(nil), o.In...), o.Out...) {
			if prog.vals[v].Layout == layoutHeads {
				return nil // a head-major input or output the row-major copy does not read
			}
		}
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
	l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
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
	q := g.ropePairs(rows, qw, p.HeadDim, p.NRot, p.RopeNeox, false, gaIn0, gaQOff, gaOut0)
	if q == nil {
		return nil
	}
	return append(q, kv...)
}

// genLayerNorm is a LayerNorm in three launches, each row's statistics in
// the scratch's parts: the parts' sums, their squared deviations from the
// row's mean, and the apply with the op's weight and bias.
func (g *devTier) genLayerNorm(o *devOp, bs *blockScratch, rows, w int, eps float32) []genLaunch {
	parts := bs.parts
	if o.V == nil || parts < 1 || w%parts != 0 {
		return nil
	}
	sb := &rowBuild{r: compose.Row{Width: parts * rows, Rows: 1}}
	sx := sb.buf(compose.BufRow, gaIn0)
	sb.r.Store(sb.buf(compose.BufOut, gaPart), sb.r.SumParts(sx, w, parts))
	vb := &rowBuild{r: compose.Row{Width: parts * rows, Rows: 1}}
	vx, vs := vb.buf(compose.BufRow, gaIn0), vb.buf(compose.BufRow, gaPart)
	vb.r.Store(vb.buf(compose.BufOut, gaPart2), vb.r.VarParts(vx, vs, w, parts))
	ab := &rowBuild{r: compose.Row{Width: w, Rows: rows}}
	ax, wt := ab.buf(compose.BufRow, gaIn0), ab.buf(compose.BufVec, gaV)
	sum, vr := ab.buf(compose.BufRow, gaPart), ab.buf(compose.BufRow, gaPart2)
	bias := -1
	if o.VB != nil {
		bias = ab.buf(compose.BufVec, gaVB)
	}
	ab.r.Store(ab.buf(compose.BufOut, gaOut0), ab.r.LayerNormApplyParts(ax, wt, bias, sum, vr, w, parts, eps))
	var ls []genLaunch
	for _, rb := range []*rowBuild{sb, vb, ab} {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return nil
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
		copy(l.args[:], rb.args)
		ls = append(ls, l)
	}
	return ls
}

// genConv is a recurrent block's causal convolution over a chunk or a
// decode row: the rows against the sequence's window, then the window moved
// to the chunk's end, into the half of the state the step writes. A ragged
// step's runs have no form yet.
func (g *devTier) genConv(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	r := bs.p.Recurrent
	if g.genRag || r.Conv < 2 || len(o.In) != 1 || prog.vals[o.In[0]].Width != r.Chans {
		return nil
	}
	cb := &rowBuild{r: compose.Row{Width: rows * r.Chans, Rows: 1}}
	s, x, w := cb.buf(compose.BufRow, gaConvIn), cb.buf(compose.BufRow, gaIn0), cb.buf(compose.BufRow, gaConvW)
	out := cb.buf(compose.BufOut, gaOut0)
	cb.r.Store(out, cb.r.ConvRows(s, x, w, cb.buf(compose.BufU32, gaRecDesc), r.Conv, r.Chans, rows))
	sb := &rowBuild{r: compose.Row{Width: (r.Conv - 1) * r.Chans, Rows: 1}}
	ss, sx, sout := sb.buf(compose.BufRow, gaConvIn), sb.buf(compose.BufRow, gaIn0), sb.buf(compose.BufOut, gaConvOut)
	v, at := sb.r.ConvShift(ss, sx, sb.buf(compose.BufU32, gaRecDesc), r.Conv, r.Chans, rows)
	sb.r.StoreAt(sout, v, at)
	var ls []genLaunch
	for _, rb := range []*rowBuild{cb, sb} {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return nil
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
		copy(l.args[:], rb.args)
		// The window's shift has its own grid; the rows' share the step's.
		l.whole = rb == sb
		ls = append(ls, l)
	}
	return ls
}

// genScan is a recurrent block's update over the step's rows: the state
// through the rows, each row's output, the next state into the half the step
// writes. Its form follows the block's recurrence: x, B and C as buffers of
// their own; one decay a head written as x*dt with a skip; or the delta
// rule, its q, k and v and gates formed from the convolution's row, at one
// row (a chunk's delta rule runs its inputs as launches of their own, not
// formed yet). A ragged step's runs are not formed yet.
func (g *devTier) genScan(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	r := bs.p.Recurrent
	if g.genRag || r.Conv < 2 && !r.Mamba1 {
		return nil
	}
	sc := compose.Scan{VHeads: r.VHeads, VDim: r.VDim, K: r.KDim, KHeads: r.KHeads, KeyTiled: r.KeyTiled,
		Slot: r.VHeads * r.VDim * r.KDim, Chans: r.Chans, ChanDecay: r.ChanDecay, Bound: float32(r.DecayBound)}
	if r.TailLen > 0 {
		sc.Slot = r.StateLen
	}
	switch {
	case r.Mamba1:
		sc.Separate, sc.KHeads = true, 1
		if len(o.In) != 4 {
			return nil
		}
	case r.SSD:
		if len(o.In) != 2 {
			return nil
		}
	default:
		sc.Delta = true
		if len(o.In) != 2+b2i(r.ChanDecay) {
			return nil
		}
		eps := bs.p.RMSEps
		if r.ChanDecay {
			eps = 1e-6
		}
		sc.Eps, sc.Scale, sc.BARep = float32(eps/float64(r.KDim)), float32(1/math.Sqrt(float64(r.KDim))), r.VHeads/max(r.KHeads, 1)
	}
	lanes := 1
	for l := g.composeTarget().Lanes; lanes*2 <= l && r.KDim%(lanes*2) == 0; {
		lanes *= 2
	}
	if sc.Delta && lanes == 1 {
		return nil
	}
	rb := &rowBuild{r: compose.Row{Width: lanes, Rows: r.VHeads * r.VDim}}
	f := rb.r.ScanBuffers(sc)
	bind := map[int]genArg{f.S: gaStateIn, f.SOut: gaStateOut, f.Out: gaOut0, f.Desc: gaRecDesc, f.Dt: gaRecDt, f.A: gaRecA}
	ins := []int{f.Conv, f.BA}
	switch {
	case sc.Separate:
		ins = []int{f.X, f.B, f.C, f.DtRaw}
	case !sc.Delta:
		ins = []int{f.Conv, f.DtRaw}
		bind[f.ConvBias] = gaRecCB
	case sc.ChanDecay:
		ins = []int{f.Conv, f.Alpha, f.Beta}
	}
	if f.D >= 0 {
		bind[f.D] = gaRecD
	}
	for i, b := range ins {
		bind[b] = gaIn0 + genArg(i)
	}
	rb.args = make([]genArg, len(rb.r.Bufs))
	for b, a := range bind {
		rb.args[b] = a
	}
	rb.r.Scan(sc, f)
	k := g.rowKern(&rb.r)
	if k.k == nil {
		return nil
	}
	l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args)), whole: true}
	copy(l.args[:], rb.args)
	return []genLaunch{l}
}

// genGatedNorm is a recurrence's output normed and gated by z: each value
// head RMS-normed then times the gate (SiLU of z, or its sigmoid), or a
// grouped norm before or after the SiLU gate, or the gate alone.
func (g *devTier) genGatedNorm(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	r := bs.p.Recurrent
	inner := r.VHeads * r.VDim
	if g.genRag || len(o.In) != 2 || prog.vals[o.In[0]].Width != inner || prog.vals[o.In[1]].Width != inner {
		return nil
	}
	eps := float32(bs.p.RMSEps)
	var ls []genLaunch
	add := func(rb *rowBuild) bool {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return false
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args)), flat: k.flat}
		copy(l.args[:], rb.args)
		ls = append(ls, l)
		return true
	}
	gate := func(z, y, out genArg, sigmoid bool) bool {
		rb := &rowBuild{r: compose.Row{Width: rows * inner, Rows: 1}}
		zv, yv := rb.load(compose.BufRow, z), rb.load(compose.BufRow, y)
		v := rb.node(compose.Node{Kind: compose.NBinary, Op: compose.OpActMul, Act: kernels.ActSiLU, A: zv, B: yv})
		if sigmoid {
			v = rb.bin(compose.OpMul, rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpAct, Act: kernels.ActSigmoid, A: zv}), yv)
		}
		rb.store(v, out)
		return add(rb)
	}
	norm := func(in, out genArg, width, groups int) bool {
		lanes := g.composeTarget().Lanes
		if lanes == 0 || width%lanes != 0 {
			lanes = 1
		}
		rb := &rowBuild{r: compose.Row{Width: lanes, Rows: rows * inner / width}}
		x, w, y := rb.buf(compose.BufRow, in), rb.buf(compose.BufRow, gaRecNorm), rb.buf(compose.BufOut, out)
		rb.r.LaneRMSNorm(x, w, y, width, groups, eps)
		k := g.rowKernWith(&rb.r, compose.Facts{RowGroup: max(lanes, 32)})
		if k.k == nil {
			return false
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
		copy(l.args[:], rb.args)
		ls = append(ls, l)
		return true
	}
	ok := true
	switch {
	case r.Mamba1 || r.SSD && r.NoNorm:
		ok = gate(gaIn1, gaIn0, gaOut0, false)
	case r.SSD && r.NormBeforeGate:
		w := inner / max(r.NormGroups, 1)
		ok = norm(gaIn0, gaRecMid, w, inner/w) && gate(gaIn1, gaRecMid, gaOut0, false)
	case r.SSD:
		w := inner / max(r.NormGroups, 1)
		ok = gate(gaIn1, gaIn0, gaRecMid, false) && norm(gaRecMid, gaOut0, w, inner/w)
	default:
		ok = norm(gaIn0, gaRecMid, r.VDim, 1) && gate(gaIn1, gaRecMid, gaOut0, r.ChanDecay || r.SigmoidGate)
	}
	if !ok {
		return nil
	}
	return ls
}

// genNormParts is an RMSNorm in two launches, each row's sum of squares in
// the scratch's parts and then the apply, with the int8 operand quantized
// after where the op makes one.
func (g *devTier) genNormParts(o *devOp, prog *devProgram, bs *blockScratch, rows, w int, eps float32, parts int) []genLaunch {
	if o.V == nil || parts < 1 || w%parts != 0 {
		return nil
	}
	sb := &rowBuild{r: compose.Row{Width: parts * rows, Rows: 1}}
	sx := sb.buf(compose.BufRow, gaIn0)
	sb.r.Store(sb.buf(compose.BufOut, gaPart), sb.r.SquareSumParts(sx, w, parts))
	ab := &rowBuild{r: compose.Row{Width: w, Rows: rows}}
	ax, wt, part := ab.buf(compose.BufRow, gaIn0), ab.buf(compose.BufVec, gaV), ab.buf(compose.BufRow, gaPart)
	ab.r.Store(ab.buf(compose.BufOut, gaOut0), ab.r.NormApplyParts(ax, wt, part, w, parts, eps, false))
	var ls []genLaunch
	for _, rb := range []*rowBuild{sb, ab} {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return nil
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
		copy(l.args[:], rb.args)
		ls = append(ls, l)
	}
	if o.Kind == dkNormQuant {
		q := g.genQuant(o, prog, rows, gaOut0, gaQ8A, gaQ8X, 1, bs.p.ActWin)
		if q == nil {
			return nil
		}
		ls = append(ls, q...)
	}
	return ls
}

// genSSMRow is a selective scan's row op: the convolved x through its
// bias and SiLU, or dt, B or C RMS-normed by its weight.
func (g *devTier) genSSMRow(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	if len(o.In) != 1 || len(o.Out) != 1 {
		return nil
	}
	w := prog.vals[o.In[0]].Width
	var rb *rowBuild
	f := compose.Facts{}
	if o.Kind == dkConvBias {
		rb = &rowBuild{r: compose.Row{Width: w, Rows: rows}}
		x, b := rb.load(compose.BufRow, gaIn0), rb.load(compose.BufVec, gaRecCB)
		rb.store(rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpAct, Act: kernels.ActSiLU, A: rb.bin(compose.OpAdd, x, b)}), gaOut0)
	} else {
		if o.V == nil {
			return nil
		}
		lanes := g.composeTarget().Lanes
		if lanes == 0 || w%lanes != 0 {
			lanes = 1
		}
		rb = &rowBuild{r: compose.Row{Width: lanes, Rows: rows}}
		x, nw, y := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufRow, gaV), rb.buf(compose.BufOut, gaOut0)
		rb.r.LaneRMSNorm(x, nw, y, w, 1, float32(bs.p.RMSEps))
		f.RowGroup = max(lanes, 32)
	}
	k := g.rowKernWith(&rb.r, f)
	if k.k == nil {
		return nil
	}
	l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
	copy(l.args[:], rb.args)
	return []genLaunch{l}
}

// ropePairs is the rotary over each head's first nrot dimensions of rows
// rows of width iw (src), a thread a pair (NORM or NEOX pairs; two tables
// where split), each row landing at its offset (off) in dst; the
// dimensions the pairs do not turn copied there first.
func (g *devTier) ropePairs(rows, iw, hd, nrot int, neox, split bool, src, off, dst genArg) []genLaunch {
	heads := iw / hd
	var ls []genLaunch
	add := func(rb *rowBuild) bool {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return false
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
		copy(l.args[:], rb.args)
		ls = append(ls, l)
		return true
	}
	if nrot < hd {
		cb := &rowBuild{r: compose.Row{Width: iw, Rows: rows}}
		x, o := cb.buf(compose.BufRow, src), cb.buf(compose.BufU32, off)
		cb.r.StoreAt(cb.buf(compose.BufOut, dst), cb.r.Load(x), cb.r.UBin(compose.OpAdd, cb.r.Load(o), cb.r.Column()))
		if !add(cb) {
			return nil
		}
	}
	rb := &rowBuild{r: compose.Row{Width: heads * nrot / 2, Rows: rows}}
	in, cs, o := rb.buf(compose.BufRow, src), rb.buf(compose.BufVec, gaCS), rb.buf(compose.BufU32, off)
	rb.r.RotaryPairs(in, cs, o, rb.buf(compose.BufOut, dst), heads, hd, nrot, neox, split)
	if !add(rb) {
		return nil
	}
	return ls
}

// pinned is a tile the tier states, 0 (the planner's rule) for one.
func pinned(t int) int {
	if t > 1 {
		return t
	}
	return 0
}

// composedComment marks a kernel as the composer's: what a kernel's
// instructions are held against main's by (model.TestDumpEmittedKernels)
// counts only these.
func composedComment(c string) string { return strings.TrimSpace("composed " + c) }

// headTiles makes bs's table of each token group's head over a chunk of
// rows head-major (heads runs of rows columns, groupTok a group). Callers
// hold g.mu; nothing is made inside a submission.
func (g *devTier) headTiles(bs *blockScratch, heads, rows int) bool {
	n := heads * rows / groupTok
	if bs.headTilesN == n && bs.headTiles != nil {
		return true
	}
	if g.genInSub {
		return false
	}
	t := make([]uint32, n)
	for i := range t {
		t[i] = uint32(i * groupTok / rows)
	}
	if bs.headTiles != nil {
		bs.headTiles.Free()
	}
	b, err := g.dev.Alloc(4 * n)
	if err == nil {
		err = b.Write(u32view(t))
	}
	if err != nil {
		return false
	}
	bs.headTiles, bs.headTilesN = b, n
	return true
}
