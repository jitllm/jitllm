package tier

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A part of a model the composer serves in the tier's own submission: a
// whole block, its graph of generic ops the model's front end built from the
// container (nn.LayerPlan.Graph), and the head. Each is asked of the
// composer through its one keyed cache, and what it forms launches on the
// tier's own buffers -- the block's weights and vectors by role, its rotary
// table and its paged history.

type partKind uint8

const (
	partHead partKind = iota + 1
	// partBlock is a whole block, its graph the model's front end's.
	partBlock
	numParts
)

var partNames = [...]string{partHead: "head", partBlock: "block"}

// composedPart is a part of the model as the composer is asked for it: its
// graph of generic ops, the formats of the weights it binds, and the tier's
// buffers it binds them to -- weights and vectors by role, the step's tables
// by the composer's name.
type composedPart struct {
	kind    partKind
	graph   *graph.Graph
	formats map[jlm.Role]jlm.Type
	weights map[jlm.Role]*resident
	vectors map[jlm.Role]backend.Buf
	tables  map[string]backend.Buf
	// history, keys and page are a block's history layout and its step's
	// key bound and page.
	history    jit.History
	keys, page int
}

// partForm is a part's kernels compiled on the device, in launch order, and
// the composer's own buffers they share: the values between its kernels,
// its scratch and its constants. The part's inputs and outputs are the
// caller's.
type partForm struct {
	codes []partCode
	args  [][]partArg
	bufs  []backend.Buf
	// width is each output's width in floats at the form's rows.
	width []int
}

// partCode is one of a form's kernels, its launch shape and its arguments,
// reused from step to step.
type partCode struct {
	k             backend.Kernel
	groups, width int
	bufs          []backend.Buf
}

// partArg is what one kernel parameter binds: one of the form's own
// buffers, the caller's input or output i, or the block's weight region or
// vector of a role, or a step's table.
type partArg struct {
	kind   uint8
	i      int
	role   jlm.Role
	region int
	table  string
}

const (
	partOwn uint8 = iota
	partIn
	partOut
	partWeight
	partVector
	partTable
)

// jlmType is a resident's format as the container names it: a packed type,
// or a float weight the device keeps verbatim, row-major, as the container
// does.
func jlmType(q kernels.Quant) (jlm.Type, bool) {
	switch q {
	case kernels.Float32:
		return jlm.TypeF32, true
	case kernels.Float16:
		return jlm.TypeF16, true
	case kernels.BFloat16:
		return jlm.TypeBF16, true
	}
	for _, t := range quant.PackedTypes {
		if k, ok := kernels.QuantOf(t); ok && k == q {
			return jlm.TypeOf(t)
		}
	}
	return 0, false
}

// askPart asks r for sl at a step of rows: the request's Key, and what it
// forms.
func (g *devTier) askPart(r *jit.Registry, sl *composedPart, rows int) (jit.Key, []jit.Code, error) {
	return r.Form(jit.Request{
		Graph:   sl.graph,
		Formats: sl.formats,
		Step:    jit.Step{Rows: rows, Page: max(sl.page, 64), Keys: max(sl.keys, 1), ActWindow: g.actWin},
		Device:  jit.Device{Kind: jit.DeviceGPU, Slots: g.dev.Slots(), Lanes: g.pickLanes()},
		History: sl.history,
	})
}

// compilePart compiles what the composer formed for sl at a step of rows.
func (g *devTier) compilePart(sl *composedPart, codes []jit.Code, rows int) (*partForm, error) {
	gr := sl.graph
	f := &partForm{}
	io := map[graph.ValueID]partArg{}
	for i, v := range gr.In {
		io[v] = partArg{kind: partIn, i: i}
	}
	for i, v := range gr.Out {
		io[v] = partArg{kind: partOut, i: i}
		f.width = append(f.width, rows*gr.Values[v].Width)
	}
	vals := map[graph.ValueID]int{}
	own := map[jit.Operand]int{}
	// Every buffer of the form is poisoned where the tier's scratch is
	// (WithPoisonScratch): a kernel reading what no kernel wrote shows.
	alloc := func(n int) (int, error) {
		b, err := g.dev.Alloc(max(n, 4))
		if err != nil {
			return 0, err
		}
		g.poisonBuf(b, max(n, 4))
		f.bufs = append(f.bufs, b)
		return len(f.bufs) - 1, nil
	}
	for _, c := range codes {
		if c.GPU == nil {
			f.close()
			return nil, fmt.Errorf("%s has no GPU code", c.Name)
		}
		kern, err := g.dev.Compile(c.GPU)
		if err != nil {
			f.close()
			return nil, fmt.Errorf("compiling %s: %w", c.Name, err)
		}
		f.codes = append(f.codes, partCode{k: kern, groups: c.Groups, width: c.GPU.Group[0]})
		var args []partArg
		for _, bd := range c.Binds {
			o := bd.Operand
			switch {
			case o.Kind == jit.OperandValue:
				if a, ok := io[o.Value]; ok {
					args = append(args, a)
					continue
				}
				i, ok := vals[o.Value]
				if !ok {
					if i, err = alloc(4 * rows * gr.Values[o.Value].Width); err != nil {
						f.close()
						return nil, err
					}
					vals[o.Value] = i
				}
				args = append(args, partArg{kind: partOwn, i: i})
			case o.Kind == jit.OperandWeight && o.Region >= 0 && o.Region < 3:
				args = append(args, partArg{kind: partWeight, role: o.Role, region: o.Region})
			case o.Kind == jit.OperandVector:
				args = append(args, partArg{kind: partVector, role: o.Role})
			case o.Kind == jit.OperandTable && sl.tables[o.Name] != nil:
				args = append(args, partArg{kind: partTable, table: o.Name})
			case bd.Words != nil || o.Kind == jit.OperandScratch:
				i, ok := own[o]
				if !ok {
					if i, err = alloc(int(max(bd.Bytes, 4*int64(len(bd.Words))))); err != nil {
						f.close()
						return nil, err
					}
					if bd.Words != nil {
						if err := f.bufs[i].Write(u32b(bd.Words)); err != nil {
							f.close()
							return nil, err
						}
					}
					own[o] = i
				}
				args = append(args, partArg{kind: partOwn, i: i})
			default:
				f.close()
				return nil, fmt.Errorf("%s binds %+v, which the part's call site does not provide", c.Name, o)
			}
		}
		f.args = append(f.args, args)
		f.codes[len(f.codes)-1].bufs = make([]backend.Buf, len(args))
	}
	return f, nil
}

// launch binds the form's kernels to the caller's inputs and outputs and
// the part's weights, vectors and tables, and launches them in order.
func (f *partForm) launch(lc *launcher, sl *composedPart, in, out []backend.Buf) error {
	for i := range f.codes {
		fc := &f.codes[i]
		for j, a := range f.args[i] {
			var b backend.Buf
			switch a.kind {
			case partOwn:
				b = f.bufs[a.i]
			case partIn:
				b = in[a.i]
			case partOut:
				b = out[a.i]
			case partVector:
				b = sl.vectors[a.role]
			case partTable:
				b = sl.tables[a.table]
			case partWeight:
				if r := sl.weights[a.role]; r != nil {
					b = [3]backend.Buf{r.qs, r.d, r.sc}[a.region]
				}
			}
			if b == nil {
				return fmt.Errorf("the composer's kernel %d reads %v, which the call site does not hold", i, a.role)
			}
			fc.bufs[j] = b
		}
		if err := lc.launch(fc.k, fc.groups, fc.width, fc.bufs...); err != nil {
			return err
		}
	}
	return nil
}

func (f *partForm) close() {
	for _, c := range f.codes {
		c.k.Close()
	}
	for _, b := range f.bufs {
		b.Free()
	}
	f.codes, f.bufs = nil, nil
}

// partsClose frees every served part's form. Callers hold g.mu.
func (g *devTier) partsClose() {
	for k, f := range g.servedForms {
		if f != nil {
			f.close()
		}
		delete(g.servedForms, k)
	}
	clear(g.served)
}

// Parts names the parts of a model the composer serves (Config.Composed).
type Parts uint8

const (
	ComposedHead Parts = 1 << partHead
	// ComposedBlock serves a block whole from its graph (nn.LayerPlan.Graph).
	ComposedBlock Parts = 1 << partBlock
)

// ServedPart is how many steps the composer served the parts of k in, how
// many one-row decode steps it was not asked for, and how many other block
// steps asked it for nothing (Stats.PartsServed, PartsDeclined,
// PartsUnasked).
func (s Stats) ServedPart(k Parts) (served, declined, unasked int) {
	for i := range s.PartsServed {
		if k&(1<<i) != 0 {
			served += s.PartsServed[i]
			declined += s.PartsDeclined[i]
			unasked += s.PartsUnasked[i]
		}
	}
	return served, declined, unasked
}

// servedKey is a served part's shape: the part, its graph, the step's rows,
// a block's key bound, page and history precision, and the head weight's
// format. It names the part so a warm step builds no graph to key it; the
// graph is what the part computes, so the key names it, not a block index.
type servedKey struct {
	kind       partKind
	graph      *graph.Graph
	rows       int
	keys, page int
	half       bool
	t          jlm.Type
}

// servedPart is a served part and its form, with the
// arguments its launch binds, reused every step.
type servedPart struct {
	sl      composedPart
	f       *partForm
	in, out [4]backend.Buf
	n, m    int
}

// errServedCold is a served part asked for inside a capture before it
// formed: the capture is abandoned and the step issued unrecorded.
var errServedCold = errors.New("tier: a served part forms outside a capture")

// servedFor is the served part of key sk, built by build on its first
// step and formed through the composer's one keyed cache. Callers hold g.mu.
func (g *devTier) servedFor(sk servedKey, build func() (composedPart, error)) (*servedPart, error) {
	if sv, ok := g.served[sk]; ok {
		if sv == nil {
			return nil, fmt.Errorf("the composed %s did not form earlier (%s)", partNames[sk.kind], g.LastErr)
		}
		return sv, nil
	}
	if g.capturing {
		return nil, errServedCold
	}
	if g.served == nil {
		g.served, g.servedForms = map[servedKey]*servedPart{}, map[jit.Key]*partForm{}
	}
	g.served[sk] = nil
	if g.composer == nil {
		g.composer = jit.Default()
	}
	r := g.composer
	if g.ComposedFloat {
		if g.composerFloat == nil {
			g.composerFloat = g.composer.WithoutInt8(jit.DeviceGPU)
		}
		r = g.composerFloat
	}
	sl, err := build()
	if err != nil {
		g.LastErr = err.Error()
		return nil, err
	}
	key, codes, err := g.askPart(r, &sl, sk.rows)
	if err != nil {
		g.LastErr = err.Error()
		return nil, fmt.Errorf("the composed %s: %w", partNames[sk.kind], err)
	}
	f, ok := g.servedForms[key]
	if !ok {
		if f, err = g.compilePart(&sl, codes, sk.rows); err != nil {
			g.LastErr = err.Error()
			return nil, fmt.Errorf("the composed %s: %w", partNames[sk.kind], err)
		}
		g.servedForms[key] = f
	}
	sv := &servedPart{sl: sl, f: f, n: len(sl.graph.In), m: len(sl.graph.Out)}
	g.served[sk] = sv
	return sv, nil
}

// run launches the served part on in and out, the block's own buffers.
func (sv *servedPart) run(g *devTier, lc *launcher, in, out []backend.Buf) error {
	copy(sv.in[:sv.n], in)
	copy(sv.out[:sv.m], out)
	if err := sv.f.launch(lc, &sv.sl, sv.in[:sv.n], sv.out[:sv.m]); err != nil {
		return err
	}
	g.PartsServed[sv.sl.kind]++
	return nil
}

// headServes says a step's head is the one the composer is asked for: one
// row, and the model's head graph the device's share of main's head -- a
// norm and the vocabulary's projection, its softcap applied after it by the
// tier and its bias and scale on the host -- not the label pick, not inside
// the persistent block.
func headServes(R int, gr *graph.Graph, norm backend.Buf, layerNorm, pickOnly, persist bool) bool {
	return R == 1 && normThenProjection(gr) && norm != nil && !layerNorm && !pickOnly && !persist
}

// normThenProjection says gr is a norm and a projection of its output, with
// no bias, the projection the graph's output.
func normThenProjection(gr *graph.Graph) bool {
	if gr == nil || len(gr.Ops) != 2 || len(gr.Out) != 1 {
		return false
	}
	n, mv := gr.Ops[0], gr.Ops[1]
	return n.Kind == graph.OpNorm && len(n.Weight) == 1 && len(n.Out) == 1 &&
		mv.Kind == graph.OpMatVec && len(mv.Weight) == 1 && len(mv.In) == 1 && len(mv.Out) == 1 &&
		mv.In[0] == n.Out[0] && mv.Out[0] == gr.Out[0]
}

// servedHead is the model's head graph gr served at one row: its norm bound
// to the tier's norm and its projection to the head's weight w, under the
// roles the graph names (the embedding's where the model ties them).
func (g *devTier) servedHead(gr *graph.Graph, w *resident, norm backend.Buf) (*servedPart, error) {
	t, ok := jlmType(w.t)
	if !ok {
		return nil, fmt.Errorf("the composed head: format %d has no container type", w.t)
	}
	normRole, outRole := gr.Ops[0].Weight[0].Role, gr.Ops[1].Weight[0].Role
	sk := servedKey{kind: partHead, graph: gr, rows: 1, t: t}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		return composedPart{kind: partHead, graph: gr, formats: map[jlm.Role]jlm.Type{outRole: t},
			weights: map[jlm.Role]*resident{outRole: w}, vectors: map[jlm.Role]backend.Buf{normRole: norm}}, nil
	})
	if err != nil {
		return nil, err
	}
	sv.sl.weights[outRole] = w
	sv.sl.vectors[normRole] = norm
	return sv, nil
}

// inPlace says the form may write output out over input in: every kernel
// that binds in launches before the first that binds out.
func (f *partForm) inPlace(in, out int) bool {
	lastRead, firstWrite := -1, len(f.codes)
	for i, args := range f.args {
		for _, a := range args {
			if a.kind == partIn && a.i == in {
				lastRead = i
			}
			if a.kind == partOut && a.i == out {
				firstWrite = min(firstWrite, i)
			}
		}
	}
	return lastRead < firstWrite
}

// roleWeight is the block's matrix its graph binds under role, nil for one
// the tier does not hold.
func (l *layer) roleWeight(r jlm.Role) *resident {
	switch r {
	case jlm.RoleAttnQ:
		return l.wq
	case jlm.RoleAttnK:
		return l.wk
	case jlm.RoleAttnV:
		return l.wv
	case jlm.RoleAttnOut:
		return l.wo
	case jlm.RoleFFNGate:
		return l.gate
	case jlm.RoleFFNUp:
		return l.up
	case jlm.RoleFFNDown:
		return l.down
	}
	return nil
}

// roleVector is the block's vector its graph binds under role, nil for one
// the tier does not hold.
func (l *layer) roleVector(r jlm.Role) backend.Buf {
	switch r {
	case jlm.RoleAttnNorm:
		return l.nAttn
	case jlm.RoleFFNNorm:
		return l.nFFN
	case jlm.RoleAttnQNorm:
		return l.nQ
	case jlm.RoleAttnKNorm:
		return l.nK
	case jlm.RoleAttnQBias:
		return l.mvq.bias
	case jlm.RoleAttnKBias:
		return l.mvk.bias
	case jlm.RoleAttnVBias:
		return l.mvv.bias
	case jlm.RoleAttnOutBias:
		return l.mvo.bias
	case jlm.RoleFFNUpBias:
		return l.mvu.bias
	case jlm.RoleFFNDownBias:
		return l.mvd.bias
	case jlm.RoleAttnSinks:
		return l.sinks
	}
	return nil
}

// blockServes says block li of this step is one the composer serves whole:
// a graph from the front end, a one-row step of the paged decode, none of
// its history at home, no window (not yet held to the tier), outside the
// persistent block and a fault probe.
func (g *devTier) blockServes(l *layer, li int, p *nn.LayerPlan, bs *blockScratch, persist bool) bool {
	pk := bs.pkv
	return l.graph != nil && !persist && len(g.kb.skip) == 0 && pk != nil && !pk.prefill && pk.st == nil &&
		g.kvp != nil && g.kvp.layers[li] != nil && !g.kvp.layers[li].geom.mla && l.kvSrc == li && !l.windowed && p.Window(li) == 0
}

// prewarmComposed forms every placed block's part and the head's at
// placement, as prewarmDecodePlans builds the decode plans: a form met first
// at a token compiles inside it. Callers hold g.mu.
func (g *devTier) prewarmComposed() {
	bs := g.bs
	if g.Composed == 0 || bs == nil || bs.pkv == nil || g.kvp == nil {
		return
	}
	if g.Composed&ComposedBlock != 0 {
		for li, l := range g.layers {
			if l == nil || !g.blockServes(l, li, &bs.p, bs, false) {
				continue
			}
			rcs := bs.cs
			if bs.csSWA != nil && bs.p.LocalRope(li) {
				rcs = bs.csSWA
			}
			if _, err := g.servedBlock(l, li, rcs, g.kvp.layers[li], bs.pkv.pRow, bs.zero, bs.p.MaxSeq, g.kvp.p); err != nil {
				g.LastErr = "the composed block: " + err.Error()
				return
			}
		}
	}
	if g.Composed&ComposedHead != 0 && bs.head != nil && headServes(1, bs.headGraph, bs.hNorm, bs.normVar != nil, false, false) {
		if _, err := g.servedHead(bs.headGraph, bs.head, bs.hNorm); err != nil {
			g.LastErr = "the composed head: " + err.Error()
		}
	}
}

// servedBlock is block li served whole: its graph formed as one unit on the
// block's weights and vectors by role, its rotary table cs and its history
// in the tier's pool layer pl, for a session of maxSeq positions.
//
// The form is the session's: its attention runs as many trips as the row's
// live positions need, a count it reads at run time (jit.Trips), so the
// session's capacity is its key bound and no position compiles a form.
func (g *devTier) servedBlock(l *layer, li int, cs backend.Buf, pl *kvLayerPool, rows, qOff backend.Buf, maxSeq, page int) (*servedPart, error) {
	keys := (maxSeq + scoreGrain - 1) / scoreGrain * scoreGrain
	sk := servedKey{kind: partBlock, graph: l.graph, rows: 1, keys: keys, page: page, half: pl.geom.f16}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		// The request names no block: blocks of one structure and formats
		// share a form, each launched on its own bindings, one after
		// another on the device's queue.
		gr := *l.graph
		gr.Block = 0
		sl := composedPart{kind: partBlock, graph: &gr, formats: map[jlm.Role]jlm.Type{},
			weights: map[jlm.Role]*resident{}, vectors: map[jlm.Role]backend.Buf{},
			tables:  map[string]backend.Buf{"rope.cs": cs, "kv.k": pl.k, "kv.v": pl.v, "pages": pl.tab, "rows": rows, "q.off": qOff},
			history: jit.History{Keys: jit.Pool{ByDim: true}, Values: jit.Pool{Half: pl.geom.f16}},
			keys:    keys, page: page}
		for _, o := range l.graph.Ops {
			for _, w := range o.Weight {
				if w.Rows > 0 {
					r := l.roleWeight(w.Role)
					if r == nil {
						return sl, fmt.Errorf("the composed block %d binds %v, which the tier holds no matrix for", li, w.Role)
					}
					t, ok := jlmType(r.t)
					if !ok {
						return sl, fmt.Errorf("the composed block %d: format %d of %v has no container type", li, r.t, w.Role)
					}
					sl.weights[w.Role], sl.formats[w.Role] = r, t
					continue
				}
				b := l.roleVector(w.Role)
				if b == nil {
					return sl, fmt.Errorf("the composed block %d binds %v, which the tier holds no vector for", li, w.Role)
				}
				// The tier holds every vector as float32, whatever the file.
				sl.vectors[w.Role], sl.formats[w.Role] = b, jlm.TypeF32
			}
		}
		return sl, nil
	})
	if err != nil {
		return nil, err
	}
	// Bound again every step: a page-in or a move replaces a block's buffers.
	for r := range sv.sl.weights {
		sv.sl.weights[r] = l.roleWeight(r)
	}
	for r := range sv.sl.vectors {
		sv.sl.vectors[r] = l.roleVector(r)
	}
	t := sv.sl.tables
	t["rope.cs"], t["kv.k"], t["kv.v"], t["pages"], t["rows"], t["q.off"] = cs, pl.k, pl.v, pl.tab, rows, qOff
	return sv, nil
}
