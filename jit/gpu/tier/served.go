package tier

import (
	"errors"
	"fmt"
	"maps"
	"slices"

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
	// partShadow is the shadow arm's comparison of a block's two outputs
	// (shadow.go).
	partShadow
	// partPrompt is a whole block at a prompt chunk's rows: one sequence's
	// positions, each row its own descriptor.
	partPrompt
	numParts
)

var partNames = [...]string{partHead: "head", partBlock: "block", partShadow: "shadow", partPrompt: "prompt block"}

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
// the plan of what each binds: the caller's inputs and outputs, the part's
// weights, vectors and tables, and the form's own buffers -- the values
// between its kernels, its scratch and its constants -- which each lane
// holds its own of (partInst). One per composer Key on a device, every
// lane's.
type partForm struct {
	codes []partCode
	args  [][]partArg
	// own is each of the form's own buffers: its bytes, and the words a
	// constant is written with.
	own []partBuf
	// width is each output's width in floats at the form's rows.
	width []int
	// key is the composer's Key the form is kept under, and refs the served
	// parts, of every lane, that launch it: the form goes with the last.
	key  jit.Key
	refs int
}

// partBuf is one of a form's own buffers: n bytes, holding words where it
// is a constant.
type partBuf struct {
	n     int
	words []uint32
}

// partCode is one of a form's kernels and its launch shape.
type partCode struct {
	k             backend.Kernel
	groups, width int
}

// partInst is one lane's instance of a form: the form's own buffers and
// what they charge the device's budget.
type partInst struct {
	bufs  []backend.Buf
	bytes uint64
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

// units is the device's multiprocessors where its backend says, else 0.
func units(d backend.Device) int {
	if m, ok := d.(backend.MultiprocessorDevice); ok {
		return m.Multiprocessors()
	}
	return 0
}

// askPart asks r for sl at a step of rows: the request's Key, and what it
// forms.
func (g *devTier) askPart(r *jit.Registry, sl *composedPart, rows int) (jit.Key, []jit.Code, error) {
	lanes, err := g.composedLanes()
	if err != nil {
		return 0, nil, err
	}
	g.Stats.ComposedLanes = lanes
	return r.Form(jit.Request{
		Graph:   sl.graph,
		Formats: sl.formats,
		Step:    jit.Step{Rows: rows, Page: max(sl.page, 64), Keys: max(sl.keys, 1), ActWindow: g.actWin},
		Device: jit.Device{Kind: jit.DeviceGPU, Slots: g.dev.Slots(), Lanes: lanes, Units: units(g.dev),
			ExactDiv: kernels.DivExact(g.dev.API())},
		History: sl.history,
	})
}

// composedLanes is the subgroup the composer forms for: the device's lanes,
// or Config.ComposedLanes where that narrows them to a power of two they
// hold whole.
func (g *devTier) composedLanes() (int, error) {
	own, n := g.pickLanes(), g.Config.ComposedLanes
	switch {
	case n == 0:
		return own, nil
	case n < 0 || n&(n-1) != 0 || n > own:
		return 0, fmt.Errorf("a composed subgroup of %d lanes on a device that guarantees %d: the forms need a power of two it holds whole", n, own)
	}
	return n, nil
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
	buf := func(n int, words []uint32) int {
		f.own = append(f.own, partBuf{n: max(n, 4), words: words})
		return len(f.own) - 1
	}
	// A scratch binding no kernel reads or writes -- a sum a pass carried in
	// registers, still a parameter -- is bound to a word: at a prompt's rows
	// the sub-block dots alone are gigabytes.
	live := map[jit.Operand]bool{}
	for _, c := range codes {
		for _, bd := range c.Binds {
			if bd.Read || bd.Writes {
				live[bd.Operand] = true
			}
		}
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
					i = buf(4*rows*gr.Values[o.Value].Width, nil)
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
					n := int(max(bd.Bytes, 4*int64(len(bd.Words))))
					if bd.Words == nil && !live[o] {
						n = 4
					}
					i = buf(n, bd.Words)
					own[o] = i
				}
				args = append(args, partArg{kind: partOwn, i: i})
			default:
				f.close()
				return nil, fmt.Errorf("%s binds %+v, which the part's call site does not provide", c.Name, o)
			}
		}
		f.args = append(f.args, args)
	}
	return f, nil
}

// instance is a lane's instance of form f: its own buffers, poisoned where
// the tier's scratch is (WithPoisonScratch) so that a kernel reading what no
// kernel wrote shows, its constants written, and charged to the device's
// budget like a block's weights -- one the budget cannot hold is refused.
// Callers hold g.mu.
func (g *devTier) instance(f *partForm) (*partInst, error) {
	var n uint64
	for _, b := range f.own {
		n += uint64(b.n)
	}
	if !g.room(n) {
		return nil, fmt.Errorf("its buffers need %d bytes, which the device's budget does not hold", n)
	}
	in := &partInst{}
	for _, b := range f.own {
		d, err := g.dev.Alloc(b.n)
		if err != nil {
			in.free()
			return nil, err
		}
		g.poisonBuf(d, b.n)
		in.bufs = append(in.bufs, d)
		if b.words != nil {
			if err := d.Write(u32b(b.words)); err != nil {
				in.free()
				return nil, err
			}
		}
	}
	in.bytes = n
	g.charge(n)
	return in, nil
}

// launchArgs is an argument list for each of f's kernels, bound again every
// step.
func (f *partForm) launchArgs() [][]backend.Buf {
	out := make([][]backend.Buf, len(f.args))
	for i, args := range f.args {
		out[i] = make([]backend.Buf, len(args))
	}
	return out
}

// free frees an instance's buffers; what they charged is the caller's to
// give back.
func (in *partInst) free() {
	for _, b := range in.bufs {
		b.Free()
	}
	in.bufs = nil
}

// launch binds the form's kernels to the caller's inputs and outputs, the
// lane's instance inst and the part's weights, vectors and tables, and
// launches them in order.
func (sv *servedPart) launch(lc *launcher, inst *partInst, in, out []backend.Buf) error {
	f, sl := sv.f, &sv.sl
	for i := range f.codes {
		fc, bufs := &f.codes[i], sv.args[i]
		for j, a := range f.args[i] {
			var b backend.Buf
			switch a.kind {
			case partOwn:
				b = inst.bufs[a.i]
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
			bufs[j] = b
		}
		if err := lc.issue(fc.k, fc.groups, fc.width, bufs...); err != nil {
			return err
		}
	}
	return nil
}

func (f *partForm) close() {
	for _, c := range f.codes {
		c.k.Close()
	}
	f.codes = nil
}

// clone is a copy of the part whose bindings, bound again every step, are
// its own.
func (sl composedPart) clone() composedPart {
	sl.weights, sl.vectors, sl.tables = maps.Clone(sl.weights), maps.Clone(sl.vectors), maps.Clone(sl.tables)
	return sl
}

// dropPart frees a lane's served part sv: its instance, whose bytes go back
// to the budget, and its form where no part launches it any more, with every
// lane's prompt instances of it. Callers hold g.mu, with no submission
// running.
func (g *devTier) dropPart(sv *servedPart) {
	if sv == nil {
		return
	}
	if in := sv.inst; in != nil {
		in.free()
		g.refund(in.bytes)
		in.bytes = 0
	}
	if sv.f.refs--; sv.f.refs == 0 {
		g.eachLane(func(l *lane) {
			for _, bs := range l.bbs {
				g.dropPromptInst(bs, sv.f)
			}
		})
		delete(g.servedForms, sv.f.key)
		sv.f.close()
	}
}

// promptInst is the lane's instance of prompt form f at bs's rows, made
// where the scratch holds none, nil where the budget does not hold it.
// Callers hold g.mu.
func (g *devTier) promptInst(bs *blockScratch, f *partForm) *partInst {
	if in := bs.prompt[f]; in != nil {
		return in
	}
	in, err := g.instance(f)
	if err != nil {
		g.LastErr = "the composed prompt block: " + err.Error()
		return nil
	}
	if bs.prompt == nil {
		bs.prompt = map[*partForm]*partInst{}
	}
	bs.prompt[f] = in
	return in
}

// dropPromptInst frees bs's instance of form f and gives its bytes back.
// Callers hold g.mu, with no submission running.
func (g *devTier) dropPromptInst(bs *blockScratch, f *partForm) {
	if in := bs.prompt[f]; in != nil {
		in.free()
		g.refund(in.bytes)
		delete(bs.prompt, f)
	}
}

// partsClose frees every lane's served parts and every form. Callers hold
// g.mu.
func (g *devTier) partsClose() {
	g.eachLane(func(l *lane) { g.dropLaneParts(l) })
	for k, f := range g.servedForms {
		f.close()
		delete(g.servedForms, k)
	}
}

// dropLaneParts frees lane l's served parts. Callers hold g.mu, with no
// submission running.
func (g *devTier) dropLaneParts(l *lane) {
	for sk, sv := range l.served {
		delete(l.served, sk)
		g.dropPart(sv)
	}
	clear(l.formErr)
}

// dropServed forgets the served parts of graph gr in every lane, a failed
// one's included, and frees each form no part launches any more: a block's
// parts go with the block, and a block placed again forms them again.
// Callers hold g.mu and drop the graph, so no recording launches what it
// frees.
func (g *devTier) dropServed(gr *graph.Graph) {
	g.eachLane(func(l *lane) {
		for sk, sv := range l.served {
			if sk.graph == gr {
				delete(l.served, sk)
				delete(l.formErr, sk)
				g.dropPart(sv)
			}
		}
	})
}

// cloneParts gives lane nl an instance of every part lane0 serves, on the
// device's forms: a clone's first step forms nothing. Callers hold g.mu.
func (g *devTier) cloneParts(nl *lane) error { return g.spreadParts(g.lane0, nl, nil) }

// spreadParts gives lane nl an instance of each part lane src serves that it
// does not, of graph gr where gr is not nil, on the device's forms: a prompt
// part's instance is its scratch's (promptInst). Callers hold g.mu.
func (g *devTier) spreadParts(src, nl *lane, gr *graph.Graph) error {
	for sk, sv := range src.served {
		if sv == nil || gr != nil && sk.graph != gr || nl.served[sk] != nil {
			continue
		}
		var in *partInst
		if sk.kind != partPrompt {
			var err error
			if in, err = g.instance(sv.f); err != nil {
				return fmt.Errorf("the composed %s: %w", partNames[sk.kind], err)
			}
		}
		sv.f.refs++
		if nl.served == nil {
			nl.served = map[servedKey]*servedPart{}
		}
		nl.served[sk] = &servedPart{sl: sv.sl.clone(), f: sv.f, inst: in, args: sv.f.launchArgs(), n: sv.n, m: sv.m}
	}
	return nil
}

// Parts names the parts of a model the composer serves (Config.Composed).
type Parts uint8

const (
	ComposedHead Parts = 1 << partHead
	// ComposedBlock serves a block whole from its graph (nn.LayerPlan.Graph).
	ComposedBlock Parts = 1 << partBlock
	// PartShadow selects the shadow arm's comparisons in Stats.ServedPart.
	// Config.Composed takes no such bit: Config.ComposedShadow is its own
	// switch.
	PartShadow Parts = 1 << partShadow
	// ComposedPrompt serves a prompt chunk's blocks whole at the chunk's
	// rows: a form a width, as the tier keeps a scratch a width.
	ComposedPrompt Parts = 1 << partPrompt
)

// partDecline is why a decode step's or a prompt chunk's composed part is
// not asked for: what the serving conditions leave to the tier's own
// kernels, counted by reason (Stats.BlockDeclines, HeadDeclines). A head
// over a step of several rows -- a chunk's folded head, a ragged step's,
// speculative rows' -- is the tier's, declined by its path's name.
type partDecline uint8

const (
	partServes partDecline = iota
	partNoGraph
	partPersist
	partSkip
	partNoPool
	partPrefill
	partStream
	partLatent
	partShared
	partWindow
	partHeadShape
	partHeadNorm
	partPick
	partRagged
	partRowLogits
	partCaller
	partNoRoom
	partFolded
	partHeadRows
	partPromptHead
	partShadowed
	numPartDeclines
)

var partDeclineNames = [numPartDeclines]string{
	partServes:     "served",
	partNoGraph:    "no graph from the front end",
	partPersist:    "inside the persistent block",
	partSkip:       "a fault probe skips blocks",
	partNoPool:     "no paged history on the device",
	partPrefill:    "the history is mid-prefill",
	partStream:     "the history streams",
	partLatent:     "a latent attention's history",
	partShared:     "the history of another block",
	partWindow:     "a graph whose window is not the plan's",
	partHeadShape:  "a head graph other than a norm and its projection",
	partHeadNorm:   "no norm buffer, or a LayerNorm head",
	partPick:       "a head that picks labels only",
	partRagged:     "a ragged step's rows",
	partRowLogits:  "every row's logits (speculative rows)",
	partCaller:     "the caller keeps it on the tier's kernels",
	partNoRoom:     "its buffers do not fit the device's budget",
	partFolded:     "a prompt chunk's folded head",
	partHeadRows:   "a head over several rows",
	partPromptHead: "a prompt's one-row head",
	partShadowed:   "a prompt outside the shadow arm",
}

// DeclineName is the reason a decline count is kept under, by its index in
// Stats.BlockDeclines or HeadDeclines.
func DeclineName(i int) string {
	if i < 0 || i >= int(numPartDeclines) {
		return fmt.Sprintf("decline %d", i)
	}
	return partDeclineNames[i]
}

// ServedPart is how many steps the composer served the parts of k in, how
// many steps where they serve it was not asked for, and how many other block
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

// ReplayedPart is how many steps a recording replayed the parts of k in
// (Stats.PartsReplayed).
func (s Stats) ReplayedPart(k Parts) int {
	n := 0
	for i := range s.PartsReplayed {
		if k&(1<<i) != 0 {
			n += s.PartsReplayed[i]
		}
	}
	return n
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

// servedPart is a served part and its form, with the arguments its launch
// binds, reused every step: its own instance, or for a prompt part none,
// its instance its scratch's (promptInst).
type servedPart struct {
	sl      composedPart
	f       *partForm
	inst    *partInst
	args    [][]backend.Buf
	in, out [4]backend.Buf
	n, m    int
}

// errServedCold is a served part asked for inside a capture before it
// formed: the capture is abandoned and the step issued unrecorded.
var errServedCold = errors.New("tier: a served part forms outside a capture")

// servedFor is the lane's served part of key sk, built by build on its
// first step and formed through the composer's one keyed cache, its form the
// device's and its instance the lane's. A submission beside other sessions'
// takes the device's lock for the build (subLock); callers otherwise hold
// g.mu.
func (g *devTier) servedFor(sk servedKey, build func() (composedPart, error)) (*servedPart, error) {
	if sv, ok := g.served[sk]; ok {
		if sv == nil {
			return nil, fmt.Errorf("the composed %s did not form earlier: %w", partNames[sk.kind], g.formErr[sk])
		}
		return sv, nil
	}
	if g.capturing {
		return nil, errServedCold
	}
	if g.served == nil {
		g.served = map[servedKey]*servedPart{}
	}
	g.served[sk] = nil
	fail := func(err error) error {
		if g.formErr == nil {
			g.formErr = map[servedKey]error{}
		}
		g.formErr[sk] = err
		g.LastErr = err.Error()
		return err
	}
	defer g.subUnlock(g.subLock())
	if g.servedForms == nil {
		g.servedForms = map[jit.Key]*partForm{}
	}
	if !g.prewarming {
		g.FormedAtStep++
	}
	if g.composer == nil {
		g.composer = jit.Default()
		// Poisoned scratch poisons the forms' workgroup arrays too: a kernel
		// reading one it never wrote changes the logits.
		if g.kb.poisonScratch {
			g.composer.PoisonWorkgroupMemory()
		}
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
		return nil, fail(err)
	}
	key, codes, err := g.askPart(r, &sl, sk.rows)
	if err != nil {
		return nil, fail(fmt.Errorf("the composed %s: %w", partNames[sk.kind], err))
	}
	f, ok := g.servedForms[key]
	if !ok {
		if f, err = g.compilePart(&sl, codes, sk.rows); err != nil {
			return nil, fail(fmt.Errorf("the composed %s: %w", partNames[sk.kind], err))
		}
		f.key = key
		g.servedForms[key] = f
	}
	var in *partInst
	if sk.kind != partPrompt {
		if in, err = g.instance(f); err != nil {
			if f.refs == 0 {
				delete(g.servedForms, key)
				f.close()
			}
			return nil, fail(fmt.Errorf("the composed %s: %w", partNames[sk.kind], err))
		}
	}
	f.refs++
	sv := &servedPart{sl: sl, f: f, inst: in, args: f.launchArgs(), n: len(sl.graph.In), m: len(sl.graph.Out)}
	g.served[sk] = sv
	return sv, nil
}

// run launches the served part on in and out, the block's own buffers, in
// instance inst.
func (sv *servedPart) run(g *devTier, lc *launcher, inst *partInst, in, out []backend.Buf) error {
	return sv.runAs(g, lc, inst, sv.sl.kind, in, out)
}

// runAs is run, counted as a step of part kind: a prompt's one-row call
// launches a decode step's form and is the prompt's.
func (sv *servedPart) runAs(g *devTier, lc *launcher, inst *partInst, kind partKind, in, out []backend.Buf) error {
	copy(sv.in[:sv.n], in)
	copy(sv.out[:sv.m], out)
	if err := sv.launch(lc, inst, sv.in[:sv.n], sv.out[:sv.m]); err != nil {
		return err
	}
	g.PartsServed[kind]++
	return nil
}

// headServes is why a one-row step's head is not the one the composer is
// asked for, serves where it is: the model's head graph the device's share
// of main's head -- a norm and the vocabulary's projection, its softcap
// applied after it by the tier and its bias and scale on the host -- not
// the label pick, not inside the persistent block.
func headServes(gr *graph.Graph, norm backend.Buf, layerNorm, pickOnly, persist bool) partDecline {
	switch {
	case !normThenProjection(gr):
		return partHeadShape
	case norm == nil || layerNorm:
		return partHeadNorm
	case pickOnly:
		return partPick
	case persist:
		return partPersist
	}
	return partServes
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

// attendWindow is the window gr's attention states, and whether it is an
// aligned chunk.
func attendWindow(gr *graph.Graph) (int, bool) {
	if gr != nil {
		for _, o := range gr.Ops {
			if o.Kind == graph.OpAttend {
				return o.Attr.Window, o.Attr.Chunked
			}
		}
	}
	return 0, false
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
	case jlm.RoleAttnNormBias:
		return l.nAttnB
	case jlm.RoleFFNNormBias:
		return l.nFFNB
	case jlm.RolePostAttnNorm:
		return l.nPostAttn
	case jlm.RolePostFFNNorm:
		return l.nPostFFN
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
	case jlm.RoleAttnSubNorm:
		return l.nSubAttn
	case jlm.RoleFFNSubNorm:
		return l.nSubFFN
	case jlm.RoleXIELU:
		return l.xielu
	}
	return nil
}

// blockServes is why block li of this step is not one the composer serves
// whole, serves where it is: a graph from the front end, a block the caller
// leaves to it, the paged decode -- or a prompt chunk's, which the tier's
// own kernels would run through its paged prefill attention -- none of its
// history at home, the window the plan holds the block to (windowHeld),
// outside the persistent block and a fault probe.
func (g *devTier) blockServes(l *layer, li int, p *nn.LayerPlan, bs *blockScratch, persist, prompt bool) partDecline {
	pk := bs.pkv
	switch {
	case l.graph == nil:
		return partNoGraph
	case slices.Contains(g.TierBlocks, li):
		return partCaller
	case persist:
		return partPersist
	case len(g.kb.skip) != 0:
		return partSkip
	case pk == nil || g.kvp == nil || g.kvp.layers[li] == nil:
		return partNoPool
	case pk.prefill && !prompt:
		return partPrefill
	case pk.st != nil:
		return partStream
	case g.kvp.layers[li].geom.mla:
		return partLatent
	case l.kvSrc != li:
		return partShared
	case !l.windowHeld(li, p):
		return partWindow
	}
	return partServes
}

// windowHeld says the window block l's graph states is the plan's for block
// li, whose descriptors the tier stages (pagedStage), or one that cuts
// nothing in the plan's session, where the full descriptors start where the
// window would.
func (l *layer) windowHeld(li int, p *nn.LayerPlan) bool {
	if w := p.Window(li); w > 0 {
		return l.win == w && l.chunked == p.SWAChunked
	}
	return l.win == 0 || l.win >= p.MaxSeq
}

// servedStep is block li's served part at this step of R rows and the
// lane's instance it launches in, nil where the budget does not hold the
// instance of a prompt's form.
func (g *devTier) servedStep(bs *blockScratch, l *layer, li int, p *nn.LayerPlan, R int) (*servedPart, *partInst, error) {
	rcs := bs.cs
	if bs.csSWA != nil && p.LocalRope(li) {
		rcs = bs.csSWA
	}
	qOff := bs.zero
	if R > 1 {
		qOff = bs.roff
	}
	sv, err := g.servedBlock(l, li, rcs, g.kvp.layers[li], bs.pkv.pRow, bs.pkv.windowDesc(p, li), qOff, p.MaxSeq, g.kvp.p, R)
	if err != nil {
		return nil, nil, err
	}
	if !sv.f.inPlace(0, 0) {
		return nil, nil, fmt.Errorf("tier: the composed block %d reads x after writing its output over it", li)
	}
	if R == 1 {
		return sv, sv.inst, nil
	}
	return sv, g.promptInst(bs, sv.f), nil
}

// promptServes is why block li of a prompt's call is not one the composer
// serves whole, serves where it is: outside the shadow arm, which compares a
// decode step's forms alone; the rows one sequence's (a ragged step's rows,
// and speculative rows wanting every row's logits, stay on the tier's
// kernels, by name); and what blockServes asks.
func (g *devTier) promptServes(l *layer, li int, p *nn.LayerPlan, bs *blockScratch, persist, ragged, rowLogits bool) partDecline {
	switch {
	case g.shadowing():
		return partShadowed
	case ragged:
		return partRagged
	case rowLogits:
		return partRowLogits
	}
	return g.blockServes(l, li, p, bs, persist, true)
}

// prewarmComposed forms every placed block's part and the head's at
// placement, as prewarmDecodePlans builds the decode plans: a form met first
// at a token compiles inside it. Callers hold g.mu.
func (g *devTier) prewarmComposed() {
	for li := range g.layers {
		g.prewarmComposedBlock(li)
	}
	g.prewarmComposedHead()
}

// prewarmComposedBlock forms block li's part where the composer serves it:
// at the placement, and at every placement of the block after it (a move, a
// seam grown back), which a step would otherwise meet first. A form that
// does not build is named in LastErr and leaves the other blocks to form.
// Callers hold g.mu.
func (g *devTier) prewarmComposedBlock(li int) {
	bs, l := g.bs, g.layers[li]
	if g.NoPromptPrewarm || g.Composed&(ComposedBlock|ComposedPrompt) == 0 || bs == nil || bs.pkv == nil ||
		g.kvp == nil || l == nil {
		return
	}
	g.prewarming = true
	defer func() { g.prewarming = false }()
	if g.Composed&ComposedBlock != 0 && g.blockServes(l, li, &bs.p, bs, false, false) == partServes {
		rcs := bs.cs
		if bs.csSWA != nil && bs.p.LocalRope(li) {
			rcs = bs.csSWA
		}
		if _, err := g.servedBlock(l, li, rcs, g.kvp.layers[li], bs.pkv.pRow, bs.pkv.windowDesc(&bs.p, li), bs.zero, bs.p.MaxSeq,
			g.kvp.p, 1); err != nil {
			g.LastErr = "the composed block: " + err.Error()
			return
		}
	}
	// Every prompt width the lane holds a scratch for: its first chunk
	// forms and allocates nothing.
	if g.Composed&ComposedPrompt != 0 {
		for _, bb := range g.bbs {
			g.prewarmPromptBlock(bb, li)
		}
	}
	// The shadow arm's comparison of the block's two outputs.
	if g.shadowing() {
		if _, err := g.servedShadow(l, bs.p.NEmbd); err != nil {
			g.LastErr = "the composed shadow: " + err.Error()
			return
		}
	}
	// Every other lane's instance too: a session's first step after a move
	// forms nothing.
	g.eachLane(func(o *lane) {
		if o == g.lane {
			return
		}
		if err := g.spreadParts(g.lane, o, l.graph); err != nil {
			g.LastErr = "a lane's composed block: " + err.Error()
		}
	})
}

// prewarmPromptBlock forms block li's part at the rows of bb, a prompt
// width's scratch, where the composer serves it, and the scratch's instance
// of its form. Callers hold g.mu and say whether this is the placement's
// (g.prewarming).
func (g *devTier) prewarmPromptBlock(bb *blockScratch, li int) {
	l := g.layers[li]
	if l == nil || bb == nil || bb.rows < 2 || bb.pkv == nil || g.kvp == nil ||
		g.promptServes(l, li, &bb.p, bb, false, false, false) != partServes {
		return
	}
	rcs := bb.cs
	if bb.csSWA != nil && bb.p.LocalRope(li) {
		rcs = bb.csSWA
	}
	sv, err := g.servedBlock(l, li, rcs, g.kvp.layers[li], bb.pkv.pRow, bb.pkv.windowDesc(&bb.p, li), bb.roff, bb.p.MaxSeq,
		g.kvp.p, bb.rows)
	if err != nil {
		g.LastErr = "the composed prompt block: " + err.Error()
		return
	}
	g.promptInst(bb, sv.f)
}

// promptForms forms every placed block's part at bb's rows where the
// composer serves it, with bb's instance of its form: a scratch built for a
// chunk (keepExact) allocates here, outside the submission, and a block
// placed after the placement's width loop (a seam grown back) forms here, as
// the walk compiles the tier's own kernels for it. At a placement
// (reserveScratch) that is the placement's forming. Callers hold g.mu.
func (g *devTier) promptForms(bb *blockScratch) {
	if g.Composed&ComposedPrompt == 0 || bb == nil || bb.rows < 2 {
		return
	}
	was := g.prewarming
	g.prewarming = was || g.reserving
	defer func() { g.prewarming = was }()
	for li := range g.layers {
		g.prewarmPromptBlock(bb, li)
	}
}

// keyBound is a block form's key bound for a session of maxSeq positions.
func keyBound(maxSeq int) int { return (maxSeq + scoreGrain - 1) / scoreGrain * scoreGrain }

// prewarmComposedHead forms the head's part where the composer serves it,
// at placement and again at PrepHead, which a model calls after placing its
// blocks: a head met first at a token formed inside it. Callers hold g.mu.
func (g *devTier) prewarmComposedHead() {
	bs := g.bs
	if g.NoPromptPrewarm || g.Composed&ComposedHead == 0 || bs == nil || bs.head == nil ||
		headServes(bs.headGraph, bs.hNorm, bs.normVar != nil, false, false) != partServes {
		return
	}
	g.prewarming = true
	defer func() { g.prewarming = false }()
	if _, err := g.servedHead(bs.headGraph, bs.head, bs.hNorm); err != nil {
		g.LastErr = "the composed head: " + err.Error()
	}
}

// servedBlock is block li served whole at a step of R rows: its graph formed
// as one unit on the block's weights and vectors by role, its rotary table
// cs and its history in the tier's pool layer pl, through the rows'
// descriptors, a windowed attention through rowsW
// (pagedScratch.windowDesc), for a session of maxSeq positions.
//
// The form is the session's: its attention runs as many trips as each row's
// live positions need, a count it reads at run time (jit.Trips), so the
// session's capacity is its key bound and no position compiles a form. A
// prompt chunk's form is its width's, as the tier's kernels are.
func (g *devTier) servedBlock(l *layer, li int, cs backend.Buf, pl *kvLayerPool, rows, rowsW, qOff backend.Buf, maxSeq, page, R int) (*servedPart, error) {
	keys := keyBound(maxSeq)
	kind := partBlock
	if R > 1 {
		kind = partPrompt
	}
	sk := servedKey{kind: kind, graph: l.graph, rows: R, keys: keys, page: page, half: pl.geom.f16}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		// The request names no block: blocks of one structure and formats
		// share a form, each launched on its own bindings, one after
		// another on the device's queue.
		gr := *l.graph
		gr.Block = 0
		sl := composedPart{kind: kind, graph: &gr, formats: map[jlm.Role]jlm.Type{},
			weights: map[jlm.Role]*resident{}, vectors: map[jlm.Role]backend.Buf{},
			tables: map[string]backend.Buf{"rope.cs": cs, "kv.k": pl.k, "kv.v": pl.v, "pages": pl.tab, "rows": rows,
				"rows.win": rowsW, "q.off": qOff},
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
	t["rope.cs"], t["kv.k"], t["kv.v"], t["pages"], t["rows"], t["rows.win"], t["q.off"] = cs, pl.k, pl.v, pl.tab, rows, rowsW, qOff
	return sv, nil
}
