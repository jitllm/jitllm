package model

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/cpu"
)

// The CPU is a device like a card: a block placed on it runs what the
// composer forms for it from the block's graph (Model.Composing), through
// the one keyed cache, on the State's own buffers -- the residual, the
// block's weights and vectors by role, its rotary table and its history in
// the host's KV pages (kvArena) -- each kernel mapped once for the model and
// called through the trampoline, the calls of a kernel over heads spread
// over the State's pool. Every block step on the CPU is counted served or
// declined by name (CPUBlockSteps), as the tier counts a card's.

// CPUBlockSteps is how many block steps on the CPU the composer served, and
// how many it declined, by reason (jit.DeclineName). With the composed CPU
// off (WithComposedCPU) nothing is counted.
func (s *State) CPUBlockSteps() (served int, declined [jit.NumDeclines]int) {
	return s.cpu.served, s.cpu.declined
}

// cpuDeclined counts a block step on the CPU that main's kernels run, where
// the composed CPU is on: a step of several rows, which the composer does
// not form for the CPU yet.
func (s *State) cpuDeclined(why jit.Decline) {
	if s.m.opt.composedCPU {
		s.cpu.declined[why]++
	}
}

// CPUFormErr is the first error the composer gave forming a block of this
// State's model for the CPU, nil for none.
func (s *State) CPUFormErr() error { return s.cpu.formErr }

// The argument sources of a composed CPU kernel's call words.
const (
	cpuArgValue uint8 = iota
	cpuArgX
	cpuArgOwn
	cpuArgWeight
	cpuArgVector
	cpuArgRope
	cpuArgKeys
	cpuArgValues
	cpuArgPages
	cpuArgRows
	cpuArgZero
	cpuArgCoord
	cpuArgWord
)

// cpuArg is one call word: its byte field in cpu.Args and where its value
// comes from, with the elements it is offset by.
type cpuArg struct {
	field  uintptr
	src    uint8
	region uint8
	coord  uint8
	role   jlm.Role
	i      int
	off    int
	elem   int
	val    uint64
}

// cpuCode is one kernel of a form, mapped, and its call words; reads and
// writes say it reads the block's input and writes its output.
type cpuCode struct {
	reads, writes bool
	code          *cpu.Code
	args          []cpuArg
	calls         []int
	n             int
	// serial is a kernel over calls that write a buffer every call shares.
	serial bool
}

// cpuBlockForm is a block's kernels for the CPU at one composer Key, mapped
// once for the model, and the words of each of its own buffers.
type cpuBlockForm struct {
	codes []cpuCode
	own   []int
	// consts is an own buffer's constant words, by its index.
	consts map[int][]uint32
}

func (f *cpuBlockForm) close() {
	for i := range f.codes {
		f.codes[i].code.Close()
	}
}

// cpuForms is a model's composed CPU forms by Key, and the registry they
// are asked of.
type cpuForms struct {
	mu  sync.Mutex
	reg *jit.Registry
	m   map[jit.Key]*cpuBlockForm
	err map[jit.Key]error
}

func (c *cpuForms) close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, f := range c.m {
		f.close()
	}
	c.m = nil
}

// cpuState is a State's composed CPU blocks: each block's form and its own
// buffers, the history tables, and the census.
type cpuState struct {
	blocks   []*cpuBlock
	composed []int
	served   int
	declined [jit.NumDeclines]int
	formErr  error
	pages    []uint32
	rows     [4]uint32
	zero     [4]uint32
	// own is each form's own buffers, by form: blocks of one form run one
	// after another, so they share them.
	own map[*cpuBlockForm][][]uint32
	// args is a call's words for each pool worker.
	args  []cpu.Args
	run   cpuRegion
	bound cpuBound
	// fault is a violation a gate sets: the bindings rewritten so a kernel
	// reads the wrong history page or the wrong first row.
	fault cpuFault
}

// cpuFault is a gate's violation of the composed CPU path's bindings.
type cpuFault uint8

const (
	cpuFaultNone cpuFault = iota
	// cpuFaultPage points the history's first page at the arena's last
	// slot once the first page is full: its positions read another page's
	// (or nothing). A table permuted from the first step only renames the
	// pages, and keys carry their rotation, so a swap of written pages
	// only reorders them.
	cpuFaultPage
	// cpuFaultRow moves the first key row a step attends by one.
	cpuFaultRow
)

// cpuBlock is block li's form on the CPU, its graph the one it was asked
// for.
type cpuBlock struct {
	graph *graph.Graph
	f     *cpuBlockForm
	why   jit.Decline
}

// cpuRegion is the kernel a pool region calls, its calls the region's range.
type cpuRegion struct {
	s  *State
	b  *cpuBound
	c  *cpuCode
	fn func(worker, lo, hi int)
}

// cpuBound is a block's step as its words read it.
type cpuBound struct {
	li  int
	l   *layer
	own [][]uint32
	cs  []float32
}

// composedCPUDevice is the CPU as the composer forms for it.
func composedCPUDevice(workers int) jit.Device {
	return jit.Device{Kind: jit.DeviceCPU, Slots: max(workers, 1), Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
}

// cpuStep runs block li on the CPU the composer's way, or says why not. The
// residual is s.x; the rotary table cs.
func (s *State) cpuStep(li int, l *layer, cs []float32) (jit.Decline, error) {
	cb, why := s.cpuBlockOf(li)
	if why != jit.DeclineServes {
		return why, nil
	}
	pg := &s.kv.layers[li]
	pos := s.pos
	if pg.p > 0 {
		if why := s.cpuHistory(li, pos); why != jit.DeclineServes {
			return why, nil
		}
	}
	b := &s.cpu.bound
	*b = cpuBound{li: li, l: l, own: s.cpuOwn(cb.f), cs: cs}
	if why := s.cpuWeights(cb.f, l); why != jit.DeclineServes {
		return why, nil
	}
	s.cpu.rows = [4]uint32{0, uint32(pg.keyStart(pos)), uint32(pos + 1), uint32(pos)}
	switch s.cpu.fault {
	case cpuFaultPage:
		if n := len(s.cpu.pages); n > 2 && pg.p > 0 && pos == pg.p+pg.p/4 {
			s.cpu.pages[0] = s.cpu.pages[n-1]
		}
	case cpuFaultRow:
		if pos > 0 {
			s.cpu.rows[1]++
		}
	}
	for i := range cb.f.codes {
		s.cpuCall(b, &cb.f.codes[i])
	}
	if pg.p > 0 && pos+1 > pg.written {
		pg.written = pos + 1
	}
	return jit.DeclineServes, nil
}

// cpuHistory makes the pages block li's step reads and writes resident,
// in the layer's arena, or says why they are not.
func (s *State) cpuHistory(li, pos int) jit.Decline {
	pg := &s.kv.layers[li]
	s.kvWritable(li, pos)
	n, _ := pg.page(pos)
	pg.grow(n)
	pg.cow(n)
	start := pg.keyStart(pos)
	s.kvEnsureWindow(li, start, pos-start)
	for p := start / pg.p; p <= n; p++ {
		if !pg.arenaPage(p) {
			return jit.DeclinePageOut
		}
	}
	return jit.DeclineServes
}

// cpuWeights says every weight f binds is one block l holds whole.
func (s *State) cpuWeights(f *cpuBlockForm, l *layer) jit.Decline {
	for i := range f.codes {
		for _, a := range f.codes[i].args {
			switch a.src {
			case cpuArgWeight:
				w := cpuRoleWeight(l, a.role)
				if w == nil || w.packed == nil && len(w.data) == 0 {
					return jit.DeclineNoWeight
				}
				if p := w.packed; p != nil && (p.Row != 0 || p.Stride != 0 && p.Stride != w.rows || p.Sheet != nil) {
					return jit.DeclineCutWeight
				}
			case cpuArgVector:
				if len(cpuRoleVector(l, a.role)) == 0 {
					return jit.DeclineNoWeight
				}
			}
		}
	}
	return jit.DeclineServes
}

// cpuOwn is form f's own buffers for this State.
func (s *State) cpuOwn(f *cpuBlockForm) [][]uint32 {
	if b, ok := s.cpu.own[f]; ok {
		return b
	}
	if s.cpu.own == nil {
		s.cpu.own = map[*cpuBlockForm][][]uint32{}
	}
	b := make([][]uint32, len(f.own))
	for i, n := range f.own {
		b[i] = make([]uint32, max(n, 1))
		copy(b[i], f.consts[i])
	}
	s.cpu.own[f] = b
	return b
}

// cpuCall runs one kernel of a block: once, or over its calls on the pool.
func (s *State) cpuCall(b *cpuBound, c *cpuCode) {
	if c.n <= 1 || c.serial || s.jit == nil || s.jit.Workers() <= 1 {
		a := &s.cpu.args[0]
		for at := range max(c.n, 1) {
			s.cpuWords(a, b, c, at)
			c.code.Call(a)
		}
		return
	}
	r := &s.cpu.run
	r.s, r.b, r.c = s, b, c
	s.jit.ParallelWorkers(c.n, 1, r.fn)
}

// cpuWords writes call at's words of kernel c into a.
func (s *State) cpuWords(a *cpu.Args, b *cpuBound, c *cpuCode, at int) {
	base := unsafe.Pointer(a)
	for i := range c.args {
		ar := &c.args[i]
		var p unsafe.Pointer
		switch ar.src {
		case cpuArgValue:
			*(*uint64)(unsafe.Add(base, ar.field)) = ar.val
			continue
		case cpuArgCoord:
			v := at
			for ax := len(c.calls) - 1; ax > int(ar.coord); ax-- {
				v /= c.calls[ax]
			}
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(v % c.calls[ar.coord])
			continue
		case cpuArgWord:
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(b.own[ar.i][ar.off])
			continue
		case cpuArgX:
			p = unsafe.Pointer(&s.x[0])
		case cpuArgOwn:
			p = unsafe.Pointer(&b.own[ar.i][0])
		case cpuArgWeight:
			w := cpuRoleWeight(b.l, ar.role)
			if w.packed == nil {
				p = unsafe.Pointer(&w.data[0])
			} else {
				p = unsafe.Pointer(&[3][]byte{w.packed.QS, w.packed.D, w.packed.SC}[ar.region][0])
			}
		case cpuArgVector:
			p = unsafe.Pointer(&cpuRoleVector(b.l, ar.role)[0])
		case cpuArgRope:
			p = unsafe.Pointer(&b.cs[0])
		case cpuArgKeys:
			p = unsafe.Pointer(&s.kv.layers[b.li].arena.k[0])
		case cpuArgValues:
			p = unsafe.Pointer(&s.kv.layers[b.li].arena.v[0])
		case cpuArgPages:
			p = unsafe.Pointer(&s.cpu.pages[0])
		case cpuArgRows:
			p = unsafe.Pointer(&s.cpu.rows[0])
		case cpuArgZero:
			p = unsafe.Pointer(&s.cpu.zero[0])
		}
		*(*uint64)(unsafe.Add(base, ar.field)) = uint64(uintptr(p)) + uint64(ar.off*ar.elem)
	}
}

// region runs the calls [lo, hi) of the kernel the State's run names, on
// worker w's words.
func (r *cpuRegion) region(w, lo, hi int) {
	a := &r.s.cpu.args[w]
	for at := lo; at < hi; at++ {
		r.s.cpuWords(a, r.b, r.c, at)
		r.c.code.Call(a)
	}
}

// cpuPlace composes a new State's blocks on the CPU: each block of the
// State's range no device holds, once, at its placement (cpuCompose).
func (s *State) cpuPlace() {
	if !s.m.opt.composedCPU {
		return
	}
	s.cpu.blocks = make([]*cpuBlock, s.c.NLayer)
	s.cpu.composed = make([]int, s.c.NLayer)
	w := 1
	if s.jit != nil {
		w = s.jit.Workers()
	}
	s.cpu.args = make([]cpu.Args, max(w, 1))
	s.cpu.run.fn = s.cpu.run.region
	for li := s.lo; li < min(s.hi, s.c.NLayer); li++ {
		if !s.devAt(li) {
			s.cpuCompose(li)
		}
	}
}

// cpuRelocated follows block li onto the CPU, composed for it there, or off
// it, its composition released.
func (s *State) cpuRelocated(li int, onCPU bool) {
	if li < 0 || li >= len(s.cpu.blocks) {
		return
	}
	if onCPU {
		s.cpuCompose(li)
		return
	}
	s.cpu.blocks[li] = nil
}

// cpuCompose composes block li for the CPU device: its form, through the
// model's keyed cache, and its bindings, counted (CPUCompositions).
func (s *State) cpuCompose(li int) {
	cb := &cpuBlock{}
	s.cpu.blocks[li] = cb
	cb.why = s.cpuWhy(li)
	if cb.why != jit.DeclineServes {
		return
	}
	f, err := s.m.formCPUBlock(s, li, cb)
	if err != nil {
		if s.cpu.formErr == nil {
			s.cpu.formErr = err
		}
		var d cpuDeclineError
		if errors.As(err, &d) {
			cb.why = d.why
		} else {
			cb.why = jit.DeclineNotFormed
		}
		return
	}
	cb.f = f
	s.cpu.composed[li]++
	if need := s.kv.layers[li].arenaSlots(); len(s.cpu.pages) < need {
		s.cpu.pages = make([]uint32, need)
		for i := range s.cpu.pages {
			s.cpu.pages[i] = uint32(i)
		}
	}
}

// cpuBlockOf is block li's composition on the CPU, made at its placement.
func (s *State) cpuBlockOf(li int) (*cpuBlock, jit.Decline) {
	if li >= len(s.cpu.blocks) || s.cpu.blocks[li] == nil {
		return nil, jit.DeclineUnplaced
	}
	cb := s.cpu.blocks[li]
	return cb, cb.why
}

// CPUCompositions is how many times each block was composed for the CPU in
// this State: once at its placement there, and once more for each
// relocation back to it.
func (s *State) CPUCompositions() []int { return s.cpu.composed }

// cpuWhy is what keeps block li off the composer's CPU path before it is
// asked: what its graph and history need that the CPU device does not hold.
func (s *State) cpuWhy(li int) jit.Decline {
	c := s.c
	g := s.m.Composing().BlockGraph(li)
	pg := &s.kv.layers[li]
	switch {
	case cpu.CPU().ISA()&cpu.ISATierAVX2 != cpu.ISATierAVX2:
		return jit.DeclineISA
	case g == nil:
		return jit.DeclineNoGraph
	case len(g.In) != 1 || len(g.Out) != 1:
		return jit.DeclineFeeds
	case c.LayerKind(li).Recurrent():
		return jit.DeclineRecurrent
	case pg.latent:
		return jit.DeclineLatent
	case c.KVShared(li):
		return jit.DeclineShared
	case pg.win > 0:
		return jit.DeclineHistoryWindow
	case s.nseq != 1:
		return jit.DeclineBatch
	case s.kvl.headMajor:
		return jit.DeclineHeadMajor
	case s.kvl.fmt != cpu.KVF16:
		// The CPU's profile keeps a history binary16 itself (jit.Halves).
		return jit.DeclineKVFormat
	case pg.p > 0 && pg.arena == nil:
		return jit.DeclinePageOut
	}
	return jit.DeclineServes
}

// cpuDeclineError is a form the CPU device refuses by a named reason.
type cpuDeclineError struct {
	why jit.Decline
	err error
}

func (e cpuDeclineError) Error() string {
	return fmt.Sprintf("%s: %v", e.why.String(), e.err)
}

// cpuBlockForm is block li's form on the CPU through the model's keyed cache.
func (m *Model) formCPUBlock(s *State, li int, cb *cpuBlock) (*cpuBlockForm, error) {
	g := m.Composing().BlockGraph(li)
	cb.graph = g
	gr := *g
	gr.Block = 0
	formats := map[jlm.Role]jlm.Type{}
	f := m.Composing()
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if e, ok := f.Find(w.Role, int32(li), -1); ok {
				formats[w.Role] = e.Type
			}
		}
	}
	pg := &s.kv.layers[li]
	page := max(pg.p, 1)
	workers := 1
	if s.jit != nil {
		workers = s.jit.Workers()
	}
	req := jit.Request{Graph: &gr, Formats: formats,
		Step:   jit.Step{Rows: 1, Page: page, Keys: max(pg.arenaSlots(), 1) * page, ActWindow: 32},
		Device: composedCPUDevice(workers)}
	c := &m.cpuForms
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.reg == nil {
		c.reg = jit.Default()
	}
	if c.m == nil {
		c.m, c.err = map[jit.Key]*cpuBlockForm{}, map[jit.Key]error{}
	}
	key, codes, err := c.reg.Form(req)
	if err != nil {
		return nil, err
	}
	if fm, ok := c.m[key]; ok {
		return fm, nil
	}
	if err, ok := c.err[key]; ok {
		return nil, err
	}
	fm, err := buildCPUForm(&gr, codes)
	if err != nil {
		c.err[key] = err
		return nil, err
	}
	c.m[key] = fm
	return fm, nil
}

// buildCPUForm maps codes and plans each call word's source.
func buildCPUForm(gr *graph.Graph, codes []jit.Code) (*cpuBlockForm, error) {
	f := &cpuBlockForm{consts: map[int][]uint32{}}
	// A kernel's constant block is its own: two kernels' blocks may share a
	// name and hold different words.
	type ownKey struct {
		o    jit.Operand
		code int
	}
	own := map[ownKey]int{}
	ownOf := func(o jit.Operand, code, words int, consts []uint32) int {
		key := ownKey{o, -1}
		if consts != nil {
			key.code = code
		}
		if i, ok := own[key]; ok {
			f.own[i] = max(f.own[i], words)
			return i
		}
		f.own = append(f.own, words)
		i := len(f.own) - 1
		own[key] = i
		if consts != nil {
			f.consts[i] = consts
		}
		return i
	}
	for ci, c := range codes {
		if c.CPU == nil {
			f.close()
			return nil, fmt.Errorf("%s has no CPU code", c.Name)
		}
		cc := cpuCode{calls: c.Calls, n: 1}
		for _, n := range c.Calls {
			cc.n *= n
		}
		bytes := map[jit.Operand]int64{}
		for _, bd := range c.Binds {
			if o := bd.Operand; o.Kind == jit.OperandValue {
				cc.reads = cc.reads || bd.Read && len(gr.In) > 0 && o.Value == gr.In[0]
				cc.writes = cc.writes || bd.Writes && len(gr.Out) > 0 && o.Value == gr.Out[0]
			}
			o := bd.Operand
			o.Copy = 0
			bytes[o] = max(bytes[o], bd.Bytes)
			if bd.Writes && bd.Operand.Kind == jit.OperandScratch {
				cc.serial = true
			}
		}
		for _, a := range c.Args {
			// A pair's second head's operand is the first's storage at its
			// offset.
			o := a.Operand
			o.Copy = 0
			ar := cpuArg{field: uintptr(a.Field), off: a.Offset, elem: 4}
			if a.Field < 0 || uintptr(a.Field)+8 > unsafe.Sizeof(cpu.Args{}) {
				f.close()
				return nil, fmt.Errorf("%s: a call word at %d past the argument block", c.Name, a.Field)
			}
			switch {
			case a.Coord > 0:
				ar.src, ar.coord = cpuArgCoord, uint8(a.Coord-1)
			case a.Length:
				ar.src, ar.val = cpuArgValue, uint64(a.Extent)
			case a.Value:
				f.close()
				return nil, fmt.Errorf("%s: a count the caller computes", c.Name)
			case a.Moving:
				f.close()
				return nil, fmt.Errorf("%s: an offset of %v the caller computes", c.Name, o)
			case a.Words != nil:
				ar.src, ar.i = cpuArgOwn, ownOf(o, ci, len(a.Words), a.Words)
			case o.Kind == jit.OperandValue && len(gr.In) > 0 && o.Value == gr.In[0],
				o.Kind == jit.OperandValue && len(gr.Out) > 0 && o.Value == gr.Out[0]:
				ar.src = cpuArgX
			case o.Kind == jit.OperandValue:
				ar.src, ar.i = cpuArgOwn, ownOf(o, ci, max(gr.Values[o.Value].Width, int(bytes[o]/4))+a.Offset, nil)
			case o.Kind == jit.OperandScratch:
				ar.src, ar.i = cpuArgOwn, ownOf(o, ci, int(bytes[o]/4)+a.Offset+1, nil)
			case o.Kind == jit.OperandWeight && o.Region >= 0 && o.Region < 3:
				ar.src, ar.role, ar.region = cpuArgWeight, o.Role, uint8(o.Region)
			case o.Kind == jit.OperandVector:
				ar.src, ar.role = cpuArgVector, o.Role
			case o.Kind == jit.OperandTable:
				switch o.Name {
				case "rope.cs":
					ar.src = cpuArgRope
				case "kv.k":
					ar.src = cpuArgKeys
				case "kv.v":
					ar.src = cpuArgValues
				case "pages":
					ar.src = cpuArgPages
				case "rows":
					ar.src = cpuArgRows
				case "q.off":
					ar.src = cpuArgZero
				default:
					f.close()
					return nil, cpuDeclineError{jit.DeclineTable, fmt.Errorf("%s reads the table %s", c.Name, o.Name)}
				}
			default:
				f.close()
				return nil, fmt.Errorf("%s binds %+v, which the CPU's call site does not provide", c.Name, o)
			}
			if a.Word {
				if ar.src != cpuArgOwn {
					f.close()
					return nil, fmt.Errorf("%s passes a word of %+v by value, which the CPU's call site holds no words of", c.Name, o)
				}
				ar.src = cpuArgWord
			}
			cc.args = append(cc.args, ar)
		}
		code, err := cpu.Map(c.CPU)
		if err != nil {
			f.close()
			return nil, fmt.Errorf("mapping %s: %w", c.Name, err)
		}
		cc.code = code
		f.codes = append(f.codes, cc)
	}
	if !f.inPlace() {
		f.close()
		return nil, fmt.Errorf("the block reads its residual after writing its output over it")
	}
	return f, nil
}

// inPlace says the form may write its output over its input, the
// residual: every kernel that reads the input runs before the first that
// writes the output.
func (f *cpuBlockForm) inPlace() bool {
	lastRead, firstWrite := -1, len(f.codes)
	for i, c := range f.codes {
		if c.reads {
			lastRead = i
		}
		if c.writes {
			firstWrite = min(firstWrite, i)
		}
	}
	return lastRead < firstWrite
}

// cpuRoleWeight is block l's matrix of role r, nil for one it does not hold.
func cpuRoleWeight(l *layer, r jlm.Role) *tensor {
	switch r {
	case jlm.RoleAttnQ:
		return &l.wq
	case jlm.RoleAttnK:
		return &l.wk
	case jlm.RoleAttnV:
		return &l.wv
	case jlm.RoleAttnOut:
		return &l.wo
	case jlm.RoleFFNGate:
		return &l.gate
	case jlm.RoleFFNUp:
		return &l.up
	case jlm.RoleFFNDown:
		return &l.down
	}
	return nil
}

// cpuRoleVector is block l's vector of role r, nil for one it does not
// hold.
func cpuRoleVector(l *layer, r jlm.Role) []float32 {
	switch r {
	case jlm.RoleAttnNorm:
		return l.attnNorm
	case jlm.RoleFFNNorm:
		return l.ffnNorm
	case jlm.RoleAttnNormBias:
		return l.attnNormB
	case jlm.RoleFFNNormBias:
		return l.ffnNormB
	case jlm.RolePostAttnNorm:
		return l.postAttnNorm
	case jlm.RolePostFFNNorm:
		return l.postFFNNorm
	case jlm.RoleAttnQNorm:
		return l.qNorm
	case jlm.RoleAttnKNorm:
		return l.kNorm
	case jlm.RoleAttnQBias:
		return l.bq
	case jlm.RoleAttnKBias:
		return l.bk
	case jlm.RoleAttnVBias:
		return l.bv
	case jlm.RoleAttnOutBias:
		return l.bo
	case jlm.RoleFFNUpBias:
		return l.upB
	case jlm.RoleFFNDownBias:
		return l.downB
	case jlm.RoleAttnSinks:
		return l.sinks
	case jlm.RoleAttnSubNorm:
		return l.attnSubNorm
	case jlm.RoleFFNSubNorm:
		return l.ffnSubNorm
	case jlm.RoleXIELU:
		return l.xielu
	}
	return nil
}

// arenaSlots is how many pages the layer's arena holds, 0 for none.
func (s *kvPages) arenaSlots() int {
	if s.arena == nil {
		return 0
	}
	return s.arena.slots
}
