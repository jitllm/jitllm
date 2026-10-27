package model

import (
	"errors"
	"fmt"
	"slices"
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
	cpuArgCall
	// cpuArgRecState and cpuArgRecConv are a linear block's recurrent states
	// and its convolution's windows, a slot a sequence, read and written in
	// place; cpuArgRecRows the call's recurrent descriptor, which names the
	// call's sequence's slot.
	cpuArgRecState
	cpuArgRecConv
	cpuArgRecRows
	// cpuArgIO is an input or output of a graph whose output is not its
	// input (the head's, the sampler's): the bound buffer of that place,
	// the inputs' then the outputs' (cpuBound.io).
	cpuArgIO
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
	// per is the elements the address moves a unit of each call
	// coordinate (jit.CallArg.Per), moves whether any does.
	per   [2]int
	moves bool
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
	// live is the code's hold on a call's rows (jit.Code's Live and
	// LiveUnits): a call of rows rows runs the units of the leading axis
	// they reach, each of inner calls (jit.RowCalls).
	live  jit.Code
	inner int
	// tiles is a kernel whose inner calls tile a weight (jit.Code.TileMajor):
	// its calls run a tile for every live row before the next tile.
	tiles bool
}

// cpuBlockForm is a block's kernels for the CPU at one composer Key, mapped
// once for the model, and the words of each of its own buffers.
type cpuBlockForm struct {
	codes []cpuCode
	// rows is the rows the form holds: 1 for decode's, else its capacity;
	// keys the positions of a sequence's history it attends at most.
	rows, keys int
	// unit is the most rows a unit of a kernel's leading axis holds
	// (jit.RowJam): a call's rows past a whole number of them are run
	// from a copy of the residual, so a unit's rows past the call's write
	// no other rows.
	unit int
	own  []int
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
	zero     []uint32
	// rowsTab is a call's rows' descriptors at the rows' form, call its
	// "rows.n" word, and rseq and rpos a step's rows' sequences and
	// positions.
	rowsTab []uint32
	call    [4]uint32
	// recRows is a linear block's call's recurrent descriptor: one row, the
	// slot it reads and the slot it writes, its sequence's.
	recRows    [3]uint32
	rseq, rpos []int
	// xpad is a call's residual rows where its kernels' units reach past
	// them (cpuBlockForm.unit).
	xpad []float32
	// runX is a copy of a step's rows whose keys a first pass writes, where
	// a bidirectional run takes more calls than one (cpuRowsRun).
	runX []float32
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
	head  cpuHeadState
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
	// cpuFaultRec binds every linear block to the recurrent state and
	// window of the first: each steps a summary that is not its own.
	cpuFaultRec
	// cpuFaultChunk starts each prompt chunk after the first from a summary
	// of zeros: the recurrence not carried across the chunk boundary.
	cpuFaultChunk
	// cpuFaultHead binds the head's input to the State's norm row in place
	// of the residual.
	cpuFaultHead
)

// cpuBlock is block li's forms on the CPU, its graph the one they were
// asked for: decode's, of one row, and the rows', at the CPU's capacity
// (cpuRowsCap), whose calls run the rows their "rows.n" word states.
type cpuBlock struct {
	graph *graph.Graph
	f     *cpuBlockForm
	rows  *cpuBlockForm
	why   jit.Decline
	// rowsWhy is why the rows' form is not served, where it is not.
	rowsWhy jit.Decline
}

// cpuRowsCap is the rows the CPU's form of several rows is composed for: a
// step of more runs in calls of this many, one after another.
const cpuRowsCap = 64

// cpuRegion is the kernel a pool region calls, its calls the region's range.
type cpuRegion struct {
	s    *State
	b    *cpuBound
	c    *cpuCode
	live int
	fn   func(worker, lo, hi int)
}

// cpuBound is a block's step as its words read it: n rows of the residual
// x, their rotary tables cs and their descriptors rows, their history the
// layer's pages kv through the page table pages.
type cpuBound struct {
	li    int
	l     *layer
	own   [][]uint32
	cs    []float32
	x     []float32
	rows  []uint32
	n     int
	kv    *kvPages
	pages []uint32
	// recLi is the block whose recurrent states a linear block's step binds
	// and recOf the State that holds them.
	recLi int
	recOf *State
	// io is the buffers of a graph apart from its residual (cpuArgIO), its
	// inputs' then its outputs'.
	io [cpuIO]unsafe.Pointer
}

// cpuIO is the most inputs and outputs a graph apart from its residual
// binds: the sampler's candidates read three and write two.
const cpuIO = 6

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
	*b = cpuBound{li: li, l: l, own: s.cpuOwn(cb.f), cs: cs, x: s.x, rows: s.cpu.rows[:], n: 1, kv: pg, pages: s.cpu.pages,
		recLi: s.cpuRecLi(li), recOf: s}
	if why := s.cpuWeights(cb.f, l); why != jit.DeclineServes {
		return why, nil
	}
	s.cpu.rows = [4]uint32{0, uint32(pg.keyStart(pos)), uint32(pos + 1), uint32(pos)}
	s.cpu.recRows = [3]uint32{1, 0, 0}
	s.cpu.call[0] = 1
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

// cpuRows runs block li on the CPU the composer's way for the n rows of x,
// row i at position pos[i] of sequence seq[i], its rotary table the i-th of
// cs, or says why not: through the rows' form, a call of at most its
// capacity at a time, each row's history in its own sequence's pages.
func (s *State) cpuRows(li int, l *layer, x, cs, csSWA []float32, seq, pos []int) (jit.Decline, error) {
	f, why := s.cpuRowsForm(li, l)
	if why != jit.DeclineServes {
		return why, nil
	}
	if why := s.cpuRowsReady(li, f, pos); why != jit.DeclineServes {
		return why, nil
	}
	cs, nrot := s.cpuTable(li, cs, csSWA)
	s.cpuRowsRun(li, l, f, x, cs, nrot, seq, pos)
	return jit.DeclineServes, nil
}

// cpuTable is the rotary table of a step's rows block li reads and its
// width: the sliding layers' (csSWA) where the model turns them by a
// rotary of their own.
func (s *State) cpuTable(li int, cs, csSWA []float32) ([]float32, int) {
	if csSWA != nil && s.c.SWA(li) {
		return csSWA, s.c.NRotSWA
	}
	return cs, s.c.RopeW()
}

// cpuSessions is cpuRows for a step across sessions (stepHost): row i's
// history is s.rowOwn[i]'s, its one sequence, and the first nlogit rows are
// each a session's last. Each session runs its own rows on its own block's
// form and buffers, its history bound as its form was composed for. Every
// session's pages are made ready before any row runs; the runs of a
// session's consecutive rows run one after another, the rows after the
// logit rows first, so a session's last row reads the keys its earlier
// rows wrote.
func (s *State) cpuSessions(li int, l *layer, x, cs, csSWA []float32, seq, pos []int, nlogit int) (jit.Decline, error) {
	cs, nrot := s.cpuTable(li, cs, csSWA)
	n := len(pos)
	if why := s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, 0, n, false); why != jit.DeclineServes {
		return why, nil
	}
	s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, nlogit, n, true)
	s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, 0, nlogit, true)
	return jit.DeclineServes, nil
}

// cpuOwnerRuns makes ready (run unset) or runs block li over the rows
// [lo, hi) of a step across sessions, a run of one session's consecutive
// rows at a time, on that session's form.
func (s *State) cpuOwnerRuns(li int, l *layer, x, cs []float32, nrot int, seq, pos []int, lo, hi int, run bool) jit.Decline {
	w := s.c.ResidW()
	for a := lo; a < hi; {
		o, b := s.rowOwn[a], a+1
		for b < hi && s.rowOwn[b] == o {
			b++
		}
		if o.m != s.m || !o.m.opt.composedCPU {
			return jit.DeclineSessions
		}
		f, why := o.cpuRowsForm(li, l)
		if why != jit.DeclineServes {
			return why
		}
		if run {
			o.cpuRowsRun(li, l, f, x[a*w:], cs[a*nrot:], nrot, seq[a:b], pos[a:b])
		} else if why := o.cpuRowsReady(li, f, pos[a:b]); why != jit.DeclineServes {
			return why
		}
		a = b
	}
	return jit.DeclineServes
}

// cpuRecRows runs linear block li over the n rows of x on its form of one
// row, a row at a time in order (each row's index; 0.. where order is nil),
// each stepping its own sequence's recurrent state -- its owner's, as
// rowOwner names it -- or says why not: the recurrence is sequential, and
// the form of a row is the block's whole step. A prompt's rows (base its
// first row in the prompt, p0 its first position; base -1 for none) keep
// each row's summary as main's row loop does: the speculative rows'
// snapshot and the restore points.
func (s *State) cpuRecRows(li int, l *layer, x, cs []float32, seq, pos, order []int, base, p0 int) (jit.Decline, error) {
	cb, why := s.cpuBlockOf(li)
	if why != jit.DeclineServes {
		return why, nil
	}
	if why := s.cpuWeights(cb.f, l); why != jit.DeclineServes {
		return why, nil
	}
	w, nrot := s.c.ResidW(), s.c.RopeW()
	b := &s.cpu.bound
	pg := &s.kv.layers[li]
	for i := range pos {
		r := i
		if order != nil {
			r = order[i]
		}
		// A block that attends as well writes and reads the row's history.
		if pg.p > 0 {
			if why := s.cpuHistory(li, pos[r]); why != jit.DeclineServes {
				return why, nil
			}
		}
		o, slot := s.rowOwner(r, seq)
		if s.cpu.fault == cpuFaultChunk && base > 0 && i == 0 {
			clear(o.rstate[li][slot*s.dg.stateStride():][:s.dg.stateStride()])
			clear(o.rconv[li][slot*s.dg.convStateLen():][:s.dg.convStateLen()])
		}
		*b = cpuBound{li: li, l: l, own: s.cpuOwn(cb.f), x: x[r*w:], rows: s.cpu.rows[:], n: 1, kv: &s.kv.layers[li],
			pages: s.cpu.pages, recLi: s.cpuRecLi(li), recOf: o}
		if nrot > 0 {
			b.cs = cs[r*nrot:]
		}
		s.cpu.rows = [4]uint32{0, uint32(pg.keyStart(pos[r])), uint32(pos[r] + 1), uint32(pos[r])}
		s.cpu.recRows = [3]uint32{1, uint32(slot), uint32(slot)}
		s.cpu.call[0] = 1
		for k := range cb.f.codes {
			s.cpuCall(b, &cb.f.codes[k])
		}
		if pg.p > 0 && pos[r]+1 > pg.written {
			pg.written = pos[r] + 1
		}
		if base >= 0 {
			s.rows.snapRow(s, li, slot, base+r)
			s.recRowPoint(li, slot, p0+r+1)
		}
	}
	return jit.DeclineServes, nil
}

// cpuRecLi is the block whose recurrent state block li's step binds: its
// own, or under cpuFaultRec the first linear block's.
func (s *State) cpuRecLi(li int) int {
	if s.cpu.fault != cpuFaultRec {
		return li
	}
	for r := range s.rstate {
		if s.rstate[r] != nil {
			return r
		}
	}
	return li
}

// cpuRowsForm is block li's rows' form, its weights the block's, or why
// not.
func (s *State) cpuRowsForm(li int, l *layer) (*cpuBlockForm, jit.Decline) {
	cb, why := s.cpuBlockOf(li)
	if why != jit.DeclineServes {
		return nil, why
	}
	if cb.rows == nil {
		return nil, cb.rowsWhy
	}
	if why := s.cpuWeights(cb.rows, l); why != jit.DeclineServes {
		return nil, why
	}
	return cb.rows, jit.DeclineServes
}

// cpuRowsReady makes the pages of the history the rows at pos write and
// read resident, in block li's arena, or says why f cannot run them: a
// history past f's key bound, a page outside the arena.
func (s *State) cpuRowsReady(li int, f *cpuBlockForm, pos []int) jit.Decline {
	pg := &s.kv.layers[li]
	if pg.p == 0 {
		return jit.DeclineServes
	}
	// Every row's page is written before any is read: a step's rows stand
	// in any order, and a page holds every sequence's positions.
	start, end := pos[0], 0
	for _, p := range pos {
		s.cpuWritable(li, p)
		start, end = min(start, pg.keyStart(p)), max(end, p+1)
	}
	if end > f.keys || max(pg.arenaSlots(), 1)*max(s.nseq, 1) > len(s.cpu.pages) {
		return jit.DeclineSessions
	}
	if why := s.cpuReadable(li, start, end-1); why != jit.DeclineServes {
		return why
	}
	if n := len(s.cpu.pages); s.cpu.fault == cpuFaultPage && n > 2 && end > pg.p+pg.p/4 {
		s.cpu.pages[0] = s.cpu.pages[n-1]
	}
	return jit.DeclineServes
}

// cpuRowsRun runs block li's rows' form f over the rows of x, their history
// the State's, a call of at most f's capacity at a time. A bidirectional
// run (bidir.go) counts its rows' keys to its end; where the step takes
// more calls than one, a first pass over a copy of the rows writes every
// row's keys, its outputs dropped and each row reading only the keys
// before it, and the second reads them all.
func (s *State) cpuRowsRun(li int, l *layer, f *cpuBlockForm, x, cs []float32, nrot int, seq, pos []int) {
	runs := s.runsAt(li) && s.bidirIn(pos[0], len(pos))
	if runs && len(pos) > f.rows {
		n := len(pos) * s.c.ResidW()
		if len(s.cpu.runX) < n {
			s.cpu.runX = make([]float32, n)
		}
		copy(s.cpu.runX, x[:n])
		s.cpuRowsCalls(li, l, f, s.cpu.runX[:n], cs, nrot, seq, pos, false)
	}
	s.cpuRowsCalls(li, l, f, x, cs, nrot, seq, pos, runs)
}

// cpuRowsCalls is cpuRowsRun's calls, each row's keys counted to its run's
// end where runs is set.
func (s *State) cpuRowsCalls(li int, l *layer, f *cpuBlockForm, x, cs []float32, nrot int, seq, pos []int, runs bool) {
	n := len(pos)
	pg := &s.kv.layers[li]
	w := s.c.ResidW()
	slots := max(pg.arenaSlots(), 1)
	b := &s.cpu.bound
	end := 0
	for r0 := 0; r0 < n; r0 += f.rows {
		k := min(f.rows, n-r0)
		for i := range k {
			p := pos[r0+i]
			e := p + 1
			if runs {
				e = max(e, s.bidirEnd(p))
			}
			end = max(end, p+1)
			d := s.cpu.rowsTab[i*4 : i*4+4]
			d[0], d[1], d[2], d[3] = uint32(seq[r0+i]*slots), uint32(pg.keyStart(p)), uint32(e), uint32(p)
			if s.cpu.fault == cpuFaultRow && p > 0 {
				d[1]++
			}
		}
		bx := x[r0*w:]
		staged := f.unit > 1 && k%f.unit != 0
		if staged {
			if len(s.cpu.xpad) < f.rows*w {
				s.cpu.xpad = make([]float32, f.rows*w)
			}
			copy(s.cpu.xpad, bx[:k*w])
			bx = s.cpu.xpad
		}
		*b = cpuBound{li: li, l: l, own: s.cpuOwn(f), cs: cs[r0*nrot:], x: bx, rows: s.cpu.rowsTab, n: k,
			kv: pg, pages: s.cpu.pages}
		s.cpu.call[0] = uint32(k)
		for i := range f.codes {
			s.cpuCall(b, &f.codes[i])
		}
		if staged {
			copy(x[r0*w:(r0+k)*w], s.cpu.xpad[:k*w])
		}
	}
	if pg.p > 0 && end > pg.written {
		pg.written = end
	}
}

// cpuRowsOf is the sequences and positions of a chunk's n rows of slot
// from position p0, in the State's own rows.
func (s *State) cpuRowsOf(slot, p0, n int) (seq, pos []int) {
	if cap(s.cpu.rpos) < n {
		s.cpu.rseq, s.cpu.rpos = make([]int, n), make([]int, n)
	}
	seq, pos = s.cpu.rseq[:n], s.cpu.rpos[:n]
	for i := range n {
		seq[i], pos[i] = slot, p0+i
	}
	return seq, pos
}

// cpuHistory makes the pages block li's step reads and writes resident,
// in the layer's arena, or says why they are not.
func (s *State) cpuHistory(li, pos int) jit.Decline {
	s.cpuWritable(li, pos)
	return s.cpuReadable(li, s.kv.layers[li].keyStart(pos), pos)
}

// cpuWritable makes the page block li writes pos into resident and its own.
func (s *State) cpuWritable(li, pos int) {
	pg := &s.kv.layers[li]
	s.kvWritable(li, pos)
	n, _ := pg.page(pos)
	pg.grow(n)
	pg.cow(n)
}

// cpuReadable makes the pages of block li's positions [start, pos]
// resident, in the layer's arena, or says why they are not.
func (s *State) cpuReadable(li, start, pos int) jit.Decline {
	pg := &s.kv.layers[li]
	n, _ := pg.page(pos)
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
				w := s.cpuRoleWeight(l, a.role)
				if w == nil || w.packed == nil && len(w.data) == 0 {
					return jit.DeclineNoWeight
				}
				if p := w.packed; p != nil && (p.Row != 0 || p.Stride != 0 && p.Stride != w.rows || p.Sheet != nil) {
					return jit.DeclineCutWeight
				}
			case cpuArgVector:
				if len(s.cpuRoleVector(l, a.role)) == 0 {
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
	n, live := c.n, 0
	if c.live.Live > 0 {
		live = c.live.LiveUnitsOf(b.n)
		n = live * c.inner
	}
	if !c.tiles {
		live = 0
	}
	if n <= 1 || c.serial || s.jit == nil || s.jit.Workers() <= 1 {
		a := &s.cpu.args[0]
		for i := range max(n, 1) {
			s.cpuWords(a, b, c, c.callAt(i, live))
			c.code.Call(a)
		}
		return
	}
	r := &s.cpu.run
	r.s, r.b, r.c, r.live = s, b, c, live
	s.jit.ParallelWorkers(n, 1, r.fn)
}

// callAt is the call the i-th of a kernel's calls runs: in order, or for a
// kernel tiling a weight over live units of rows (tiles), a tile's every
// unit before the next tile's.
func (c *cpuCode) callAt(i, live int) int {
	if live <= 0 {
		return i
	}
	return i%live*c.inner + i/live
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
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(cpuCoord(c.calls, at, int(ar.coord)))
			continue
		case cpuArgWord:
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(b.own[ar.i][ar.off])
			continue
		case cpuArgX:
			p = unsafe.Pointer(&b.x[0])
		case cpuArgIO:
			p = b.io[ar.i]
		case cpuArgOwn:
			p = unsafe.Pointer(&b.own[ar.i][0])
		case cpuArgWeight:
			w := s.cpuRoleWeight(b.l, ar.role)
			if w.packed == nil {
				p = unsafe.Pointer(&w.data[0])
			} else {
				p = unsafe.Pointer(&[3][]byte{w.packed.QS, w.packed.D, w.packed.SC}[ar.region][0])
			}
		case cpuArgVector:
			p = unsafe.Pointer(&s.cpuRoleVector(b.l, ar.role)[0])
		case cpuArgRope:
			p = unsafe.Pointer(&b.cs[0])
		case cpuArgKeys:
			p = unsafe.Pointer(&b.kv.arena.k[0])
		case cpuArgValues:
			p = unsafe.Pointer(&b.kv.arena.v[0])
		case cpuArgPages:
			p = unsafe.Pointer(&b.pages[0])
		case cpuArgRows:
			p = unsafe.Pointer(&b.rows[0])
		case cpuArgCall:
			p = unsafe.Pointer(&s.cpu.call[0])
		case cpuArgRecState:
			p = unsafe.Pointer(&b.recOf.rstate[b.recLi][0])
		case cpuArgRecConv:
			p = unsafe.Pointer(&b.recOf.rconv[b.recLi][0])
		case cpuArgRecRows:
			p = unsafe.Pointer(&s.cpu.recRows[0])
		case cpuArgZero:
			p = unsafe.Pointer(&s.cpu.zero[0])
		}
		off := ar.off
		if ar.moves {
			for ax := range c.calls {
				if ax < len(ar.per) && ar.per[ax] != 0 {
					off += cpuCoord(c.calls, at, ax) * ar.per[ax]
				}
			}
		}
		*(*uint64)(unsafe.Add(base, ar.field)) = uint64(uintptr(p)) + uint64(off*ar.elem)
	}
}

// cpuCoord is call at's coordinate along axis ax of the calls' space calls,
// the last axis fastest.
func cpuCoord(calls []int, at, ax int) int {
	for a := len(calls) - 1; a > ax; a-- {
		at /= calls[a]
	}
	return at % calls[ax]
}

// region runs the calls [lo, hi) of the kernel the State's run names, on
// worker w's words.
func (r *cpuRegion) region(w, lo, hi int) {
	a := &r.s.cpu.args[w]
	for i := lo; i < hi; i++ {
		r.s.cpuWords(a, r.b, r.c, r.c.callAt(i, r.live))
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
	s.cpu.zero = make([]uint32, 4)
	for li := s.lo; li < min(s.hi, s.c.NLayer); li++ {
		if !s.devAt(li) {
			s.cpuCompose(li)
		}
	}
	s.cpuPlaceHead()
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
	f, err := s.m.formCPUBlock(s, li, cb, 1)
	if err != nil {
		cb.why = s.cpuFormDecline(err)
		return
	}
	cb.f = f
	if s.c.LayerKind(li).Recurrent() {
		cb.rowsWhy = jit.DeclineRecurrent
	} else if cb.rows, err = s.m.formCPUBlock(s, li, cb, cpuRowsCap); err != nil {
		cb.rowsWhy = s.cpuFormDecline(err)
	}
	s.cpu.composed[li]++
	s.cpuPages(s.kv.layers[li].arenaSlots())
	if cb.rows != nil && len(s.cpu.rowsTab) < cpuRowsCap*4 {
		s.cpu.rowsTab = make([]uint32, cpuRowsCap*4)
		s.cpu.zero = make([]uint32, cpuRowsCap*4)
	}
}

// cpuFormDecline is the reason a form's error declines its block by, the
// first error kept (CPUFormErr).
func (s *State) cpuFormDecline(err error) jit.Decline {
	if s.cpu.formErr == nil {
		s.cpu.formErr = err
	}
	var d cpuDeclineError
	if errors.As(err, &d) {
		return d.why
	}
	return jit.DeclineNotFormed
}

// cpuPages lays out the page table the forms read for slots pages a
// sequence: sequence q's table starts at q*slots, its page j the arena's
// slot j holding q's positions -- the pool addressed a sequence's page at a
// time, as a page holds every sequence's positions in order (kvLayout).
func (s *State) cpuPages(slots int) {
	nseq := max(s.nseq, 1)
	if len(s.cpu.pages) >= slots*nseq {
		return
	}
	s.cpu.pages = make([]uint32, slots*nseq)
	for q := range nseq {
		for j := range slots {
			s.cpu.pages[q*slots+j] = uint32(j*nseq + q)
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
	case c.MoEAt(li):
		// The host keeps every expert in a page of its own: there is no
		// bank whole for the graph's expert matvecs to read.
		return jit.DeclineExperts
	case c.LayerKind(li).Recurrent() && (li >= len(s.rstate) || s.rstate[li] == nil || s.rconv[li] == nil || s.dg.tail != 0):
		return jit.DeclineRecurrent
	case pg.latent:
		return jit.DeclineLatent
	case c.KVShared(li):
		return jit.DeclineShared
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

// formCPUBlock is block li's form on the CPU at rows rows through the
// model's keyed cache: its history the State's sequences' pages
// (cpuPages), each a history of the State's context.
func (m *Model) formCPUBlock(s *State, li int, cb *cpuBlock, rows int) (*cpuBlockForm, error) {
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
		Step:   jit.Step{Rows: rows, Page: page, Keys: max(pg.arenaSlots(), 1) * page, ActWindow: 32, Seqs: max(s.nseq, 1)},
		Device: composedCPUDevice(workers)}
	return m.cpuFormOf(req, &gr, rows, false)
}

// cpuFormOf is req's form on the CPU, mapped once for the model by its Key.
func (m *Model) cpuFormOf(req jit.Request, gr *graph.Graph, rows int, apart bool) (*cpuBlockForm, error) {
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
	fm, err := buildCPUForm(gr, codes, rows, apart)
	if err != nil {
		c.err[key] = err
		return nil, err
	}
	fm.keys = req.Step.Keys
	c.m[key] = fm
	return fm, nil
}

// buildCPUForm maps codes and plans each call word's source: the graph's
// input and output the residual, or with apart each an io place of its own.
func buildCPUForm(gr *graph.Graph, codes []jit.Code, rows int, apart bool) (*cpuBlockForm, error) {
	if apart && len(gr.In)+len(gr.Out) > cpuIO {
		return nil, fmt.Errorf("a graph of %d inputs and %d outputs, past the %d places the CPU's call site binds", len(gr.In), len(gr.Out), cpuIO)
	}
	f := &cpuBlockForm{consts: map[int][]uint32{}, rows: rows}
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
		if rows > 1 {
			// A form of several rows runs the units of its calls' leading
			// axis the call's rows reach: a kernel not held to them would
			// read and write rows past the call's.
			if c.Live == 0 || len(c.Calls) == 0 || c.Calls[0] != c.LiveUnits {
				f.close()
				return nil, fmt.Errorf("%s at %d rows: its calls %v hold no rows of a call (live %d of %d units)", c.Name, rows, c.Calls, c.Live, c.LiveUnits)
			}
			cc.live, cc.inner = jit.Code{Live: c.Live, LiveUnits: c.LiveUnits}, cc.n/c.Calls[0]
			cc.tiles = c.TileMajor && cc.inner > 1
			f.unit = max(f.unit, c.Live/c.LiveUnits)
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
			if a.Elem > 0 {
				ar.elem = a.Elem
			}
			if len(a.Per) > len(ar.per) || len(a.Per) > len(c.Calls) {
				f.close()
				return nil, fmt.Errorf("%s: an address moving along %d axes of a calls' space of %d", c.Name, len(a.Per), len(c.Calls))
			}
			for ax, n := range a.Per {
				ar.per[ax] = n
				ar.moves = ar.moves || n != 0
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
			case apart && o.Kind == jit.OperandValue && slices.Contains(gr.In, o.Value):
				ar.src, ar.i = cpuArgIO, slices.Index(gr.In, o.Value)
			case apart && o.Kind == jit.OperandValue && slices.Contains(gr.Out, o.Value):
				ar.src, ar.i = cpuArgIO, len(gr.In)+slices.Index(gr.Out, o.Value)
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
				case "rows", "rows.win":
					// A row's descriptor is its own layer's: its range the
					// layer's window where it has one (kvPages.keyStart).
					ar.src = cpuArgRows
				case "rows.n":
					ar.src = cpuArgCall
				case "rec.state", "rec.state.out":
					ar.src = cpuArgRecState
				case "rec.conv", "rec.conv.out":
					ar.src = cpuArgRecConv
				case "rec.rows":
					ar.src = cpuArgRecRows
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
	if !apart && !f.inPlace() {
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

// cpuRoleWeight is block l's matrix of role r, nil for one it does not hold;
// with no block, the head's.
func (s *State) cpuRoleWeight(l *layer, r jlm.Role) *tensor {
	if l == nil {
		if r == jlm.RoleOutput || r == jlm.RoleTokenEmbd {
			return s.outW
		}
		return nil
	}
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
	case jlm.RoleAttnQKV:
		if l.ssmIn.rows != 0 {
			return &l.ssmIn
		}
		return &l.wq
	case jlm.RoleSCX:
		return &l.wq
	case jlm.RoleAttnGate, jlm.RoleSCB:
		return &l.ssmGate
	case jlm.RoleSSMBA, jlm.RoleSCC:
		return &l.ssmBA
	case jlm.RoleSSMXDt:
		return &l.ssmXDt
	case jlm.RoleSSMXB:
		return &l.ssmXB
	case jlm.RoleSSMXC:
		return &l.ssmXC
	case jlm.RoleSSMDtProj:
		return &l.ssmDtProj
	case jlm.RoleSSMOut:
		return &l.ssmOut
	case jlm.RoleSSMFA:
		return &l.ssmFA
	case jlm.RoleSSMFB:
		return &l.ssmFB
	case jlm.RoleSSMGA:
		return &l.ssmGA
	case jlm.RoleSSMGB:
		return &l.ssmGB
	case jlm.RoleSSMConv1d:
		return &l.ssmConvTaps
	}
	return nil
}

// cpuRoleVector is block l's vector of role r, nil for one it does not
// hold; with no block, the head's.
func (s *State) cpuRoleVector(l *layer, r jlm.Role) []float32 {
	if l == nil {
		switch r {
		case jlm.RoleOutputNorm:
			return s.outNorm
		case jlm.RoleOutputNormBias:
			return s.m.outNormB
		case jlm.RoleOutputBias:
			return s.m.outB
		}
		return nil
	}
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
	case jlm.RoleLayerOutScale:
		return l.outScaleRow
	case jlm.RoleSSMA:
		return l.ssmA
	case jlm.RoleSSMDtBias:
		return l.ssmDtBias
	case jlm.RoleSSMNorm:
		return l.ssmNorm
	case jlm.RoleSSMD:
		return l.ssmD
	case jlm.RoleSSMConvBias:
		return l.ssmConvB
	case jlm.RoleSSMDtNorm:
		return l.ssmDtNorm
	case jlm.RoleSSMBNorm:
		return l.ssmBNorm
	case jlm.RoleSSMCNorm:
		return l.ssmCNorm
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
