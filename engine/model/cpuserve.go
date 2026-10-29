package model

import (
	"errors"
	"fmt"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	hw "github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/probe"
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

// CPURecSteps is how many steps of several rows the CPU's linear blocks
// ran whole on their rows' form, and how many they stepped a row at a time.
func (s *State) CPURecSteps() (whole, stepped int) { return s.cpu.recWhole, s.cpu.recStepped }

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
	// call's sequence's slot. cpuArgRecConvOut is where a call's windows are
	// written: in place for a row, apart for a step of several (its runs
	// past the step's write a run's window again from the window read).
	cpuArgRecState
	cpuArgRecConv
	cpuArgRecConvOut
	// cpuArgShare is the rows a call of a Share code takes (jit.Code.Share).
	cpuArgShare
	cpuArgRecRows
	// cpuArgPoolPos is the positions table a pooled readout reads its live
	// count from (cpupool.go).
	cpuArgPoolPos
	// cpuArgIO is an input or output of a graph whose output is not its
	// input (the head's, the sampler's): the bound buffer of that place,
	// the inputs' then the outputs' (cpuBound.io).
	cpuArgIO
	// cpuArgSide is a block graph's input past its residual streams, the
	// rows' per-layer inputs (cpuBound.side).
	cpuArgSide
	// cpuArgRequest is the step's request words (cpuBound.req), cpuArgPos
	// its positions (cpuBound.pos).
	cpuArgRequest
	cpuArgPos
	// cpuArgPlanes is the block's rotary planes (nn.Rope.TabPlanes), which
	// a kernel turning at positions of its own reads (cpuForms.planes).
	cpuArgPlanes
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
	// stream is the residual stream a cpuArgX address is in.
	stream int
	// per is the elements the address moves a unit of each call
	// coordinate (jit.CallArg.Per), moves whether any does.
	per   [2]int
	moves bool
}

// cpuCode is one kernel of a form, mapped, and its call words; reads and
// writes are the residual streams whose input it reads and whose output it
// writes, a bit a stream.
type cpuCode struct {
	reads, writes uint64
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
	// name is the composer's name for the kernel; nanos and runs its time
	// and launches under WithProfile.
	name        string
	desc        string
	nanos, runs int64
	// pcalls, pcallNs, pstartNs and pmaxNs are its calls under WithProfile:
	// their count, time, start after the launch's and each launch's last start.
	pcalls, pcallNs, pstartNs, pmaxNs int64
	// rd and wr are the buffers the kernel's calls read and write, as the
	// call site binds them; join is how many kernels after it run in its
	// pool phase, none reading or writing what another of the phase writes,
	// and ends their calls' running totals (cpuGroup).
	rd, wr []cpuBuf
	join   int
	// chain is how many kernels after it, some reading what one before
	// writes, run in its phase unit by unit (chainPhases), of units units;
	// sliced is the rows of each buffer a call reaches alone (cpuCode.slice).
	chain, units int
	sliced       []cpuSlice
	// share is the code's hold on a weight's rows sized at the launch
	// (jit.Code.Share).
	share jit.Code
}

type cpuSlice struct {
	buf cpuBuf
	per int
}

// cpuBuf is a buffer a call word binds, by where the call site finds it.
type cpuBuf struct {
	src    uint8
	region uint8
	role   jlm.Role
	i      int
	stream int
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
	// keysEnd is the codes up to the last that writes the history: a
	// bidirectional run of more rows than a call runs them over every row
	// first (cpuRowsRun). keysX says one of them writes the residual.
	keysEnd int
	keysX   bool
	// cross is the own buffers codes[:keysEnd] write and the rest read:
	// what a bidirectional run of more rows than a call holds for each
	// call between its two passes (cpuRowsRun).
	cross []int
	// A form over banks in the host's expert pages runs in two halves
	// (cpuForm): its codes before split, up to the selection, then the rest,
	// which read the banks once the selected experts' pages are resident
	// and held. sel is the own buffer holding the selection, of selN words,
	// and wsel the picks' weights' (weightsOf), -1 where none holds them,
	// read by the codes wreads.
	split, sel, selN, wsel int
	wreads                 []int
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
	// reg32 is reg with histories kept at f32 (jit.Registry.WithoutHalves).
	reg32 *jit.Registry
	m     map[jit.Key]*cpuBlockForm
	err   map[jit.Key]error
	// planes is each block's rotary planes where a form of it reads them
	// (cpuArgPlanes), set once under mu before the block's first call.
	planes [][]float32
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
	launch    int
	launchSet bool
	blocks    []*cpuBlock
	composed  []int
	served    int
	declined  [jit.NumDeclines]int
	formErr   error
	pages     []uint32
	// fpages is pages under cpuFaultPage (cpuFaultTable).
	fpages []uint32
	rows   [4]uint32
	zero   []uint32
	// rowsTab is a call's rows' descriptors at the rows' form, call its
	// "rows.n" word, and rseq and rpos a step's rows' sequences and
	// positions.
	rowsTab []uint32
	call    [4]uint32
	// recRows is a linear block's call's recurrent descriptor: one row, the
	// slot it reads and the slot it writes, its sequence's.
	recRows [3]uint32
	// recDesc is a call of the rows' form's run descriptor (cpuRecDesc),
	// lastPos each sequence's last position a step's rows reach
	// (cpuRecWhole).
	recDesc []uint32
	lastPos []int
	// convOut is a step of several rows' convolution windows, written apart
	// and copied over the runs' slots after the call.
	convOut []float32
	// recWhole counts the steps of several rows a linear block ran on its
	// rows' form, recStepped those it stepped a row at a time.
	recWhole, recStepped int
	rseq, rpos           []int
	// sel is a call's experts as the pager's fault path takes them, picked
	// marking those already in it, by expert (cpuExperts).
	sel    []int32
	picked []bool
	// banks is each block's routed banks' planes in the host's expert
	// pages, by bank (gate, up, down) and plane (jlm.File.BankView), set at
	// the block's composition.
	banks [][3][3]unsafe.Pointer
	// xpad is a call's residual rows where its kernels' units reach past
	// them (cpuBlockForm.unit).
	xpad []float32
	// sidePad is xpad's per-layer inputs.
	sidePad []float32
	// stash is the values crossing a form's last history write, a call's
	// after another, between a bidirectional run's two passes (cpuRowsRun);
	// stashing says which pass is running.
	stash    []uint32
	stashing uint8
	// runWindows says each run of the State's (bidir) is a window its rows
	// attend inside, none of the positions before it: a tower's windowed
	// block's (cpuTowerRows).
	runWindows bool
	// own is each form's own buffers, by form: blocks of one form run one
	// after another, so they share them.
	own map[*cpuBlockForm][][]uint32
	// args is a call's words for each pool worker.
	args []cpu.Args
	// part is the workers the launch running sizes its calls for, and ends
	// its phase's kernels' calls' running totals (cpuGroup).
	part int
	ends []int
	// t0 and maxStart are the profiled launch's start and its latest call's.
	t0       time.Time
	maxStart int64
	run      cpuRegion
	bound    cpuBound
	// fault is a violation a gate sets: the bindings rewritten so a kernel
	// reads the wrong history page or the wrong first row.
	fault   cpuFault
	head    cpuHeadState
	streams cpuStreamsState
	proj    cpuProjState
	pool    cpuPoolState
}

// cpuFault is a gate's violation of the composed CPU path's bindings.
type cpuFault uint8

const (
	cpuFaultNone cpuFault = iota
	// cpuFaultPage points a layer's first page at the last slot its own
	// arena holds once that page is full (cpuFaultTable): its positions
	// read another page's (or nothing). A table permuted from the first
	// step only renames the pages, and keys carry their rotation, so a swap
	// of written pages only reorders them.
	cpuFaultPage
	// cpuFaultRow moves the first key row a step attends by one, and past
	// its second row the last by one: the step reads neither its first row
	// nor its newest, the one a block attending mostly to itself reads.
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
	// cpuFaultExpert makes resident and holds another expert's page than
	// the first the selection names: a bank's sheet read where the pager
	// holds nothing of it, or holds it evicted.
	cpuFaultExpert
	// cpuFaultStreams swaps streams 1 and 2 where the input rows' expansion
	// writes them, and reads stream 1 in stream 2's place where the head's
	// collapse reads them (a mix the same over any order of its streams sees
	// a stream read twice).
	cpuFaultStreams
	// cpuFaultEntryRows gathers each patch row of the tower's entry from the
	// patch after its own.
	cpuFaultEntryRows
	// cpuFaultKeysFirst runs a bidirectional run of several calls' first
	// pass over none of its codes: each call's rows read only the keys of
	// the calls before them and their own.
	cpuFaultKeysFirst
	// cpuFaultWrite writes a step's new key and value at the row before its
	// own, its own row left as the history held it.
	cpuFaultWrite
	// cpuFaultPicks gives a call's first row the second's picks, read with
	// its own weights.
	cpuFaultPicks
	// cpuFaultWeight drops each row's first pick's weight (weightsOf).
	cpuFaultWeight
	// cpuFaultFilterT binds a convolutional block's depthwise filters with
	// their taps' rows and columns swapped.
	cpuFaultFilterT
)

// A bidirectional run's passes (cpuState.stashing).
const (
	stashNone uint8 = iota
	stashSave
	stashLoad
)

// cpuBlock is block li's forms on the CPU, its graph the one they were
// asked for: decode's, of one row, and the rows', at the CPU's capacity
// (cpuRowsCap), whose calls run the rows their "rows.n" word states.
type cpuBlock struct {
	graph *graph.Graph
	f     *cpuBlockForm
	rows  *cpuBlockForm
	// sideOff is where the block's slice of the rows' per-layer inputs
	// starts in each row: the call binds the side input there, so blocks
	// that differ by their slice alone share a form.
	sideOff int
	why     jit.Decline
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
	// g is a phase's kernels (cpuGroup), gfn the region that runs them;
	// cfn runs them unit by unit, units units (cpuChain).
	g     []cpuCode
	gfn   func(worker, lo, hi int)
	cfn   func(worker, lo, hi int)
	units int
}

// cpuBound is a block's step as its words read it: n rows of the residual
// x, their rotary tables cs and their descriptors rows, their history the
// layer's pages kv through the page table pages.
type cpuBound struct {
	li  int
	l   *layer
	own [][]uint32
	cs  []float32
	x   []float32
	// stride is the elements from one residual stream of x to the next.
	stride int
	rows   []uint32
	n      int
	kv     *kvPages
	pages  []uint32
	// recLi is the block whose recurrent states a linear block's step binds
	// and recOf the State that holds them.
	recLi int
	recOf *State
	// rec, where set, is the call's run descriptor in place of recRows.
	rec []uint32
	// side is the bound rows' per-layer inputs, a row each (cpuArgSide).
	side []float32
	// io is the buffers of a graph apart from its residual (cpuArgIO), its
	// inputs' then its outputs'.
	io [cpuIO]unsafe.Pointer
	// req is the step's request words (cpuArgRequest), pos its live rows,
	// then a row's position (cpuArgPos).
	req, pos []uint32
}

// cpuIO is the most inputs and outputs a graph apart from its residual
// binds: the sampler's candidates read three and write two.
const cpuIO = 6

// composedCPUDevice is the CPU as the composer forms for it.
func composedCPUDevice(workers int) jit.Device {
	return jit.Device{Kind: jit.DeviceCPU, Slots: max(workers, 1), Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
}

// composedCPU is this State's CPU as the composer forms for it, its launch
// cost its probe row's.
func (s *State) composedCPU() jit.Device {
	workers := 1
	if s.jit != nil {
		workers = s.jit.Workers()
	}
	d := composedCPUDevice(workers)
	if !s.cpu.launchSet {
		if name, err := probe.HostName(); err == nil {
			if p, ok := hw.Known(name, runtime.GOARCH); ok {
				s.cpu.launch = p.Device.LaunchWords()
			}
		}
		s.cpu.launchSet = true
	}
	d.Launch = s.cpu.launch
	return d
}

// cpuFaultRowWords applies the State's key-row fault to one row's
// descriptor (first key, end, write row), the step's first row left whole.
func (s *State) cpuFaultRowWords(t []uint32) {
	if t[3] == 0 {
		return
	}
	switch s.cpu.fault {
	case cpuFaultRow:
		t[1]++
		if t[2] > t[1]+1 {
			t[2]--
		}
	case cpuFaultWrite:
		t[3]--
	}
}

// cpuStep runs block li on the CPU the composer's way, or says why not. The
// residual is s.x; the rotary table cs.
func (s *State) cpuStep(li int, l *layer, cs []float32) (jit.Decline, error) {
	cb, why := s.cpuBlockOf(li)
	if why != jit.DeclineServes {
		return why, nil
	}
	// A block sharing another's history reads its source's pages.
	hl := s.c.KVSource(li)
	pg := &s.kv.layers[hl]
	pos := s.pos
	if pg.p > 0 {
		if why := s.cpuHistory(hl, pos); why != jit.DeclineServes {
			return why, nil
		}
	}
	b := &s.cpu.bound
	*b = cpuBound{li: li, l: l, own: s.cpuOwn(cb.f), cs: cs, x: s.resid(), stride: s.c.NEmbd, rows: s.cpu.rows[:], n: 1, kv: pg, pages: s.cpu.pages,
		recLi: s.cpuRecLi(li), recOf: s, side: s.ple}
	if b.side != nil {
		b.side = b.side[cb.sideOff:]
	}
	if why := s.cpuWeights(cb.f, li, l); why != jit.DeclineServes {
		return why, nil
	}
	s.cpu.rows = [4]uint32{0, uint32(pg.keyStart(pos)), uint32(pos + 1), uint32(pos)}
	s.cpu.recRows = [3]uint32{1, 0, 0}
	s.cpu.call[0] = 1
	switch s.cpu.fault {
	case cpuFaultPage:
		b.pages = s.cpuFaultTable(pg, pos+1)
	case cpuFaultRow, cpuFaultWrite:
		s.cpuFaultRowWords(s.cpu.rows[:])
	}
	if err := s.cpuForm(b, cb.f, li); err != nil {
		return jit.DeclineServes, err
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
	return jit.DeclineServes, s.cpuRowsRun(li, l, f, x, len(pos)*s.c.NEmbd, s.bple, cs, nrot, seq, pos)
}

// cpuTowerRows runs tower block li on the CPU the composer's way over the
// picture's n rows in s.bx, one run of rows attending both ways (bidir.go),
// or says why not. Its history is the State's arena page, lent to the block
// for its run.
func (s *State) cpuTowerRows(li int, l *layer, n int) (jit.Decline, error) {
	v := s.vis
	var wins []nn.KeyRun
	if s.visWindowed(li) {
		v.winRuns = v.winRuns[:0]
		for w := 0; w+1 < len(v.segs); w += 2 {
			v.winRuns = append(v.winRuns, nn.KeyRun{Lo: v.segs[w], Hi: v.segs[w+1]})
		}
		wins = v.winRuns
	}
	cs, nrot := s.cpuTable(li, s.bcs, nil)
	return s.cpuRunRows(li, l, s.bx[:n*s.c.ResidW()], n, cs, nrot, wins)
}

// cpuRunRows runs non-causal block li on the CPU the composer's way over
// the n rows of x, one run attending both ways or, where wins is not nil,
// each window a run its rows attend inside; cs is the rows' rotary table,
// nrot wide. Its history is the segment's arena (ncArena), lent to the
// block for its run.
func (s *State) cpuRunRows(li int, l *layer, x []float32, n int, cs []float32, nrot int, wins []nn.KeyRun) (jit.Decline, error) {
	f, why := s.cpuRowsForm(li, l)
	if why != jit.DeclineServes {
		return why, nil
	}
	if n > f.keys {
		return jit.DeclineSessions, nil
	}
	s.cpuPages(1)
	a := s.nc()
	a.run[0] = nn.KeyRun{Lo: 0, Hi: n}
	runs := a.run[:]
	if wins != nil {
		runs, s.cpu.runWindows = wins, true
	}
	if !a.causalFault {
		s.bidir = runs
	}
	seq, pos := s.cpuRowsOf(0, 0, n)
	err := s.cpuRowsRun(li, l, f, x, n*s.c.NEmbd, nil, cs, nrot, seq, pos)
	s.bidir, s.cpu.runWindows = nil, false
	return jit.DeclineServes, err
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
	why, err := s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, 0, n, false)
	if why != jit.DeclineServes || err != nil {
		return why, err
	}
	if why, err = s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, nlogit, n, true); err != nil {
		return why, err
	}
	return s.cpuOwnerRuns(li, l, x, cs, nrot, seq, pos, 0, nlogit, true)
}

// cpuOwnerRuns makes ready (run unset) or runs block li over the rows
// [lo, hi) of a step across sessions, a run of one session's consecutive
// rows at a time, on that session's form.
func (s *State) cpuOwnerRuns(li int, l *layer, x, cs []float32, nrot int, seq, pos []int, lo, hi int, run bool) (jit.Decline, error) {
	d, pw := s.c.NEmbd, s.c.pleWidth()
	for a := lo; a < hi; {
		o, b := s.rowOwn[a], a+1
		for b < hi && s.rowOwn[b] == o {
			b++
		}
		if o.m != s.m || !o.m.opt.composedCPU {
			return jit.DeclineSessions, nil
		}
		f, why := o.cpuRowsForm(li, l)
		if why != jit.DeclineServes {
			return why, nil
		}
		if run {
			var side []float32
			if pw != 0 {
				side = s.bple[a*pw:]
			}
			if err := o.cpuRowsRun(li, l, f, x[a*d:], len(pos)*d, side, cs[a*nrot:], nrot, seq[a:b], pos[a:b]); err != nil {
				return jit.DeclineServes, err
			}
		} else if why := o.cpuRowsReady(li, f, pos[a:b]); why != jit.DeclineServes {
			return why, nil
		}
		a = b
	}
	return jit.DeclineServes, nil
}

// cpuRecRows runs linear block li over the n rows of x, or says why not:
// on its rows' form, the scan stepping each sequence's rows in order, where
// the step's rows of each sequence stand in order and all are this State's
// (cpuRecWhole); else on its form of one row, a row at a time in order
// (each row's index; 0.. where order is nil), each stepping its own
// sequence's recurrent state -- its owner's, as rowOwner names it. A
// prompt's rows (base its first row in the prompt, p0 its first position;
// base -1 for none) keep each row's summary as main's row loop does: the
// speculative rows' snapshot, which takes the row at a time, and the
// restore points, where the rows' form ends a call.
func (s *State) cpuRecRows(li int, l *layer, x, cs []float32, seq, pos, order []int, base, p0 int) (jit.Decline, error) {
	cb, why := s.cpuBlockOf(li)
	if why != jit.DeclineServes {
		return why, nil
	}
	if f := s.cpuRecWhole(li, l, cb, seq, pos, order, base); f != nil {
		why, err := s.cpuRecRun(li, l, f, x, cs, seq, pos, base, p0)
		if why == jit.DeclineServes && err == nil {
			s.cpu.recWhole++
		}
		return why, err
	}
	if len(pos) > 1 {
		s.cpu.recStepped++
	}
	if why := s.cpuWeights(cb.f, li, l); why != jit.DeclineServes {
		return why, nil
	}
	// x is stream-major: row r's stream k at (k*rows + r)*n_embd.
	d, nrot := s.c.NEmbd, s.c.RopeW()
	stride := len(x) / s.c.Streams()
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
		*b = cpuBound{li: li, l: l, own: s.cpuOwn(cb.f), x: x[r*d:], stride: stride, rows: s.cpu.rows[:], n: 1, kv: &s.kv.layers[li],
			pages: s.cpu.pages, recLi: s.cpuRecLi(li), recOf: o}
		if nrot > 0 {
			b.cs = cs[r*nrot:]
		}
		s.cpu.rows = [4]uint32{0, uint32(pg.keyStart(pos[r])), uint32(pos[r] + 1), uint32(pos[r])}
		s.cpu.recRows = [3]uint32{1, uint32(slot), uint32(slot)}
		s.cpu.call[0] = 1
		if err := s.cpuForm(b, cb.f, li); err != nil {
			return jit.DeclineServes, err
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

// cpuRecWhole is block li's rows' form where the step's rows run on it
// whole: several rows, all this State's, in causal order, each sequence's
// rows in order of position, and no speculative snapshot to take after
// each row; nil where not.
func (s *State) cpuRecWhole(li int, l *layer, cb *cpuBlock, seq, pos, order []int, base int) *cpuBlockForm {
	if len(pos) < 2 || cb.rows == nil || len(s.rowOwn) > 0 || base >= 0 && s.rows != nil {
		return nil
	}
	for i, r := range order {
		if r != i {
			return nil
		}
	}
	if len(s.cpu.lastPos) < max(s.nseq, 1) {
		s.cpu.lastPos = make([]int, max(s.nseq, 1))
	}
	last := s.cpu.lastPos[:max(s.nseq, 1)]
	for i := range last {
		last[i] = -1
	}
	for i, q := range seq {
		if q < 0 || q >= len(last) || pos[i] <= last[q] {
			return nil
		}
		last[q] = pos[i]
	}
	if s.cpuWeights(cb.rows, li, l) != jit.DeclineServes {
		return nil
	}
	return cb.rows
}

// cpuRecRun runs linear block li's rows' form f over the step's rows, a
// call at most f's rows; a prompt's (base its first row in the prompt, p0
// its first position) end a call at each restore point inside them and seal
// it there.
func (s *State) cpuRecRun(li int, l *layer, f *cpuBlockForm, x, cs []float32, seq, pos []int, base, p0 int) (jit.Decline, error) {
	if why := s.cpuRowsReady(li, f, pos); why != jit.DeclineServes {
		return why, nil
	}
	n, d, nrot, pw := len(pos), s.c.NEmbd, s.c.RopeW(), s.c.pleWidth()
	if s.cpu.fault == cpuFaultChunk && base > 0 {
		slot := seq[0]
		clear(s.rstate[li][slot*s.dg.stateStride():][:s.dg.stateStride()])
		clear(s.rconv[li][slot*s.dg.convStateLen():][:s.dg.convStateLen()])
	}
	points := base >= 0 && s.recPointsOn()
	for a := 0; a < n; {
		b := n
		for e := a + 1; points && e < n; e++ {
			if s.recPoint(p0 + e) {
				b = e
				break
			}
		}
		side := s.bple
		if pw != 0 && side != nil {
			side = side[a*pw:]
		}
		if err := s.cpuRowsRun(li, l, f, x[a*d:], n*d, side, cs[a*nrot:], nrot, seq[a:b], pos[a:b]); err != nil {
			return jit.DeclineServes, err
		}
		if base >= 0 {
			s.recRowPoint(li, seq[a], p0+b)
		}
		a = b
	}
	return jit.DeclineServes, nil
}

// cpuRecDesc is the run descriptor (kernels.RunDescWords' layout) of a
// call's rows of sequences seq: a run a sequence, its rows in the call's
// order, its slot read and written the sequence's; the entries past the
// runs repeat the last.
func (s *State) cpuRecDesc(seq []int) []uint32 {
	k := len(seq)
	if len(s.cpu.recDesc) < 1+7*k {
		s.cpu.recDesc = make([]uint32, 1+7*max(k, cpuRowsCap))
	}
	w := s.cpu.recDesc[:1+7*k]
	w[0] = uint32(k)
	at, runs := 0, 0
	for i, q := range seq {
		if slices.Contains(seq[:i], q) {
			continue
		}
		start := at
		for r := i; r < k; r++ {
			if seq[r] == q {
				w[1+4*k+at], w[1+5*k+r], w[1+6*k+r] = uint32(r), uint32(at), uint32(runs)
				at++
			}
		}
		w[1+2*runs], w[2+2*runs] = uint32(q), uint32(q)
		w[1+2*k+2*runs], w[2+2*k+2*runs] = uint32(start), uint32(at-start)
		runs++
	}
	for j := runs; j < k; j++ {
		copy(w[1+2*j:3+2*j], w[2*runs-1:2*runs+1])
		copy(w[1+2*k+2*j:3+2*k+2*j], w[2*k+2*runs-1:2*k+2*runs+1])
	}
	return w
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
	if why := s.cpuWeights(cb.rows, li, l); why != jit.DeclineServes {
		return nil, why
	}
	return cb.rows, jit.DeclineServes
}

// cpuRowsReady makes the pages of the history the rows at pos write and
// read resident, in block li's arena, or says why f cannot run them: a
// history past f's key bound, a page outside the arena.
func (s *State) cpuRowsReady(li int, f *cpuBlockForm, pos []int) jit.Decline {
	hl := s.c.KVSource(li)
	pg := &s.kv.layers[hl]
	if pg.p == 0 {
		return jit.DeclineServes
	}
	// Every row's page is written before any is read: a step's rows stand
	// in any order, and a page holds every sequence's positions.
	start, end := pos[0], 0
	for _, p := range pos {
		s.cpuWritable(hl, p)
		start, end = min(start, pg.keyStart(p)), max(end, p+1)
	}
	if end > f.keys || max(pg.arenaSlots(), 1)*max(s.nseq, 1) > len(s.cpu.pages) {
		return jit.DeclineSessions
	}
	if why := s.cpuReadable(hl, start, end-1); why != jit.DeclineServes {
		return why
	}
	return jit.DeclineServes
}

// cpuFaultTable is the page table a step of the layer whose pages pg
// holds reads, its rows' history ending before end: under cpuFaultPage,
// once the layer's first page is full, a table whose first page is the last
// slot the layer's own arena holds, where that arena holds more than two --
// its positions read another page's (or nothing). Each layer is pointed at
// a slot its own arena holds, so a model whose arenas differ (a sliding
// layer's beside a global one's) is reached.
func (s *State) cpuFaultTable(pg *kvPages, end int) []uint32 {
	n := pg.arenaSlots() * max(s.nseq, 1)
	if s.cpu.fault != cpuFaultPage || n <= 2 || n > len(s.cpu.pages) || pg.p == 0 || end <= pg.p+pg.p/4 {
		return s.cpu.pages
	}
	if len(s.cpu.fpages) != len(s.cpu.pages) {
		s.cpu.fpages = make([]uint32, len(s.cpu.pages))
	}
	copy(s.cpu.fpages, s.cpu.pages)
	s.cpu.fpages[0] = s.cpu.pages[n-1]
	return s.cpu.fpages
}

// cpuRowsRun runs block li's rows' form f over the rows of x, their history
// the State's, a call of at most f's capacity at a time. A bidirectional
// run (bidir.go) counts its rows' keys to its end; where the step takes
// more calls than one, the form's codes through its last history write
// (keysEnd) first run over every row, each reading only the keys before
// it, so that every call's rows then read all of them. Where those codes
// write the residual too they run over a copy of it.
func (s *State) cpuRowsRun(li int, l *layer, f *cpuBlockForm, x []float32, stride int, side, cs []float32, nrot int, seq, pos []int) error {
	runs := s.runsAt(li) && s.bidirIn(pos[0], len(pos))
	if runs && len(pos) > f.rows {
		keys := f.keysEnd
		if s.cpu.fault == cpuFaultKeysFirst {
			keys = 0
		}
		s.cpu.stashing = stashSave
		err := s.cpuRowsCalls(li, l, f, 0, keys, x, stride, side, cs, nrot, seq, pos, true)
		s.cpu.stashing = stashLoad
		if err == nil {
			err = s.cpuRowsCalls(li, l, f, keys, len(f.codes), x, stride, side, cs, nrot, seq, pos, true)
		}
		s.cpu.stashing = stashNone
		return err
	}
	return s.cpuRowsCalls(li, l, f, 0, len(f.codes), x, stride, side, cs, nrot, seq, pos, runs)
}

// cpuStash saves (stashSave) or restores (stashLoad) call c's values form
// f's codes before keysEnd write and the rest read (cpuBlockForm.cross).
func (s *State) cpuStash(f *cpuBlockForm, own [][]uint32, c int) {
	at := 0
	for _, i := range f.cross {
		at += len(own[i]) * c
	}
	for _, i := range f.cross {
		n := len(own[i])
		if end := at + n; len(s.cpu.stash) < end {
			s.cpu.stash = append(s.cpu.stash, make([]uint32, end-len(s.cpu.stash))...)
		}
		if s.cpu.stashing == stashSave {
			copy(s.cpu.stash[at:at+n], own[i])
		} else {
			copy(own[i], s.cpu.stash[at:at+n])
		}
		at += n
	}
}

// cpuRowsCalls is cpuRowsRun's calls of f's codes [lo, hi), each row's keys
// counted to its run's end where runs is set; x is the rows' residual,
// stream-major, stride elements from one stream to the next; side is the
// rows' per-layer inputs, nil for a model of none. A form over banks
// (cpuBlockForm.split) holds the experts each call's rows pick, from its
// selection to the call's end (cpuExperts).
func (s *State) cpuRowsCalls(li int, l *layer, f *cpuBlockForm, lo, hi int, x []float32, stride int, side, cs []float32, nrot int, seq, pos []int, runs bool) error {
	n := len(pos)
	pg := &s.kv.layers[s.c.KVSource(li)]
	d, ns, pw := s.c.NEmbd, s.c.Streams(), s.c.pleWidth()
	so := s.cpuSideOff(li)
	slots := max(pg.arenaSlots(), 1)
	b := &s.cpu.bound
	end := 0
	for r0 := 0; r0 < n; r0 += f.rows {
		k := min(f.rows, n-r0)
		for i := range k {
			p := pos[r0+i]
			e, st := p+1, pg.keyStart(p)
			if runs {
				e = max(e, s.bidirEnd(p))
				if s.cpu.runWindows {
					st = max(st, s.bidirStart(p))
				}
			}
			end = max(end, p+1)
			t := s.cpu.rowsTab[i*4 : i*4+4]
			t[0], t[1], t[2], t[3] = uint32(seq[r0+i]*slots), uint32(st), uint32(e), uint32(p)
			s.cpuFaultRowWords(t)
		}
		bx, bstride := x[r0*d:], stride
		var bside []float32
		if side != nil {
			bside = side[r0*pw:]
		}
		staged := f.unit > 1 && k%f.unit != 0
		if staged {
			if len(s.cpu.xpad) < f.rows*d*ns {
				s.cpu.xpad = make([]float32, f.rows*d*ns)
			}
			for j := range ns {
				copy(s.cpu.xpad[j*f.rows*d:], bx[j*stride:j*stride+k*d])
			}
			bx, bstride = s.cpu.xpad, f.rows*d
			if side != nil {
				if len(s.cpu.sidePad) < f.rows*pw {
					s.cpu.sidePad = make([]float32, f.rows*pw)
				}
				copy(s.cpu.sidePad, bside[:k*pw])
				bside = s.cpu.sidePad
			}
		}
		if bside != nil {
			bside = bside[so:]
		}
		*b = cpuBound{li: li, l: l, own: s.cpuOwn(f), cs: cs[r0*nrot:], x: bx, stride: bstride, rows: s.cpu.rowsTab, n: k,
			kv: pg, pages: s.cpuFaultTable(pg, end), side: bside}
		rec := s.c.LayerKind(li).Recurrent()
		if rec {
			b.recLi, b.recOf, b.rec = s.cpuRecLi(li), s, s.cpuRecDesc(seq[r0:r0+k])
			if n := len(s.rconv[b.recLi]); len(s.cpu.convOut) < n {
				s.cpu.convOut = make([]float32, n)
			}
		}
		s.cpu.call[0] = uint32(k)
		if s.cpu.stashing == stashLoad {
			s.cpuStash(f, b.own, r0/f.rows)
		}
		for i := lo; i < hi; i++ {
			if i == f.split && f.split > 0 {
				if err := s.cpuExperts(b, f, li, k); err != nil {
					s.expHold.release()
					return err
				}
			}
			if s.cpu.fault == cpuFaultWeight && slices.Contains(f.wreads, i) {
				// Each row's first pick weighted by zero where a kernel
				// reads the weights.
				w := f.selN / f.rows
				for r := range k {
					b.own[f.wsel][r*w] = 0
				}
			}
			s.cpuCall(b, &f.codes[i])
		}
		s.expHold.release()
		if rec {
			// The runs' windows over their slots.
			L := s.dg.convStateLen()
			for i, q := range seq[r0 : r0+k] {
				if !slices.Contains(seq[r0:r0+i], q) {
					copy(s.rconv[b.recLi][q*L:(q+1)*L], s.cpu.convOut[q*L:(q+1)*L])
				}
			}
		}
		if s.cpu.stashing == stashSave {
			s.cpuStash(f, b.own, r0/f.rows)
		}
		if staged {
			for j := range ns {
				copy(x[j*stride+r0*d:j*stride+(r0+k)*d], s.cpu.xpad[j*f.rows*d:j*f.rows*d+k*d])
			}
		}
	}
	if pg.p > 0 && end > pg.written {
		pg.written = end
	}
	return nil
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
func (s *State) cpuWeights(f *cpuBlockForm, li int, l *layer) jit.Decline {
	for i := range f.codes {
		for _, a := range f.codes[i].args {
			switch a.src {
			case cpuArgWeight:
				if jlm.ExpertBank(a.role) {
					// A bank is read in the host's expert pages (cpuBanksPaged).
					if li >= len(s.cpu.banks) || s.cpu.banks[li][a.role-jlm.RoleExpGateBank][a.region] == nil {
						return jit.DeclineNoWeight
					}
					continue
				}
				w := s.cpuRoleWeight(li, l, a.role)
				if w == nil || w.packed == nil && len(w.data) == 0 {
					return jit.DeclineNoWeight
				}
				if p := w.packed; p != nil && (p.Row != 0 || p.Stride != 0 && p.Stride != w.rows || p.Sheet != nil) {
					return jit.DeclineCutWeight
				}
			case cpuArgVector:
				if len(s.cpuRoleVector(li, l, a.role)) == 0 {
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
	if s.prof {
		t0 := time.Now()
		s.cpu.t0, s.cpu.maxStart = t0, 0
		s.cpuLaunch(b, c)
		c.nanos += int64(time.Since(t0))
		c.runs++
		c.pmaxNs += s.cpu.maxStart
		return
	}
	s.cpuLaunch(b, c)
}

func (s *State) cpuLaunch(b *cpuBound, c *cpuCode) {
	n, live := c.n, 0
	if c.live.Live > 0 {
		live = c.live.LiveUnitsOf(b.n)
		n = live * c.inner
	}
	if !c.tiles {
		live = 0
	}
	s.cpu.part = s.Workers()
	if c.serial {
		s.cpu.part = 1
	}
	if c.share.Share > 0 {
		_, n = c.share.ShareRows(s.cpu.part)
	}
	if n <= 1 || c.serial || s.jit == nil || s.jit.Workers() <= 1 {
		a := &s.cpu.args[0]
		for i := range max(n, 1) {
			s.cpuWords(a, b, c, c.callAt(i, live))
			c.code.Call(a)
		}
		return
	}
	// A prefill's pool may be wider than the decode pool the words were
	// sized for at placement (the P and E cores'): a word block for each of
	// its workers.
	if w := s.jit.Workers(); len(s.cpu.args) < w {
		s.cpu.args = append(s.cpu.args, make([]cpu.Args, w-len(s.cpu.args))...)
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
	rows, _ := c.share.ShareRows(s.cpu.part)
	for i := range c.args {
		ar := &c.args[i]
		var p unsafe.Pointer
		switch ar.src {
		case cpuArgValue:
			*(*uint64)(unsafe.Add(base, ar.field)) = ar.val
			continue
		case cpuArgShare:
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(min(rows, c.share.Share-at*rows))
			continue
		case cpuArgCoord:
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(c.coord(at, int(ar.coord), rows))
			continue
		case cpuArgWord:
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(b.own[ar.i][ar.off])
			continue
		case cpuArgX:
			p = unsafe.Pointer(&b.x[ar.stream*b.stride])
		case cpuArgIO:
			p = b.io[ar.i]
		case cpuArgSide:
			p = unsafe.Pointer(&b.side[0])
		case cpuArgOwn:
			p = unsafe.Pointer(&b.own[ar.i][0])
		case cpuArgWeight:
			if jlm.ExpertBank(ar.role) {
				p = s.cpu.banks[b.li][ar.role-jlm.RoleExpGateBank][ar.region]
				break
			}
			w := s.cpuRoleWeight(b.li, b.l, ar.role)
			if w.packed == nil {
				p = unsafe.Pointer(&w.data[0])
			} else {
				p = unsafe.Pointer(&[3][]byte{w.packed.QS, w.packed.D, w.packed.SC}[ar.region][0])
			}
		case cpuArgVector:
			p = unsafe.Pointer(&s.cpuRoleVector(b.li, b.l, ar.role)[0])
		case cpuArgRope:
			p = unsafe.Pointer(&b.cs[0])
		case cpuArgPlanes:
			p = unsafe.Pointer(&s.m.cpuForms.planes[b.li][0])
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
		case cpuArgRecConvOut:
			p = unsafe.Pointer(&b.recOf.rconv[b.recLi][0])
			if b.rec != nil {
				p = unsafe.Pointer(&s.cpu.convOut[0])
			}
		case cpuArgPoolPos:
			p = unsafe.Pointer(&s.cpu.pool.pos[0])
		case cpuArgRecRows:
			p = unsafe.Pointer(&s.cpu.recRows[0])
			if b.rec != nil {
				p = unsafe.Pointer(&b.rec[0])
			}
		case cpuArgZero:
			p = unsafe.Pointer(&s.cpu.zero[0])
		case cpuArgRequest:
			p = unsafe.Pointer(&b.req[0])
		case cpuArgPos:
			p = unsafe.Pointer(&b.pos[0])
		}
		off := ar.off
		if ar.moves {
			for ax := range c.calls {
				if ax < len(ar.per) && ar.per[ax] != 0 {
					off += c.coord(at, ax, rows) * ar.per[ax]
				}
			}
		}
		*(*uint64)(unsafe.Add(base, ar.field)) = uint64(uintptr(p)) + uint64(off*ar.elem)
	}
}

// coord is call at's coordinate along axis ax: a Share code's first row
// for calls of rows rows, else cpuCoord's.
func (c *cpuCode) coord(at, ax, rows int) int {
	if rows > 0 {
		return at * rows / max(c.share.ShareUnit, 1)
	}
	return cpuCoord(c.calls, at, ax)
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
		if r.s.prof {
			r.s.cpuProfCall(r.c, a)
			continue
		}
		r.c.code.Call(a)
	}
}

func (s *State) cpuProfCall(c *cpuCode, a *cpu.Args) {
	t := time.Now()
	c.code.Call(a)
	off := int64(t.Sub(s.cpu.t0))
	atomic.AddInt64(&c.pcalls, 1)
	atomic.AddInt64(&c.pcallNs, int64(time.Since(t)))
	atomic.AddInt64(&c.pstartNs, off)
	for {
		m := atomic.LoadInt64(&s.cpu.maxStart)
		if off <= m || atomic.CompareAndSwapInt64(&s.cpu.maxStart, m, off) {
			return
		}
	}
}

// groupRegion runs the calls [lo, hi) of the phase's kernels on worker w's
// words.
func (r *cpuRegion) groupRegion(w, lo, hi int) {
	r.s.cpuGroupRun(&r.s.cpu.args[w], r.b, r.g, lo, hi)
}

// cpuPlace composes a new State's blocks on the CPU: each block of the
// State's range no device holds, once, at its placement (cpuCompose).
func (s *State) cpuPlace() {
	if !s.m.opt.composedCPU {
		return
	}
	// A segment's blocks are past the text model's in the one block space.
	n := max(s.c.NLayer, s.hi)
	s.cpu.blocks = make([]*cpuBlock, n)
	s.cpu.composed = make([]int, n)
	w := 1
	if s.jit != nil {
		w = s.jit.Workers()
	}
	s.cpu.args = make([]cpu.Args, max(w, 1))
	s.cpu.ends = make([]int, 0, 16)
	s.cpu.run.fn = s.cpu.run.region
	s.cpu.run.gfn = s.cpu.run.groupRegion
	s.cpu.run.cfn = s.cpu.run.chainRegion
	s.cpu.zero = make([]uint32, 4)
	for li := s.lo; li < min(s.hi, n); li++ {
		if !s.devAt(li) {
			s.cpuCompose(li)
		}
	}
	if s.nc() != nil {
		if s.vis != nil {
			s.cpuPlaceProjector()
		}
		if s.enc != nil {
			s.cpuPlacePool()
		}
		return
	}
	s.cpuPlaceHead()
	s.cpuPlaceStreams()
	s.cpuPlacePool()
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
	if s.m.layers[li].mn != nil {
		s.cpuComposeConv(li, cb)
		return
	}
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
	if cb.rows, err = s.m.formCPUBlock(s, li, cb, cpuRowsCap); err != nil {
		cb.rowsWhy = s.cpuFormDecline(err)
	}
	s.cpu.composed[li]++
	s.cpuPages(s.kv.layers[s.c.KVSource(li)].arenaSlots())
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
	// A block sharing another's history reads its source's pages.
	pg := &s.kv.layers[c.KVSource(li)]
	if s.nc() != nil {
		return s.cpuTowerWhy(g)
	}
	switch {
	case !jit.HostCPULowers():
		return jit.DeclineISA
	case g == nil:
		return jit.DeclineNoGraph
	case len(g.Out) != c.Streams() || len(g.In) > len(g.Out)+1 || len(g.In) > len(g.Out) && (c.PLEDim == 0 || s.ple == nil):
		return jit.DeclineFeeds
	case c.MoEAt(li) && !s.cpuBanksPaged(li):
		// The forms read a bank as one view of its experts in the host's
		// expert pages (jlm.File.BankView); a bank with none is not read.
		return jit.DeclineExperts
	case c.LayerKind(li).Recurrent() && (li >= len(s.rstate) || s.rstate[li] == nil || s.rconv[li] == nil):
		return jit.DeclineRecurrent
	case s.kvl.headMajor:
		return jit.DeclineHeadMajor
	case s.kvl.fmt != cpu.KVF16 && s.kvl.fmt != cpu.KVF32:
		// The CPU's profile keeps a history binary16 itself (jit.Halves),
		// or at f32 without that pass.
		return jit.DeclineKVFormat
	case pg.p > 0 && pg.arena == nil:
		return jit.DeclinePageOut
	}
	return jit.DeclineServes
}

// sideAtBinding moves the offset of every slice of gr's side input (its
// input past its residual streams) into the input's binding: where each
// slice reads from one offset, the slices read from 0 and the offset is
// returned for the call to bind the input at. A graph whose slices of the
// side differ, or that reads it otherwise, keeps its offsets (0).
func sideAtBinding(gr *graph.Graph) int {
	if len(gr.In) != len(gr.Out)+1 {
		return 0
	}
	side := gr.In[len(gr.Out)]
	off := -1
	for _, o := range gr.Ops {
		if !slices.Contains(o.In, side) {
			continue
		}
		if o.Kind != graph.OpSlice || o.Attr.HeadDim != 0 || off >= 0 && o.Attr.Offset != off {
			return 0
		}
		off = o.Attr.Offset
	}
	if off <= 0 {
		return 0
	}
	gr.Ops = slices.Clone(gr.Ops)
	for i := range gr.Ops {
		if slices.Contains(gr.Ops[i].In, side) {
			gr.Ops[i].Attr.Offset = 0
		}
	}
	return off
}

// cpuSideOff is block li's sideOff, 0 for a block not composed.
func (s *State) cpuSideOff(li int) int {
	if li < len(s.cpu.blocks) && s.cpu.blocks[li] != nil {
		return s.cpu.blocks[li].sideOff
	}
	return 0
}

// cpuTowerWhy is cpuWhy for a block of a non-causal segment (a tower's or an
// encoder's): its history is the block's run's own, written and read inside
// the one call over the segment's rows (lendNC), so no other path reads its
// format, and its page is the State's arena.
func (s *State) cpuTowerWhy(g *graph.Graph) jit.Decline {
	switch {
	case !jit.HostCPULowers():
		return jit.DeclineISA
	case g == nil:
		return jit.DeclineNoGraph
	case len(g.Out) != 1 || len(g.In) != 1:
		return jit.DeclineFeeds
	case s.nc().arena == nil:
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
	cb.sideOff = sideAtBinding(&gr)
	formats := map[jlm.Role]jlm.Type{}
	f := m.Composing()
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if m.towerAt(li) >= 0 {
				if t, ok := towerFormat(&m.layers[li], w.Role); ok {
					formats[w.Role] = t
				}
				continue
			}
			if e, ok := f.Find(w.Role, int32(li), -1); ok {
				formats[w.Role] = e.Type
			}
		}
	}
	pg := &s.kv.layers[li]
	page := max(pg.p, 1)
	if a := s.nc(); a != nil {
		page = a.page
	}
	req := jit.Request{Graph: &gr, Formats: formats, Strides: m.cpuStrides(li, g),
		Step:    jit.Step{Rows: rows, Page: page, Keys: max(pg.arenaSlots(), 1) * page, ActWindow: s.jit.ActWindow(), Seqs: max(s.nseq, 1)},
		History: jit.History{ValuesInKeys: pg.latent},
		Device:  s.composedCPU()}
	fm, err := m.cpuFormOf(req, &gr, rows, false, false, s.kvl.fmt == cpu.KVF32)
	if err == nil && readsPlanes(g) && !m.cpuPlanes(li) {
		return nil, cpuDeclineError{jit.DeclineTable, fmt.Errorf("block %d turns by a rotary with no planes", li)}
	}
	return fm, err
}

// readsPlanes reports whether a form of g reads the rotary's planes: a
// pooled selection turning its blocks, or a rotary at positions of its own.
func readsPlanes(g *graph.Graph) bool {
	return slices.ContainsFunc(g.Ops, func(o graph.Op) bool {
		return o.Kind == graph.OpSelect && o.Attr.Pooled && o.Attr.NRot > 0 || o.Kind == graph.OpRope && o.Attr.Shift > 0
	})
}

// cpuPlanes sets block li's rotary planes (cpuArgPlanes) where they are not,
// and reports whether the block has them.
func (m *Model) cpuPlanes(li int) bool {
	c := &m.cpuForms
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.planes == nil {
		c.planes = make([][]float32, len(m.layers))
	}
	if li >= len(c.planes) {
		return false
	}
	if c.planes[li] == nil {
		r := m.ropeAt(li)
		c.planes[li] = r.TabPlanes(r.NRot / 2)
	}
	return len(c.planes[li]) > 0
}

// cpuFormOf is req's form on the CPU, mapped once for the model by its Key;
// whole is buildCPUForm's.
func (m *Model) cpuFormOf(req jit.Request, gr *graph.Graph, rows int, apart, whole, f32 bool) (*cpuBlockForm, error) {
	c := &m.cpuForms
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.reg == nil {
		c.reg = jit.Default()
		c.reg32 = c.reg.WithoutHalves(jit.DeviceCPU)
	}
	if c.m == nil {
		c.m, c.err = map[jit.Key]*cpuBlockForm{}, map[jit.Key]error{}
	}
	reg := c.reg
	if f32 {
		reg = c.reg32
	}
	key, codes, err := reg.Form(req)
	if err != nil {
		return nil, err
	}
	if fm, ok := c.m[key]; ok {
		return fm, nil
	}
	if err, ok := c.err[key]; ok {
		return nil, err
	}
	fm, err := buildCPUForm(gr, codes, rows, apart, whole)
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
// A whole form is run at its rows and no fewer (a convolutional block's,
// over its grid), so its kernels need not be held to a call's rows.
func buildCPUForm(gr *graph.Graph, codes []jit.Code, rows int, apart, whole bool) (*cpuBlockForm, error) {
	if apart && len(gr.In)+len(gr.Out) > cpuIO {
		return nil, fmt.Errorf("a graph of %d inputs and %d outputs, past the %d places the CPU's call site binds", len(gr.In), len(gr.Out), cpuIO)
	}
	f := &cpuBlockForm{consts: map[int][]uint32{}, rows: rows, sel: -1, wsel: -1}
	selV, wsV := selOf(gr), weightsOf(gr)
	wOwn := map[graph.ValueID]int{}
	type ownUse struct{ code, own int }
	var wrote, read []ownUse
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
		cc := cpuCode{calls: c.Calls, n: 1, name: c.Name, share: jit.Code{Share: c.Share, ShareUnit: c.ShareUnit}}
		if c.Share > 0 && len(c.Calls) != 1 {
			f.close()
			return nil, fmt.Errorf("%s: a share of rows over calls %v", c.Name, c.Calls)
		}
		for _, n := range c.Calls {
			cc.n *= n
		}
		if rows > 1 && !c.Scan && !whole {
			// A form of several rows runs the units of its calls' leading
			// axis the call's rows reach: a kernel not held to them would
			// read and write rows past the call's. A scan's calls are its
			// state's rows, each reading the call's rows its descriptor
			// names.
			if c.Live == 0 || len(c.Calls) == 0 || c.Calls[0] != c.LiveUnits {
				f.close()
				return nil, fmt.Errorf("%s at %d rows: its calls %v hold no rows of a call (live %d of %d units)", c.Name, rows, c.Calls, c.Live, c.LiveUnits)
			}
			cc.live, cc.inner = jit.Code{Live: c.Live, LiveUnits: c.LiveUnits}, cc.n/c.Calls[0]
			cc.tiles = c.TileMajor && cc.inner > 1
			f.unit = max(f.unit, c.Live/c.LiveUnits)
		}
		bytes := map[jit.Operand]int64{}
		reach := map[jit.Operand][]int{}
		for _, r := range c.Reach {
			reach[r.Operand] = r.Per
		}
		// rw is each operand's use: 1 read, 2 written.
		rw := map[jit.Operand]uint8{}
		for _, bd := range c.Binds {
			if o := bd.Operand; bd.Writes && o.Kind == jit.OperandTable && (o.Name == "kv.k" || o.Name == "kv.v") {
				f.keysEnd = ci + 1
			}
			if o := bd.Operand; o.Kind == jit.OperandValue {
				if k := streamIn(gr, o.Value); k >= 0 && bd.Read {
					cc.reads |= 1 << k
				}
				if k := slices.Index(gr.Out, o.Value); k >= 0 && bd.Writes {
					cc.writes |= 1 << k
				}
			}
			o := bd.Operand
			o.Copy = 0
			bytes[o] = max(bytes[o], bd.Bytes)
			if bd.Read {
				rw[o] |= 1
			}
			if bd.Writes {
				rw[o] |= 2
			}
			if bd.Writes && bd.Operand.Kind == jit.OperandScratch && !bd.Own && !callsOwnScratch(c, bd.Operand, bd.Bytes) {
				cc.serial = true
			}
			if bd.Writes {
				n := o.Name
				if o.Kind == jit.OperandValue && int(o.Value) < len(gr.Values) {
					n = gr.Values[o.Value].Name
				}
				cc.desc += " " + n
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
			case a.Share:
				ar.src = cpuArgShare
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
			case o.Kind == jit.OperandValue && streamIn(gr, o.Value) >= 0:
				ar.src, ar.stream = cpuArgX, streamIn(gr, o.Value)
			case o.Kind == jit.OperandValue && slices.Contains(gr.Out, o.Value):
				ar.src, ar.stream = cpuArgX, slices.Index(gr.Out, o.Value)
			case o.Kind == jit.OperandValue && len(gr.In) == len(gr.Out)+1 && o.Value == gr.In[len(gr.Out)]:
				ar.src = cpuArgSide
			case o.Kind == jit.OperandValue:
				ar.src, ar.i = cpuArgOwn, ownOf(o, ci, max(gr.Values[o.Value].Width, int(bytes[o]/4))+a.Offset, nil)
				switch o.Value {
				case selV:
					f.sel = ar.i
				default:
					if slices.Contains(wsV, o.Value) {
						wOwn[o.Value] = ar.i
					}
				}
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
				case "rope.planes":
					ar.src = cpuArgPlanes
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
				case "rec.conv":
					ar.src = cpuArgRecConv
				case "rec.conv.out":
					ar.src = cpuArgRecConvOut
				case "rec.rows":
					ar.src = cpuArgRecRows
				case "q.off":
					ar.src = cpuArgZero
				case "request":
					ar.src = cpuArgRequest
				case "rope.pos":
					// A pooled readout reads its live count (cpuPoolState), any
					// other graph its rows' positions (cpuBound.pos).
					ar.src = cpuArgPos
					if slices.ContainsFunc(gr.Ops, func(o graph.Op) bool { return o.Kind == graph.OpPool }) {
						ar.src = cpuArgPoolPos
					}
				default:
					f.close()
					return nil, cpuDeclineError{jit.DeclineTable, fmt.Errorf("%s reads the table %s", c.Name, o.Name)}
				}
			default:
				f.close()
				return nil, fmt.Errorf("%s binds %+v, which the CPU's call site does not provide", c.Name, o)
			}
			if ar.src == cpuArgOwn && a.Words == nil {
				if rw[o]&2 != 0 {
					wrote = append(wrote, ownUse{ci, ar.i})
				}
				if rw[o] != 2 {
					read = append(read, ownUse{ci, ar.i})
				}
			}
			if a.Word {
				if ar.src != cpuArgOwn {
					f.close()
					return nil, fmt.Errorf("%s passes a word of %+v by value, which the CPU's call site holds no words of", c.Name, o)
				}
				ar.src = cpuArgWord
			}
			cc.args = append(cc.args, ar)
			if ar.src != cpuArgValue && ar.src != cpuArgCoord && ar.src != cpuArgShare {
				buf := cpuBuf{src: ar.src, region: ar.region, role: ar.role, i: ar.i, stream: ar.stream}
				if buf.src == cpuArgWord {
					buf.src = cpuArgOwn
				}
				use := rw[o]
				if use == 0 || a.Word {
					// Unbound, or a word read when its call is set up: a
					// read, and, unbound, a write too.
					use |= 1
					if rw[o] == 0 {
						use |= 2
					}
				}
				if use&1 != 0 {
					cc.rd = append(cc.rd, buf)
				}
				if use&2 != 0 {
					cc.wr = append(cc.wr, buf)
				}
				cc.slice(buf, reach[o])
			}
		}
		code, err := cpu.MapNamed(c.CPU, c.Name)
		if err != nil {
			f.close()
			return nil, fmt.Errorf("mapping %s: %w", c.Name, err)
		}
		cc.code = code
		f.codes = append(f.codes, cc)
	}
	// The weights as the first of their values a kernel reads.
	for _, v := range wsV {
		own, ok := wOwn[v]
		for _, r := range read {
			if ok && r.own == own && !slices.Contains(f.wreads, r.code) {
				f.wsel, f.wreads = own, append(f.wreads, r.code)
			}
		}
		if f.wsel >= 0 {
			break
		}
	}
	for _, cc := range f.codes[:f.keysEnd] {
		f.keysX = f.keysX || cc.writes != 0
	}
	for _, w := range wrote {
		if w.code >= f.keysEnd || slices.Contains(f.cross, w.own) {
			continue
		}
		for _, r := range read {
			if r.own == w.own && r.code >= f.keysEnd {
				f.cross = append(f.cross, w.own)
				break
			}
		}
	}
	if !apart && !f.inPlace() {
		f.close()
		return nil, fmt.Errorf("the block reads its residual after writing its output over it")
	}
	if err := f.bankHalves(gr, rows); err != nil {
		f.close()
		return nil, err
	}
	if rows == 1 {
		f.chainPhases()
		f.joinPhases()
	}
	return f, nil
}

// callsOwnScratch reports whether code c's calls each write a slice of
// scratch o no other call writes: every address of o the code is passed
// moves along every axis of its calls, each axis's step past the whole of
// the axes inside it, as an operand laid out by its primitive's leading axes
// is, and the code reaching no more of o than its calls' slices; an axis of
// one call moves nothing. A scratch some call's address does not move along
// is every call's.
func callsOwnScratch(c jit.Code, o jit.Operand, bytes int64) bool {
	if len(c.Calls) == 0 {
		return false
	}
	o.Copy = 0
	seen := false
	for _, a := range c.Args {
		x := a.Operand
		x.Copy = 0
		if x != o {
			continue
		}
		if a.Word || a.Coord > 0 || a.Length || a.Value || a.Moving || len(a.Per) != len(c.Calls) {
			return false
		}
		span := 0
		for ax := range c.Calls {
			if c.Calls[ax] == 1 {
				continue
			}
			inner := 1
			if ax+1 < len(c.Calls) {
				inner = a.Per[ax+1] * c.Calls[ax+1]
			}
			if a.Per[ax] <= 0 || a.Per[ax] < inner {
				return false
			}
			span = max(span, a.Per[ax]*c.Calls[ax])
		}
		elem := int64(4)
		if a.Elem > 0 {
			elem = int64(a.Elem)
		}
		if bytes <= 0 || bytes > int64(span)*elem {
			return false
		}
		seen = true
	}
	return seen
}

// joinPhases runs each run of consecutive kernels that share no buffer one
// of them writes in one pool phase: their calls one space, claimed as one
// kernel's. A run does not cross the selection's split.
func (f *cpuBlockForm) joinPhases() {
	for g := 0; g < len(f.codes); {
		if f.codes[g].chain > 0 {
			g += f.codes[g].chain + 1
			continue
		}
		j := g + 1
		if f.codes[g].joinable() {
			for j < len(f.codes) && j != f.split && f.codes[j].joinable() && f.codes[j].chain == 0 && apart(f.codes[g:j], &f.codes[j]) {
				j++
			}
		}
		f.codes[g].join = j - g - 1
		g = j
	}
}

// slice records the rows of buffer b a call reaches alone along the one
// axis of its calls past a single call (jit.Code.Reach), -1 for none.
func (c *cpuCode) slice(b cpuBuf, reach []int) {
	per := -1
	if ax := oneAxis(c.calls); ax >= 0 && ax < len(reach) && reach[ax] > 0 {
		per = reach[ax]
	}
	for i := range c.sliced {
		if c.sliced[i].buf == b {
			if c.sliced[i].per != per {
				c.sliced[i].per = -1
			}
			return
		}
	}
	c.sliced = append(c.sliced, cpuSlice{b, per})
}

// oneAxis is the one axis of calls past a single call, -1 for none or more.
func oneAxis(calls []int) int {
	ax := -1
	for i, n := range calls {
		if n == 1 {
			continue
		}
		if ax >= 0 {
			return -1
		}
		ax = i
	}
	return ax
}

// chainPhases runs each run of kernels that chained admits, one reading
// what another writes, in one pool phase: unit u runs its calls of each
// kernel in turn. A run does not cross the selection's split.
func (f *cpuBlockForm) chainPhases() {
	for g := 0; g < len(f.codes); {
		j := g + 1
		for j < len(f.codes) && j != f.split && f.codes[j].joinable() && chained(f.codes[g:j+1]) {
			j++
		}
		dep := false
		for k := g + 1; k < j; k++ {
			dep = dep || !apart(f.codes[g:k], &f.codes[k])
		}
		if dep {
			f.codes[g].chain, f.codes[g].units = j-g-1, chainUnits(f.codes[g:j])
		}
		g = j
	}
}

// chainUnits is the units of a phase of kernels cs: the fewest calls of
// one, every kernel's calls along one axis and a whole number a unit; 0
// where they are not.
func chainUnits(cs []cpuCode) int {
	u := cs[0].n
	for i := range cs {
		u = min(u, cs[i].n)
	}
	ax := oneAxis(cs[0].calls)
	for i := range cs {
		if !cs[i].joinable() || cs[i].share.Share > 0 || ax < 0 || oneAxis(cs[i].calls) != ax || len(cs[i].calls) != len(cs[0].calls) || cs[i].n%u != 0 {
			return 0
		}
	}
	return u
}

// chained says kernels cs may run in one phase unit by unit: every buffer
// one writes and another reads or writes, every call of both reaches only
// its own slice of, a unit's calls' slices one span in each.
func chained(cs []cpuCode) bool {
	u := chainUnits(cs)
	if u <= 1 {
		return false
	}
	span := func(c *cpuCode, b cpuBuf) int {
		for _, s := range c.sliced {
			if s.buf == b && s.per > 0 {
				return s.per * (c.n / u)
			}
		}
		return -1
	}
	for i := range cs {
		for _, b := range cs[i].wr {
			for j := range cs {
				if j == i || !slices.Contains(cs[j].rd, b) && !slices.Contains(cs[j].wr, b) {
					continue
				}
				if s := span(&cs[i], b); s < 0 || s != span(&cs[j], b) {
					return false
				}
			}
		}
	}
	return true
}

// joinable says a kernel's calls may share a phase with another's: calls
// on the pool, each free of the others and of the step's rows. A kernel of
// one call runs on the caller, in no phase.
func (c *cpuCode) joinable() bool {
	return c.live.Live == 0 && !c.tiles && c.n > 1 && !c.serial
}

// apart says kernel c reads nothing the kernels gs write and writes nothing
// they read or write.
func apart(gs []cpuCode, c *cpuCode) bool {
	for i := range gs {
		g := &gs[i]
		for _, w := range g.wr {
			if slices.Contains(c.rd, w) || slices.Contains(c.wr, w) {
				return false
			}
		}
		for _, w := range c.wr {
			if slices.Contains(g.rd, w) {
				return false
			}
		}
	}
	return true
}

// selOf is the selection gr's banks are read at, -1 for a graph that reads
// none.
func selOf(gr *graph.Graph) graph.ValueID {
	for _, o := range gr.Ops {
		if o.Kind == graph.OpExpertMatVec && len(o.In) == 2 && len(o.Weight) > 0 && jlm.ExpertBank(o.Weight[0].Role) {
			return o.In[1]
		}
	}
	return -1
}

// weightsOf is the picks' weights gr's banks' outputs are multiplied by,
// then the values of their width each is an elementwise map of, back to the
// first that is not: nil for a graph that weights none.
func weightsOf(gr *graph.Graph) []graph.ValueID {
	bank := map[graph.ValueID]bool{}
	made := map[graph.ValueID]int{}
	w := graph.ValueID(-1)
	for i, o := range gr.Ops {
		for _, v := range o.Out {
			made[v] = i
		}
		if o.Kind == graph.OpExpertMatVec && len(o.In) == 2 && len(o.Weight) > 0 && jlm.ExpertBank(o.Weight[0].Role) {
			bank[o.Out[0]] = true
		}
		if w >= 0 || o.Kind != graph.OpMap || o.Attr.Fn != graph.MapMul || len(o.In) != 2 {
			continue
		}
		switch {
		case bank[o.In[0]]:
			w = o.In[1]
		case bank[o.In[1]]:
			w = o.In[0]
		}
	}
	var ws []graph.ValueID
	for w >= 0 && len(ws) < len(gr.Ops) {
		ws = append(ws, w)
		i, ok := made[w]
		if o := gr.Ops[i]; !ok || o.Kind != graph.OpMap || len(o.In) == 0 || gr.Values[o.In[0]].Width != gr.Values[w].Width {
			break
		}
		w = gr.Ops[i].In[0]
	}
	return ws
}

// bankHalves finds where a form over banks is cut (cpuBlockForm.split): at
// its first code that binds a bank, with the selection written by a code
// before it.
func (f *cpuBlockForm) bankHalves(gr *graph.Graph, rows int) error {
	sel := selOf(gr)
	if sel < 0 {
		return nil
	}
	f.split = -1
	for ci, c := range f.codes {
		for _, a := range c.args {
			if a.src == cpuArgWeight && jlm.ExpertBank(a.role) && f.split < 0 {
				f.split = ci
			}
		}
	}
	if f.sel < 0 || f.split <= 0 {
		return fmt.Errorf("a form over banks with no selection of its own (%d) or no code before its banks (%d)", f.sel, f.split)
	}
	f.selN = rows * gr.Values[sel].Width
	return nil
}

// cpuBanksPaged says block li's routed banks are each one view of its
// experts in the host's expert pages (jlm.File.BankView).
func (s *State) cpuBanksPaged(li int) bool {
	l := &s.m.layers[li]
	if s.m.container == nil || l.legacyExperts {
		return false
	}
	if len(s.cpu.banks) < len(s.m.layers) {
		s.cpu.banks = make([][3][3]unsafe.Pointer, len(s.m.layers))
	}
	for i, t := range []*tensor{&l.gate, &l.up, &l.down} {
		if t.e == nil {
			continue
		}
		planes, _, ok := s.m.container.BankView(t.e)
		if !ok {
			return false
		}
		s.cpu.banks[li][i] = planes
	}
	return l.up.e != nil && l.down.e != nil
}

// cpuStrides is each bank of block li's graph in the host's expert pages and
// the words from one expert's sheet to the next's there.
func (m *Model) cpuStrides(li int, g *graph.Graph) map[jlm.Role]int {
	var st map[jlm.Role]int
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			t := m.bankTensor(li, w.Role)
			if t == nil || t.e == nil {
				continue
			}
			if _, stride, ok := m.container.BankView(t.e); ok {
				if st == nil {
					st = map[jlm.Role]int{}
				}
				st[w.Role] = int(stride / 4)
			}
		}
	}
	return st
}

// bankTensor is block li's routed bank of role r, nil for any other role.
func (m *Model) bankTensor(li int, r jlm.Role) *tensor {
	if li >= len(m.layers) || !jlm.ExpertBank(r) {
		return nil
	}
	l := &m.layers[li]
	return [...]*tensor{&l.gate, &l.up, &l.down}[r-jlm.RoleExpGateBank]
}

// cpuForm runs form f of block li over the step b: its codes in order, and
// for a form over banks (cpuBlockForm.split) the selection's experts made
// resident and held between its halves -- the pager's own fault path
// (ensureExperts), each read from the file where the budget evicted it --
// and counted as the host's own mixture counts them.
func (s *State) cpuForm(b *cpuBound, f *cpuBlockForm, li int) error {
	if f.split <= 0 {
		s.cpuCalls(b, f.codes)
		return nil
	}
	s.cpuCalls(b, f.codes[:f.split])
	err := s.cpuExperts(b, f, li, 1)
	defer s.expHold.release()
	if err != nil {
		return err
	}
	if !s.rowsOfChunk {
		s.hot.add(li, s.cpu.sel)
	}
	s.cpuCalls(b, f.codes[f.split:])
	return nil
}

// cpuExperts makes the experts the n rows of a call of form f select
// (cpuBlockForm.sel, a row's picks after another's) resident and held until
// s.expHold's release, each once however many rows pick it -- the pager's
// own fault path (ensureExperts), each read from the file where the budget
// evicted it -- and counts them resident as main's mixture does. s.cpu.sel
// is the experts held, in the order the rows first pick them.
func (s *State) cpuExperts(b *cpuBound, f *cpuBlockForm, li, n int) error {
	k := f.selN / max(f.rows, 1)
	picks := b.own[f.sel][:n*k]
	ne := s.c.NExpert
	if s.cpu.fault == cpuFaultPicks && n > 1 {
		// The first row given the second's picks, with its own weights.
		copy(picks[:k], picks[k:2*k])
	}
	if len(s.cpu.picked) < ne {
		s.cpu.picked = make([]bool, ne)
	}
	s.cpu.sel = s.cpu.sel[:0]
	for _, x := range picks {
		if int(x) >= ne {
			// No page: ensureExperts names it.
			s.cpu.sel = append(s.cpu.sel, int32(x))
		} else if !s.cpu.picked[x] {
			s.cpu.picked[x] = true
			s.cpu.sel = append(s.cpu.sel, int32(x))
		}
	}
	other := int32(-1)
	if s.cpu.fault == cpuFaultExpert {
		// Another expert's page made resident and held than the first the
		// kernels read: one no row picks, where there is one.
		e := s.cpu.sel[0]
		for j := 1; j < ne && other < 0; j++ {
			if x := (e + int32(j)) % int32(ne); !s.cpu.picked[x] {
				other = x
			}
		}
	}
	for _, x := range s.cpu.sel {
		if int(x) < ne {
			s.cpu.picked[x] = false
		}
	}
	if other >= 0 {
		s.cpu.sel[0] = other
	}
	if err := s.m.ensureExperts(li, s.cpu.sel, &s.expHold); err != nil {
		return err
	}
	s.res.touch(li, ne, s.cpu.sel)
	return nil
}

// cpuCalls runs kernels cs in order, each with those its phase joins
// (cpuBlockForm.joinPhases).
func (s *State) cpuCalls(b *cpuBound, cs []cpuCode) {
	for i := 0; i < len(cs); i++ {
		c := &cs[i]
		if c.chain > 0 {
			g := cs[i : i+c.chain+1]
			i += c.chain
			if !s.prof {
				s.cpuChain(b, g, c.units)
				continue
			}
			t0 := time.Now()
			s.cpuChain(b, g, c.units)
			c.nanos += int64(time.Since(t0))
			c.runs++
			continue
		}
		if c.join == 0 {
			s.cpuCall(b, c)
			continue
		}
		g := cs[i : i+c.join+1]
		i += c.join
		if !s.prof {
			s.cpuGroup(b, g)
			continue
		}
		t0 := time.Now()
		s.cpu.t0, s.cpu.maxStart = t0, 0
		s.cpuGroup(b, g)
		d := int64(time.Since(t0))
		g[0].pmaxNs += s.cpu.maxStart
		ends := s.cpu.ends
		for j := range g {
			n := ends[j]
			if j > 0 {
				n -= ends[j-1]
			}
			g[j].nanos += d * int64(n) / int64(ends[len(ends)-1])
			g[j].runs++
		}
	}
}

// cpuGroup runs the kernels g as one phase over the pool: their calls one
// space, in order of the kernels.
func (s *State) cpuGroup(b *cpuBound, g []cpuCode) {
	s.cpu.part = s.Workers()
	s.cpu.ends = s.cpu.ends[:0]
	n := 0
	for j := range g {
		if g[j].share.Share > 0 {
			_, k := g[j].share.ShareRows(s.cpu.part)
			n += k
		} else {
			n += max(g[j].n, 1)
		}
		s.cpu.ends = append(s.cpu.ends, n)
	}
	if s.jit == nil || s.jit.Workers() <= 1 {
		s.cpuGroupRun(&s.cpu.args[0], b, g, 0, n)
		return
	}
	if w := s.jit.Workers(); len(s.cpu.args) < w {
		s.cpu.args = append(s.cpu.args, make([]cpu.Args, w-len(s.cpu.args))...)
	}
	r := &s.cpu.run
	r.s, r.b, r.g = s, b, g
	s.jit.ParallelWorkers(n, 1, r.gfn)
}

// cpuChain runs the kernels g as one phase over the pool, units units:
// a unit its calls of each kernel in turn (chainPhases).
func (s *State) cpuChain(b *cpuBound, g []cpuCode, units int) {
	s.cpu.part = s.Workers()
	if s.jit == nil || s.jit.Workers() <= 1 {
		s.cpuChainRun(&s.cpu.args[0], b, g, units, 0, units)
		return
	}
	if w := s.jit.Workers(); len(s.cpu.args) < w {
		s.cpu.args = append(s.cpu.args, make([]cpu.Args, w-len(s.cpu.args))...)
	}
	r := &s.cpu.run
	r.s, r.b, r.g, r.units = s, b, g, units
	s.jit.ParallelWorkers(units, 1, r.cfn)
}

// cpuChainRun runs the units [lo, hi) of the kernels g's phase on a.
func (s *State) cpuChainRun(a *cpu.Args, b *cpuBound, g []cpuCode, units, lo, hi int) {
	for u := lo; u < hi; u++ {
		for k := range g {
			q := g[k].n / units
			for at := u * q; at < (u+1)*q; at++ {
				s.cpuWords(a, b, &g[k], at)
				g[k].code.Call(a)
			}
		}
	}
}

// chainRegion runs the units [lo, hi) of the phase's kernels on worker w's
// words.
func (r *cpuRegion) chainRegion(w, lo, hi int) {
	r.s.cpuChainRun(&r.s.cpu.args[w], r.b, r.g, r.units, lo, hi)
}

// cpuGroupRun runs the calls [lo, hi) of the kernels g's one space on a.
func (s *State) cpuGroupRun(a *cpu.Args, b *cpuBound, g []cpuCode, lo, hi int) {
	ends := s.cpu.ends
	k := 0
	for i := lo; i < hi; i++ {
		for i >= ends[k] {
			k++
		}
		at := i
		if k > 0 {
			at -= ends[k-1]
		}
		s.cpuWords(a, b, &g[k], at)
		if s.prof {
			s.cpuProfCall(&g[k], a)
			continue
		}
		g[k].code.Call(a)
	}
}

// inPlace says the form may write each residual stream's output over its
// input: every kernel that reads a stream's input runs before the first
// that writes its output.
func (f *cpuBlockForm) inPlace() bool {
	for k := range 64 {
		lastRead, firstWrite := -1, len(f.codes)
		for i, c := range f.codes {
			if c.reads&(1<<k) != 0 {
				lastRead = i
			}
			if c.writes&(1<<k) != 0 {
				firstWrite = min(firstWrite, i)
			}
		}
		if lastRead >= firstWrite {
			return false
		}
	}
	return true
}

// streamIn is the residual stream whose input v is in block graph gr, -1
// for a value that is none: the graph's first inputs, one a stream it
// writes.
func streamIn(gr *graph.Graph, v graph.ValueID) int {
	if k := slices.Index(gr.In, v); k < len(gr.Out) {
		return k
	}
	return -1
}

// cpuRoleWeight is block li's (l's) matrix of role r, nil for one it does
// not hold; with no block, the head's. An encoder segment's blocks hold
// theirs in the encoder (encRoleWeight).
func (s *State) cpuRoleWeight(li int, l *layer, r jlm.Role) *tensor {
	if l == nil {
		switch m := s.m; {
		case r == jlm.RoleOutput || r == jlm.RoleTokenEmbd:
			return s.outW
		case r == jlm.RoleStreamExpand && m.altProj.rows != 0:
			return &m.altProj
		case r == jlm.RolePLETokEmbd && m.pleTok.e != nil:
			return &m.pleTok
		case r == jlm.RolePLEModelProj && m.pleProj.e != nil:
			return &m.pleProj
		case r >= jlm.RoleStreamCollapse1 && int(r-jlm.RoleStreamCollapse1) < len(m.altUnembd):
			return &m.altUnembd[r-jlm.RoleStreamCollapse1]
		case s.vis != nil && r == jlm.RoleVProj:
			return &s.vis.t.projW
		case s.vis != nil && r == jlm.RoleVPatchEmbd:
			return &s.vis.t.patchW
		case s.vis != nil && r == jlm.RoleVPosEmbd:
			return &s.cpu.proj.posW
		case s.vis != nil && r == jlm.RoleVProj2:
			return &s.vis.t.projW2
		case s.vis != nil && r == jlm.RoleVProj3:
			return &s.vis.t.projW3
		case r == jlm.RoleEmbdDense1:
			return &m.dense1
		case r == jlm.RoleEmbdDense2:
			return &m.dense2
		case m.ds4Head != nil && r == jlm.RoleHCHeadFn:
			return &m.ds4Head.fn
		case m.q4eHead != nil && r == jlm.RoleHCHeadDown:
			return &m.q4eHead.hc.down
		case m.q4eHead != nil && r == jlm.RoleHCHeadUp:
			return &m.q4eHead.hc.up
		}
		return nil
	}
	if s.enc != nil {
		return s.m.encRoleWeight(li, r)
	}
	if l.mn != nil {
		if s.cpu.fault == cpuFaultFilterT {
			if t := l.mn.filterT(r); t != nil {
				return t
			}
		}
		return convRoleWeight(l, r)
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
	case jlm.RoleRouter:
		return &l.router
	case jlm.RoleShExpGate:
		return &l.shGate
	case jlm.RoleShExpUp:
		return &l.shUp
	case jlm.RoleShExpDown:
		return &l.shDown
	case jlm.RoleFFNRoutedDown:
		return &l.routedDown
	case jlm.RoleFFNRoutedUp:
		return &l.routedUp
	case jlm.RoleAttnQKV:
		if l.ssmIn.rows != 0 {
			return &l.ssmIn
		}
		return &l.wq
	case jlm.RoleSCX:
		return &l.wq
	case jlm.RoleAttnGate, jlm.RoleSCB:
		// A full block of a model of latent attention gates its output by
		// its own; a linear block's is its state's gate.
		if l.mlaGate.rows != 0 && !s.c.LayerKind(li).Recurrent() {
			return &l.mlaGate
		}
		return &l.ssmGate
	case jlm.RoleAttnQA:
		return &l.wqa
	case jlm.RoleAttnQB:
		return &l.wqb
	case jlm.RoleAttnKVA:
		return &l.wkva
	case jlm.RoleAttnKB:
		return &l.wkb
	case jlm.RoleAttnVB:
		return &l.wvb
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
	case jlm.RolePLEGate:
		return &l.pleGate
	case jlm.RolePLEProj:
		return &l.pleProj
	case jlm.RoleStreamRouter:
		return &l.altRouter
	case jlm.RoleLowRankIn:
		return &l.laurelL
	case jlm.RoleLowRankOut:
		return &l.laurelR
	case jlm.RoleIdxQ:
		return &l.idxQ
	case jlm.RoleIdxQB:
		return &l.idxQB
	case jlm.RoleIdxK:
		return &l.idxK
	case jlm.RoleIdxProj:
		return &l.idxProj
	}
	if h := l.q4e; h != nil {
		switch r {
		case jlm.RoleHCAttnDown:
			return &h.attn.down
		case jlm.RoleHCAttnUp:
			return &h.attn.up
		case jlm.RoleHCAttnInject:
			return &h.attn.inj
		case jlm.RoleHCFFNDown:
			return &h.ffn.down
		case jlm.RoleHCFFNUp:
			return &h.ffn.up
		case jlm.RoleHCFFNInject:
			return &h.ffn.inj
		}
	}
	return nil
}

// cpuRoleVector is block li's (l's) vector of role r, nil for one it does
// not hold; with no block, the head's.
func (s *State) cpuRoleVector(li int, l *layer, r jlm.Role) []float32 {
	if l == nil {
		switch r {
		case jlm.RoleOutputNorm:
			return s.outNorm
		case jlm.RoleOutputNormBias:
			return s.m.outNormB
		case jlm.RoleOutputBias:
			return s.m.outB
		case jlm.RoleOutputResScore:
			return s.m.resOut
		case jlm.RolePLEProjNorm:
			return s.m.pleNorm
		case jlm.RoleHCHeadNorm:
			if s.m.q4eHead != nil {
				return s.m.q4eHead.hc.norm
			}
		case jlm.RoleHCHeadBase, jlm.RoleHCHeadScale:
			if hd := s.m.ds4Head; hd != nil {
				if r == jlm.RoleHCHeadBase {
					return hd.base[:s.c.HCMult]
				}
				return hd.scale[:1]
			}
		}
		if s.vis == nil {
			return nil
		}
		switch t := s.vis.t; r {
		case jlm.RoleVPatchBias:
			return t.patchB
		case jlm.RoleVClassEmbd:
			return t.clsW
		case jlm.RoleVPreNorm:
			return t.preLnW
		case jlm.RoleVPreNormBias:
			return t.preLnB
		case jlm.RoleVProjBias:
			return t.projB
		case jlm.RoleVProj2Bias:
			return t.projB2
		case jlm.RoleVProjNorm:
			return t.projNorm
		case jlm.RoleVProjNormBias:
			return t.projNormB
		case jlm.RoleVProjInNorm:
			return t.projInNorm
		case jlm.RoleVProjInNormBias:
			return t.projInNormB
		case jlm.RoleVStdBias:
			return t.stdBias()
		case jlm.RoleVStdScale:
			return t.stdScale()
		}
		return nil
	}
	if s.enc != nil {
		return s.m.encRoleVector(li, r)
	}
	if l.mn != nil {
		return convRoleVector(l, r)
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
	case jlm.RoleAttnQANorm:
		return l.qaNorm
	case jlm.RoleAttnKVANorm:
		return l.kvaNorm
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
	case jlm.RoleVFFNGateBias:
		return l.gateB
	case jlm.RoleVClamp:
		return l.clamp
	case jlm.RoleAttnSinks:
		return l.sinks
	case jlm.RoleRouterBias:
		return l.routerB
	case jlm.RoleExpGateBias:
		return l.expGateB
	case jlm.RoleExpUpBias:
		return l.expUpB
	case jlm.RoleExpDownBias:
		return l.expDownB
	case jlm.RoleExpProbsB:
		return l.expProbsB
	case jlm.RoleShRouter:
		return l.shRouter
	case jlm.RoleAttnSubNorm:
		return l.attnSubNorm
	case jlm.RoleFFNSubNorm:
		return l.ffnSubNorm
	case jlm.RoleXIELU:
		return l.xielu
	case jlm.RoleIdxQNorm:
		return l.idxQNorm
	case jlm.RoleIdxKNorm:
		return l.idxKNorm
	case jlm.RoleIdxKNormB:
		return l.idxKNormB
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
	case jlm.RolePLEPostNorm:
		return l.plePost
	case jlm.RoleStreamRouterNorm:
		return l.altRouterNorm
	case jlm.RoleStreamPredictCoef:
		return l.altPredT
	case jlm.RoleStreamCorrectCoef:
		return l.altCorrT
	case jlm.RoleStreamCorrectScale:
		return l.altCorrScale
	case jlm.RoleLowRankNorm:
		return l.laurelPost
	case jlm.RoleFFNNorm2:
		return l.ffnNorm2
	case jlm.RolePostFFNNorm1:
		return l.postFFNNorm1
	case jlm.RolePostFFNNorm2:
		return l.postFFNNorm2
	case jlm.RoleRouterNorm:
		return l.routerNorm
	case jlm.RoleExpScale:
		return l.expScale
	case jlm.RoleAttnResScore:
		return l.resAttn
	case jlm.RoleFFNResScore:
		return l.resFFN
	case jlm.RoleFFNRoutedNorm:
		return l.routedNorm
	}
	if h := l.q4e; h != nil {
		switch r {
		case jlm.RoleHCAttnNorm:
			return h.attn.norm
		case jlm.RoleHCFFNNorm:
			return h.ffn.norm
		}
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
