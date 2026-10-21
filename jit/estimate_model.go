package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// Class is a kind of instruction an estimate counts against a device's
// rate for it.
type Class uint8

const (
	// ClassFloat is float arithmetic: add, multiply, fma, min, max,
	// compares and selects.
	ClassFloat Class = iota
	// ClassSpecial is float work a device runs slower: divide, square root,
	// exponential, conversions and rounding.
	ClassSpecial
	// ClassInt is integer and bit work on data: masks, shifts, packing.
	ClassInt
	// ClassIndex is integer work on addresses and counts.
	ClassIndex
	// ClassLanes is work across a vector's lanes: reductions, shuffles,
	// broadcasts, interleaves.
	ClassLanes
	// ClassGroup is work across a workgroup: reductions, exchanges.
	ClassGroup
	// ClassMem is the issue of a load or a store.
	ClassMem
	NumClasses
)

// Rate is how fast a device serves a resource: what one thread gets alone,
// and the whole device's ceiling.
type Rate struct {
	Thread, Total float64
}

// at is the rate threads running together get.
func (r Rate) at(threads float64) float64 {
	if r.Thread <= 0 {
		return r.Total
	}
	if r.Total <= 0 {
		return r.Thread * threads
	}
	return min(r.Total, r.Thread*threads)
}

// Cache is one level of a device's caches: its capacity and the bytes a
// second it serves.
type Cache struct {
	Bytes int64
	Rate  Rate
}

// Cost is an estimate of a placement from its final IR, nothing lowered:
// the bytes its kernels move by where they are served from, the
// instructions they issue by class, their launches and the bytes carried
// between devices, and the seconds the devices' descriptions make of them.
// Seconds is the larger of each kernel's memory and compute time plus its
// launch, summed; Unknown counts loops whose trips the step does not
// decide, taken as one.
type Cost struct {
	Bytes, Carried int64
	// Served is the bytes by level: 0..2 the caches nearest first, 3 the
	// device's memory.
	Served                                    [4]int64
	Issued                                    [NumClasses]int64
	Launches                                  int
	Seconds                                   float64
	MemSeconds, ComputeSeconds, LaunchSeconds float64
	Unknown                                   int
}

func (c Cost) add(o Cost) Cost {
	c.Bytes += o.Bytes
	c.Carried += o.Carried
	for i := range c.Served {
		c.Served[i] += o.Served[i]
	}
	for i := range c.Issued {
		c.Issued[i] += o.Issued[i]
	}
	c.Launches += o.Launches
	c.Seconds += o.Seconds
	c.MemSeconds += o.MemSeconds
	c.ComputeSeconds += o.ComputeSeconds
	c.LaunchSeconds += o.LaunchSeconds
	c.Unknown += o.Unknown
	return c
}

// costModel estimates a run of kernels on one device at one step, keeping
// what each leaves in the device's caches for the next.
type costModel struct {
	d Device
	s Step
	// recent is the operands last touched, most recent last, with their
	// bytes: a reuse distance stack.
	recent []touch
}

type touch struct {
	o     Operand
	bytes int64
}

// level is where operand o's bytes are served from: the nearest cache the
// operand and everything touched since its last use fit in, 3 for memory.
func (m *costModel) level(o Operand, bytes int64) int {
	var since int64
	for i := len(m.recent) - 1; i >= 0; i-- {
		t := m.recent[i]
		if t.o == o {
			for l, c := range m.d.Caches {
				if c.Bytes > 0 && since+bytes <= c.Bytes {
					return l
				}
			}
			return 3
		}
		since += t.bytes
	}
	return 3
}

func (m *costModel) touch(o Operand, bytes int64) {
	for i, t := range m.recent {
		if t.o == o {
			m.recent = append(m.recent[:i], m.recent[i+1:]...)
			break
		}
	}
	m.recent = append(m.recent, touch{o, bytes})
}

// rate is level l's rate for threads running together.
func (m *costModel) rate(l int, threads float64) float64 {
	if l < 3 {
		return m.d.Caches[l].Rate.at(threads)
	}
	return Rate{Thread: m.d.Thread, Total: m.d.Bandwidth}.at(threads)
}

// cost is kernel k's estimate in the run.
func (m *costModel) cost(k *Kernel) Cost {
	e := estimator{k: k, s: m.s, trips: map[int]float64{}, homes: k.homes()}
	threads := float64(1)
	if m.d.Kind == DeviceGPU {
		threads = float64(max(1, k.elements()))
		if k.Work {
			threads *= float64(max(1, k.Group))
		}
	}
	c := Cost{Launches: 1}
	type use struct {
		touched, unique float64
		hull            span
		write           bool
	}
	uses := map[int]*use{}
	order := []int{}
	note := func(b int, touched, unique float64, h span, write bool) {
		u, ok := uses[b]
		if !ok {
			u = &use{hull: span{math.Inf(1), math.Inf(-1)}}
			uses[b] = u
			order = append(order, b)
		}
		u.touched += touched
		u.unique += unique
		u.hull = span{min(u.hull.lo, h.lo), max(u.hull.hi, h.hi)}
		u.write = u.write || write
	}
	live := k.live()
	words := k.vectorWords()
	for i, nd := range k.Nodes {
		if !live[i] {
			continue
		}
		n := e.runs(i)
		if nd.Op == OpLoad {
			if k.Bufs[nd.Buf].Kind == BufShared {
				continue
			}
			w := e.width(i, nd.Buf)
			note(nd.Buf, n*w, e.footprint([]int{nd.A}, nd.Ax, e.homes[i], nd.Buf)*w, e.reach(i, nd.A, nd.Ax, nd.Buf, w), false)
			c.Issued[ClassMem] += int64(n)
			continue
		}
		if cl, ok := classOf(k, i, words); ok {
			c.Issued[cl] += int64(n)
		}
	}
	for _, s := range k.Stores {
		if s.Barrier || s.Run || k.Bufs[s.Buf].Kind == BufShared {
			continue
		}
		n := e.within(s.At)
		b := float64(elemBytes(k.Bufs[s.Buf]))
		w := b
		if s.At >= 0 {
			w = b * float64(max(1, max(k.Loops[s.At].Lanes, k.Loops[s.At].Wide)))
		}
		if s.Narrow {
			w = 16
		}
		note(s.Buf, n*w, e.footprint([]int{s.Ix}, s.Ax, s.At, s.Buf)*w, e.storeReach(s, w), true)
		c.Issued[ClassMem] += int64(n)
	}
	for _, b := range order {
		u := uses[b]
		buf := k.Bufs[b]
		if shape := buf.Shape; shape != nil {
			n := int64(elemBytes(buf))
			for _, x := range shape {
				n *= int64(x)
			}
			u.unique = min(u.unique, float64(n))
		}
		u.unique = min(u.unique, u.touched, max(0, u.hull.hi-u.hull.lo))
		uniq := int64(u.unique)
		l := m.fit(uniq)
		if !u.write {
			l = m.level(buf.Operand, uniq)
		}
		c.Bytes += uniq
		c.Served[l] += uniq
		c.MemSeconds += float64(uniq) / m.rate(l, threads)
		if again := u.touched - u.unique; again > 0 {
			near := m.fit(uniq)
			c.Served[near] += int64(again)
			c.MemSeconds += again / m.rate(near, threads)
		}
		m.touch(buf.Operand, uniq)
	}
	for cl, n := range c.Issued {
		if r := m.d.Rates[cl].at(threads); r > 0 && n > 0 {
			c.ComputeSeconds += float64(n) / r
		}
	}
	c.LaunchSeconds = m.d.Launch
	c.Seconds = max(c.MemSeconds, c.ComputeSeconds) + c.LaunchSeconds
	c.Unknown = e.unknown
	return c
}

// fit is the nearest cache bytes fit in, 3 for none.
func (m *costModel) fit(bytes int64) int {
	for l, c := range m.d.Caches {
		if c.Bytes > 0 && bytes <= c.Bytes {
			return l
		}
	}
	return 3
}

// estimator counts how often a kernel's code runs at a step.
type estimator struct {
	k       *Kernel
	s       Step
	trips   map[int]float64
	homes   []int
	unknown int
	// elem is the element whose counts eval reads; byElem the loops whose
	// counts read an element's coordinates, and runsIn each loop's runs.
	elem   []int
	byElem map[int]bool
	runsIn map[int]float64
	// folding is the folds whose interval is being taken: a fold's step
	// that reads the fold adds nothing to it.
	folding map[int]bool
}

// runs is how many times node i's instruction runs in a launch: its
// element's count times the trips of the loops it sits in.
func (e *estimator) runs(i int) float64 {
	if i >= len(e.homes) {
		return e.within(-1)
	}
	return e.within(e.homes[i])
}

// within is the runs of an instruction in loop c over the launch: each
// element's product of the trips of the loops around c, a spread loop's
// dealt over the group, summed over the elements -- one element's times
// their count where no count reads an element's coordinates.
func (e *estimator) within(c int) float64 {
	if e.runsIn == nil {
		e.runsIn = map[int]float64{}
	}
	if n, ok := e.runsIn[c]; ok {
		return n
	}
	k := e.k
	group := 1.0
	if k.Work {
		group = float64(max(1, k.Group))
	}
	product := func() float64 {
		n := 1.0
		for d := c; d >= 0; d = k.Loops[d].Parent {
			t := e.loopTrips(d)
			if k.Loops[d].Spread && k.Group > 0 {
				t /= float64(k.Group)
			}
			n *= t
		}
		return n
	}
	elems := max(1, k.elements())
	varies := false
	for d := c; d >= 0; d = k.Loops[d].Parent {
		varies = varies || e.countByElem(d)
	}
	var n float64
	if !varies || elems > 1<<16 {
		e.elem = make([]int, len(k.Space))
		n = product() * float64(elems) * group
	} else {
		for x := range elems {
			e.elem = e.coords(x)
			e.trips = map[int]float64{}
			n += product() * group
		}
		e.trips = map[int]float64{}
		e.elem = make([]int, len(k.Space))
	}
	e.runsIn[c] = n
	return n
}

// coords is element x's coordinates, the last axis fastest.
func (e *estimator) coords(x int) []int {
	sp := e.k.Space
	out := make([]int, len(sp))
	for a := len(sp) - 1; a >= 0; a-- {
		out[a] = x % max(1, sp[a])
		x /= max(1, sp[a])
	}
	return out
}

// countByElem reports whether loop c's count reads an element's
// coordinates.
func (e *estimator) countByElem(c int) bool {
	if e.byElem == nil {
		e.byElem = map[int]bool{}
	}
	if v, ok := e.byElem[c]; ok {
		return v
	}
	k := e.k
	reads := false
	if l := k.Loops[c]; l.Count > 0 {
		seen := map[int]bool{}
		var walk func(i int)
		walk = func(i int) {
			if i < 0 || seen[i] || reads {
				return
			}
			seen[i] = true
			nd := k.Nodes[i]
			if nd.Op == OpCoord {
				reads = true
				return
			}
			for _, a := range k.args(&nd) {
				walk(*a)
			}
		}
		walk(l.Count - 1)
	}
	e.byElem[c] = reads
	return reads
}

// footprint is how many distinct places an access at home h, indexed by
// ix or by coordinates ax, reaches over a launch: the extents of the
// coordinates, threads and loops its index reads; unbounded where a loop
// around h walks buffer b by a cursor, or the index reads the launch's
// thread, whose places the index no longer bounds.
func (e *estimator) footprint(ix, ax []int, h, b int) float64 {
	k := e.k
	if ax != nil {
		ix = ax
	}
	seen := map[int]bool{}
	axes, loops := map[int]bool{}, map[int]bool{}
	thread, flat := false, false
	var walk func(i int)
	walk = func(i int) {
		if i < 0 || seen[i] {
			return
		}
		seen[i] = true
		nd := k.Nodes[i]
		switch nd.Op {
		case OpCoord:
			axes[nd.A] = true
		case OpIter:
			for c := nd.At; c >= 0; c = k.Loops[c].Parent {
				loops[c] = true
			}
		case OpThread, OpLocal:
			thread = true
		case OpFlat:
			flat = true
		}
		if nd.Op == OpLoad && nd.Ax == nil {
			walk(nd.A)
			return
		}
		for _, a := range k.args(&nd) {
			walk(*a)
		}
	}
	for _, i := range ix {
		walk(i)
	}
	for c := h; c >= 0; c = k.Loops[c].Parent {
		for _, cu := range k.Loops[c].Cursors {
			if cu.Buf == b {
				return math.Inf(1)
			}
		}
	}
	if flat {
		return math.Inf(1)
	}
	f := 1.0
	for a := range axes {
		f *= float64(k.Space[a])
	}
	for c := range loops {
		f *= e.loopTrips(c)
	}
	if thread {
		f *= float64(max(1, k.Group))
	}
	return f
}

// loopTrips is loop c's trips at the step for the current element: its
// count's value, else its fixed trips; a loop over a range whose count the
// step does not decide its live positions.
func (e *estimator) loopTrips(c int) float64 {
	if t, ok := e.trips[c]; ok {
		return t
	}
	l := e.k.Loops[c]
	var t float64
	switch {
	case l.Count > 0:
		if v, ok := e.eval(l.Count-1, 0); ok {
			t = v
		} else if l.Range {
			t = float64(e.s.Keys)
		} else if l.Trip > 0 {
			t = float64(l.Trip)
		} else {
			e.unknown++
			t = 1
		}
	case l.Trip > 0 || l.Kept:
		t = float64(max(1, l.Trip))
	case l.Range:
		t = float64(e.s.Keys)
	default:
		t = 1
	}
	e.trips[c] = t
	return t
}

// eval is node i's value at the step for the current element and its
// first thread, where the step decides it.
func (e *estimator) eval(i, depth int) (float64, bool) {
	k := e.k
	if depth > 96 {
		return 0, false
	}
	nd := k.Nodes[i]
	bin := func(f func(a, b float64) float64) (float64, bool) {
		a, ok := e.eval(nd.A, depth+1)
		if !ok {
			return 0, false
		}
		b, ok := e.eval(nd.B, depth+1)
		if !ok {
			return 0, false
		}
		return f(a, b), true
	}
	truth := func(v bool) float64 {
		if v {
			return 1
		}
		return 0
	}
	switch nd.Op {
	case OpConst, OpLength:
		return float64(nd.Imm), true
	case OpCoord:
		if nd.A < len(e.elem) {
			return float64(e.elem[nd.A]), true
		}
		return 0, true
	case OpThread, OpLocal, OpFlat, OpIter:
		return 0, true
	case OpSpan:
		if nd.Imm == 1 {
			return float64(min(e.s.Page, e.s.Keys)), true
		}
		return 0, true
	case OpParam:
		for _, a := range k.Args {
			if a.Buf < 0 && a.Field == int(nd.Imm) && a.Node > 0 {
				return e.eval(a.Node-1, depth+1)
			}
		}
		return 0, false
	case OpLoad:
		if nd.Ax != nil {
			return 0, false
		}
		ix, ok := e.eval(nd.A, depth+1)
		if !ok {
			return 0, false
		}
		buf := k.Bufs[nd.Buf]
		if buf.Offset > 0 {
			off, ok := e.eval(buf.Offset-1, depth+1)
			if !ok {
				return 0, false
			}
			ix += off
		}
		return stepWord(buf.Operand, int(ix), e.s)
	case OpSelect:
		p, ok := e.eval(nd.A, depth+1)
		if !ok {
			return 0, false
		}
		if p != 0 {
			return e.eval(nd.B, depth+1)
		}
		return e.eval(nd.C, depth+1)
	case OpLess:
		return bin(func(a, b float64) float64 { return truth(a < b) })
	case OpEq:
		return bin(func(a, b float64) float64 { return truth(a == b) })
	case OpAdd:
		return bin(func(a, b float64) float64 { return a + b })
	case OpSub:
		return bin(func(a, b float64) float64 { return max(0, a-b) })
	case OpMul:
		return bin(func(a, b float64) float64 { return a * b })
	case OpDiv:
		return bin(func(a, b float64) float64 {
			if b == 0 {
				return 0
			}
			return float64(int64(a) / int64(b))
		})
	case OpRem:
		return bin(func(a, b float64) float64 {
			if b == 0 {
				return 0
			}
			return float64(int64(a) % int64(b))
		})
	case OpShr:
		return bin(func(a, b float64) float64 { return float64(int64(a) >> uint(b)) })
	case OpShl:
		return bin(func(a, b float64) float64 { return float64(int64(a) << uint(b)) })
	case OpAnd:
		return bin(func(a, b float64) float64 { return float64(int64(a) & int64(b)) })
	case OpMin:
		return bin(func(a, b float64) float64 { return min(a, b) })
	case OpMax:
		return bin(func(a, b float64) float64 { return max(a, b) })
	}
	return 0, false
}

// span is a closed interval of values an index takes.
type span struct{ lo, hi float64 }

var anywhere = span{0, math.Inf(1)}

// hull is the interval the index of an access takes over the launch: its
// coordinates, threads and loops' iterations over their whole ranges, its
// loads of the step's tables at the words those ranges reach.
func (e *estimator) hull(ix, ax []int, b int) span {
	k := e.k
	if ax != nil {
		sh := k.Bufs[b].Shape
		if len(sh) != len(ax) {
			return anywhere
		}
		var out span
		stride := 1.0
		for j := len(ax) - 1; j >= 0; j-- {
			v := e.ival(ax[j], 0)
			out.lo += v.lo * stride
			out.hi += v.hi * stride
			stride *= float64(sh[j])
		}
		return out
	}
	if len(ix) != 1 {
		return anywhere
	}
	return e.ival(ix[0], 0)
}

// ival is the interval node i takes over the launch; a fold's, the hull
// of its start and its steps.
func (e *estimator) ival(i, depth int) span {
	k := e.k
	if depth > 96 {
		return anywhere
	}
	nd := k.Nodes[i]
	if v, ok := e.eval(i, 0); ok && e.constant(i) {
		return span{v, v}
	}
	a := func() span { return e.ival(nd.A, depth+1) }
	b := func() span { return e.ival(nd.B, depth+1) }
	switch nd.Op {
	case OpConst, OpLength:
		return span{float64(nd.Imm), float64(nd.Imm)}
	case OpCoord:
		return span{0, float64(max(1, k.Space[nd.A]) - 1)}
	case OpThread, OpLocal:
		return span{0, float64(max(1, k.Group) - 1)}
	case OpFlat:
		return span{0, float64(max(1, k.elements()*max(1, k.Group)) - 1)}
	case OpIter:
		return span{0, max(0, e.maxTrips(nd.At)-1)}
	case OpParam:
		if v, ok := e.eval(i, 0); ok {
			return span{v, v}
		}
		return anywhere
	case OpLoad:
		if nd.Ax != nil {
			return anywhere
		}
		w := e.ival(nd.A, depth+1)
		if math.IsInf(w.hi, 1) || w.hi-w.lo > 1<<16 {
			return anywhere
		}
		out := span{math.Inf(1), math.Inf(-1)}
		buf := k.Bufs[nd.Buf]
		off := 0.0
		if buf.Offset > 0 {
			v, ok := e.eval(buf.Offset-1, 0)
			if !ok {
				return anywhere
			}
			off = v
		}
		for x := w.lo; x <= w.hi; x++ {
			v, ok := stepWord(buf.Operand, int(x+off), e.s)
			if !ok {
				return anywhere
			}
			out.lo, out.hi = min(out.lo, v), max(out.hi, v)
		}
		return out
	case OpAdd:
		x, y := a(), b()
		return span{x.lo + y.lo, x.hi + y.hi}
	case OpSub:
		x, y := a(), b()
		return span{max(0, x.lo-y.hi), max(0, x.hi-y.lo)}
	case OpMul:
		x, y := a(), b()
		return span{x.lo * y.lo, x.hi * y.hi}
	case OpDiv, OpShr:
		x, y := a(), b()
		if nd.Op == OpShr {
			y = span{math.Exp2(y.lo), math.Exp2(y.hi)}
		}
		if y.lo <= 0 {
			return anywhere
		}
		return span{math.Floor(x.lo / y.hi), math.Floor(x.hi / y.lo)}
	case OpShl:
		x, y := a(), b()
		return span{x.lo * math.Exp2(y.lo), x.hi * math.Exp2(y.hi)}
	case OpRem:
		x, y := a(), b()
		if y.lo == y.hi && y.lo > 0 && math.Floor(x.lo/y.lo) == math.Floor(x.hi/y.lo) {
			m := math.Floor(x.lo/y.lo) * y.lo
			return span{x.lo - m, x.hi - m}
		}
		return span{0, max(0, y.hi-1)}
	case OpAnd:
		x, y := a(), b()
		return span{0, min(x.hi, y.hi)}
	case OpMin:
		x, y := a(), b()
		return span{min(x.lo, y.lo), min(x.hi, y.hi)}
	case OpMax:
		x, y := a(), b()
		return span{max(x.lo, y.lo), max(x.hi, y.hi)}
	case OpSelect:
		x, y := e.ival(nd.B, depth+1), e.ival(nd.C, depth+1)
		return span{min(x.lo, y.lo), max(x.hi, y.hi)}
	case OpFold:
		if e.folding == nil {
			e.folding = map[int]bool{}
		}
		if e.folding[i] {
			return span{math.Inf(1), math.Inf(-1)}
		}
		e.folding[i] = true
		x, y := e.ival(nd.B, depth+1), e.ival(nd.A, depth+1)
		delete(e.folding, i)
		return span{min(x.lo, y.lo), max(x.hi, y.hi)}
	}
	return anywhere
}

// constant reports whether node i reads nothing that varies over a launch.
func (e *estimator) constant(i int) bool {
	k := e.k
	seen := map[int]bool{}
	var walk func(i int) bool
	walk = func(i int) bool {
		if seen[i] {
			return true
		}
		seen[i] = true
		nd := k.Nodes[i]
		switch nd.Op {
		case OpCoord, OpThread, OpLocal, OpFlat, OpIter, OpFold:
			return false
		}
		for _, a := range k.args(&nd) {
			if !walk(*a) {
				return false
			}
		}
		return true
	}
	return walk(i)
}

// maxTrips is the most trips loop c makes for any element.
func (e *estimator) maxTrips(c int) float64 {
	if !e.countByElem(c) || e.k.elements() > 1<<16 {
		return e.loopTrips(c)
	}
	saved, savedTrips := e.elem, e.trips
	best := 0.0
	for x := range max(1, e.k.elements()) {
		e.elem, e.trips = e.coords(x), map[int]float64{}
		best = max(best, e.loopTrips(c))
	}
	e.elem, e.trips = saved, savedTrips
	return best
}

// walked reports whether a loop around h walks buffer b by a cursor.
func (e *estimator) walked(h, b int) bool {
	for c := h; c >= 0; c = e.k.Loops[c].Parent {
		for _, cu := range e.k.Loops[c].Cursors {
			if cu.Buf == b {
				return true
			}
		}
	}
	return false
}

// reach is the bytes load i's places span: its index's hull in elements
// of buffer b, widened by the w bytes one run reads; unbounded where a
// cursor walks the buffer.
func (e *estimator) reach(i, ix int, ax []int, b int, w float64) span {
	if e.k.Nodes[i].Ptr > 0 || e.walked(e.homes[i], b) {
		return anywhere
	}
	h := e.hull([]int{ix}, ax, b)
	eb := float64(elemBytes(e.k.Bufs[b]))
	return span{h.lo * eb, h.hi*eb + w}
}

func (e *estimator) storeReach(s Store, w float64) span {
	if s.Ptr > 0 || e.walked(s.At, s.Buf) {
		return anywhere
	}
	h := e.hull([]int{s.Ix}, s.Ax, s.Buf)
	eb := float64(elemBytes(e.k.Bufs[s.Buf]))
	return span{h.lo * eb, h.hi*eb + w}
}

// width is the bytes one run of load i reads: a vector's lanes of the
// buffer's elements in a vector loop, one element else.
func (e *estimator) width(i, b int) float64 {
	k := e.k
	w := float64(elemBytes(k.Bufs[b]))
	h := e.homes[i]
	if h >= 0 && k.Nodes[i].Op == OpLoad && k.varies(i, h) {
		w *= float64(max(1, max(k.Loops[h].Lanes, k.Loops[h].Wide)))
	}
	return w
}

// classOf is the class node i's instruction counts in, false for a value
// that is no instruction: a constant, a coordinate, an argument word.
func classOf(k *Kernel, i int, words map[int]bool) (Class, bool) {
	nd := k.Nodes[i]
	switch nd.Op {
	case OpConst, OpCoord, OpIter, OpFold, OpParam, OpLength, OpSpan, OpLanes, OpThread, OpLocal, OpFlat:
		return 0, false
	case OpGroupSum, OpGroupMax, OpExchange:
		return ClassGroup, true
	case OpHSum, OpHMax, OpHMin, OpShuffle, OpBroadcast, OpZip, OpUnpack, OpUpper, OpDupEven, OpDupOdd, OpSwapPairs, OpFirstLane:
		return ClassLanes, true
	case OpDiv, OpSqrt, OpExp, OpCvtF32, OpCvtI32, OpRoundI32, OpHalf:
		if nd.Type == ir.F32 || k.Nodes[nd.A].Type == ir.F32 || words[i] {
			return ClassSpecial, true
		}
	}
	switch {
	case nd.Type == ir.F32:
		return ClassFloat, true
	case nd.Type == ir.Pred && k.Nodes[nd.A].Type == ir.F32:
		return ClassFloat, true
	case words[i]:
		return ClassInt, true
	}
	return ClassIndex, true
}
