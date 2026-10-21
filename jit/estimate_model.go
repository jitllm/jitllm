package jit

import (
	"github.com/jitllm/jitllm/engine/graph"
	"math"
	"slices"

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

// Cache is one level of a device's caches: its capacity, the bytes a
// second it serves, the seconds a read it serves waits, the seconds a
// short stream of reads from it waits beyond its bytes at its rate while
// the prefetchers start, and the fraction of a buffer it keeps after reads
// of twice its size since the buffer's last use: 0 for a cache that keeps
// only the most recent, more for one whose replacement resists a stream;
// and the share of its capacity a buffer larger than it, read over and
// over, finds in it. Kernel is set on a cache a launch starts empty, a
// GPU's L1: it serves a kernel's re-reads and nothing a kernel before it
// read.
type Cache struct {
	Bytes                         int64
	Rate                          Rate
	Latency, Start, Retain, Cycle float64
	Kernel                        bool
}

// Cost is an estimate of a placement from its final IR, nothing lowered:
// the bytes its kernels move by where they are served from, the
// instructions they issue by class, their launches and the bytes carried
// between devices, and the seconds the devices' descriptions make of them.
// A kernel's compute time is its busiest class's, or all its instructions'
// at the rate a mix issues, whichever is longer, a GPU kernel's at the
// share of the multiprocessors its workgroups fill; its chain time is its
// longest fold's: the trips of one entry into the fold's loop, each waiting
// for the instructions between the fold and its next value, or the longest
// path through its instructions outside any loop, each waiting for those
// it reads -- on a GPU, the schedule of one thread whose subgroup issues
// in order.
// A kernel's memory time is its bytes at the rate of the level each is
// served from, and the longest start of a stream among them: a level's
// start, or for a buffer of a few lines its lines read one at a time. A
// GPU's levels run as one pipeline, every byte from farther out passing
// through each nearer level: its memory time is its busiest level's.
// Seconds is the largest of each kernel's memory time, compute time and
// longest chain of dependent reads' waits, plus the wait for its code's
// first line where the code is not in the nearest cache: that and its call
// on the host; on a GPU, that and the device's floor, or the launch where
// the host issues launches slower, summed; Unknown counts loops whose trips the step does not decide, taken
// as one.
type Cost struct {
	Bytes, Carried int64
	// Served is the bytes by level: 0..2 the caches nearest first, 3 the
	// device's memory.
	Served                                    [4]int64
	Issued                                    [NumClasses]int64
	Launches                                  int
	Seconds                                   float64
	MemSeconds, ComputeSeconds, LaunchSeconds float64
	WaitSeconds, FetchSeconds, ChainSeconds   float64
	FloorSeconds                              float64
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
	c.WaitSeconds += o.WaitSeconds
	c.FetchSeconds += o.FetchSeconds
	c.ChainSeconds += o.ChainSeconds
	c.FloorSeconds += o.FloorSeconds
	c.Unknown += o.Unknown
	return c
}

// costModel estimates a run of kernels on one device at one step, keeping
// what each leaves in the device's caches for the next.
type costModel struct {
	d Device
	s Step
	// reach is, by buffer, the bytes the last kernel estimated reaches
	// past its address.
	reach map[int]int64
	// recent is the operands last touched, most recent last, with their
	// bytes: a reuse distance stack.
	recent []touch
	// alias is, by part and value, the part and value whose storage a
	// part's boundary value is: a block reads the residual the one before
	// it wrote.
	alias map[[2]int][2]int
}

// partAliases is each part's boundary value's storage: the output of the
// part before it, and that one's in turn.
func partAliases(parts []Part) map[[2]int][2]int {
	out := map[[2]int][2]int{}
	for i := 1; i < len(parts); i++ {
		a, b := parts[i-1].Graph, parts[i].Graph
		if len(a.Out) == 0 || len(b.In) == 0 {
			continue
		}
		from := [2]int{i - 1, int(a.Out[0])}
		if x, ok := out[from]; ok {
			from = x
		}
		out[[2]int{i, int(b.In[0])}] = from
	}
	return out
}

// key is the storage operand o names: a graph value by its part and id
// alone, a boundary value as the value it is the storage of.
func (m *costModel) key(o Operand) Operand {
	if o.Kind != OperandValue {
		return o
	}
	at := [2]int{o.Of, int(o.Value)}
	if x, ok := m.alias[at]; ok {
		at = x
	}
	return Operand{Kind: OperandValue, Of: at[0], Value: graph.ValueID(at[1])}
}

// touch is an operand's bytes or, where code is set, a kernel's code.
type touch struct {
	o     Operand
	code  *Kernel
	bytes int64
}

// serve is the fraction of x's bytes each level serves, the caches
// nearest first and memory last. A level holds them all where they and
// everything touched since their last use fit in it; past that, the
// fraction its replacement keeps falls as Retain to the power of how many
// times over the level those bytes are, and a buffer larger than the level
// finds no more of itself there than the Cycle share of the level.
func (m *costModel) serve(x touch) [4]float64 {
	var out [4]float64
	i := len(m.recent) - 1
	for ; i >= 0; i-- {
		if t := m.recent[i]; t.o == x.o && t.code == x.code {
			break
		}
	}
	if i < 0 {
		out[3] = 1
		return out
	}
	since := x.bytes
	for _, t := range m.recent[i+1:] {
		since += t.bytes
	}
	left := 1.0
	for l, c := range m.d.Caches {
		if c.Bytes <= 0 || c.Kernel {
			continue
		}
		keep := 1.0
		if over := float64(since)/float64(c.Bytes) - 1; over > 0 {
			keep = math.Pow(c.Retain, over)
		}
		if x.bytes > c.Bytes {
			keep = min(keep, c.Cycle*float64(c.Bytes)/float64(x.bytes))
		}
		out[l] = left * keep
		left -= out[l]
	}
	out[3] = left
	return out
}

func (m *costModel) touch(o Operand, bytes int64) {
	m.touchOf(touch{o: o, bytes: bytes})
}

func (m *costModel) touchOf(x touch) {
	for i, t := range m.recent {
		if t.o == x.o && t.code == x.code {
			m.recent = append(m.recent[:i], m.recent[i+1:]...)
			break
		}
	}
	m.recent = append(m.recent, x)
}

// sectorBytes is the bytes a GPU moves for a lane at least: a sector.
const sectorBytes = 32

// lineBytes is the bytes a cache moves at once: a host's line, a GPU L1's.
func lineBytes(kind DeviceKind) int64 {
	if kind == DeviceGPU {
		return 128
	}
	return 64
}

// codeBytes is the bytes an instruction of a device's code takes on
// average: x86's encodings with their VEX prefixes, and a GPU's fixed
// sixteen.
func codeBytes(kind DeviceKind) int64 {
	if kind == DeviceGPU {
		return 16
	}
	return 6
}

// code is the bytes of kernel k's code: an instruction for each live node
// and store in each copy an unrolling made of its loop, and a loop's count
// and branch.
func (e *estimator) code(live map[int]bool, kind DeviceKind) int64 {
	k := e.k
	copies := func(h int) int64 {
		n := int64(1)
		for c := h; c >= 0; c = k.Loops[c].Parent {
			n *= int64(max(1, k.Loops[c].Copies))
		}
		return n
	}
	var n int64
	for i := range k.Nodes {
		if live[i] {
			n += copies(e.homes[i])
		}
	}
	for _, s := range k.Stores {
		n += copies(s.At)
	}
	for c := range k.Loops {
		n += 3 * copies(k.Loops[c].Parent)
	}
	return n * codeBytes(kind)
}

// rate is level l's rate for threads running together.
func (m *costModel) rate(l int, threads float64) float64 {
	if l < 3 {
		return m.d.Caches[l].Rate.at(threads)
	}
	return Rate{Thread: m.d.Thread, Total: m.d.Bandwidth}.at(threads)
}

// start is the seconds a stream from level l waits to start.
func (m *costModel) start(l int) float64 {
	if l < 3 {
		return m.d.Caches[l].Start
	}
	return m.d.Start
}

// latency is the seconds a read level l serves waits.
func (m *costModel) latency(l int) float64 {
	if l < 3 {
		return m.d.Caches[l].Latency
	}
	return m.d.Latency
}

// chain is the longest a fold of the kernel e counts makes its loop wait:
// a trip's critical path from the fold to the value it carries to the next
// trip, by the delay of each instruction's class, times the trips of an
// entry into the loop.
func (m *costModel) chain(e *estimator, live map[int]bool, words map[int]bool) float64 {
	k := e.k
	var out float64
	for f, nd := range k.Nodes {
		if !live[f] || nd.Op != OpFold || nd.At < 0 {
			continue
		}
		memo := map[int]float64{}
		var path func(i int) float64
		path = func(i int) float64 {
			if i == f {
				return 0
			}
			if v, ok := memo[i]; ok {
				return v
			}
			memo[i] = -1
			x := k.Nodes[i]
			best := -1.0
			for _, a := range k.args(&x) {
				best = max(best, path(*a))
			}
			if best < 0 {
				return -1
			}
			switch {
			case x.Op == OpLoad:
				best += m.d.Caches[0].Latency
			default:
				if cl, ok := classOf(k, i, words); ok {
					best += m.d.Delay[cl]
				}
			}
			memo[i] = best
			return best
		}
		if step := path(nd.A); step > 0 {
			out = max(out, step*e.maxTrips(nd.At))
		}
	}
	return out
}

// critical is the longest path through kernel k's instructions outside
// any loop, each waiting for the ones it reads by its class's delay, a
// load by the mean latency of its buffer's reads: what one thread takes
// with nothing else to issue.
func (m *costModel) critical(e *estimator, live, words map[int]bool, reads map[int]float64) float64 {
	k := e.k
	homes := e.homes
	memo := make([]float64, len(k.Nodes))
	done := make([]bool, len(k.Nodes))
	var depth func(i int) float64
	depth = func(i int) float64 {
		if done[i] {
			return memo[i]
		}
		done[i] = true
		nd := k.Nodes[i]
		var d float64
		for _, a := range k.args(&nd) {
			if *a < len(k.Nodes) && (*a >= len(homes) || homes[*a] < 0) {
				d = max(d, depth(*a))
			}
		}
		switch {
		case nd.Op == OpLoad:
			d += reads[nd.Buf]
		case e.folded(i):
		default:
			if cl, ok := classOf(k, i, words); ok {
				d += m.d.Delay[cl]
			}
		}
		memo[i] = d
		return d
	}
	var out float64
	for i := range k.Nodes {
		if live[i] && (i >= len(homes) || homes[i] < 0) {
			out = max(out, depth(i))
		}
	}
	return out
}

// inOrder is the seconds one GPU thread takes through kernel e counts,
// its subgroup issuing in order: each instruction, in the kernel's order,
// issues a slot after the one before it at the rate one thread issues its
// class, and no sooner than the instructions it reads are done, each its
// class's delay after it issued, a load its buffer's mean read latency --
// over the loads a thread keeps in flight where the load is in a loop and
// its place reads nothing the loop carries. A loop takes the thread's trips
// of it times a trip, a spread loop's trips dealt over the group; what it
// computes is done when it ends.
func (m *costModel) inOrder(e *estimator, live, words map[int]bool, reads map[int]float64) float64 {
	k := e.k
	// carried reports whether node i reads a value a loop carries: a load
	// at such a place waits for the trip before it, where a load at any
	// other is issued ahead of its use, the device's Overlap of them in
	// flight.
	memo := map[int]bool{}
	var carried func(i int) bool
	carried = func(i int) bool {
		if i < 0 || i >= len(k.Nodes) {
			return false
		}
		if v, ok := memo[i]; ok {
			return v
		}
		memo[i] = false
		nd := k.Nodes[i]
		v := nd.Op == OpFold
		for _, a := range k.args(&nd) {
			v = v || carried(*a)
		}
		memo[i] = v
		return v
	}
	cost := func(i int) (issue, delay float64) {
		nd := k.Nodes[i]
		if e.folded(i) {
			return 0, 0
		}
		cl, ok := classOf(k, i, words)
		if nd.Op == OpLoad {
			cl, ok = ClassMem, true
		}
		if !ok {
			return 0, 0
		}
		if r := m.d.Rates[cl].Thread; r > 0 {
			issue = 1 / r
		}
		delay = m.d.Delay[cl]
		if nd.Op == OpLoad {
			if lat, ok := reads[nd.Buf]; ok {
				delay = lat
			}
			if e.homes[i] >= 0 && !carried(nd.A) && !slices.ContainsFunc(nd.Ax, carried) {
				delay /= max(1, m.d.Overlap)
			}
		}
		return issue, delay
	}
	scope := func(i int) int {
		if i < len(e.homes) {
			return e.homes[i]
		}
		return -1
	}
	// within reports whether loop c sits in loop of, or is it.
	within := func(c, of int) bool {
		for ; c >= 0; c = k.Loops[c].Parent {
			if c == of {
				return true
			}
		}
		return of < 0
	}
	var run func(of int) float64
	run = func(of int) float64 {
		ready := map[int]float64{}
		var t, end float64
		seen := map[int]bool{}
		for i := range k.Nodes {
			if !live[i] {
				continue
			}
			h := scope(i)
			if h != of {
				child := h
				for child >= 0 && k.Loops[child].Parent != of {
					child = k.Loops[child].Parent
				}
				if child < 0 || !within(h, of) || seen[child] {
					continue
				}
				seen[child] = true
				trips := e.maxTrips(child)
				if k.Loops[child].Spread {
					trips = math.Ceil(trips / float64(max(1, k.Group)))
				}
				t += trips * run(child)
				end = max(end, t)
				continue
			}
			issue, delay := cost(i)
			start := t + issue
			nd := k.Nodes[i]
			for _, a := range k.args(&nd) {
				if r, ok := ready[*a]; ok {
					start = max(start, r)
				} else if *a < len(k.Nodes) && scope(*a) != of && within(scope(*a), of) {
					start = max(start, t)
				}
			}
			t = start
			ready[i] = start + delay
			end = max(end, ready[i])
		}
		return max(t, end)
	}
	return run(-1)
}

// exposed is the seconds a GPU kernel's threads wait for their loads: a
// subgroup issues in order, so a thread waits for each run of a load
// before what uses it, but for the device's Overlap of them at once, while
// different loads of a trip are in flight together. Each load's runs a
// thread times the latency of its buffer's reads, over the overlap; the
// longest of them.
func (m *costModel) exposed(e *estimator, live map[int]bool, wait map[int]float64) float64 {
	k := e.k
	elems := float64(max(1, k.elements()))
	var out float64
	for i, nd := range k.Nodes {
		if !live[i] || nd.Op != OpLoad || k.Bufs[nd.Buf].Kind == BufShared {
			continue
		}
		out = max(out, e.runs(i)/elems*wait[nd.Buf]/max(1, m.d.Overlap))
	}
	return out
}

// wait is the longest chain of dependent reads in kernel k: each load
// waits for the loads its index reads, then for the levels its buffer is
// served from, each by its share.
func (m *costModel) wait(k *Kernel, live map[int]bool, served map[int][4]float64) float64 {
	memo := make([]float64, len(k.Nodes))
	state := make([]uint8, len(k.Nodes))
	var chain func(i int) float64
	chain = func(i int) float64 {
		if state[i] != 0 {
			return memo[i]
		}
		state[i] = 1
		nd := k.Nodes[i]
		var w float64
		for _, a := range k.args(&nd) {
			w = max(w, chain(*a))
		}
		if nd.Op == OpLoad {
			for l, f := range served[nd.Buf] {
				w += f * m.latency(l)
			}
		}
		memo[i] = w
		return w
	}
	var out float64
	for i := range k.Nodes {
		if live[i] {
			out = max(out, chain(i))
		}
	}
	return out
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
		touched, unique, loads float64
		hull                   span
		write                  bool
	}
	uses := map[int]*use{}
	order := []int{}
	note := func(b int, loads, touched, unique float64, h span, write bool) {
		u, ok := uses[b]
		if !ok {
			u = &use{hull: span{math.Inf(1), math.Inf(-1)}}
			uses[b] = u
			order = append(order, b)
		}
		u.loads += loads
		u.touched += touched
		u.unique += unique
		u.hull = span{min(u.hull.lo, h.lo), max(u.hull.hi, h.hi)}
		u.write = u.write || write
	}
	live := k.live()
	words := k.vectorWords()
	// over is the bytes a GPU kernel's subgroups move past what their
	// lanes read: a lane whose place is a sector or more from its
	// neighbour's moves a sector of its own, served from the first cache
	// a launch does not start empty.
	var over float64
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
			if m.d.Kind == DeviceGPU {
				if st := e.laneStride(i); st > w {
					over += n * (min(st, sectorBytes) - min(w, sectorBytes))
				}
			}
			note(nd.Buf, n, n*w, e.footprint([]int{nd.A}, nd.Ax, e.homes[i], nd.Buf)*w, e.reach(i, nd.A, nd.Ax, nd.Buf, w), false)
			c.Issued[ClassMem] += int64(n)
			continue
		}
		if cl, ok := classOf(k, i, words); ok && !e.folded(i) {
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
		note(s.Buf, 0, n*w, e.footprint([]int{s.Ix}, s.Ax, s.At, s.Buf)*w, e.storeReach(s, w), true)
		c.Issued[ClassMem] += int64(n)
	}
	m.reach = map[int]int64{}
	served := map[int][4]float64{}
	// reads is, by buffer, the mean latency of a read the kernel makes of
	// it: a read of each line its bytes span from where they are served,
	// every other read from the nearest cache.
	reads := map[int]float64{}
	var start float64
	for _, b := range order {
		u := uses[b]
		buf := k.Bufs[b]
		if !math.IsInf(u.hull.hi, 1) {
			m.reach[b] = int64(u.hull.hi)
		} else if !math.IsInf(u.touched, 1) {
			m.reach[b] = int64(u.touched)
		}
		if shape := buf.Shape; shape != nil {
			n := int64(elemBytes(buf))
			for _, x := range shape {
				n *= int64(x)
			}
			u.unique = min(u.unique, float64(n))
		}
		u.unique = min(u.unique, u.touched, max(0, u.hull.hi-u.hull.lo))
		uniq := int64(u.unique)
		var from [4]float64
		from[m.fit(uniq)] = 1
		if !u.write {
			from = m.serve(touch{o: m.key(buf.Operand), bytes: uniq})
			served[b] = from
		}
		for l, f := range from {
			if f <= 0 {
				continue
			}
			c.Served[l] += int64(f * float64(uniq))
			c.MemSeconds += f * float64(uniq) / m.rate(l, threads)
			if !u.write && uniq > 0 && f > 0.05 {
				lines := math.Ceil(f * float64(uniq) / 64)
				start = max(start, min(m.start(l), lines*m.latency(l)))
			}
		}
		c.Bytes += uniq
		if again := u.touched - u.unique; again > 0 {
			near := m.fit(uniq)
			c.Served[near] += int64(again)
			c.MemSeconds += again / m.rate(near, threads)
		}
		if u.loads > 0 {
			var far float64
			for l, f := range from {
				far += f * m.latency(l)
			}
			lines := min(u.loads, math.Ceil(float64(uniq)/float64(lineBytes(m.d.Kind))))
			reads[b] = (lines*far + (u.loads-lines)*m.latency(0)) / u.loads
		}
		m.touch(m.key(buf.Operand), uniq)
	}
	if over > 0 {
		l := 0
		for l < len(m.d.Caches) && m.d.Caches[l].Kernel {
			l++
		}
		c.Served[l] += int64(over)
		c.MemSeconds += over / m.rate(l, threads)
	}
	if m.d.Kind == DeviceGPU {
		c.MemSeconds = 0
		var through float64
		for l := len(c.Served) - 1; l >= 0; l-- {
			if l < len(m.d.Caches) && m.d.Caches[l].Bytes <= 0 {
				continue
			}
			through += float64(c.Served[l])
			if r := m.rate(l, threads); r > 0 {
				c.MemSeconds = max(c.MemSeconds, through/r)
			}
		}
	}
	c.MemSeconds += start
	share := 1.0
	if m.d.Kind == DeviceGPU && m.d.Units > 0 {
		groups := max(1, k.elements())
		if !k.Work {
			groups = (groups + max(1, k.Group) - 1) / max(1, k.Group)
		}
		share = float64(min(groups, m.d.Units)) / float64(m.d.Units)
	}
	var all float64
	for cl, n := range c.Issued {
		if r := min(m.d.Rates[cl].at(threads), share*m.d.Rates[cl].Total); r > 0 && n > 0 {
			c.ComputeSeconds = max(c.ComputeSeconds, float64(n)/r)
		}
		all += float64(n)
	}
	if r := m.d.Issue.at(threads); r > 0 {
		c.ComputeSeconds = max(c.ComputeSeconds, all/r)
	}
	c.LaunchSeconds = m.d.Launch
	c.WaitSeconds = m.wait(k, live, served)
	if m.d.Kind == DeviceGPU {
		c.WaitSeconds = max(c.WaitSeconds, m.exposed(&e, live, reads))
	}
	code := touch{code: k, bytes: e.code(live, m.d.Kind)}
	for l, f := range m.serve(code) {
		if l > 0 {
			c.FetchSeconds += f * m.latency(l)
		}
	}
	m.touchOf(code)
	c.ChainSeconds = max(m.chain(&e, live, words), m.critical(&e, live, words, reads))
	if m.d.Kind == DeviceGPU {
		c.ChainSeconds = max(c.ChainSeconds, m.inOrder(&e, live, words, reads))
	}
	work := max(c.MemSeconds, c.ComputeSeconds, c.WaitSeconds, c.ChainSeconds) + c.FetchSeconds
	c.Seconds = work + c.LaunchSeconds
	if m.d.Kind == DeviceGPU {
		c.FloorSeconds = m.d.Floor
		c.Seconds = max(work+c.FloorSeconds, c.LaunchSeconds)
	}
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
	elem []int
	// lane is the thread of the element's group eval takes.
	lane   int
	byElem map[int]bool
	// fold memoises folded: 1 folded, 2 not, 3 being decided.
	fold   []int8
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
	case OpThread, OpLocal:
		return float64(e.lane), true
	case OpFlat:
		x := 0
		for a, c := range e.elem {
			x = x*max(1, k.Space[a]) + c
		}
		return float64(x), true
	case OpIter:
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

// folded reports whether node i is a value the device's compiler computes
// before the kernel runs: a constant, or pure arithmetic on constants. Such
// an instruction becomes an immediate and is never issued.
func (e *estimator) folded(i int) bool {
	k := e.k
	if i >= len(k.Nodes) {
		return false
	}
	if e.fold == nil {
		e.fold = make([]int8, len(k.Nodes))
	}
	switch e.fold[i] {
	case 1:
		return true
	case 2, 3:
		return false
	}
	e.fold[i] = 3
	nd := k.Nodes[i]
	ok := true
	switch nd.Op {
	case OpConst:
	case OpLoad, OpParam, OpLength, OpCoord, OpThread, OpLocal, OpFlat, OpIter, OpFold, OpSpan, OpLanes,
		OpGroupSum, OpGroupMax, OpExchange:
		ok = false
	default:
		for _, a := range k.args(&nd) {
			if !e.folded(*a) {
				ok = false
				break
			}
		}
	}
	e.fold[i] = 2
	if ok {
		e.fold[i] = 1
	}
	return ok
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

// laneStride is the bytes between the places load i reads for two
// neighbouring lanes of a subgroup at the first trip: the next thread of
// a group where a group shares the element, the next element else; 0 where
// the step does not decide it.
func (e *estimator) laneStride(i int) float64 {
	k := e.k
	nd := k.Nodes[i]
	at := func() (float64, bool) {
		if nd.Ax == nil {
			return e.eval(nd.A, 0)
		}
		sh := k.Bufs[nd.Buf].Shape
		if len(sh) != len(nd.Ax) {
			return 0, false
		}
		var x float64
		for j, a := range nd.Ax {
			v, ok := e.eval(a, 0)
			if !ok {
				return 0, false
			}
			x = x*float64(sh[j]) + v
		}
		return x, true
	}
	elem, lane := e.elem, e.lane
	defer func() { e.elem, e.lane = elem, lane }()
	e.elem, e.lane = e.coords(0), 0
	a, ok := at()
	if !ok {
		return 0
	}
	if k.Work {
		e.lane = 1
	} else if k.elements() > 1 {
		e.elem = e.coords(1)
	} else {
		return 0
	}
	b, ok := at()
	if !ok {
		return 0
	}
	return math.Abs(b-a) * float64(elemBytes(k.Bufs[nd.Buf]))
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
		if nd.Op == OpDiv && k.Nodes[nd.B].Op != OpConst {
			return ClassSpecial, true
		}
	case OpRem:
		if k.Nodes[nd.B].Op != OpConst {
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
