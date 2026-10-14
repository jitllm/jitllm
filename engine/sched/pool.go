// Package sched runs work across the machine's performance cores.
//
// The pool is portable: a sequence number, a spin and a dynamic chunk claim.
// Only topology differs per OS (which cores are performance cores), and that
// lives in topo_*.go.
//
// It does not bind a thread to a CPU (see worker). Confining the process is the
// caller's job (scripts/cap uses taskset); where a worker runs inside that mask
// is the Go scheduler's business.
//
// Core selection matters more than thread count: a few P-cores reach most of
// DRAM read bandwidth, E-cores add none and become the straggler behind every
// barrier, and leaving the OS no core is worse than using fewer. So this never
// uses runtime.NumCPU() unpartitioned.
package sched

import (
	"context"
	"runtime"
	"runtime/pprof"
	"sync/atomic"
	"time"
)

// Option configures a Pool or the core set it runs on. The package reads no
// environment; cmd/jitllm maps JITLLM_CORESET, JITLLM_CORES and JITLLM_SPIN_US
// onto these options.
type Option func(*config)

type config struct {
	coreset string
	cores   int
	spin    time.Duration
	spinSet bool
	numa    bool
}

func newConfig(opts []Option) config {
	var c config
	for _, o := range opts {
		o(&c)
	}
	return c
}

// WithCoreSet picks which cores the pool may use: "" or "p" for P-cores, "pe"
// to add the E-cores, "smt" for hyperthread siblings, "all" for everything.
func WithCoreSet(name string) Option { return func(c *config) { c.coreset = name } }

// WithCores caps how many of them to take. It exists to measure scaling, not to
// tune: a scaling curve tells "memory-bound and saturated" apart from
// "threading is broken".
func WithCores(n int) Option { return func(c *config) { c.cores = n } }

// WithNUMA makes DoNodes node-affine on a host with more than one node. It
// binds nothing: see drainNodes for why.
func WithNUMA(on bool) Option { return func(c *config) { c.numa = on } }

// SetSpinning switches the workers between spinning for the next region (the
// pool's budget, WithSpin's or the derived one) and parking at once. A caller
// whose regions are separated by blocking reads parks them, so a worker does
// not burn a core waiting on a disk; one whose regions follow each other
// closely spins, because waking a parked worker at every region costs more
// than the spin. It takes effect at each worker's next wait.
func (p *Pool) SetSpinning(on bool) {
	p = p.w()
	if on {
		p.spin.Store(int64(p.spinDef))
	} else {
		p.spin.Store(0)
	}
}

// SetSpinFor has the workers spin for d before parking, until the next
// SetSpinning: for a caller whose regions are separated by a wait that uses
// no core -- a device it is suspended inside of -- longer than the pool's
// budget, so the workers are spinning when the next region arrives rather
// than woken one by one for it.
func (p *Pool) SetSpinFor(d time.Duration) {
	p.w().spin.Store(int64(max(d, 0)))
}

// Spinning reports whether the workers spin for the next region rather than
// park at once.
func (p *Pool) Spinning() bool { return p.w().spin.Load() > 0 }

// WithSpin sets how long a worker spins before parking. Zero parks immediately.
func WithSpin(d time.Duration) Option {
	return func(c *config) { c.spin, c.spinSet = d, true }
}

// CoreSet returns the CPUs of a named pool layout:
//
//	p    physical P-cores only          (default; decode is bandwidth-bound)
//	pe   P-cores then E-cores
//	smt  both hyperthreads of each P-core
//	all  everything the OS reports
//
// The non-default sets exist so the claims behind the default stay falsifiable.
func CoreSet(name string) []int {
	switch name {
	case "pe":
		return append(PCores(), ECores()...)
	case "smt":
		return SMTSiblings()
	case "all":
		return append(SMTSiblings(), ECores()...)
	}
	return PCores()
}

// DecodeCores is the pool for batch-1 decode: every P-core, or WithCores(n).
func DecodeCores(opts ...Option) []int {
	c := newConfig(opts)
	p := CoreSet(c.coreset)
	if len(p) == 0 {
		return []int{0}
	}
	// There is no portable right number, so the pool takes them all and the
	// tuned Pool.SetParticipants decides how many drain a given region.
	n := len(p)
	if c.cores > 0 {
		n = c.cores
	}
	if len(p) > n {
		p = p[:n]
	}
	return p
}

type job struct {
	fn func(worker, lo, hi int)
	// rfn is DoRange's body, which does not want the worker id. It is a field
	// of its own so the caller's func is stored as it is: adapting it to fn's
	// signature takes a closure, and that closure is a heap allocation on every
	// region.
	rfn   func(lo, hi int)
	total int
	chunk int
	limit int    // participants that drain this job; the rest wake and go back
	label string // pprof label, empty when profiling is off
	block int    // >0: DoNodes -- [0,total) is blocks of this many, block b on node (first+b)%nodes
	first int
}

// Pool is a set of workers, one per core the pool was built for, plus the
// calling goroutine, which is participant 0: len(cpus) participants means
// len(cpus)-1 goroutines. A blocked caller would make len(cpus)+1 runnable
// goroutines compete for len(cpus) cores.
//
// A job is published by bumping seq and workers spin on it; a channel per
// worker cost far more per empty region (see
// docs/engineering-history/scheduling-and-measurement.md). The channel survives
// only as the parking fallback once the spin budget runs out.
type Pool struct {
	// nParallel and nSerial count Do calls that dispatched and Do calls that
	// ran inline because the region was too small to be worth waking anyone.
	nParallel, nSerial atomic.Int64
	nAffine            atomic.Int64 // regions DoNodes dispatched node-affine

	cpus  []int
	start []chan struct{} // parking fallback, one per spawned worker
	quit  chan struct{}
	j     job
	next  atomic.Int64
	// limit is p.j.limit, published before the sequence bump so a worker can
	// decide whether to participate without reading p.j.
	limit atomic.Int64

	seq atomic.Uint64 // bumped by Do to publish a job
	// gate is the current region's door: its epoch (the seq that published
	// it) above gateEpochShift, gateClosed, and how many workers joined. A
	// worker touches p.j only after joining an open gate of the epoch it saw;
	// left counts the joined workers that are done.
	gate    atomic.Uint64
	left    atomic.Int64
	parked  []atomic.Bool // spawned worker i is blocked on start[i]
	stopped atomic.Bool
	// spin is how long a worker spins before parking, in nanoseconds, read on
	// every wait; spinDef is the pool's own budget, which SetSpinning restores.
	spin    atomic.Int64
	spinDef time.Duration
	part    atomic.Int64 // participants that drain; <= len(cpus)

	// NUMA, set only by WithNUMA on a host of two or more nodes: cpuNode maps
	// a CPU to its node index (into NUMANodes()), nodeSet is each node's CPUs,
	// and nnext is DoNodes' per-node claim.
	cpuNode []int
	nodeSet [][]int
	nodeIDs []int
	nnext   []atomic.Int64

	// crew is the pool whose workers a shared view (Shared) runs on; nil for a
	// pool that owns its workers. A view has its own participant count and
	// counters; the crew's lock makes its views take turns, a region each.
	crew *Pool
	// mu is held by a view's region on the crew, shared with every crew
	// whose cores overlap this one's (Shared); key and views are the crew's
	// registry entry and how many views are open. wake is a view's channel
	// for mu's handoff.
	mu    *turn
	wake  chan struct{}
	key   string
	views int
	// beside is a view opened beside another of the same caller's
	// (SharedBeside), which the turn's users do not count.
	beside bool
	// fz is an open fused region's state (fuse.go), on the crew.
	fz fused
}

// w is the pool whose workers run p's regions.
func (p *Pool) w() *Pool {
	if p.crew != nil {
		return p.crew
	}
	return p
}

// spinBudget is how long a worker re-reads seq before parking. It has to cover
// the tail of the gap between regions (layer boundaries, attention, sampling),
// not its mean; spinning much past 1 ms gets worse again, because a worker that
// never parks competes with the caller's serial stretches. The sweep is in
// docs/engineering-history/scheduling-and-measurement.md.
func spinBudget(c config) time.Duration {
	if c.spinSet && c.spin >= 0 {
		return c.spin
	}
	return 250 * time.Microsecond
}

// New starts one worker per cpu beyond the first; the caller is the
// participant on cpus[0]. The workers live until Close.
func New(cpus []int, opts ...Option) *Pool {
	cfg := newConfig(opts)
	if len(cpus) == 0 {
		cpus = DecodeCores(opts...)
	}
	p := &Pool{
		cpus:   cpus,
		start:  make([]chan struct{}, len(cpus)-1),
		parked: make([]atomic.Bool, len(cpus)-1),
		quit:   make(chan struct{}),
	}
	p.spinDef = spinBudget(cfg)
	p.part.Store(int64(len(cpus)))
	if cfg.numa {
		p.initNUMA()
		// A longer spin across sockets, unless the caller chose one: waking
		// parked workers on two sockets costs more than the spin.
		if p.cpuNode != nil && !cfg.spinSet {
			p.spinDef = 2 * time.Millisecond
		}
	}
	p.spin.Store(int64(p.spinDef))
	for i := range p.start {
		p.start[i] = make(chan struct{}, 1)
		go p.worker(i)
	}
	return p
}

// N is the participant count, the caller included.
func (p *Pool) N() int {
	if p == nil {
		return 1
	}
	return int(p.part.Load())
}

// Max is how many participants the pool could use, ignoring any limit.
func (p *Pool) Max() int {
	if p == nil {
		return 1
	}
	return len(p.w().cpus)
}

// SetParticipants caps how many participants drain a region without rebuilding
// the pool, so the count can be tuned rather than asserted: the knee is not a
// property of the core count (a prefill GEMM on six P-cores scales to five and
// collapses at six, leaving Go's runtime threads nowhere to run).
//
// Workers above the limit still wake and still acknowledge; they skip the drain.
func (p *Pool) SetParticipants(n int) {
	if p == nil {
		return
	}
	if n < 1 {
		n = 1
	}
	if n > p.Max() {
		n = p.Max()
	}
	p.part.Store(int64(n))
}

func (p *Pool) worker(id int) {
	// No LockOSThread and no sched_setaffinity: a per-CPU binding was never
	// shown to help, and it leaked -- the mask outlived the unlocked thread in
	// the runtime's pool and confined whatever ran on it next. See AGENTS.md
	// RULE 5.
	seen := uint64(0)
	for {
		s, ok := p.await(id, seen)
		if !ok {
			return
		}
		seen = s
		// A participant joins the region before it reads p.j, and only while
		// the region's gate is open: the caller closes it once it has no chunk
		// left to claim and waits only for the workers that joined. A worker
		// the OS or the Go scheduler kept off a CPU past that point finds the
		// gate closed and touches nothing, so a region never waits on a worker
		// that was not running (it once waited on every worker, participant
		// or not, and an oversubscribed host stalled a region per scheduling
		// slice of the slowest one). A capped-out worker never joins, so it
		// cannot be reading p.j while Do writes the next job (it once crashed
		// with a negative index under MoE, whose participant limit swings per
		// region).
		workerWake(p, id)
		if id+1 < int(p.limit.Load()) && p.join(s) {
			p.run(id + 1)
			p.left.Add(1)
		}
	}
}

// Layout of Pool.gate.
const (
	gateEpochShift = 17
	gateClosed     = 1 << 16
	gateCount      = gateClosed - 1
)

// join enters the region published at seq s, reporting false when its gate is
// already closed or belongs to another region.
func (p *Pool) join(s uint64) bool {
	for {
		g := p.gate.Load()
		if g>>gateEpochShift != s&(1<<(64-gateEpochShift)-1) || g&gateClosed != 0 {
			return false
		}
		if p.gate.CompareAndSwap(g, g+1) {
			return true
		}
	}
}

// run drains the current job, under a pprof label when one is set.
//
// Generated code is invisible to the profiler: its PCs are outside Go's
// moduledata, so every sample lands in runtime._ExternalCode. Labels are
// attached to the goroutine rather than derived from the PC, so
// `pprof -tagfocus` can separate kernels that the call graph cannot.
func (p *Pool) run(id int) {
	// A worker that joined a fused region (fuse.go) runs its phases. Asked of
	// the region rather than of p.j: a late joiner finds p.j already holding
	// the first phase's job.
	if p.fz.owner.Load() != nil {
		p.phaseLoop(id)
		return
	}
	if id >= p.j.limit {
		return // above the tuned participant count for this region
	}
	if p.j.label == "" {
		p.drain(id)
		return
	}
	pprof.Do(context.Background(),
		pprof.Labels("kernel", p.j.label),
		func(context.Context) { p.drain(id) })
}

// await blocks until Do publishes a sequence newer than seen.
//
// The park handshake is Dekker's: the worker stores parked then re-reads seq,
// and Do bumps seq then reads parked. Go's sync/atomic is sequentially
// consistent, so the two cannot both observe the stale value -- either the
// worker sees the new seq and does not park, or Do sees parked and sends. A
// send that arrives after the worker un-parked is left in the buffered channel
// and drained on the next pass, where it reads as a spurious wake and the outer
// loop simply re-checks seq.
func (p *Pool) await(id int, seen uint64) (uint64, bool) {
	for {
		// The clock is read once per 1024 spins, not per spin: a job that is
		// already waiting must be picked up in nanoseconds, and time.Now costs
		// more than the atomic load it would be guarding.
		var deadline int64
		for n := 0; ; n++ {
			if s := p.seq.Load(); s != seen {
				return s, p.alive()
			}
			if n&1023 == 1023 {
				now := time.Now().UnixNano()
				if deadline == 0 {
					deadline = now + p.spin.Load()
				} else if now > deadline {
					break
				}
			}
		}
		p.parked[id].Store(true)
		if s := p.seq.Load(); s != seen {
			p.parked[id].Store(false)
			select {
			case <-p.start[id]:
			default:
			}
			return s, p.alive()
		}
		select {
		case <-p.quit:
			return 0, false
		case <-p.start[id]:
		}
		p.parked[id].Store(false)
		if !p.alive() {
			return 0, false
		}
	}
}

func (p *Pool) alive() bool { return !p.stopped.Load() }

// drain claims chunks until the work is gone. Dynamic claiming, never a static
// split: the P/E throughput ratio differs by kernel, so a slower core simply
// claims fewer chunks.
func (p *Pool) drain(id int) {
	// Snapshot the job once: reading p.j.chunk twice in one expression can see
	// two values if a straggler overlaps the next job, giving a negative lo.
	chunk, total, fn, rfn := p.j.chunk, p.j.total, p.j.fn, p.j.rfn
	if block := p.j.block; block > 0 {
		p.drainNodes(id, chunk, total, block, p.j.first, fn)
		return
	}
	for {
		lo := int(p.next.Add(int64(chunk))) - chunk
		if lo >= total {
			return
		}
		hi := lo + chunk
		if hi > total {
			hi = total
		}
		if rfn != nil {
			rfn(lo, hi)
		} else {
			fn(id, lo, hi)
		}
	}
}

// Do runs fn over [0, total) split into chunks, across every participant, and
// returns when all of it is done. Not reentrant: one Pool runs one region at a
// time. DoLabeled is Do with a pprof label applied for the duration of the
// region, so time spent in generated code can be attributed to a kernel.
// DoNodes is DoLabeled for work laid out across NUMA nodes: [0,total) is
// blocks of `block` indices, and block b's memory is on node index
// (first+b) % Nodes(). Each participant drains the node it is running on
// first and then steals, so the answer never depends on the placement -- only
// the traffic does. A pool without WithNUMA, or a one-node host, is plain
// DoLabeled.
func (p *Pool) DoNodes(label string, total, chunk, block, first int, fn func(worker, lo, hi int)) {
	if p == nil || p.w().cpuNode == nil || block <= 0 || chunk <= 0 {
		p.DoLabeled(label, total, chunk, fn)
		return
	}
	if !regionLabels.Load() {
		label = ""
	}
	p.nAffine.Add(1)
	p.dispatch(job{fn: fn, label: label, total: total, chunk: chunk, block: block, first: first})
}

// Affine is how many regions DoNodes dispatched node-affine: the check that
// the NUMA path RAN, which a rate cannot tell from one that never did.
func (p *Pool) Affine() int64 {
	if p == nil {
		return 0
	}
	return p.nAffine.Load()
}

// Nodes is how many NUMA nodes DoNodes distributes over; 0 without WithNUMA
// on a multi-node host.
func (p *Pool) Nodes() int {
	if p == nil {
		return 0
	}
	return len(p.w().nodeIDs)
}

// NodeIndex maps a node ID (as sched.NodeOf reports it) to DoNodes' index, or
// -1 for a node the pool does not span.
func (p *Pool) NodeIndex(node int) int {
	if p != nil {
		for i, n := range p.w().nodeIDs {
			if n == node {
				return i
			}
		}
	}
	return -1
}

func (p *Pool) initNUMA() {
	ids, sets := nodeCPUs()
	if len(ids) < 2 {
		return
	}
	maxCPU := 0
	for _, set := range sets {
		for _, c := range set {
			maxCPU = max(maxCPU, c)
		}
	}
	p.cpuNode = make([]int, maxCPU+1)
	p.nodeSet, p.nodeIDs = sets, ids
	for i := range p.cpuNode {
		p.cpuNode[i] = -1
	}
	for n, set := range sets {
		for _, c := range set {
			p.cpuNode[c] = n
		}
	}
	p.nnext = make([]atomic.Int64, len(ids))
}

// drainNodes claims chunks from the node the participant is running on first,
// then from every other node in turn. Node m owns blocks m', m'+N, ... (m' its
// offset from first); its claim counter walks those blocks' indices in order,
// and a chunk that runs off the end of one continues in the node's next, as
// two calls (claimNode).
//
// The node is asked before every claim, not once per region: a worker can
// resume on another thread between chunks, and a once-per-region answer sends
// chunks to the wrong node. getcpu is ~0.1 us against a chunk of tens of
// microseconds.
//
// No worker is bound to its node: locked threads piled onto a few CPUs and
// could not give their thread up when a spinning worker was preempted.
//
// A chunk may cross a block: forcing chunk to divide block leaves too few
// chunks for the workers on tall matrices, enlarging the region.
func (p *Pool) drainNodes(id, chunk, total, block, first int, fn func(worker, lo, hi int)) {
	n := len(p.nnext)
	for {
		me := 0
		if c := currentCPU(); c >= 0 && c < len(p.cpuNode) && p.cpuNode[c] >= 0 {
			me = p.cpuNode[c]
		}
		claimed := false
		for s := 0; s < n && !claimed; s++ {
			claimed = p.claimNode((me+s)%n, chunk, total, block, first, id, fn)
		}
		if !claimed {
			return
		}
	}
}

// claimNode runs one chunk of node m's blocks, as one call or two when it
// crosses into the node's next block, and reports false once m has none left.
func (p *Pool) claimNode(m, chunk, total, block, first, id int, fn func(worker, lo, hi int)) bool {
	n := len(p.nnext)
	b0 := ((m-first)%n + n) % n
	k := int(p.nnext[m].Add(int64(chunk))) - chunk
	ran := false
	for rem := chunk; rem > 0; {
		off := k % block
		lo := (b0+(k/block)*n)*block + off
		if lo >= total {
			break
		}
		l := min(rem, block-off)
		fn(id, lo, min(lo+l, total))
		k, rem, ran = k+l, rem-l, true
	}
	return ran
}

func (p *Pool) DoLabeled(label string, total, chunk int, fn func(worker, lo, hi int)) {
	if !regionLabels.Load() {
		p.do("", total, chunk, fn)
		return
	}
	p.do(label, total, chunk, fn)
}

// regionLabels is whether a region carries a pprof label at all. Off by
// default: pprof.Do allocates a label map and a context per participant per
// region (most of a token's allocations, and on macOS an madvise per span).
// Only a CPU profile needs them; cmd/jitllm turns them on with -cpuprofile
// and -tgprofile.
var regionLabels atomic.Bool

// RegionLabels reports whether regions carry pprof labels.
func RegionLabels() bool { return regionLabels.Load() }

// SetRegionLabels turns pprof region labels on or off, and reports what it was.
func SetRegionLabels(on bool) bool { return regionLabels.Swap(on) }

func (p *Pool) Do(total, chunk int, fn func(worker, lo, hi int)) {
	p.do("", total, chunk, fn)
}

// do is Do with the pprof label passed as an argument rather than a Pool field,
// so it cannot leak between callers as a mislabelled profile.
func (p *Pool) do(label string, total, chunk int, fn func(worker, lo, hi int)) {
	p.dispatch(job{fn: fn, label: label, total: total, chunk: chunk})
}

// DoRange is Do for a body that does not use the worker id.
func (p *Pool) DoRange(total, chunk int, fn func(lo, hi int)) {
	p.dispatch(job{rfn: fn, total: total, chunk: chunk})
}

// call runs one chunk of j's body.
func (j *job) call(worker, lo, hi int) {
	if j.rfn != nil {
		j.rfn(lo, hi)
		return
	}
	j.fn(worker, lo, hi)
}

// dispatch publishes j, whose body, label, total, chunk and NUMA block are the
// caller's; it fills in the participant limit.
func (p *Pool) dispatch(j job) {
	total, chunk := j.total, j.chunk
	if p == nil || total <= 0 {
		if total > 0 {
			j.call(0, 0, total)
		}
		return
	}
	if chunk < 1 {
		chunk = 1
		j.chunk = 1
	}
	// Below roughly one chunk per worker, dispatch costs more than it saves.
	if total <= chunk {
		p.nSerial.Add(1)
		j.call(0, 0, total)
		return
	}
	part := int(p.part.Load())
	// Inside a fused region the caller's region is a phase of it (fuse.go).
	if c := p.w(); c.fz.owner.Load() == p {
		c.phase(j, part)
		return
	}
	if p.crew != nil {
		// A view runs on its crew, one view's region at a time: the workers
		// and the published job are the crew's. The participant count is the
		// view's own, so each user's tuner keeps its answer.
		c := p.crew
		c.mu.Lock(p.wake)
		defer c.mu.Unlock()
		if c.stopped.Load() {
			p.nSerial.Add(1)
			j.call(0, 0, total)
			return
		}
		p.nParallel.Add(1)
		c.publish(j, part)
		return
	}
	// A stopped pool runs the region inline: its workers have returned.
	if p.stopped.Load() {
		p.nSerial.Add(1)
		j.call(0, 0, total)
		return
	}
	p.nParallel.Add(1)
	p.publish(j, part)
}

// publish runs j on p's workers, at most part participants, and returns when
// all of it is done.
func (p *Pool) publish(j job, part int) {
	total, chunk := j.total, j.chunk
	// The tuned participant count is an upper bound; never use more
	// participants than there are chunks.
	limit := part
	if c := (total + chunk - 1) / chunk; c < limit {
		limit = c
	}
	j.limit = limit
	p.j = j
	// Published before the sequence bump; see the worker loop.
	p.limit.Store(int64(limit))
	p.next.Store(0)
	for i := range p.nnext {
		p.nnext[i].Store(0)
	}
	// The gate opens for the new epoch before the sequence bump publishes it;
	// see the worker loop.
	p.left.Store(0)
	p.gate.Store((p.seq.Load() + 1) << gateEpochShift)
	p.seq.Add(1)
	// Only participants are woken: the rest never join.
	for i := range min(len(p.parked), limit-1) {
		if p.parked[i].Load() {
			select {
			case p.start[i] <- struct{}{}:
			default:
			}
		}
	}
	// The caller is participant 0 and claims chunks like everyone else.
	p.run(0)
	// No chunk is left to claim, so a worker that has not joined has nothing
	// to do: close the gate and wait for the ones that did.
	joined := int64(p.gate.Or(gateClosed) & gateCount)
	for p.left.Load() != joined {
		runtime.Gosched()
	}
}

// Close stops the workers. It is idempotent and safe to call from any
// goroutine; a Do that races it runs the region inline rather than hanging.
//
// It does not join, which would block Close behind an arbitrary caller's fn;
// it guarantees only that no further region is dispatched.
func (p *Pool) Close() {
	if p == nil {
		return
	}
	if p.crew != nil {
		p.crew.release(p)
		return
	}
	if !p.stopped.CompareAndSwap(false, true) {
		return // already closed; quit is already closed too
	}
	p.seq.Add(1)
	close(p.quit)
	for i := range p.start {
		select {
		case p.start[i] <- struct{}{}:
		default:
		}
	}
}

// Regions reports how many Do calls dispatched to workers and how many ran
// inline, since the last ResetRegions.
//
// Only the parallel ones cost a round trip, so only those are what fusion
// could remove.
func (p *Pool) Regions() (parallel, serial int64) {
	if p == nil {
		return 0, 0
	}
	return p.nParallel.Load(), p.nSerial.Load()
}

func (p *Pool) ResetRegions() {
	if p == nil {
		return
	}
	p.nParallel.Store(0)
	p.nSerial.Store(0)
}
