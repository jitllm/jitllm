package cpu

import (
	"runtime"

	"github.com/jitllm/jitllm/engine/sched"
)

// Pool is the CPU device's workers: a view of the process's workers on the
// decode cores (sched.Shared), every instance's a view of the same crew, so
// sessions on one host take turns a region at a time rather than spinning
// two crews on the same cores. Beside it, the prefill pool where a prompt
// runs on more cores than a token, and the E-cores' pool a served prefill
// takes beside a session decoding on the P-cores. One caller drives it, one
// phase at a time.
type Pool struct {
	// run is the pool regions go to: decode, or the prefill or E-cores' pool
	// while a prefill holds it.
	run, decode, prefill, e *sched.Pool
	// procs is the GOMAXPROCS request a prefill made, live until it ends.
	procs sched.ProcsHold
	cfg   poolConfig
	onE   bool
	// qos lowers a served prefill's thread to the E-cores (LowerToE).
	qos sched.QoS
}

type poolConfig struct {
	sched   []sched.Option
	prefill []int
	split   bool
	part    int
}

// PoolOption configures NewPool.
type PoolOption func(*poolConfig)

// WithSched passes options to the pools: their cores (sched.WithCores,
// sched.WithCoreSet), NUMA and spinning.
func WithSched(o ...sched.Option) PoolOption {
	return func(c *poolConfig) { c.sched = append(c.sched, o...) }
}

// WithPrefillCPUs runs a prompt's regions on cpus. Without it the prompt
// takes the host's default: the P and E cores on Apple Silicon, where
// prefill is compute the E-cores add to, and the decode cores elsewhere,
// where the split measured neutral (docs/engineering-history/cpu-kernels.md).
func WithPrefillCPUs(cpus []int) PoolOption {
	return func(c *poolConfig) { c.prefill = cpus }
}

// WithServingSplit lets a served prefill run on the E-cores beside a session
// decoding on the P-cores (BeginPrefillE).
func WithServingSplit(on bool) PoolOption { return func(c *poolConfig) { c.split = on } }

// WithParticipants caps how many workers drain each region; 0 takes them all.
func WithParticipants(n int) PoolOption { return func(c *poolConfig) { c.part = n } }

// NewPool is the CPU device's workers on the cores opts name.
func NewPool(opts ...PoolOption) *Pool {
	p := &Pool{}
	for _, o := range opts {
		o(&p.cfg)
	}
	c := &p.cfg
	p.qos = sched.QoSOf(c.sched...)
	p.decode = sched.Shared(sched.DecodeCores(c.sched...), c.sched...)
	cs := c.prefill
	if cs == nil && runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		cs = append(sched.PCores(), sched.ECores()...)
	}
	// Not when the caller narrowed the decode pool below the P cores: a
	// prefill pool wider than what was asked for would quietly ignore the
	// request.
	if len(cs) > p.decode.Max() && p.decode.Max() >= len(sched.DecodeCores()) && len(cs) <= runtime.NumCPU() {
		p.prefill = sched.SharedBeside(cs, p.decode, c.sched...)
	}
	if c.part > 0 {
		p.decode.SetParticipants(c.part)
		p.prefill.SetParticipants(c.part)
	}
	p.run = p.decode
	return p
}

// Workers is how many workers drain a region now, the caller included.
func (p *Pool) Workers() int { return p.run.N() }

// Max is the most workers any of the pool's regions may take: the scratch a
// region's workers each hold is sized by it.
func (p *Pool) Max() int {
	n := p.decode.Max()
	if p.prefill != nil {
		n = max(n, p.prefill.Max())
	}
	if p.cfg.split && p.qos.CanSplitE() {
		n = max(n, len(sched.ECores()))
	}
	return n
}

// Do runs fn over [0, n) in ranges of chunk, each with the id, below
// Workers, of the worker running it.
func (p *Pool) Do(n, chunk int, fn func(worker, lo, hi int)) {
	if n > 0 {
		p.run.Do(n, chunk, fn)
	}
}

// Fuse opens a fused region (sched.Pool.Fuse): until Unfuse every region is
// a phase of one. It reports whether it opened one; a pool another user
// shares, a NUMA pool and a one-core pool do not fuse.
func (p *Pool) Fuse() bool { return p.run.Fuse() }

// Unfuse closes the region a true Fuse opened.
func (p *Pool) Unfuse() { p.run.Unfuse() }

// SetSpinning switches the workers between spinning for the next region and
// parking at once (sched.Pool.SetSpinning).
func (p *Pool) SetSpinning(on bool) { p.run.SetSpinning(on) }

// Spinning reports what SetSpinning last chose.
func (p *Pool) Spinning() bool { return p.run.Spinning() }

// Regions is how many parallel and serial regions the pool has run.
func (p *Pool) Regions() (parallel, serial int64) { return p.run.Regions() }

// Phases is how many phases fused regions have run: the regions fusion
// folded into one, which Regions does not count.
func (p *Pool) Phases() int64 { return p.run.Phases() }

// ResetRegions zeroes the region counters.
func (p *Pool) ResetRegions() { p.run.ResetRegions() }

// NUMA is the node count the pool distributes over (0 when it does not) and
// how many regions it has dispatched node-affine.
func (p *Pool) NUMA() (nodes int, affine int64) { return p.run.Nodes(), p.run.Affine() }

// BeginPrefill runs every region until EndPrefill on the prefill pool, where
// there is one (WithPrefillCPUs), and raises GOMAXPROCS for it: a wide pool
// on too few Ps is far slower, and leaving GOMAXPROCS wide costs decode. The
// raise goes through sched.ProcsHold, which serves several models'
// overlapping prefills and restores the value once the last one ends.
func (p *Pool) BeginPrefill() {
	if p.prefill == nil {
		return
	}
	p.run = p.prefill
	p.procs.Raise(p.prefill.Max())
}

// EndPrefill returns to the decode pool and GOMAXPROCS; see BeginPrefill.
func (p *Pool) EndPrefill() {
	p.run = p.decode
	p.procs.Release()
}

// BeginPrefillE runs every region until EndPrefillE on the E-cores' pool,
// and reports whether it did: not unless WithServingSplit is set and the
// host has E-cores a thread can ask for (sched.CanSplitE). The decode view
// on the P-cores is parked meanwhile (sched.Pool.Park), so a session
// decoding there is its cores' only user and can fuse. The caller runs the
// prefill on a thread at the E-core class (sched.LowerToE): it is
// participant 0 of every region.
func (p *Pool) BeginPrefillE() bool {
	if p.onE || !p.cfg.split || !p.qos.CanSplitE() {
		return false
	}
	if p.e == nil {
		p.e = sched.Shared(sched.ECores(), p.cfg.sched...)
		p.e.SetRegionLabels(p.decode.RegionLabels())
	}
	p.decode.Park()
	p.run, p.onE = p.e, true
	p.procs.Raise(p.e.Max())
	return true
}

// EndPrefillE returns from the E-cores' pool to the decode pool and takes
// the decode view back; see BeginPrefillE.
func (p *Pool) EndPrefillE() {
	if !p.onE {
		return
	}
	p.run, p.onE = p.decode, false
	p.decode.Unpark()
	p.procs.Release()
}

// Close releases the pool's views of the crew.
// LowerToE puts the calling goroutine's thread at the E-core class for a
// prefill on the E-cores (sched.QoS.LowerToE).
func (p *Pool) LowerToE() sched.Lowered { return p.qos.LowerToE() }

// SetRegionLabels turns the pprof region labels of every pool this one runs
// its regions on on or off (sched.Pool.SetRegionLabels), and reports what
// they were.
func (p *Pool) SetRegionLabels(on bool) bool {
	was := p.decode.SetRegionLabels(on)
	if p.prefill != nil {
		p.prefill.SetRegionLabels(on)
	}
	if p.e != nil {
		p.e.SetRegionLabels(on)
	}
	return was
}

func (p *Pool) Close() {
	p.decode.Close()
	p.prefill.Close()
	p.e.Close()
	p.run, p.decode, p.prefill, p.e = nil, nil, nil, nil
	p.procs.Release()
}
