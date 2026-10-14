//go:build amd64 || arm64

package nn

import (
	"maps"
	"runtime"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/jit/cpu"
)

// gemmKey identifies a prefill kernel: quantization, blocks per row, and the
// register tile. The tile is part of the key because it is tuned: the same
// shape can hold several candidates while the tuner decides between them.
type gemmKey struct {
	t          quant.Type
	nb, mr, nr int
}

// gemmTile is a register tile: mr weight rows by nr groups of eight tokens.
type gemmTile struct{ mr, nr int }

// rateKey identifies a matmul call site for throughput comparison. A kernel
// depends only on (type, blocks-per-row) plus its tile, but a measured rate
// also depends on the row count (shapes sharing nb differ far more than any
// tile does, and pooling them made the IQR gate refuse every candidate) and on
// the batch width, since a short batch is zero-padded to fill the tile.
type rateKey struct {
	t              quant.Type
	nb, rows, ntok int
}

type shapeKey struct {
	t  quant.Type
	nb int
	w  int // interleave width; 0 in the shape inventory, >=2 for a kernel
}

// JIT is the generated-code tier: the kernels plus the scratch buffers they
// need. One per session, because the buffers are not safe to share.
//
// There is no interpreted tier behind it. A shape with no optimized kernel gets
// a basic generated one, and a host with no code generator at all cannot run a
// model (jit_stub.go); the Go arithmetic every kernel is gated against lives in
// internal/oracle, which only tests may import.
type JIT struct {
	hot hotRegions // see hotRegions
	// tier is the instruction-set tier this JIT was built for, recorded once
	// at construction, and em is that tier's emitter table: every kernel this
	// JIT emits comes from em, so a JIT never mixes tiers. Force a tier (a
	// jitllmtest instrument) BEFORE building the JIT that should run it.
	tier cpu.Tier
	em   *cpu.Emitters

	code  map[quant.Type]*cpu.Code // one row per iteration, K at runtime
	code4 map[quant.Type]*cpu.Code // four rows interleaved, K at runtime
	// shaped holds pack kernels: several rows interleaved with the row stride
	// baked in, keyed by (type, blocks per row, width). Baking the stride costs
	// one block body of code, where unrolling the whole loop would blow the L1i.
	shaped map[shapeKey]*cpu.Code
	// The attention kernels: attn is the default set (AddAttnKV), and attnSets
	// the other geometries a model's layers attend at (AttnSetFor; Gemma 4's
	// sliding and global layers differ).
	attn     AttnSet
	attnSets []*AttnSet
	// delta is the gated delta rule, provisioned per model by AddDelta; deltaN
	// is the state width baked into it, so a caller asking for another width
	// gets a decline rather than a kernel that walks the wrong stride.
	delta  *cpu.Code
	deltaN int
	// deltaC is the per-channel decay twin (Kimi-Linear), provisioned by
	// AddDeltaChan. A model uses one or the other for its whole life.
	deltaC *cpu.Code
	// ssd is Mamba-2's selective update (AddSSD), baked at the state size
	// ssdN like the delta rule.
	ssd  *cpu.Code
	ssdN int
	// selScan is Mamba-1's per-channel scan, provisioned at its state size
	// selScanN like the SSD kernel.
	selScan  *cpu.Code
	selScanN int
	// conv is the causal convolution, provisioned per model by AddConv1d.
	conv                *cpu.Code
	convTaps, convChans int
	// dw is the depthwise convolution rows, one per baked shape, provisioned
	// by AddDWConv before the calls that read them (a convolutional tower's
	// load), so the calls only read the map.
	dw map[cpu.DWShape]*cpu.Code
	// attnKV records the cache format the attention kernels were emitted for.
	attnKV cpu.KVFmt
	// The qt-wide score kernels, one per geometry (AttnTiledFor), for
	// bidirectional callers; attnT is the default AddAttnTiled made.
	attnTiled []*AttnTiledSet
	attnT     *AttnTiledSet
	// The generated activation packer, per (tile width, half-sums).
	packMu    sync.Mutex
	packCode  map[packKey]*cpu.Code
	packKonst map[quant.Type][]float32
	konst     map[quant.Type][]byte
	// packed holds the kernels that read a jlm container's device layout,
	// and pkonst their shared constant block. Keyed by type like code, because
	// the scale arithmetic differs per format.
	packed      map[quant.Type]*cpu.Code
	packedTail  map[quant.Type]*cpu.Code
	packedWide  map[quant.Type]*cpu.Code
	packedFused map[quant.Type]*cpu.Code
	// built marks the types NewJIT has already emitted for. Not f.code[t] !=
	// nil: the packed family can exist for a type the row-major one declines.
	built map[quant.Type]bool
	piacc []int32
	pdscr []float32
	// winQS, winD and winSC hold a packed row window: the last rows of a tensor
	// whose row count the tail kernel's group does not divide (packedWindow).
	// winP is the tensor over them and winOut its rows' results, kept here
	// because both reach a pool region (MatVecPacked) and so would escape --
	// two allocations a token on every model whose vocabulary is ragged, as
	// granite's 49159-row head is.
	winQS, winD, winSC []byte
	winP               Packed
	winOut             [cpu.PackedTail]float32
	pmscr              []float32
	pkonst             []byte

	// shapes is the (type, blocks-per-row) inventory, width zero. It is both
	// the list emitWidth walks when the tuner climbs and the model's identity
	// in the cache key.
	shapes []shapeKey
	// keyShapes is every (type, blocks per row) the model has, whatever its
	// layout: the tuners' cache keys are built from it, so a width measured on
	// one model is not reused for another. shapes above is the row-major
	// family's own list and is empty for a container.
	keyShapes []shapeKey
	tuner     *packTuner

	// The prefill GEMM's buffers, all sized for one token tile: prefill is
	// batched by tile, not by sequence, so none of these grow with the prompt.
	gemm     map[gemmKey]*cpu.Code
	gq       []int8
	gx       []float32 // zero-padded staging for a short final token tile
	gscale   []float32
	gsum     []int32
	ghalf    []int32
	gout     []float32
	gscratch []float32
	// device is the optional GPU tier; nil is the normal case.
	device Device
	gtile  map[rateKey]gemmTile
	gtune  map[rateKey]*gemmTuner
	ptune  *duelTuner
	ctune  *duelTuner
	rtune  *duelTuner // the batched row chunk's cap, in KiB (newRowChunkDuel)
	wtune  *duelTuner // the weight-stationary GEMM's rows a call, or off (newGEMMRowsDuel)
	// fpf duels the fused kernel's prefetch distance; fusedAt owns every
	// fused kernel by distance, and packedFused points into one of them.
	fpf     *fpfTuner
	fusedAt map[int]map[quant.Type]*cpu.Code
	// aheadForm is whether the tier emits the fused kernel at a prefetch
	// distance (distPick).
	aheadForm bool
	// fpfNone is len(packedFused)+1 when newFPFTuner last found nothing to
	// duel, so a model with no fused kernel (every weight a float) does not
	// fingerprint an empty set again -- and allocate -- on every token.
	fpfNone int

	// qact is the generated activation quantizer, indexed by whether the
	// format needs per-16 sums; nil where the tier has no emitter.
	// qkonst caches the per-format constant block so the hot path allocates
	// nothing.
	qact       [2]*cpu.Code
	qactNarrow [2]*cpu.Code
	qkonst     map[quant.Type][]float32
	// qscr is the narrow kernel's per-worker spill space, one run of
	// cpu.QuantActNarrowScratch float32 each.
	qscr []float32
	// kvPad and kvScr are the q8 cache append's padded row and spill space
	// (QuantizeKVRows): it runs outside any region, so it shares no worker's.
	kvPad, kvScr []float32
	// kvWiden is the q8 cache's widening to float32 per head width
	// (WidenKVRows), emitted on first ask.
	kvWiden map[int]*cpu.Code

	// grq, grpairs, grhalfs and grseen are each worker's private quantized
	// activation for MatVecPackedGroup, one per format slot, and the group
	// call each was made for (groupGen).
	grq      [][]int8
	grpairs  [][]float32
	grhalfs  [][]float32
	grseen   []uint64
	gract    [][]float32 // each worker's act(x)*up for a gated group
	groupGen uint64
	q        []int8
	pairs    []float32
	kpart    []float32           // MatVecPacked's k-slice partial outputs (fusedSliced)
	ksliced  int64               // how many matvecs fusedSliced has run (KSliced)
	picks    map[pickKey]*mvPick // MatVecPacked's kernel per shape (mvpick.go)
	distSize map[quant.Type]int  // the largest shape whose distance set packedFused
	half     []float32           // per-16 activation sums, for the 16-wide k-quants
	// The same three for a batch: MatMulPacked needs every token's quantized
	// activation live at once, where the matvec path stages one and caches it.
	mq       []int8
	mpairs   []float32
	mhalf    []float32
	mmCalls  atomic.Int64
	tiledMu  sync.Mutex
	tiled    map[tiledKey]*cpu.Code
	tiledTok map[tiledKey]int
	// wsGEMM is the weight-stationary GEMM per (type, k, rows), under tiledMu;
	// a nil entry is a cached refusal. gemmCalls counts the MatMulPacked calls
	// that ran it, since a decline is correct and therefore invisible.
	wsGEMM map[wsKey]*cpu.Code
	// floatMM is the float GEMM per (type, token tile), under tiledMu; a nil
	// entry is a cached refusal (floatmm.go).
	floatMM      map[floatMMKey]*cpu.Code
	floatMMCalls atomic.Int64
	mout         []float32 // its padded [ntok][nrows+GEMMPad/4] output
	// mmj is MatMulPacked's per-call state and the mm*Fn its pool regions,
	// method values built once so a warm batched matmul allocates nothing.
	mmj                                    mmJob
	mmQuantFn, mmRunFn, mmGemmFn, mmCopyFn func(worker, lo, hi int)
	gemmCalls                              atomic.Int64
	// cfg is the caller's resolved option set. Every knob is read off this JIT
	// rather than a package variable, so two models in one process cannot
	// reconfigure each other.
	cfg Config
	// tilePin is Config.TileTok: the token tile every batched matmul on this
	// JIT is held to, or 0 for the widest one that fits.
	tilePin int
	// chunkBytes is Config.ChunkBytes: the locality cap on a batched chunk's
	// weight slice, or 0 for none. mmChunkRows records what the last batched
	// matmul actually chose, so a gate can assert the cap narrowed something.
	chunkBytes int
	// gemmRows is the weight-stationary GEMM's rows a call as the gemmrows
	// duel last chose it: 0 for the default, gemmOff for the tiled path.
	gemmRows    int
	mmChunkRows atomic.Int64
	tiledCalls  atomic.Int64

	// gen rises whenever the caller overwrites its activation buffer. The same
	// vector feeds q, k and v (and gate and up), so quantizations are cached by
	// (pointer, length, type, generation).
	gen uint64
	// actWindow is how many elements share one activation scale: 32, or 256 for
	// an all-k-quant model. See NewJIT.
	actWindow int

	cachePtr *float32
	cacheLen int
	cacheTyp quant.Type
	cacheGen uint64
	cacheOK  bool

	// pool runs on the decode core set (physical P-cores). Decode is
	// memory-bound and E-cores add no read bandwidth.
	pool *sched.Pool
	// look is the fused runs' idle work (lookahead.go); lookOff turns it off.
	look    lookahead
	lookOff bool
	// narrow is the decode pool and wide, when non-nil, the prefill pool;
	// pool is whichever of them the current phase runs on. See BeginPrefill.
	narrow, wide *sched.Pool
	// procs releases the GOMAXPROCS request BeginPrefill made, nil when none
	// is live; see sched.RaiseProcs.
	procs func()
}

var jitOnce sync.Once

// Available reports whether this build has a code generator. It does: amd64
// and arm64 are the architectures with an emitter (jit_stub.go is the rest).
func Available() bool { return true }

// Profiling counters, enabled by WithProfile, so the serial fraction of a
// parallel region is measured rather than guessed. The totals are process-wide
// by API (ProfileNanos has no receiver); the gate, Config.Profile, is per JIT.
var nsQuantize, nsKernel, nsOut atomic.Int64

// ProfileNanos reports time spent quantizing activations, inside generated code,
// and widening results, in nanoseconds. Only the middle one parallelizes.
func ProfileNanos() (quantize, kernel, out int64) {
	return nsQuantize.Load(), nsKernel.Load(), nsOut.Load()
}

// ResetProfile zeroes the phase counters, so a caller can bracket one arm of a
// comparison instead of reading the whole process.
func ResetProfile() { nsQuantize.Store(0); nsKernel.Store(0); nsOut.Store(0) }

// quantStart opens a quantize span and quantSince records it, for the
// container paths (prepAct, MatMulPacked, MatVecPackedGather) that do not go
// through MatVec's own accounting.
func (f *JIT) quantStart() int64 {
	if f.cfg.Profile {
		return f.now()
	}
	return 0
}

func (f *JIT) quantSince(t0 int64) {
	if f.cfg.Profile {
		nsQuantize.Add(f.now() - t0)
	}
}

// ResetForTest clears what is still process-wide, so a test that built a JIT
// with non-default options does not leave anything set for the next one.
// Options live on the JIT, so only two things remain that are not per-JIT:
//
//	jitOnce      package initialisation, once per process
//	cpu.SetWidths  the print-only reporter `jitllm hardware` pins; nn no longer
//	               writes it, so this is belt and braces for a test that called
//	               cpu.SetWidths itself
func ResetForTest() {
	jitOnce = sync.Once{}
	cpu.SetWidths(0, 0)
}

// NewJIT builds the kernels a model needs, with a worker pool of its own. maxK
// is the widest matvec input and maxRows the tallest output. A model's segments
// -- its text blocks and its vision tower -- share one: a JIT holds an attention
// set per geometry (AttnSetFor), so nothing about a tower needs a second.
func NewJIT(maxK, maxRows int, types []quant.Type, opts ...Option) *JIT {
	cfg := newConfig(opts)
	jitOnce.Do(func() {
		// There is deliberately no switch to turn the JIT off: there is no
		// tier to fall back to (jit_stub.go covers hosts without an emitter).
	})
	// The activation amax window is a codegen input decided once: 256 when
	// every matvec type is a k-quant (one scale per super-block, like
	// llama.cpp's block_q8_K), 32 otherwise, since Q4_0/Q8_0 blocks are 32
	// wide. The layout is the same either way; a wider window only makes
	// consecutive (d, -sum*biasC/8) pairs share a d, which a kernel may hoist.
	window := cpu.WideActWindow(types)
	if cfg.ActWindow > 0 {
		// The graph's own window (BitNet's per-token quantization).
		window = cfg.ActWindow
	}
	tier := cpu.HostTier()
	f := &JIT{
		cfg:         cfg,
		tier:        tier,
		em:          cpu.EmittersWith(tier, cpu.EmitOpts{Prefetch: cfg.Prefetch, A64Prefetch: cfg.A64Prefetch}),
		actWindow:   window,
		tilePin:     cfg.TileTok,
		chunkBytes:  cfg.ChunkBytes,
		code:        map[quant.Type]*cpu.Code{},
		code4:       map[quant.Type]*cpu.Code{},
		shaped:      map[shapeKey]*cpu.Code{},
		gemm:        map[gemmKey]*cpu.Code{},
		gtile:       map[rateKey]gemmTile{},
		konst:       map[quant.Type][]byte{},
		packed:      map[quant.Type]*cpu.Code{},
		packedTail:  map[quant.Type]*cpu.Code{},
		packedWide:  map[quant.Type]*cpu.Code{},
		packedFused: map[quant.Type]*cpu.Code{},
		built:       map[quant.Type]bool{},
		q:           make([]int8, maxK),
		pairs:       make([]float32, 2*(maxK/cpu.Q8Block+1)),
		half:        make([]float32, maxK/16+2),
	}
	// The row-major (GGUF) and packed (container) kernel families are gated
	// independently: a host that cannot run the row-major kernel for t must
	// still get the packed one, which is the only weight path model.Open admits.
	for _, t := range types {
		if f.built[t] {
			continue
		}
		f.built[t] = true
		if f.em.RowMajorSupported(t) {
			// Emitted eagerly: every shape is known from the header before the
			// first token, and generation costs microseconds.
			if b, err := f.em.RowMajor(cpu.Spec{W: t, Rows: 1, Accs: cpu.AccChains(f.cfg.Accs, t), Cols: 1, ActWin: int16(f.actWindow)}); err == nil {
				if c, err := cpu.MapNamed(b, t.String()+"_1row"); err == nil {
					f.code[t] = c
					f.konst[t] = cpu.KernelConst(t)
					// The interleaved kernel is the fast path; the single-row
					// one remains for the tail and as the simpler thing to
					// bisect against.
					if b4, err := f.em.RowMajor(cpu.Spec{W: t, Rows: cpu.Interleave, Accs: 1, Cols: 1}); err == nil {
						if c4, err := cpu.Map(b4); err == nil {
							f.code4[t] = c4
						}
					}
				}
			}
		}
		// The packed kernel. f.em is the one place the host probe decides
		// which instruction sequence it gets (VNNI, the pre-VNNI VEX dot, or
		// the SSE tier); the emitters themselves are host-pure.
		if !f.em.PackedSupported(t) {
			continue
		}
		pb, err := f.em.PackedMatVec(t, cpu.PackedRows)
		if err != nil {
			continue
		}
		pc, err := cpu.MapNamed(pb, t.String()+"_packed")
		if err != nil {
			continue
		}
		if f.pkonst == nil {
			f.pkonst = make([]byte, cpu.PackedScratchBytes)
			if err := cpu.PackedScratch(f.pkonst); err != nil {
				f.pkonst = nil
			}
		}
		if f.pkonst == nil {
			continue
		}
		f.packed[t] = pc
		// The tail kernel finishes a row count the main tile does not divide.
		if tb, err := f.em.PackedMatVec(t, cpu.PackedTail); err == nil {
			if tc, err := cpu.MapNamed(tb, t.String()+"_packed_tail"); err == nil {
				f.packedTail[t] = tc
			}
		}
		// The wide kernel serves a whole row range off memory accumulators, so
		// its reads are sequential. It does not exist for every format: the
		// sub-block unroll is over the L1i budget for the 16-wide k-quants, and
		// the pre-VNNI path refuses it outright (it has no call site).
		if wb, err := f.em.PackedWide(t); err == nil {
			if wc, err := cpu.MapNamed(wb, t.String()+"_packed_wide"); err == nil {
				f.packedWide[t] = wc
			}
		}
		// The fused kernel keeps the accumulator in registers by running every
		// word of a sub-block in one pass; it is preferred where it exists. On
		// arm64 at a 256-wide window PackedFusedWin folds the sub-block scale in
		// integers, matching the prefill GEMM bit for bit; GEMMExact keeps the
		// float form.
		fb, err := []byte(nil), error(nil)
		if f.em.PackedFusedWin != nil && !cfg.GEMMExact {
			fb, err = f.em.PackedFusedWin(t, window)
		}
		if fb == nil || err != nil {
			fb, err = f.em.PackedFused(t)
		}
		if err == nil {
			if fc, err := cpu.MapNamed(fb, t.String()+"_packed_fused"); err == nil {
				f.packedFused[t] = fc
			}
		}
	}
	// Whether this tier's fused kernel has a prefetch form at all. The SSE
	// tier refuses every distance (cpu.EmitPackedMatVecFusedAheadSSE), and a
	// per-shape pick over distances whose kernels were never emitted would
	// time one kernel five times and settle on a distance nothing runs.
	for t := range f.packedFused {
		if _, err := f.em.PackedFusedAhead(t, cpu.FusedAheadWords); err == nil {
			f.aheadForm = true
			break
		}
	}
	// The activation quantizer. Two kernels serve every format (the only thing
	// baked is whether it needs per-16 sums); the per-format scale arithmetic
	// arrives through the constant block.
	f.qkonst = make(map[quant.Type][]float32)
	for i, half := range [2]bool{false, true} {
		if cfg.QuantActGo {
			break // the dose arm: every quantize runs the Go loop
		}
		suffix := ""
		if half {
			suffix = "_half"
		}
		if b, err := f.em.QuantAct(half); err == nil {
			if c, err := cpu.MapNamed(b, "quantact"+suffix); err == nil {
				f.qact[i] = c
			}
		}
		if b, err := f.em.QuantActNarrow(half); err == nil {
			if c, err := cpu.MapNamed(b, "quantact_narrow"+suffix); err == nil {
				f.qactNarrow[i] = c
			}
		}
	}
	// The caller's sched options reach the pool here, which is the whole
	// chain: cmd reads JITLLM_CORES -> model.Option -> nn.WithSched ->
	// sched.WithCores. Nothing between them reads the environment.
	//
	// The pool is a view of the process's workers for these cores (sched.Shared):
	// every State builds a JIT, and two States each with workers of their own
	// would be two pools spinning on the same cores. Views take turns a region
	// at a time, so sessions on one host run beside each other.
	f.pool = sched.Shared(sched.DecodeCores(cfg.Sched...), cfg.Sched...)
	// Not when the caller narrowed the decode pool below the default P set
	// (JITLLM_CORES): a prefill pool wider than what was asked for would
	// quietly ignore the request.
	if cs := sched.CoreSet(prefillCoreSet(cfg.PrefillCores)); len(cs) > f.pool.Max() &&
		f.pool.Max() >= len(sched.DecodeCores()) && len(cs) <= runtime.NumCPU() {
		f.wide = sched.SharedBeside(cs, f.pool, cfg.Sched...)
	}
	f.narrow = f.pool
	// The narrow quantizer's spill space, one run per worker, sized from the
	// wider pool's maximum: SetParticipants narrows a region without narrowing
	// the pool, and prefill swaps the wide pool in.
	workers := f.pool.Max()
	if f.wide != nil {
		workers = max(workers, f.wide.Max())
	}
	f.qscr = make([]float32, max(1, workers)*cpu.QuantActNarrowScratch)
	// The widest table default across the types present seeds the climb. It is
	// a starting point, not an answer: the first ten tokens decide.
	static := 0
	for t := range f.code {
		if w := cpu.PackWidthNative(cfg.Pack, t); w > static {
			static = w
		}
	}
	f.tuner = newPackTuner(static, cfg.Pack, cfg.Tune, cfg.Quiet)
	f.fusedAt = map[int]map[quant.Type]*cpu.Code{0: maps.Clone(f.packedFused)}
	if f.distPick() {
		// Every distance a shape may pick, emitted now rather than on its
		// first call: the variants come from the same emitter state as the
		// rest of this JIT, and no timed call pays for an emission.
		for _, d := range prefetchDistances {
			f.fusedKernels(d)
		}
	}
	return f
}

// NewInput tells the tier that the activation buffer has been rewritten, so any
// cached quantization of it is stale. Explicit rather than inferred: a cache
// that guesses when its input changed is a silent-wrong-answer generator.
func (f *JIT) NewInput() {
	if f != nil {
		f.gen++
	}
}

// Parallel runs fn over [0,n) on the pool, so elementwise passes do not run on
// one core while the others sit idle.
func (f *JIT) Parallel(n, chunk int, fn func(lo, hi int)) {
	if f == nil || n <= 0 {
		if n > 0 {
			fn(0, n)
		}
		return
	}
	f.pool.DoRange(n, chunk, fn)
}

// Workers is the pool size, 1 when unthreaded.
// SetSpinning switches the pool's workers between spinning for the next region
// and parking at once (sched.Pool.SetSpinning).
func (f *JIT) SetSpinning(on bool) {
	if f != nil && f.pool != nil {
		f.pool.SetSpinning(on)
	}
}

// Spinning reports what SetSpinning last chose.
func (f *JIT) Spinning() bool { return f != nil && f.pool != nil && f.pool.Spinning() }

func (f *JIT) Workers() int {
	if f == nil {
		return 1
	}
	return f.pool.N()
}

// AddShape generates a kernel specialized to one (type, K) the model actually
// contains. Called at load with the model's real tensor shapes: a transformer
// has only a handful of distinct ones, so this is a few kernels, not one per
// layer.
func (f *JIT) AddShape(t quant.Type, k int) {
	if f == nil || k <= 0 {
		return
	}
	// Recorded for the tuner cache key before the row-major early returns
	// below, which every container model takes; otherwise every container
	// would share one empty-shape key.
	if be := t.BlockElems(); be > 0 && k%int(be) == 0 {
		ks := shapeKey{t: t, nb: cpu.BlocksPerRow(t, k)}
		if !slices.Contains(f.keyShapes, ks) {
			f.keyShapes = append(f.keyShapes, ks)
		}
	}
	if f.code[t] == nil {
		return
	}
	if !f.em.RowMajorGGUF {
		return // the interleaved kernels are the AVX2 tier's GGUF family only
	}
	if cpu.PackWidthNative(f.cfg.Pack, t) < 2 {
		return // no interleaved kernel for this format
	}
	base := shapeKey{t: t, nb: cpu.BlocksPerRow(t, k)}
	for _, s := range f.shapes {
		if s == base {
			return
		}
	}
	f.shapes = append(f.shapes, base)
	// Only the widths in play; the rest are emitted if the climb reaches them.
	for _, w := range f.tuner.initialWidths() {
		f.emitWidth(base, w)
	}
}

// emitWidth generates the interleaved kernel for one (shape, width) pair.
func (f *JIT) emitWidth(s shapeKey, w int) {
	key := shapeKey{s.t, s.nb, w}
	if f.shaped[key] != nil {
		return
	}
	b, err := cpu.EmitInterleaved(s.t, w, s.nb)
	if err != nil {
		return
	}
	if c, err := cpu.MapNamed(b, s.t.String()+"_pack"+itoaN(w)+"_k"+itoaN(s.nb)); err == nil {
		f.shaped[key] = c
	}
}

// ensureWidth brings every known shape up to width w, for when the climb
// promotes a challenger.
func (f *JIT) ensureWidth(w int) {
	for _, s := range f.shapes {
		f.emitWidth(s, w)
	}
}

// dropUnusedWidths releases the losing ladder once the tuner settles.
func (f *JIT) dropUnusedWidths(keep int) {
	for k, c := range f.shaped {
		if k.w != keep {
			c.Close()
			delete(f.shaped, k)
		}
	}
}

// TokenStart and TokenEnd bracket one decode step so the tuner can time it.
// Both are no-ops once the width has settled, and safe on a nil tier.
func (f *JIT) TokenStart() {
	if f == nil || f.tuner == nil {
		return
	}
	if !f.tuner.keyed {
		f.tuner.keyed = true
		f.tuner.key = tuneKey("pack", f.keyShapes, f.pool.Max())
		// A decode-only session never calls MatMul, so it would never pick up a
		// participant count that prefill already measured on this machine.
		// Apply the cached one here; an unmeasured machine keeps every core.
		// A pin outranks the cache, as it does in the prefill duel.
		if f.cfg.Part >= 1 {
			f.pool.SetParticipants(f.cfg.Part)
		} else if n, ok := lookupTuned(tuneKey("part", f.keyShapes, f.pool.Max()), f.cfg.Tune); ok && n >= 1 {
			f.pool.SetParticipants(n)
		}
		if w, ok := lookupTuned(f.tuner.key, f.cfg.Tune); ok && w >= 2 {
			// Already measured on this CPU, core count and shape set.
			f.tuner.on, f.tuner.best, f.tuner.cur, f.tuner.chal = false, w, w, 0
			f.ensureWidth(w)
			f.dropUnusedWidths(w)
		}
	}
	f.tuner.begin()
	// Not on arm64: its fused kernel carries its prefetch (cpu.a64FusedAhead),
	// and a duel that swaps the kernel under every shape would be what
	// MatVecPacked's per-shape timing (mvpick.go) measured instead.
	if f.fpf == nil && f.tuner.keyed && f.tier != cpu.TierNEON && f.fpfNone != len(f.packedFused)+1 {
		if f.fpf = f.newFPFTuner(); f.fpf == nil {
			f.fpfNone = len(f.packedFused) + 1
		}
	}
	f.fpfStart()
}

func (f *JIT) TokenEnd() {
	if f == nil || f.tuner == nil {
		return
	}
	f.tuner.end(f)
	f.fpfEnd()
}

// DiscardToken tells the decode duels that the token under way is not a
// sample -- it waited on something no kernel choice controls (a mixture's
// expert pages coming off disk). Call it before TokenEnd.
func (f *JIT) DiscardToken() {
	if f == nil || f.fpf == nil {
		return
	}
	f.fpf.discard = true
}

// Shapes is how many shape-specialized kernels were generated.
func (f *JIT) Shapes() int {
	if f == nil {
		return 0
	}
	return len(f.shaped)
}

// Types reports which quantization types have a generated kernel, in either
// family: a container model has weights in f.packed and nothing in f.code.
func (f *JIT) Types() []quant.Type {
	if f == nil {
		return nil
	}
	seen := map[quant.Type]bool{}
	out := make([]quant.Type, 0, len(f.code)+len(f.packed))
	for _, m := range []map[quant.Type]*cpu.Code{f.code, f.packed} {
		for t := range m {
			if !seen[t] {
				seen[t] = true
				out = append(out, t)
			}
		}
	}
	return out
}

// Close releases the executable mappings.
func (f *JIT) Close() error {
	if f == nil {
		return nil
	}
	for _, c := range f.code {
		c.Close()
	}
	for _, c := range f.code4 {
		c.Close()
	}
	for _, c := range f.shaped {
		c.Close()
	}
	// fusedAt owns the fused kernels (packedFused only points into it);
	// Code.Close is idempotent either way.
	for _, m := range []map[quant.Type]*cpu.Code{f.packed, f.packedTail, f.packedWide, f.packedFused} {
		for _, c := range m {
			c.Close()
		}
	}
	for _, m := range f.fusedAt {
		for _, c := range m {
			c.Close()
		}
	}
	// The token-tiled kernels are emitted lazily per shape.
	f.tiledMu.Lock()
	for _, c := range f.tiled {
		if c != nil {
			c.Close()
		}
	}
	f.tiled = nil
	for _, c := range f.wsGEMM {
		if c != nil {
			c.Close()
		}
	}
	f.wsGEMM = nil
	for _, c := range f.floatMM {
		if c != nil {
			c.Close()
		}
	}
	f.floatMM = nil
	f.tiledMu.Unlock()
	for _, c := range f.kvWiden {
		c.Close()
	}
	f.kvWiden = nil
	if f.deltaC != nil {
		f.deltaC.Close()
		f.deltaC = nil
	}
	if f.selScan != nil {
		f.selScan.Close()
		f.selScan = nil
	}
	if f.ssd != nil {
		f.ssd.Close()
		f.ssd = nil
	}
	if f.delta != nil {
		f.delta.Close()
		f.delta = nil
	}
	if f.conv != nil {
		f.conv.Close()
		f.conv = nil
	}
	for _, c := range f.dw {
		c.Close()
	}
	f.dw = nil
	f.attn.close()
	for _, a := range f.attnSets {
		a.close()
	}
	f.attnSets = nil
	for _, a := range f.attnTiled {
		if a.scores != nil {
			a.scores.Close()
		}
	}
	f.attnTiled, f.attnT = nil, nil
	f.shaped = nil
	f.code, f.code4 = nil, nil
	f.narrow.Close()
	if f.wide != nil {
		f.wide.Close()
	}
	f.pool, f.narrow, f.wide = nil, nil, nil
	if f.procs != nil {
		f.procs()
		f.procs = nil
	}
	return nil
}

// BeginPrefill runs every region until EndPrefill on the prefill pool, when
// there is one (Config.PrefillCores), and EndPrefill returns to the decode
// pool. The caller runs one phase at a time, as it already must: a pool has a
// single caller.
//
// It also raises GOMAXPROCS for the phase: a wide pool on too few Ps is far
// slower, and leaving GOMAXPROCS wide costs decode (see
// docs/engineering-history/cpu-kernels.md). GOMAXPROCS is process state, so
// the raise goes through sched.RaiseProcs, which serves several models'
// overlapping prefills and restores the value once the last one ends.
func (f *JIT) BeginPrefill() {
	if f == nil || f.wide == nil {
		return
	}
	f.pool = f.wide
	if f.procs == nil {
		f.procs = sched.RaiseProcs(f.wide.Max())
	}
}

// EndPrefill returns to the decode pool and GOMAXPROCS; see BeginPrefill.
func (f *JIT) EndPrefill() {
	if f == nil || f.narrow == nil {
		return
	}
	f.pool = f.narrow
	if f.procs != nil {
		f.procs()
		f.procs = nil
	}
}

// MatVec computes out[r] = dot(row r of w, x) with generated code, reporting
// false when it cannot: no kernel for this type, a shape the kernel does not
// handle, or scratch too small.
func (f *JIT) MatVec(out []float32, t quant.Type, w []byte, x []float32, nrows, k int) bool {
	return f.matVec(out, t, w, x, nrows, k, true)
}

// MatVecHost is MatVec without the device offer: the host kernel, whatever is
// attached. It is for weights of a block the host is running; the device's
// per-matvec staging buffers are per device, not per session, so offering them
// lets two sessions disturb each other's logits.
func (f *JIT) MatVecHost(out []float32, t quant.Type, w []byte, x []float32, nrows, k int) bool {
	return f.matVec(out, t, w, x, nrows, k, false)
}

func (f *JIT) matVec(out []float32, t quant.Type, w []byte, x []float32, nrows, k int, offer bool) bool {
	if f == nil {
		return false
	}
	// A quantized weight here is the source format's row-major layout on an
	// inference path; see engine/nn/rowmajor.go.
	countRowMajor(t)
	// The accelerator gets first refusal, before any CPU-side setup. It declines
	// by returning false and the CPU path below runs untouched, which is what
	// makes this safe to attach unconditionally.
	if offer && f.device != nil && f.device.MatVec(out, t, w, x, nrows, k) {
		return true
	}
	// F32, F16 and BF16 are one generated kernel at every k (the kernel takes
	// the remainder itself). Mixture routers, Qwen3-Next's BF16 shared-expert
	// gate and small HF models' matrices ride here. The router's selection is
	// gated on the decision (TestF32RouterSelectionAgrees), not the bits.
	if t == quant.F32 || t == quant.F16 || t == quant.BF16 {
		c := f.code[t]
		if c == nil || k <= 0 || len(x) < k || len(out) < nrows || len(w) < nrows*k*int(t.BlockBytes()) {
			return false
		}
		j := &f.hot.float
		j.code, j.out, j.w, j.x, j.k, j.es = c, out, w, x[:k], k, int(t.BlockBytes())
		f.pool.DoRange(nrows, max(1, nrows/(4*f.pool.N())), f.hot.floatFn())
		j.out, j.w, j.x = nil, nil, nil
		return true
	}
	// Before any pool.Do: growing this inside a worker would be a race, and the
	// arm64 k-quant kernels read it (see ensureScratch).
	f.ensureScratch()
	code := f.code[t]
	// k must tile BOTH the weight block (256 for k-quants) and the 32-element
	// activation block, or the kernel and the activation scales disagree about
	// where a group starts.
	if code == nil || k == 0 || k%cpu.Q8Block != 0 || k%int(t.BlockElems()) != 0 {
		return false
	}
	if len(x) < k || len(out) < nrows || k > len(f.q) {
		return false
	}
	x = x[:k]
	nb := k / cpu.Q8Block
	if 2*nb > len(f.pairs) {
		return false
	}
	t0 := f.now()
	// Requantize only when the input actually changed.
	if !f.cacheOK || f.cachePtr != &x[0] || f.cacheLen != k ||
		f.cacheTyp != t || f.cacheGen != f.gen {
		// Spread quantization across the pool. Below a few blocks per worker the
		// dispatch costs more than the work, so small vectors stay serial.
		qk := f.quantKernel(t)
		var half []float32
		if cpu.NeedsHalfSums(t) {
			half = f.half[:k/16]
		}
		qs, ps, win := f.q[:k], f.pairs[:2*nb], f.actWindow
		if nb >= 4*f.pool.N() {
			f.pool.Do(nb, max(1, nb/(4*f.pool.N())), func(w, blo, bhi int) {
				qk.Run(t, qs, ps, half, f.quantScratch(w), x, k, blo, bhi, win)
			})
		} else {
			qk.Run(t, qs, ps, half, f.quantScratch(0), x, k, 0, nb, win)
		}
		f.cachePtr, f.cacheLen, f.cacheTyp, f.cacheGen, f.cacheOK = &x[0], k, t, f.gen, true
	}
	t1 := f.now()

	rowBytes := cpu.RowBytes(t, k)
	kBlocks := cpu.BlocksPerRow(t, k) // super-blocks for k-quants, blocks otherwise
	if nrows*rowBytes > len(w) {
		return false
	}
	konst := f.konst[t]

	// Two limits on the chunk size, and they pull in opposite directions.
	//
	// Upper: the I3 execution budget. A goroutine inside generated code cannot
	// be async-preempted, so one long call stalls every GC in the process.
	// Lower: load balance. Handing each worker a single chunk makes the region
	// as slow as its unluckiest core, so aim for a few chunks per worker.
	chunk := cpu.RowsPerCall(kBlocks * int(t.BlockElems()) / cpu.Q8Block)
	if bal := nrows / (4 * f.pool.N()); bal > 0 && bal < chunk {
		// Round the load-balance chunk UP to a cache line of output. Two workers
		// writing into one 64-byte line is coherence traffic, not parallelism.
		chunk = (bal + cpu.OutLine - 1) / cpu.OutLine * cpu.OutLine
	}
	// pack8 when there is one for this shape: eight rows per iteration off a
	// single base pointer, with the stride an immediate.
	pw := f.tuner.width()
	if sc := f.shaped[shapeKey{t, kBlocks, pw}]; sc != nil && pw >= 2 && nrows >= pw {
		pk := nrows / pw * pw
		pchunk := max(pw, chunk/pw*pw)
		f.pool.DoLabeled(regionLabel(t, "/pack"+itoaN(pw)), pk, pchunk, func(worker, lo, hi int) {
			args := cpu.Args{
				Out: &out[lo], W: &w[lo*rowBytes], A: &f.q[0], AScale: &f.pairs[0],
				Rows: int64((hi - lo) / pw), K: int64(kBlocks),
				RowStr: int64(rowBytes), Scr: &konst[0], AHalf: &f.half[0],
				Scratch: f.scratchFor(worker),
			}
			sc.Call(&args)
		})
		if pk < nrows {
			args := cpu.Args{
				Out: &out[pk], W: &w[pk*rowBytes], A: &f.q[0], AScale: &f.pairs[0],
				Rows: int64(nrows - pk), K: int64(kBlocks), RowStr: int64(rowBytes),
				Scr: &konst[0], AHalf: &f.half[0], Scratch: f.scratchFor(0),
			}
			code.Call(&args)
		}
		return f.done(out, nrows, t0, t1)
	}

	// The interleaved kernel handles whole groups of four rows; anything left
	// over goes through the single-row kernel. Both produce identical results —
	// the interleave changes the schedule, not the arithmetic.
	code4 := f.code4[t]
	if code4 == nil {
		// No interleaved kernel for this format: run the single-row one across
		// the pool. This path must stay parallel; falling through to the serial
		// tail below would silently drop to one core.
		f.pool.DoLabeled(regionLabel(t, "/1row"), nrows, chunk, func(worker, lo, hi int) {
			args := cpu.Args{
				Out: &out[lo], W: &w[lo*rowBytes], A: &f.q[0], AScale: &f.pairs[0],
				Rows: int64(hi - lo), K: int64(kBlocks), RowStr: int64(rowBytes),
				Scr: &konst[0], AHalf: &f.half[0], Scratch: f.scratchFor(worker),
			}
			code.Call(&args)
		})
		return f.done(out, nrows, t0, t1)
	}
	bulk := nrows / cpu.Interleave * cpu.Interleave
	if bulk > 0 {
		f.pool.Do(bulk, chunk, func(worker, lo, hi int) {
			args := cpu.Args{
				Out: &out[lo], W: &w[lo*rowBytes], A: &f.q[0], AScale: &f.pairs[0],
				Rows: int64((hi - lo) / cpu.Interleave), K: int64(kBlocks),
				RowStr: int64(rowBytes), Scr: &konst[0], AHalf: &f.half[0],
				Scratch: f.scratchFor(worker),
			}
			code4.Call(&args)
		})
	}
	if bulk < nrows {
		args := cpu.Args{
			Out: &out[bulk], W: &w[bulk*rowBytes], A: &f.q[0], AScale: &f.pairs[0],
			Rows: int64(nrows - bulk), K: int64(kBlocks), RowStr: int64(rowBytes),
			Scr: &konst[0], AHalf: &f.half[0], Scratch: f.scratchFor(0),
		}
		code.Call(&args)
	}
	return f.done(out, nrows, t0, t1)
}

// done records the region split. Every successful MatVec exit goes through it
// so the profile covers all three kernel paths.
func (f *JIT) done(out []float32, nrows int, t0, t1 int64) bool {
	t2 := f.now()
	if f.cfg.Profile {
		t3 := f.now()
		nsQuantize.Add(t1 - t0)
		nsKernel.Add(t2 - t1)
		nsOut.Add(t3 - t2)
	}
	return true
}

func itoaN(v int) string { return strconv.Itoa(v) }

// now is a zero-cost no-op unless THIS JIT is profiling.
func (f *JIT) now() int64 {
	if !f.cfg.Profile {
		return 0
	}
	return time.Now().UnixNano()
}

// Regions and ResetRegions expose the pool's dispatch counters.
func (f *JIT) Regions() (parallel, serial int64) {
	if f == nil {
		return 0, 0
	}
	return f.pool.Regions()
}

// NUMA is the node count the pool distributes over (0 when it does not) and
// how many regions it has dispatched node-affine.
func (f *JIT) NUMA() (nodes int, affine int64) {
	if f == nil {
		return 0, 0
	}
	return f.pool.Nodes(), f.pool.Affine()
}

// KSliced is how many matvecs this JIT has split over k across workers
// (fusedSliced): the selection check for a gate that means to cover that path,
// which a pool narrower than the matrix wants never takes by itself.
func (f *JIT) KSliced() int64 {
	if f == nil {
		return 0
	}
	return f.ksliced
}

// Fuse opens a fused region on the pool (sched.Pool.Fuse): until Unfuse, every
// region this JIT dispatches is a phase of one region. It reports whether it
// opened one; a pool another user shares, a NUMA pool and a one-core pool do
// not fuse, and their regions run as they always do.
func (f *JIT) Fuse() bool {
	if f == nil {
		return false
	}
	if f.lookOff {
		return f.pool.Fuse()
	}
	if w := f.pool.Max(); len(f.look.pos) < w {
		f.look.pos = make([]aheadPos, w)
	}
	return f.pool.FuseWith(&f.look)
}

// Unfuse closes the region a true Fuse opened.
func (f *JIT) Unfuse() { f.pool.Unfuse() }

// Phases is how many phases fused regions have run on this JIT's workers: the
// regions fusion folded into one, which Regions does not count.
func (f *JIT) Phases() int64 {
	if f == nil {
		return 0
	}
	return f.pool.Phases()
}

func (f *JIT) ResetRegions() {
	if f != nil {
		f.pool.ResetRegions()
	}
}

// ActWindow is how many elements share one activation scale in this session.
func (f *JIT) ActWindow() int {
	if f == nil {
		return cpu.Q8Block
	}
	return f.actWindow
}
