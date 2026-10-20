package maintier

import (
	"time"

	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// This package, and backend, vulkan and kernels below it, read no
// environment. cmd/jitllm reads the JITLLM_* names and passes options, so an
// embedder or a server can configure each tier explicitly. The one exception
// is fault.go, compiled only under `-tags jitllmfault`.
//
// The knobs are per tier: OpenWith and New copy them onto the tier's Config,
// so two models in one process keep their own pins. A reader with no tier in
// scope (choose, order) takes them as an argument.

// knobs are the measurement and bisection switches that are not Config fields.
// Zero is the shipping default for every one of them.
type knobs struct {
	api           string          // pin the backend by API name
	tune          TuneMode        // probe the devices, or keep the order given
	split         int             // pin the matvec column split; 0 measures
	accSplit      int             // pin the attention weighted-sum split
	mvAccSplit    int             // pin the matvec accumulator split
	lanes         int             // pin the subgroup width: 0 ask, 1, or 32
	poisonKV      bool            // fill a fresh KV cache with NaN (RULE 13)
	poisonScratch bool            // and the block scratch, its twin
	noAttnMMA     bool            // decline the matrix instruction for scores only
	skip          map[string]bool // produce wrong answers on purpose; see WithSkip
	batch         BatchShape      // the batched prefill's tile shape
	mmaMT         int             // warp-matrix tile, rows
	mmaNT         int             // warp-matrix tile, columns
	mvMin         uint64          // bytes per crossing at which serving matvecs pays
	mvMinSet      bool            // ... and whether a caller set it, since 0 is a value
	arena         uint64          // packed-arena retention budget
	mlaFault      MLAFault        // produce a wrong latent attention on purpose
	scaleFault    ScaleFault      // drop one of Granite's scales on purpose
	cacheDir      string          // where the tuners' answers persist; "" keeps none
	lazyTune      bool            // tune a missing shape in the background; see WithLazyTune
}

// TuneMode is whether the tier measures the devices it was given.
type TuneMode int

const (
	// TuneAuto probes each backend with a real matvec and reads the cached
	// answer when one is on disk. The default.
	TuneAuto TuneMode = iota
	// TuneOff keeps the order it was given and takes the first device. A test
	// that wants a deterministic placement sets this, and so does fakeTier:
	// timing a device that answers instantly orders it by noise.
	TuneOff
	// TuneForce re-probes, ignoring the on-disk answer.
	TuneForce
)

// BatchShape is the batched prefill's tile geometry. Every field is zero for
// "take the fitted default"; they exist so `cmd/mvbench` can sweep a shape that
// is otherwise a constant in the emitter.
type BatchShape struct {
	Tok    int // token columns per thread
	RowT   int // rows per thread
	QTile  int // attention query tile
	KTile  int // attention key tile
	ATile  int // attention accumulate tile
	Parts  int // partial reductions per row
	AttnNT int // attention output columns per thread
	// The ragged decode step's matvec (LayersRows) on a device with no integer
	// matrix instruction: token columns and rows per thread, and the k split.
	RagTok, RagRowT, RagSplit int
	// Split pins the batched prefill matvec's k split; 0 splits until the
	// grid is full (dot4Split).
	Split int
}

// WithAPI pins the backend by name ("cuda", "vulkan", "metal") instead of
// measuring. A host without it falls back to the first device, loudly.
func WithAPI(s string) Option { return func(o *openOpts) { o.kb.api = s } }

// WithDeviceTune decides whether the tier probes the devices it opened. See
// TuneMode.
func WithDeviceTune(m TuneMode) Option { return func(o *openOpts) { o.kb.tune = m } }

// WithTuneCache names the directory the device tuners keep their answers in
// (the backend probe, the split tuner and the sequence retune), so a second
// process on the same device, driver and executable reads them instead of
// timing again. An entry is keyed by the device UUID, never its name, by the
// driver version and by the executable's fingerprint; a device with no UUID or
// no driver version caches nothing. "" (the default) keeps nothing on disk.
func WithTuneCache(dir string) Option { return func(o *openOpts) { o.kb.cacheDir = dir } }

// WithLazyTune moves the split tuner off the placement path: a shape with no
// answer in the WithTuneCache directory runs the table's split at once, and a
// background goroutine times the candidates between the device's own work
// and writes the winner to the cache and to the tier, where every matvec
// built after it takes it (and every later process reads it). It needs a
// cache directory; without one the tuner measures at placement as before.
func WithLazyTune(on bool) Option { return func(o *openOpts) { o.kb.lazyTune = on } }

// WithVerbose traces every backend's open attempt, the device probe and the
// ordering to stderr. It sets package backend's trace switch, which is
// process-wide: it is the one option here that is not per tier.
func WithVerbose(on bool) Option { return func(o *openOpts) { o.verbose, o.verboseSet = on, true } }

// WithROCm names the ROCm library directory (the one holding
// libamdhip64.so) the AMD backend loads from, in place of the default search;
// only that directory is tried. A directory with no ROCm is no HIP device, and
// an explicit hip:N then fails naming it.
func WithROCm(dir string) Option { return func(o *openOpts) { o.hip.Path = dir } }

// WithVulkanDevice pins one Vulkan device by enumeration index or name
// substring, for a host with more than one.
func WithVulkanDevice(sel string) Option { return func(o *openOpts) { o.vk.Device = sel } }

// WithSubgroup overrides the subgroup-width guarantee: "off" assumes none,
// "force" assumes the driver's answer where the probe disagrees. See
// vulkan.Config.Subgroup -- "force" produces wrong answers on purpose and
// belongs only in a gate.
func WithSubgroup(mode string) Option { return func(o *openOpts) { o.vk.Subgroup = mode } }

// WithVulkanMaxGroups lowers how many workgroups one Vulkan dispatch carries
// below the device's maxComputeWorkGroupCount; 0 is the device's own. A launch
// past the limit is split into several dispatches either way, with the same
// result (vulkan.Config.MaxGroups). This sends every launch through the split
// on a device whose own limit nothing reaches, which is what a gate wants.
func WithVulkanMaxGroups(n uint32) Option { return func(o *openOpts) { o.vk.MaxGroups = n } }

// WithCentering configures weight centering: off, and the
// token-columns threshold at which it is emitted. See kernels.Center.
func WithCentering(noCenter bool, minTok int) Option {
	return func(o *openOpts) { o.kern.NoCenter, o.kern.MinTok = noCenter, minTok }
}

// WithCUDAInline chooses how CUDA device calls reach the driver: on the
// caller's thread (true, the default) or posted to the device's owner
// goroutine. It applies to the devices this option set opens.
func WithCUDAInline(on bool) Option { return func(o *openOpts) { o.cudaPosted = !on } }

// WithCUDAMaxRegs caps every CUDA kernel this option set compiles at n
// registers per thread (backend.Opts.CUDAMaxRegs); 0 leaves ptxas its choice.
// A cap below what a kernel needs spills it, which is how the register gate is
// shown to fire.
func WithCUDAMaxRegs(n int) Option { return func(o *openOpts) { o.cudaRegs = n } }

// WithMetalFastMath compiles this tier's Metal libraries with Metal's fast
// math instead of the IEEE default the other backends have.
func WithMetalFastMath(on bool) Option { return func(o *openOpts) { o.metal.FastMath = on } }

// WithMetalSpin sets how long a Metal Wait polls a command buffer before it
// blocks. Negative blocks at once; zero takes the default.
func WithMetalSpin(d time.Duration) Option { return func(o *openOpts) { o.metal.Spin = d } }

// WithMetalConcurrent opens this tier's Metal compute encoders concurrent,
// with a buffer barrier where one dispatch depends on another, rather than
// serial (metal.Opts.Concurrent).
func WithMetalConcurrent(on bool) Option { return func(o *openOpts) { o.metal.Concurrent = on } }

// WithSplit pins the matvec column split. Summing S partial results is a
// different order from summing the blocks in sequence and float addition is
// not associative, so pinning it to 1 attributes a token divergence to the
// split rather than to a kernel bug.
func WithSplit(n int) Option { return func(o *openOpts) { o.kb.split = n } }

// WithAttnAccSplit pins how many partial sums the attention weighted sum is
// computed in. Same argument as WithSplit: pinning both arms to 1 is how a real
// difference is told from f32 reassociation.
func WithAttnAccSplit(n int) Option { return func(o *openOpts) { o.kb.accSplit = n } }

// WithMatVecAccSplit pins the matvec accumulator split. It exists because
// Vulkan and Metal cannot report Slots(), so both fall back to an unvalidated
// constant; this makes it sweepable.
func WithMatVecAccSplit(n int) Option { return func(o *openOpts) { o.kb.mvAccSplit = n } }

// WithLanes pins the subgroup width the device assumes: 1 (no shuffle) or 32.
// Zero asks backend.GuaranteedLanes. 32 on a device that does not guarantee it
// is what scripts/three-way.sh --violate subgroup-width sets.
func WithLanes(n int) Option { return func(o *openOpts) { o.kb.lanes = n } }

// WithPoisonKV fills a fresh KV cache with NaN instead of trusting the
// allocator, so a kernel reading memory nothing wrote fails a gate (RULE 13).
func WithPoisonKV(on bool) Option { return func(o *openOpts) { o.kb.poisonKV = on } }

// WithPoisonScratch fills every block scratch buffer with NaN at allocation,
// WithPoisonKV's twin. The scratch is per device and outlives a session, so a
// kernel reading a region it did not write is right while the buffer is fresh
// zeros and wrong for the next session.
func WithPoisonScratch(on bool) Option { return func(o *openOpts) { o.kb.poisonScratch = on } }

// WithoutAttnMMA declines the warp matrix instruction for the attention SCORES
// only, leaving it in place for the matvecs. Config.NoMMA declines it
// everywhere; this is the narrower bisection.
func WithoutAttnMMA(on bool) Option { return func(o *openOpts) { o.kb.noAttnMMA = on } }

// MLAFault is a deliberate violation of one of multi-head latent attention's
// three load-bearing invariants. MLAFaultNone is the shipping path.
//
// It produces wrong answers on purpose so the gate comparing a placed MLA
// block against the host can be shown to fire: each of these failures is
// invisible to a shape check, an allocation ledger and a kernel gate.
type MLAFault int

const (
	// MLAFaultNone is the engine.
	MLAFaultNone MLAFault = iota
	// MLAFaultSwapBanks absorbs through attn_v_b and un-absorbs through
	// attn_k_b. It is the "W_k is not transposed" class expressed as something
	// a test can force: the container stores attn_k_b ALREADY transposed and
	// attn_v_b verbatim, so a tier that treated them alike -- or transposed the
	// wrong one -- would compute exactly this, with every shape right.
	MLAFaultSwapBanks
	// MLAFaultValueHeadMajor reads the value as NHead heads of KVLoraRank
	// inside the cache row -- head h at h*KVLoraRank -- instead of every head
	// sharing the row's leading KVLoraRank floats. It is the mistake an
	// ordinary head-major V cache invites; head 0 still reads the right floats.
	MLAFaultValueHeadMajor
	// MLAFaultSheetOff shifts the per-head sheet index of both absorb banks by
	// one, so head h absorbs through head h+1's weights. Every read is in
	// bounds, every value finite, and the model stays fluent.
	MLAFaultSheetOff
	// MLAFaultRotateNoPE rotates a NoPE model's rotary channels (Kimi-Linear
	// keeps qk_rope_head_dim channels and never rotates them) in a batched
	// chunk's and a ragged step's key and query (emitMLARows), as a positional
	// model would. Exact at position 0, where the rotation is the identity, and
	// wrong after. Decode does not take it, so its gates compare the batched
	// path against decode a row at a time.
	MLAFaultRotateNoPE
)

// WithMLAFault violates one of latent attention's invariants on purpose, so
// that the gate comparing a placed MLA block against the host can be shown to
// fire. MLAFaultNone is the default and the shipping path.
func WithMLAFault(f MLAFault) Option { return func(o *openOpts) { o.kb.mlaFault = f } }

// ScaleFault drops one of the scales a Granite model states (embedding_scale,
// residual_scale, attention.scale) on the device, so the gate comparing its
// placed blocks against the host can be shown to fire. Each leaves a fluent
// model: a scale is one multiply, and nothing about a shape or a buffer moves.
// ScaleFaultNone is the shipping path. (logit_scale divides the logits on the
// host after the device, so it has no device arm to break.)
type ScaleFault int

const (
	// ScaleFaultNone is the engine.
	ScaleFaultNone ScaleFault = iota
	// ScaleFaultResidual adds each block output to the residual unscaled.
	ScaleFaultResidual
	// ScaleFaultAttn scores attention at 1/sqrt(HeadDim) instead of the plan's
	// scale, in every kernel that bakes one (devTier.scoreScale).
	ScaleFaultAttn
	// ScaleFaultEmbd gathers a prompt's embedding rows on the device unscaled.
	ScaleFaultEmbd
	// ScaleFaultStepResidual lets a step across sessions fuse its attention
	// and FFN residual adds into the projections, unscaled, where decode and a
	// prompt chunk keep the scaled add: the step alone parts from each session
	// run alone.
	ScaleFaultStepResidual
)

// WithScaleFault drops one of Granite's scales on the device on purpose.
func WithScaleFault(f ScaleFault) Option { return func(o *openOpts) { o.kb.scaleFault = f } }

// WithSkip omits parts of a multi-block submission -- "attn", "ffn", or the
// finer keys the submission builder tests. It produces wrong answers on
// purpose, to bisect a divergence inside a submission, and belongs only in a
// gate.
func WithSkip(keys ...string) Option {
	return func(o *openOpts) {
		if o.kb.skip == nil {
			o.kb.skip = map[string]bool{}
		}
		for _, k := range keys {
			if k != "" {
				o.kb.skip[k] = true
			}
		}
	}
}

// WithBatchShape sets the batched prefill's tile geometry; see BatchShape.
func WithBatchShape(b BatchShape) Option { return func(o *openOpts) { o.kb.batch = b } }

// WithMMATile pins the warp-matrix tile as rows x columns; 0,0 takes the fitted
// default. mvbench -mma has the per-shape isolated numbers.
func WithMMATile(mt, nt int) Option {
	return func(o *openOpts) { o.kb.mmaMT, o.kb.mmaNT = mt, nt }
}

// WithMinMatVecBytes sets the average weight bytes per crossing at which
// serving matvecs one at a time breaks even, a property of the link and the
// model rather than the card. Zero is a value, not an absence: it serves every
// matvec however few bytes it moves. Not calling this takes the default.
func WithMinMatVecBytes(n uint64) Option {
	return func(o *openOpts) { o.kb.mvMin, o.kb.mvMinSet = n, true }
}

// WithArena sets the packed arena's retention budget in bytes: every block's
// weights in the device layout, host-resident, so a block paged back onto a card
// is a DMA rather than a repack. Zero retains nothing, which is what a model
// that does not page wants.
func WithArena(n uint64) Option { return func(o *openOpts) { o.kb.arena = n } }

// WithConfig applies f to the Config the tier is built with, which is where the
// A/B arms live: NoGraph, Paging, NoUnpack, NoBatch, NoTail, NoKT, NoMMA, KVF16,
// Verify, TableSplit, PerLayerSubmit, ScalarSoftmax and StageLimit. One option
// rather than a wrapper per field keeps each field's documentation in one
// place.
func WithConfig(f func(*Config)) Option { return func(o *openOpts) { o.cfg = append(o.cfg, f) } }

// apply publishes the one process-wide switch in the option set: backend's
// trace logger. Everything else is copied onto the tier by New.
func (o *openOpts) apply() {
	if o.verboseSet {
		backend.SetVerbose(o.verbose)
	}
}

// devOpts is what every device this option set opens is opened with.
func (o *openOpts) devOpts() backend.Opts {
	return backend.Opts{Vulkan: o.vk, CUDAPosted: o.cudaPosted, CUDAMaxRegs: o.cudaRegs, Metal: o.metal, HIP: o.hip}
}

// WithPersist sets whether each decode block's launches are composed into
// persistent kernels (Config.Persist); grid, when positive, caps their grid
// below the device's co-residency limit (Config.PersistGrid). Off by default.
func WithPersist(on bool, grid int) Option {
	return WithConfig(func(c *Config) { c.Persist, c.PersistDeps, c.PersistGrid = on, false, grid })
}

// WithGateUp2 sets Config.GateUp2: a dense gated FFN's gate and up as one
// Gate2 matvec writing act(gate)*up. On by default.
func WithGateUp2(on bool) Option { return WithConfig(func(c *Config) { c.GateUp2 = on }) }

// WithGateUp2Force sets Config.GateUp2Force: Gate2 wherever the shape has it,
// whatever the tuner chose.
func WithGateUp2Force(on bool) Option { return WithConfig(func(c *Config) { c.GateUp2Force = on }) }

// WithPersistDeps is WithPersist with the persistent kernels' tasks ordered
// by dependency counters rather than grid barriers (Config.PersistDeps,
// docs/design/dep-block.md). Off by default.
func WithPersistDeps(on bool, grid int) Option {
	return WithConfig(func(c *Config) { c.Persist, c.PersistDeps, c.PersistGrid = on, on, grid })
}

// WithFFNForm sets Config.FFNForm: the recipe every dense gated FFN runs,
// FFNTuned (the default) leaving it to the per-card trial.
func WithFFNForm(f FFNForm) Option { return WithConfig(func(c *Config) { c.FFNForm = f }) }

// WithGateUpFuse sets Config.GateUpFuse: a dense gated FFN's gate and up in
// one launch, down forming act(gate)*up in its prologue.
func WithGateUpFuse(on bool) Option {
	return WithConfig(func(c *Config) { c.GateUpFuse = on })
}

// WithAutoStream sets whether a mixture block no device can hold resident is
// streamed onto one (GPU.AutoStream) rather than left on the host. On by
// default.
func WithAutoStream(on bool) Option {
	return WithConfig(func(c *Config) { c.NoAutoStream = !on })
}

// WithHybridCache sets whether a hybrid block gets an expert cache on the
// card: its hottest experts run there, beside the host running the rest of
// each token's selection. On by default.
func WithHybridCache(on bool) Option {
	return WithConfig(func(c *Config) { c.NoHybridCache = !on })
}

// WithSplitAdmit caps how many experts a hybrid block's cache takes in a
// token (Config.SplitAdmit).
func WithSplitAdmit(n int) Option {
	return WithConfig(func(c *Config) { c.SplitAdmit = n })
}

// WithHybridExperts sets where a streamed block's routed experts run by
// default: on the host (true, hybrid execution) or sent to the card each token
// (false). A placement's choice for a block (GPU.PlaceExperts) wins.
func WithHybridExperts(on bool) Option {
	return WithConfig(func(c *Config) { c.HybridExperts, c.NoHybrid = on, !on })
}
