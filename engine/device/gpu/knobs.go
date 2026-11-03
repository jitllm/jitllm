package gpu

import (
	"time"

	"github.com/jitllm/jitllm/backend"
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
	verbose       bool            // trace the opens, the device probe and the ordering
	tune          TuneMode        // probe the devices, or keep the order given
	lanes         int             // pin the subgroup width: 0 ask, 1, or 32
	poisonKV      bool            // fill a fresh KV cache with NaN (RULE 13)
	poisonScratch bool            // and the block scratch, its twin
	dropSegments  bool            // every composed split sums all but its last segment
	ropeFault     [2]int          // the rotary tables read each row at k times its position plus d
	skip          map[string]bool // produce wrong answers on purpose; see WithSkip
	batch         BatchShape      // the batched prefill's tile shape
	mmaMT         int             // warp-matrix tile, rows
	mmaNT         int             // warp-matrix tile, columns
	arena         uint64          // packed-arena retention budget
	scaleFault    ScaleFault      // drop one of Granite's scales on purpose
	pageFault     PageFault       // read a routed expert's wrong page on purpose
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

// WithVerbose traces every backend's open attempt, the device probe and the
// ordering to stderr, for this tier's opens.
func WithVerbose(on bool) Option { return func(o *openOpts) { o.kb.verbose = on } }

// WithCounters names what this tier's devices count into (backend.Counters),
// so a caller can read one set across tiers; without it each tier has its own
// (GPU.Counters).
func WithCounters(c *backend.Counters) Option { return func(o *openOpts) { o.counters = c } }

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

// WithMetalUnretained selects -commandBufferWithUnretainedReferences on this
// tier's Metal devices (metal.Opts.Unretained), a measurement instrument.
func WithMetalUnretained(on bool) Option { return func(o *openOpts) { o.metal.Unretained = on } }

// WithMetalResidency attaches an MTLResidencySet to this tier's Metal devices
// (metal.Opts.Residency); the open fails where one cannot be carried.
func WithMetalResidency(on bool) Option { return func(o *openOpts) { o.metal.Residency = on } }

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

// WithSplitFault has every split the composer forms sum its segments but the
// last (composer.Registry.DropSegments): the violation a serving gate must fail on.
func WithSplitFault(on bool) Option { return func(o *openOpts) { o.kb.dropSegments = on } }

// WithRopeFault has the rotary tables read every row at k times its cache
// position plus d (ropeRowsWords): the violation a served table must fail
// on. k 0 is none.
func WithRopeFault(k, d int) Option { return func(o *openOpts) { o.kb.ropeFault = [2]int{k, d} } }

// ScaleFault drops Granite's embedding_scale from the composed input rows'
// graph on the device, so the gate comparing its placed blocks against the
// host can be shown to fire. The residual and attention scales are the
// model's statement, broken there (the gates' "not stated" arms).
// ScaleFaultNone is the shipping path.
type ScaleFault int

const (
	// ScaleFaultNone is the engine.
	ScaleFaultNone ScaleFault = iota
	// ScaleFaultEmbd gathers a prompt's embedding rows on the device unscaled.
	ScaleFaultEmbd
)

// WithScaleFault drops one of Granite's scales on the device on purpose.
func WithScaleFault(f ScaleFault) Option { return func(o *openOpts) { o.kb.scaleFault = f } }

// PageFault breaks a composed block's banks held in pages (pagedRun) on
// purpose, so the gate holding them to whole banks can be shown to fire.
type PageFault int

const (
	// PageFaultNone is the engine.
	PageFaultNone PageFault = iota
	// PageFaultSwap writes the page table with the step's first two
	// selected experts' pages swapped: a page read from the wrong expert.
	PageFaultSwap
	// PageFaultStale maps the step's selection onto pages without reading a
	// missed expert into the page it takes: an eviction not re-read.
	PageFaultStale
)

// WithPageFault breaks the composed path's expert pages on purpose.
func WithPageFault(f PageFault) Option { return func(o *openOpts) { o.kb.pageFault = f } }

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

// WithArena sets the packed arena's retention budget in bytes: every block's
// weights in the device layout, host-resident, so a block paged back onto a card
// is a DMA rather than a repack. Zero retains nothing, which is what a model
// that does not page wants.
func WithArena(n uint64) Option { return func(o *openOpts) { o.kb.arena = n } }

// WithConfig applies f to the Config the tier is built with, which is where the
// A/B arms live: NoGraph, NoBatch, NoTail, NoMMA, KVF16, PerLayerSubmit and
// StageLimit. One option
// rather than a wrapper per field keeps each field's documentation in one
// place.
func WithConfig(f func(*Config)) Option { return func(o *openOpts) { o.cfg = append(o.cfg, f) } }

// devOpts is what every device this option set opens is opened with.
func (o *openOpts) devOpts() backend.Opts {
	return backend.Opts{Vulkan: o.vk, CUDAPosted: o.cudaPosted, CUDAMaxRegs: o.cudaRegs, Metal: o.metal, HIP: o.hip,
		Counters: o.counters, Verbose: o.kb.verbose}
}

// WithPersist sets whether each decode block's launches are composed into
// persistent kernels (Config.Persist); grid, when positive, caps their grid
// below the device's co-residency limit (Config.PersistGrid). Off by default.
func WithPersist(on bool, grid int) Option {
	return WithConfig(func(c *Config) { c.Persist, c.PersistDeps, c.PersistGrid = on, false, grid })
}

// WithComposedShadow sets Config.ComposedShadow: the composer's form of each
// block it would serve runs beside the tier's own kernels and is compared with
// them at every step (Stats.ShadowWorst). It changes no answer.
func WithComposedShadow() Option { return WithConfig(func(c *Config) { c.ComposedShadow = true }) }

// WithPersistDeps is WithPersist with the persistent kernels' tasks ordered
// by dependency counters rather than grid barriers (Config.PersistDeps,
// docs/design/dep-block.md). Off by default.
func WithPersistDeps(on bool, grid int) Option {
	return WithConfig(func(c *Config) { c.Persist, c.PersistDeps, c.PersistGrid = on, on, grid })
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
