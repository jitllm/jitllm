//go:build amd64 || arm64

package nn

import (
	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// Option configures a JIT. The package reads no environment: cmd/jitllm reads
// the JITLLM_* names and passes them here, so every knob is settable by an
// embedded caller. There is no knob to turn the JIT off; there is no
// interpreted tier to turn onto.
type Option func(*Config)

// Config is the resolved set. Zero is the shipping default.
type Config struct {
	// Sched is handed to the worker pool this JIT builds.
	Sched []sched.Option
	// PrefillCores names a sched.CoreSet for a second pool that prefill
	// drains through (BeginPrefill); "" takes the host default, and "p" -- or
	// any set no wider than the decode pool -- builds none. See prefillCoreSet.
	PrefillCores string
	// ServingSplit lets a prefill run on the E-cores' own pool while another
	// session decodes on the P-cores (BeginPrefillE): where there are E-cores
	// to give (sched.CanSplitE, macOS) and the caller says another session is
	// busy. Off, BeginPrefillE does nothing.
	ServingSplit bool

	// Profile arms the quantize/kernel/widen nanosecond counters inside MatVec.
	// MatMulProf does the same for the batched path, splitting it into
	// pack / kernel / transpose.
	Profile    bool
	MatMulProf bool
	// QuantActGo builds no activation-quantize kernel; see WithQuantActGo.
	QuantActGo bool
	// RopeGo runs the float64 Go rotary table; see WithRopeGo.
	RopeGo bool

	// Tune is the width/tile/chunk tuner. Off pins the fitted defaults; Force
	// re-measures the host instead of reading the on-disk cache.
	Tune TuneMode
	// Quiet silences the tuner's per-duel lines on stderr.
	Quiet bool

	// Pack pins the matvec interleave width instead of letting the tuner climb
	// to it on real tokens. Tile pins the GEMM tile, Chunk the prefill batch
	// width, Part how many pool threads drain a region. Zero means "tune".
	Pack  int
	Tile  string
	Chunk int
	Part  int
	// FusedPrefetch pins the fused kernel's software-prefetch distance in row
	// groups: -1 is unset (tune), 0 none. See WithFusedPrefetch.
	FusedPrefetch int
	// Prefetch is the amd64 GGUF k-quant kernels' PREFETCHT0 distance in
	// super-blocks and A64Prefetch the arm64 packed matvec's PRFM distance in
	// bytes; see cpu.EmitOpts.
	Prefetch, A64Prefetch int
	// Probes is the arm64 measurement scaffolding; see cpu.Probes.
	Probes cpu.Probes

	// Accs pins how many independent accumulator chains a k-quant matvec
	// carries (1..4); 0 takes cpu.BestAccs. A measurement instrument, per JIT
	// so two States in one process can carry different values for an A/B.
	Accs int

	// NoGEMM makes MatMul always decline, so prefill falls back to a per-token
	// MatVec loop -- the pre-GEMM engine, kept as the A/B arm it was built as.
	NoGEMM bool

	// ActWindow, when not zero, is the activation amax window in place of
	// the one the weight types decide (cpu.WideActWindow): a model whose
	// graph quantizes every projection input per token (BitNet) asks for a
	// window at least as wide as its widest row, so every 32-element block
	// of a row shares the row's one scale. A multiple of 32.
	ActWindow int
	// ChunkBytes caps the weight slice one pool chunk of a batched matmul
	// covers, in bytes, 0 meaning no locality cap (the GC-preemption budget
	// alone sizes the chunk). Batched path only; see batchedChunk.
	ChunkBytes int

	// TileTok pins the token tile a batched packed matmul runs at, 0 meaning
	// as wide as the registers and the code budget allow. A prefill sweeps the
	// weights ceil(ntok/tile) times, so this knob changes traffic rather than
	// arithmetic, which makes it the dose for bus-versus-kernel questions.
	TileTok int

	// GEMMTok is how many tokens one call of the weight-stationary prefill GEMM
	// covers (cpu.EmitPackedMatMulStationary): 0 sizes it from k, a
	// negative value turns that kernel off so MatMulPacked runs the tiled or
	// per-token path it replaced -- the A/B arm, kept reachable from one binary.
	GEMMTok int
	// GEMMRows pins the rows one call of that GEMM covers (rounded down to a
	// multiple of eight, at least eight); 0 takes the measured default.
	GEMMRows int
	// GEMMExact keeps every packed kernel on its float epilogue: that GEMM,
	// and on arm64 the fused decode matvec, both of which otherwise sum a
	// k-quant's super-block in integers (cpu.StationaryIntAcc) -- the same
	// integers, so they agree with each other, but not with the tiled
	// kernel's per-sub-block float chain. It also keeps the decode matvec on
	// the whole of k unless FusedKSlices pins a count, since a k-split is
	// another order (fusedSlices). The A/B arm, and the arithmetic the
	// exactness gates and the llama.cpp coverage ratchet measure.
	GEMMExact bool
	// FusedKSplit sets how the fused decode matvec meets the per-call budget:
	// 0 walks k in the largest pieces the budget allows over a balanced row
	// chunk, a positive value pins the piece in super-blocks, and -1 is the
	// old contract (whole k per call, row chunk capped). See fusedPiece.
	FusedKSplit int
	// FusedKSlices splits a short, long-k matrix over k across workers: 0
	// decides (fusedSlices), -1 never, n pins n slices. Not bit-identical to
	// the whole-k call; see fusedSlices.
	FusedKSlices int
	// MatVecPick is how MatVecPacked chooses between its kernels per matrix
	// shape (see mvpick.go): 0 decides (timed in place on arm64, the fused
	// kernel wherever it exists elsewhere), -1 never times, and n > 0 pins
	// candidate n-1 of pickArms.
	MatVecPick int
	// MixtureBatch is whether a mixture's experts go out in one pool region
	// (MatVecPackedMulti/Gather): 0 decides (see MixtureBatches), 1 on, -1
	// off, which keeps the caller's per-expert loop.
	MixtureBatch int
}

// TuneMode is whether the width tuners run.
type TuneMode int

const (
	TuneAuto  TuneMode = iota // measure, and read the on-disk cache
	TuneOff                   // take the fitted defaults, measure nothing
	TuneForce                 // measure, ignoring the on-disk cache
)

func newConfig(opts []Option) Config {
	c := Config{FusedPrefetch: -1}
	for _, o := range opts {
		o(&c)
	}
	return c
}

// WithSched passes options to the worker pool this JIT creates.
func WithSched(o ...sched.Option) Option { return func(c *Config) { c.Sched = append(c.Sched, o...) } }

// WithQuantActGo builds no activation-quantize kernel, so every quantize runs
// cpu.QuantizeQ8Window. It is the A/B arm for those kernels and nothing else
// uses it.
func WithQuantActGo(on bool) Option { return func(c *Config) { c.QuantActGo = on } }

// WithRopeGo builds the rotary table with the float64 Go loop instead of the
// generated kernel: the A/B arm for that kernel. Its body only exists under
// the jitllmtest build tag (engine/nn/ropego.go), so a release binary handed this
// option panics rather than running Go arithmetic on the token path.
func WithRopeGo(on bool) Option { return func(c *Config) { c.RopeGo = on } }

// WithProfile arms the per-phase nanosecond counters ProfileNanos reports.
func WithProfile(on bool) Option { return func(c *Config) { c.Profile = on } }

// WithMatMulProfile splits the batched path into pack / kernel / transpose.
func WithMatMulProfile(on bool) Option { return func(c *Config) { c.MatMulProf = on } }

// WithTune chooses whether the width tuners run and whether they trust the cache.
func WithTune(m TuneMode) Option { return func(c *Config) { c.Tune = m } }

// WithQuietTuner silences the tuner's stderr lines.
func WithQuietTuner(on bool) Option { return func(c *Config) { c.Quiet = on } }

// WithPackWidth pins the matvec interleave width; 0 tunes it.
func WithPackWidth(n int) Option { return func(c *Config) { c.Pack = n } }

// WithFusedPrefetch pins how many row groups ahead the fused packed kernel
// prefetches; 0 emits no prefetch. Unset, it is dueled on real decode tokens.
func WithFusedPrefetch(n int) Option { return func(c *Config) { c.FusedPrefetch = n } }

// WithPrefetch sets the amd64 GGUF k-quant kernels' PREFETCHT0 distance in
// super-blocks; 0 emits none. See cpu.EmitOpts.
func WithPrefetch(n int) Option { return func(c *Config) { c.Prefetch = n } }

// WithA64Prefetch sets the arm64 packed matvec's PRFM distance in bytes: 0
// takes one tile, negative emits none. See cpu.EmitOpts.
func WithA64Prefetch(n int) Option { return func(c *Config) { c.A64Prefetch = n } }

// WithProbes injects n dead scalar adds, 128-bit loads and vector adds per
// block into the arm64 k-quant decode matvec, to price one class of operation
// against another. A measurement instrument: it makes the kernel slower.
func WithProbes(scalar, load, vector int) Option {
	return func(c *Config) { c.Probes = cpu.Probes{Scalar: scalar, Load: load, Vector: vector} }
}

// WithAccs pins the accumulator-chain count of the k-quant matvec; 0 measures.
func WithAccs(n int) Option { return func(c *Config) { c.Accs = n } }

// WithGEMMTile pins the GEMM tile, spelled as the tuner prints it; "" tunes it.
func WithGEMMTile(s string) Option { return func(c *Config) { c.Tile = s } }

// WithPrefillChunk pins the prefill batch width; 0 duels for it.
func WithPrefillChunk(n int) Option { return func(c *Config) { c.Chunk = n } }

// WithPrefillCoreSet names the core set prefill runs on; see Config.PrefillCores.
func WithPrefillCoreSet(name string) Option { return func(c *Config) { c.PrefillCores = name } }

// WithServingSplit allows the E-core prefill pool; see Config.ServingSplit.
func WithServingSplit(on bool) Option { return func(c *Config) { c.ServingSplit = on } }

// WithParticipants pins how many pool threads drain a region; 0 duels for it.
func WithParticipants(n int) Option { return func(c *Config) { c.Part = n } }

// WithoutGEMM makes MatMul decline, so prefill runs a per-token MatVec loop.
func WithoutGEMM(on bool) Option { return func(c *Config) { c.NoGEMM = on } }

// WithTileTokens pins the batched matmul's token tile. Zero is the default:
// the widest tile the register file and the L1i budget allow.
func WithTileTokens(n int) Option { return func(c *Config) { c.TileTok = n } }

// WithGEMMTokens sets the weight-stationary GEMM's tokens per call; negative
// turns that kernel off. Zero is the default, sized from k.
func WithGEMMTokens(n int) Option { return func(c *Config) { c.GEMMTok = n } }

// WithFusedKSlices pins the k-slice count; see Config.FusedKSlices.
func WithFusedKSlices(n int) Option { return func(c *Config) { c.FusedKSlices = n } }

// WithMatVecPick sets how the packed matvec's kernel is chosen per shape; see
// Config.MatVecPick.
func WithMatVecPick(n int) Option { return func(c *Config) { c.MatVecPick = n } }

// WithMixtureBatch forces the batched mixture dispatch on (1) or off (-1); see
// Config.MixtureBatch.
func WithMixtureBatch(n int) Option { return func(c *Config) { c.MixtureBatch = n } }

// WithFusedKSplit pins the fused matvec's k piece; see Config.FusedKSplit.
func WithFusedKSplit(n int) Option { return func(c *Config) { c.FusedKSplit = n } }

// WithGEMMExact keeps every packed kernel on its float epilogue; see
// Config.GEMMExact.
func WithGEMMExact(on bool) Option { return func(c *Config) { c.GEMMExact = on } }

// WithGEMMRows pins the weight-stationary GEMM's rows per call; 0 is the
// measured default.
func WithGEMMRows(n int) Option { return func(c *Config) { c.GEMMRows = n } }

// WithChunkBytes caps the weight slice one pool chunk of a batched matmul
// covers. Zero is the default: no locality cap, the GC budget alone.
func WithChunkBytes(n int) Option { return func(c *Config) { c.ChunkBytes = n } }

// WithActWindow sets the activation amax window in place of the one the
// weight types decide; see Config.ActWindow.
func WithActWindow(n int) Option { return func(c *Config) { c.ActWindow = n } }
