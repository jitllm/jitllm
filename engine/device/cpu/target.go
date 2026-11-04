package cpu

import (
	"sync/atomic"
	"unsafe"
)

// Target is the CPU a model's CPU device maps generated code onto: its
// features, probed when the Target is made less what its options take away,
// the tier they make, whether the engine runs on it at all (Baseline), how
// many codes it mapped per tier, and the perf map it writes when asked
// (WithPerfMap). A CPU device is opened with the options its device spec
// states (cpu:sse, cpu:novnni, cpu:nodotprod, cpu:perfmap; SpecOptions), and
// each model's CPU device maps through its own Target.
type Target struct {
	f        Features
	tier     Tier
	baseline error
	mapped   [numTiers]atomic.Int64
	perf     perfMap
}

// Option is something New states about the Target it makes.
type Option func(*options)

type options struct {
	tier                     Tier
	lower, noVNNI, noDotProd bool
	perfMap                  bool
}

// AtTier makes the Target one of tier t: what a CPU of that tier would not
// have is taken away, so the SSE tier's code is selected and run on an AVX2
// machine. A Target only loses capabilities -- it cannot execute an
// instruction its CPU lacks -- so the top tier takes nothing.
func AtTier(t Tier) Option { return func(o *options) { o.tier, o.lower = t, true } }

// WithoutVNNI takes AVX-VNNI and AVX512-VNNI away: the composer forms its
// pre-VNNI byte dots for the Target.
func WithoutVNNI() Option { return func(o *options) { o.noVNNI = true } }

// WithoutDotProd takes FEAT_DotProd away on arm64: every A64 kernel formed for
// the Target widens its SDOTs. Elsewhere there is nothing to take.
func WithoutDotProd() Option { return func(o *options) { o.noDotProd = true } }

// WithPerfMap has the Target write /tmp/perf-<pid>.map, the JIT symbol format
// Linux perf reads: one "<hex addr> <hex size> <name>" line per named code it
// maps, so samples resolve to a kernel's name and offset rather than
// runtime._ExternalCode. Off by default, because it writes a world-readable
// file into /tmp; every Target of the process appends to the one file.
func WithPerfMap() Option { return func(o *options) { o.perfMap = true } }

// New is the CPU this process runs on, as opts state it.
func New(opts ...Option) *Target {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	f := Probe()
	if o.noVNNI {
		f.AVXVNNI, f.AVX512VNNI = false, false
	}
	if o.noDotProd {
		withoutDotProd(&f)
	}
	if o.lower {
		lowerTo(&f, o.tier)
	}
	tg := &Target{f: f, tier: tierOfCPU(f), baseline: baselineOf(f)}
	if o.perfMap {
		tg.perf.open()
	}
	return tg
}

// Features is the Target's capability set.
func (tg *Target) Features() Features { return tg.f }

// Tier is the instruction-set tier the Target runs (tier.go).
func (tg *Target) Tier() Tier { return tg.tier }

// ISA is the Target's extensions as the composer lowers for them.
func (tg *Target) ISA() ISA { return tg.f.ISA() }

// Baseline reports whether the engine runs on the Target: whether the composer
// lowers for its tier.
//
// It is not Map's gate (Map checks each kernel's declared ISA, isa.go); it
// answers whether the Target has a tier at all. On amd64 an AVX2 host and an
// SSE4.1+SSSE3 host do; anything below never does. model.Open refuses
// through it, and the error names which extensions are missing, because the
// caller prints it.
//
// It exists because a pre-AVX2 host would SIGILL on the first token: the
// quantized path declined correctly, but the elementwise path asked a width
// question where a capability question belonged. See
// docs/engineering-history/cpu-kernels.md, "THE SSE TIER".
func (tg *Target) Baseline() error { return tg.baseline }

// Map makes code executable on the Target, refusing first if the Target cannot
// run it.
//
// The gate is per kernel and reads the kernel's own declaration: an SSE-tier
// kernel carries its ISA in a leading NOP (isa.go) and an undeclared kernel
// needs AVX2|FMA|F16C. Every executable byte the engine runs passes through
// a Target, which is why the check is not at the emitters: a condition
// consulted in many places gets consulted in most of them. mapExec is the
// ungated form, used only by the CPUID and XGETBV probes, which must run
// before there is a Target to ask.
func (tg *Target) Map(code []byte) (*Code, error) { return tg.MapNamed(code, "") }

// MapNamed is Map with the code's name for the perf map.
func (tg *Target) MapNamed(code []byte, name string) (*Code, error) {
	if err := tg.gate(code); err != nil {
		return nil, err
	}
	c, err := mapExec(code)
	if err != nil {
		return nil, err
	}
	c.Name = name
	t := TierNEON
	if tg.tier != TierNEON {
		t = KernelTier(code)
	}
	tg.mapped[t].Add(1)
	if name != "" {
		tg.perf.add(uintptr(unsafe.Pointer(c.entry)), len(code), name)
	}
	return c, nil
}

// MappedByTier is how many codes the Target has mapped, keyed by the tier each
// was generated for: a selection check, so a gate on a Target at the SSE tier
// can assert it mapped no AVX2 code. On amd64 the tier is read from the
// code's own declaration; on arm64 every code counts as TierNEON.
func (tg *Target) MappedByTier() map[Tier]int64 {
	out := make(map[Tier]int64, numTiers)
	for t := range tg.mapped {
		if n := tg.mapped[t].Load(); n != 0 {
			out[Tier(t)] = n
		}
	}
	return out
}

// Close ends the Target's perf map.
func (tg *Target) Close() error { return tg.perf.close() }
