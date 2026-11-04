package cpu

import (
	"sync/atomic"
	"unsafe"
)

// Host is the CPU a caller maps generated code onto: its features, probed
// when the Host is made or stated lower (AtTier, WithoutVNNI,
// WithoutDotProd), the tier they make, whether the engine runs on it at all
// (Baseline), how many codes it mapped per tier, and the perf map it writes
// when asked (WithPerfMap). Each model's CPU device maps through its own.
type Host struct {
	f        Features
	tier     Tier
	baseline error
	mapped   [numTiers]atomic.Int64
	perf     perfMap
}

// HostOption is something NewHost states about the Host it makes.
type HostOption func(*hostOpts)

type hostOpts struct {
	tier          Tier
	lower         bool
	noVNNI, noDot bool
	perfMap       bool
}

// AtTier makes the Host one of tier t: what a host of that tier would not
// have is taken away, so the SSE tier's code is selected and run on an AVX2
// machine. A Host only loses capabilities -- it cannot execute an
// instruction its CPU lacks -- so the top tier takes nothing.
func AtTier(t Tier) HostOption { return func(o *hostOpts) { o.tier, o.lower = t, true } }

// WithoutVNNI makes the Host one without AVX-VNNI or AVX512-VNNI.
func WithoutVNNI() HostOption { return func(o *hostOpts) { o.noVNNI = true } }

// WithoutDotProd makes the Host one without FEAT_DotProd, so the forms that
// widen SDOT are selected on a chip that has it.
func WithoutDotProd() HostOption { return func(o *hostOpts) { o.noDot = true } }

// WithPerfMap has the Host write /tmp/perf-<pid>.map, the JIT symbol format
// Linux perf reads: one "<hex addr> <hex size> <name>" line per named code it
// maps, so samples resolve to a kernel's name and offset rather than
// runtime._ExternalCode. Off by default, because it writes a world-readable
// file into /tmp; every Host of the process appends to the one file.
func WithPerfMap() HostOption { return func(o *hostOpts) { o.perfMap = true } }

// NewHost is the CPU this process runs on, as opts state it.
func NewHost(opts ...HostOption) *Host {
	var o hostOpts
	for _, opt := range opts {
		opt(&o)
	}
	f := CPU()
	if o.noVNNI {
		f.AVXVNNI, f.AVX512VNNI = false, false
	}
	if o.noDot {
		f.DotProd = false
	}
	if o.lower {
		lowerTo(&f, o.tier)
	}
	h := &Host{f: f, tier: tierOfHost(f), baseline: baselineOf(f)}
	if o.perfMap {
		h.perf.open()
	}
	return h
}

// Features is the Host's capability set.
func (h *Host) Features() Features { return h.f }

// Tier is the instruction-set tier the Host runs (tier.go).
func (h *Host) Tier() Tier { return h.tier }

// ISA is the Host's extensions as the composer lowers for them.
func (h *Host) ISA() ISA { return h.f.ISA() }

// Baseline reports whether the engine runs on the Host: whether the composer
// lowers for its tier.
//
// It is not Map's gate (Map checks each kernel's declared ISA, isa.go); it
// answers whether the Host has a tier at all. On amd64 an AVX2 host and an
// SSE4.1+SSSE3 host do; anything below never does. model.Open refuses
// through it, and the error names which extensions are missing, because the
// caller prints it.
//
// It exists because a pre-AVX2 host would SIGILL on the first token: the
// quantized path declined correctly, but the elementwise path asked a width
// question where a capability question belonged. See
// docs/engineering-history/cpu-kernels.md, "THE SSE TIER".
func (h *Host) Baseline() error { return h.baseline }

// Map makes code executable on the Host, refusing first if the Host cannot
// run it.
//
// The gate is per kernel and reads the kernel's own declaration: an SSE-tier
// kernel carries its ISA in a leading NOP (isa.go) and an undeclared kernel
// needs AVX2|FMA|F16C. Every executable byte the engine runs passes through
// a Host, which is why the check is not at the emitters: a condition
// consulted in many places gets consulted in most of them. mapExec is the
// ungated form, used only by the CPUID and XGETBV probes, which must run
// before there is a Host to ask.
func (h *Host) Map(code []byte) (*Code, error) { return h.MapNamed(code, "") }

// MapNamed is Map with the code's name for the perf map.
func (h *Host) MapNamed(code []byte, name string) (*Code, error) {
	if err := h.gate(code); err != nil {
		return nil, err
	}
	c, err := mapExec(code)
	if err != nil {
		return nil, err
	}
	c.Name = name
	t := TierNEON
	if h.tier != TierNEON {
		t = KernelTier(code)
	}
	h.mapped[t].Add(1)
	if name != "" {
		h.perf.add(uintptr(unsafe.Pointer(c.entry)), len(code), name)
	}
	return c, nil
}

// MappedByTier is how many codes the Host has mapped, keyed by the tier each
// was generated for: a selection check, so a gate on a Host at the SSE tier
// can assert it mapped no AVX2 code. On amd64 the tier is read from the
// code's own declaration; on arm64 every code counts as TierNEON.
func (h *Host) MappedByTier() map[Tier]int64 {
	out := make(map[Tier]int64, numTiers)
	for t := range h.mapped {
		if n := h.mapped[t].Load(); n != 0 {
			out[Tier(t)] = n
		}
	}
	return out
}

// Close ends the Host's perf map.
func (h *Host) Close() error { return h.perf.close() }

// Baseline is NewHost().Baseline() without a Host: whether the engine runs on
// this CPU.
func Baseline() error { return baselineOf(CPU()) }

// HostTier is the instruction-set tier this CPU runs (Host.Tier).
func HostTier() Tier { return tierOfHost(CPU()) }
