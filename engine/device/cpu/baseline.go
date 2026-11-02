package cpu

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
)

// Baseline reports whether this host can run the engine: whether every kernel
// a model needs exists for this host's tier.
//
// It is not Map's gate (Map checks each kernel's declared ISA, isa.go); it
// answers whether this host's tier is complete. On amd64 an AVX2 host always
// is; an SSE host is when the SSE table has a kernel for every op
// (SSEPending); anything below SSE4.1+SSSE3 never is. model.Open refuses
// through it (nn.HostRefusal), and the error names which extensions or which
// SSE-tier ops are missing, because the caller prints it.
//
// It exists because a pre-AVX2 host would SIGILL on the first token: the
// quantized path declined correctly, but the elementwise path asked a width
// question where a capability question belonged. See
// docs/engineering-history/cpu-kernels.md, "THE SSE TIER".
//
// It is a probe, not a knob: there is no way to disable the JIT. The only
// writer of the force flag below is a _test.go file in this package.
func Baseline() error {
	baselineOnce.Do(func() { baselineErr = baseline() })
	if forceNoBaseline {
		return errNoBaseline
	}
	return baselineErr
}

var (
	baselineOnce sync.Once
	baselineErr  error
)

// forceNoBaseline makes Baseline report absent. Assigned only from a _test.go
// file in this package, the same instrument as forceNoVNNI beside it: a shipped
// binary has no way to set it, so the probe is the only answer there.
var forceNoBaseline bool

// BaselineForceable reports whether this build can force the baseline absent,
// so a test needing both tiers skips rather than comparing the fast tier with
// itself (mirrors nn.Disableable).
func BaselineForceable() bool { return baselineForceable }

// Map makes code executable, refusing first if this host cannot run this code.
//
// The gate is per kernel and reads the kernel's own declaration: an SSE-tier
// kernel carries its ISA in a leading NOP (isa.go) and an undeclared kernel
// needs AVX2|FMA|F16C. Every executable byte this package produces passes
// through here, which is why the check is not at the emitters: a condition
// consulted in many places gets consulted in most of them.
//
// A refusal is a decline the caller already handles: nn's mustMap panics with
// the kernel's name, and the packed and float builders in nn.NewJIT treat it
// as "no kernel". mapExec is the ungated form, used only by the CPUID and
// XGETBV probes, which must run before the gate has an answer.
func Map(code []byte) (*Code, error) {
	if err := mapGate(code); err != nil {
		return nil, err
	}
	c, err := mapExec(code)
	if err == nil {
		c.tier = KernelTier(code)
		countMapped(code)
	}
	return c, err
}

// MapNamed is Map plus a perf-map entry, with the same gate.
func MapNamed(code []byte, name string) (*Code, error) {
	if err := mapGate(code); err != nil {
		return nil, err
	}
	c, err := mapExecNamed(code, name)
	if err == nil {
		c.tier = KernelTier(code)
		countMapped(code)
	}
	return c, err
}

// ErrNoVNNI is Map refusing a kernel that contains VPDPBUSD on a host without
// AVX-VNNI. It is a decline: the builders in nn skip the kernel (the packed
// family has a pre-VNNI sequence to use instead) and a mandatory kernel's
// mustMap names it.
var ErrNoVNNI = errors.New("cpu: this kernel contains VPDPBUSD and this host has no AVX-VNNI")

// ErrISA is what every ISAError matches under errors.Is: Map refusing a kernel
// that needs extensions this host does not have.
var ErrISA = errors.New("cpu: this kernel needs instructions this host does not have")

// ISAError is Map's refusal of a kernel whose ISA this host does not cover. It
// names the missing extensions, so the message reads as a property of the host
// rather than a bug in the engine.
type ISAError struct {
	Need     ISA  // what the kernel requires
	Missing  ISA  // Need minus what this host has
	Declared bool // whether Need came from the kernel's own declaration
	detail   string
}

func (e *ISAError) Error() string {
	where := "an undeclared (AVX2-tier: VEX-encoded, FMA, F16C) kernel"
	if e.Declared {
		where = "a kernel declared " + e.Need.String()
	}
	return fmt.Sprintf("cpu: %s cannot run here: this host lacks %s", where, e.detail)
}

// Is makes errors.Is(err, ErrISA) true for every ISAError.
func (e *ISAError) Is(target error) bool { return target == ErrISA }

// isaCheck refuses code whose ISA the host does not cover. It is the amd64
// half of mapGate, factored out so the refusal text is built in one place.
func isaCheck(code []byte) error {
	need, declared := KernelISA(code)
	f := CPU()
	if miss := need &^ f.ISA(); miss != 0 {
		return &ISAError{Need: need, Missing: miss, Declared: declared, detail: missingNames(f, need)}
	}
	return nil
}

// mapped counts successful Maps by the tier the kernel was generated for. It
// is a selection check: it lets a gate assert that a forced-SSE run mapped
// zero AVX2 kernels (a cache keyed without the tier would otherwise pass
// silently).
var mapped [numTiers]atomic.Int64

func countMapped(code []byte) {
	t := TierNEON
	if HostTier() != TierNEON {
		t = KernelTier(code)
	}
	mapped[t].Add(1)
}

// MappedByTier snapshots how many kernels Map and MapNamed have mapped in this
// process, keyed by the tier each kernel was generated for. On amd64 that is
// read from the kernel's own declaration; on arm64 every kernel counts as
// TierNEON.
func MappedByTier() map[Tier]int64 {
	out := make(map[Tier]int64, numTiers)
	for t := range mapped {
		if n := mapped[t].Load(); n != 0 {
			out[Tier(t)] = n
		}
	}
	return out
}
