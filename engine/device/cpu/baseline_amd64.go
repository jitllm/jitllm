package cpu

import (
	"errors"
	"fmt"
	"strings"
)

// errNoBaseline is the sentinel every below-floor answer wraps, and what the
// forceNoBaseline test instrument returns as it stands. It names the floor;
// baseline() adds exactly what this host or this tree lacks.
var errNoBaseline = errors.New(
	"jit: this CPU cannot run the engine (the amd64 floor is the SSE tier: " +
		"sse2, sse3, ssse3 and sse4.1, with every op generated for it)")

// baseline is the amd64 engine floor, one answer per tier.
//
// AVX2 tier: AVX2, FMA and F16C with the OS saving YMM state -- each
// load-bearing: vop hardcodes 256-bit forms, the norms and attention emit
// VFMADD231PS (FMA is a separate CPUID bit), and the packed d-plane is read
// with VCVTPH2PS (F16C). tierOf answers TierAVX2 only when all three are
// usable.
//
// SSE tier: the legacy-encoded 128-bit kernels (sse.go), for a host with SSE2,
// SSE3, SSSE3 and SSE4.1 and no usable AVX2. It passes when the SSE table has
// a kernel for every op (SSEPending is empty) and is refused naming each op
// that has none.
//
// Below that the host is refused naming the SSE-floor extensions it lacks.
func baseline() error {
	f := CPU()
	switch tierOf(f) {
	case TierAVX2:
		return nil
	case TierSSE:
		if p := SSEPending(); len(p) > 0 {
			return fmt.Errorf("%w: this host runs the SSE tier (no usable %s), and that tier "+
				"has no kernel yet for %s: %w", errNoBaseline, missingNames(f, ISATierAVX2),
				strings.Join(p, ", "), ErrNoSSEKernel)
		}
		return nil
	}
	return fmt.Errorf("%w: missing %s", errNoBaseline, missingNames(f, ISATierSSE))
}

// mapGate is Map's question on amd64: can this host run this kernel.
//
// forceNoBaseline refuses everything. Otherwise the declared (or implied) ISA
// is compared with the host's, and only an undeclared kernel gets the VNNI
// byte scan: in a legacy-encoded kernel the C4 ... 50 pattern can occur inside
// a correct instruction (PINSRW is 66 0F C4 /r ib), and a false positive there
// would remove the only kernel an SSE host has.
func mapGate(code []byte) error {
	if forceNoBaseline {
		return errNoBaseline
	}
	if err := isaCheck(code); err != nil {
		return err
	}
	if _, declared := KernelISA(code); !declared {
		return runnable(code)
	}
	return nil
}

const baselineForceable = true
