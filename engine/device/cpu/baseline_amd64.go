package cpu

import "fmt"

// errNoBaseline is the sentinel every below-floor answer wraps. It names the
// floor; baselineOf adds exactly what this host or this tree lacks.
const errNoBaseline jitError = "jit: this CPU cannot run the engine (the amd64 floor is the SSE tier: " +
	"sse2, sse3, ssse3 and sse4.1, with every op generated for it)"

// baselineOf is the amd64 engine floor for f, one answer per tier.
//
// AVX2 tier: AVX2, FMA and F16C with the OS saving YMM state -- each
// load-bearing: vop hardcodes 256-bit forms, the norms and attention emit
// VFMADD231PS (FMA is a separate CPUID bit), and the packed d-plane is read
// with VCVTPH2PS (F16C). tierOf answers TierAVX2 only when all three are
// usable.
//
// SSE tier: the legacy-encoded 128-bit code the composer lowers to for a
// host with SSE2, SSE3, SSSE3 and SSE4.1 and no usable AVX2.
//
// Below that the host is refused naming the SSE-floor extensions it lacks.
func baselineOf(f Features) error {
	switch tierOf(f) {
	case TierAVX2, TierSSE:
		return nil
	}
	return fmt.Errorf("%w: missing %s", errNoBaseline, missingNames(f, ISATierSSE))
}

// gate is Map's question on amd64: can the Target run this kernel.
//
// The declared (or implied) ISA
// is compared with the host's, and only an undeclared kernel gets the VNNI
// byte scan: in a legacy-encoded kernel the C4 ... 50 pattern can occur inside
// a correct instruction (PINSRW is 66 0F C4 /r ib), and a false positive there
// would remove the only kernel an SSE host has.
func (tg *Target) gate(code []byte) error {
	if err := isaCheck(tg.f, code); err != nil {
		return err
	}
	if _, declared := KernelISA(code); !declared {
		return runnable(tg.f, code)
	}
	return nil
}
