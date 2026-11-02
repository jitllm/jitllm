package cpu

import "errors"

var errNoBaseline = errors.New("jit: this CPU is below the arm64 baseline (ARMv8-A NEON)")

// baseline is nil on arm64: ARMv8-A mandates NEON, so the floor is the
// architecture itself. FEAT_DotProd is deliberately not asked here: it is a
// per-kernel question (SupportedA64Packed, hasDotProd), and folding it in
// would decline the elementwise kernels, which need only NEON.
func baseline() error { return nil }

const baselineForceable = true

// mapGate is Map's question on arm64. There is no x86 ISA to check: the
// floor is the architecture itself, FEAT_DotProd is a per-kernel question
// SupportedNative owns, and a declaration (isa.go) is an amd64 artefact that
// never appears in A64 code.
func mapGate(code []byte) error {
	if err := Baseline(); err != nil {
		return err
	}
	return runnable(code)
}
