//go:build arm64

package cpu

// EmitResample is the architecture-neutral name nn calls; on arm64 it is the
// NEON twin.
func EmitResample(prec int) []byte { return EmitA64Resample(prec) }
