//go:build !amd64 && !arm64

package cpu

// EmitResample on a host with no emitter: it refuses, and nn then has no
// resampling kernel for this architecture.
func EmitResample(int) []byte { return nil }
