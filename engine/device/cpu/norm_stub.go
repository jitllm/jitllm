//go:build !amd64 && !arm64

package cpu

// EmitRMSNorm on a host with no emitter: it refuses, and nn then has no RMSNorm
// kernel for this architecture.
func EmitRMSNorm(int) []byte { return nil }
