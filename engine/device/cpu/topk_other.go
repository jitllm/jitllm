//go:build !amd64 && !arm64

package cpu

// EmitMoETopK is owed on this architecture. topk.go has the AVX2 kernel and
// topk_a64.go the NEON one; a host that is neither cannot run a mixture at all,
// because engine/model/moe.go has no Go loop left to fall back to (RULE 8) and returns
// this error naming the router's shape instead.
func EmitMoETopK(n, k int, norm bool) ([]byte, error) { return nil, moeShapeErr(n, k) }

// EmitMoERoute is owed here for the same reason.
func EmitMoERoute(n, k int, g MoEGate) ([]byte, error) { return nil, moeShapeErr(n, k) }
