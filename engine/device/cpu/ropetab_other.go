//go:build !amd64 && !arm64

package cpu

// EmitRopeTable is owed on this architecture. ropetab.go has the AVX2 kernel,
// ropetab_a64.go the NEON one and sse_ropetab.go the legacy-SSE one; a host
// that is none of those cannot build a rotary table at all, because
// nn.Rope.Table has no Go loop left to fall back to (RULE 8).
func EmitRopeTable(npairs int) ([]byte, error) { return nil, ropeTabShapeErr(npairs, 0) }
