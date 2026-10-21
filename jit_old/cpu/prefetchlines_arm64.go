//go:build arm64

package cpu

// EmitPrefetchLines is the primary tier's prefetch-lines kernel (prefetchlines.go).
func EmitPrefetchLines() []byte { return EmitA64PrefetchLines() }
