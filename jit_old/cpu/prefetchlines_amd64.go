//go:build amd64

package cpu

// EmitPrefetchLines is the primary tier's prefetch-lines kernel (prefetchlines.go).
func EmitPrefetchLines() []byte { return emitPrefetchLinesX86(false) }
