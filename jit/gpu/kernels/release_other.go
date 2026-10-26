//go:build !linux

package kernels

// ReleaseOffHeap clears b, a span of a MapOffHeap mapping: what it reads when
// next touched is zeros, as on Linux, where the memory goes back to the OS.
func ReleaseOffHeap(b []byte) { clear(b) }
