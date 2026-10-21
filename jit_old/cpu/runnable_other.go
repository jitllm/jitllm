//go:build !amd64

package cpu

// runnable is the non-amd64 answer, and it is nil rather than the scan.
//
// The scan must not run here: features_stub.go returns a zero Features, so
// CPU().AVXVNNI is false, and the VEX byte pattern is ordinary data in A64
// code; a false positive would decline a good NEON kernel.
func runnable([]byte) error { return nil }
