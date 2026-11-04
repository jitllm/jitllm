//go:build !amd64 && jitllmtest

package cpu

// ForceNoVNNIForTest mirrors the amd64 hook so a test in another package
// compiles on every architecture without a build tag of its own.
//
// It is a no-op rather than absent so names match across architectures (see
// packed_other.go). arm64 has no VNNI to pretend away (SDOT is baseline), so
// this reports that nothing was forced and a caller that must see the
// pre-VNNI sequence skips.
func ForceNoVNNIForTest(bool) bool { return false }
