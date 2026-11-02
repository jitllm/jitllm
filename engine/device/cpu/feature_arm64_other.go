//go:build arm64 && !linux && !darwin && !windows

package cpu

// probeDotProd has no way to ask on this OS, so it answers "absent" -- the safe
// direction, since the alternative is a SIGILL on the first token.
func probeDotProd() bool { return false }
