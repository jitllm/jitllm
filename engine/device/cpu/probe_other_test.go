//go:build !amd64

package cpu

// reprobeHostFeatures has nothing to re-probe off amd64: Features is the x86
// capability set and CPU() is a constant there. arm64's own probe
// (hasA64DotProd) has its own Once and forceNoDotProd is read inside it.
func reprobeHostFeatures() {}
