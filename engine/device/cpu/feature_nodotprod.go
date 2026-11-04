//go:build !arm64

package cpu

// The filename must not end in _arm: Go would read that as a GOARCH constraint
// for 32-bit ARM and drop the file on amd64. The build tag is the constraint.
//
// HasDotProd is false off arm64: no chip here runs A64 code.
func HasDotProd() bool { return false }
