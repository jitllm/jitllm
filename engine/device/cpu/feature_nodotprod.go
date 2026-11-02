//go:build !arm64

package cpu

// The filename must not end in _arm: Go would read that as a GOARCH constraint
// for 32-bit ARM and drop the file on amd64. The build tag is the constraint.
//
// hasDotProd is arm64's probe. EmitA64 compiles everywhere (so the NEON encoder
// can be tested from an x86 box), so the symbol must exist here too; it never
// decides anything off arm64.
func hasDotProd() bool { return false }

// a64EmulateDot is whether the A64 emitters widen SDOT (sdotemu.go). Off arm64
// nothing executes A64 code, so only the test instrument decides: a
// disassembly test on an x86 box can then read the emulated sequence.
func a64EmulateDot() bool { return !forceDotEncoding && forceNoDotProd }

// HasDotProd is false off arm64: no chip here runs A64 code.
func HasDotProd() bool { return false }
