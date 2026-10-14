//go:build !darwin

package sched

// EQoS sets nothing off macOS, where a pool's cores are its process mask's
// (RULE 5), and reports false.
func EQoS() bool { return false }

// Lowered is nothing off macOS.
type Lowered struct{}

// LowerToE moves no thread off macOS.
func LowerToE() Lowered { return Lowered{} }

// Raise has nothing to undo off macOS.
func (Lowered) Raise() {}

// CanSplitE is false off macOS: nothing here asks for a core class.
func CanSplitE() bool { return false }

// SetEClass has nothing to set off macOS.
func SetEClass(uint32) uint32 { return 0 }

// EClass is 0 off macOS.
func EClass() uint32 { return 0 }

// QoSSelf is 0 off macOS.
func QoSSelf() uint32 { return 0 }

// allE is false off macOS: no pool asks for a core class here.
func allE([]int) bool { return false }
