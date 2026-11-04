//go:build !darwin

package sched

// QoS lowers nothing off macOS, where a pool's cores are its process mask's
// (RULE 5).
type QoS struct{}

// LoadQoS is the QoS off macOS: nothing to load.
func LoadQoS(uint32) QoS { return QoS{} }

// Ready is false off macOS.
func (QoS) Ready() bool { return false }

// EClass is 0 off macOS.
func (QoS) EClass() uint32 { return 0 }

// Self is 0 off macOS.
func (QoS) Self() uint32 { return 0 }

// EQoS sets nothing off macOS and reports false.
func (QoS) EQoS() bool { return false }

// Lowered is nothing off macOS.
type Lowered struct{}

// LowerToE moves no thread off macOS.
func (QoS) LowerToE() Lowered { return Lowered{} }

// Raise has nothing to undo off macOS.
func (Lowered) Raise() {}

// CanSplitE is false off macOS: nothing here asks for a core class.
func (QoS) CanSplitE() bool { return false }

// allE is false off macOS: no pool asks for a core class here.
func allE([]int) bool { return false }
