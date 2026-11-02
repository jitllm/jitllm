//go:build !amd64 && !arm64

package cpu

// The narrow-window quantizer's constants, so a caller sizes its scratch the
// same on every architecture.
const (
	QuantActNarrowBlocks  = 8
	QuantActNarrowScratch = 24
)

// EmitQuantActNarrow is owed here. quantact_narrow.go has the amd64 kernel and
// says what it is for: eight one-block windows sharing one pair of divisions,
// because at a 32-element window those divisions are most of the block.
func EmitQuantActNarrow(half bool) []byte { return nil }
