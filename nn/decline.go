package nn

import "fmt"

// Decline is why a device does not run what the composer forms for a part
// of a step -- a block, a head, a table, a sample, the input rows -- the one
// set of reasons every device's census counts under, a card's and the
// CPU's alike. DeclineServes is a part served.
type Decline uint8

const (
	DeclineServes Decline = iota
	DeclineNoGraph
	DeclineCaller
	DeclineNoRoom
	DeclineNoWeight
	DeclineNotFormed
	DeclineISA
	DeclineUnplaced
	NumDeclines
)

func (d Decline) String() string {
	switch d {
	case DeclineServes:
		return "served"
	case DeclineNoGraph:
		return "no graph from the front end"
	case DeclineCaller:
		return "the caller keeps it on the tier's kernels"
	case DeclineNoRoom:
		return "its buffers do not fit the device's budget"
	case DeclineNoWeight:
		return "a weight or vector the block does not hold"
	case DeclineNotFormed:
		return "the composer did not form the block"
	case DeclineUnplaced:
		return "a block not composed for the device at its placement"
	case DeclineISA:
		return "a CPU the composer's CPU lowering does not emit for (x86-64 below SSE4.1 and SSSE3)"
	}
	return fmt.Sprintf("decline %d", d)
}

// DeclineName is the reason a decline count is kept under, by its index in
// a census array.
func DeclineName(i int) string {
	return Decline(i).String()
}
