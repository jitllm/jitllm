package session

// Placement is one block of the model: which device runs it, and whether its
// page is in host memory. One byte per block is the whole state, which is why
// the signal is a []byte.
type Placement byte

const (
	// PlaceCPU is a block the CPU device runs.
	PlaceCPU Placement = 0
	// PlaceDevice is a block device 0 runs; device k is OnDevice(k).
	PlaceDevice Placement = 1
	// PlaceResident is set when the block's page is in host memory. A host
	// block without it is read from the file on its next pass.
	PlaceResident Placement = 0x80
)

// OnDevice is the placement of a block device k runs.
func OnDevice(k int) Placement { return PlaceDevice + Placement(min(max(k, 0), 126)) }

// Device is the ordinal of the device running the block, or -1 for the host.
func (p Placement) Device() int { return int(p&^PlaceResident) - 1 }

// Resident reports whether the block's page is in host memory.
func (p Placement) Resident() bool { return p&PlaceResident != 0 }
