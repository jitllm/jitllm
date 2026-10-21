package ir

// ISA names a device's instruction set by number: its family and the
// family's version (a CUDA compute capability as 70 for sm_70, an AMD gfx
// target as 942 for gfx942). Passes ask it by family and version, never by
// name; four bytes, copied as one word.
type ISA struct {
	Family  Family
	Version uint16
}

// Family is an instruction set's family.
type Family uint16

// The families.
const (
	FamilyNone Family = iota
	FamilyPTX
	FamilyGFX
	FamilySPIRV
	FamilyMSL
)

// PTX is CUDA compute capability sm (70 for sm_70).
func PTX(sm int) ISA { return ISA{FamilyPTX, uint16(sm)} }

// GFX is the AMD target gfx<n> (942 for gfx942).
func GFX(n int) ISA { return ISA{FamilyGFX, uint16(n)} }

// SPIRV and MSL are the ISAs with no version a pass distinguishes.
func SPIRV() ISA { return ISA{Family: FamilySPIRV} }
func MSL() ISA   { return ISA{Family: FamilyMSL} }

// Word is the ISA as one integer, for a cache key.
func (a ISA) Word() uint32 { return uint32(a.Family)<<16 | uint32(a.Version) }
