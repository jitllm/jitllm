package ir

type ISA struct {
	Family  Family
	Version uint16
}

type Family uint16

const (
	FamilyNone Family = iota
	FamilyPTX
	FamilyGFX
	FamilySPIRV
	FamilyMSL
)

func PTX(sm int) ISA { return ISA{FamilyPTX, uint16(sm)} }

func GFX(n int) ISA { return ISA{FamilyGFX, uint16(n)} }

func SPIRV() ISA { return ISA{Family: FamilySPIRV} }
func MSL() ISA   { return ISA{Family: FamilyMSL} }

func (a ISA) Word() uint32 { return uint32(a.Family)<<16 | uint32(a.Version) }
