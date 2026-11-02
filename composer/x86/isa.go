package x86

var isaMarker = [5]byte{0x0F, 0x1F, 0x80, 'J', 'I'}

const ISAMarkerLen = len(isaMarker) + 2

func (a *Buf) DeclareISA(mask ISA) {
	if a.declared {
		a.fail("x86: DeclareISA called twice")
	}
	if a.Len() != 0 {
		a.fail("x86: DeclareISA after %d bytes; the declaration must "+
			"be the kernel's first instruction so Map finds it at code[0]", a.Len())
	}
	if mask&ISASSE2 == 0 {
		a.fail("x86: DeclareISA(%s) without sse2; SSE2 is the x86-64 "+
			"floor every legacy vector instruction here is checked against", mask)
	}
	a.declared, a.isa = true, mask
	for _, b := range isaMarker {
		a.u8(b)
	}
	a.u8(byte(mask))
	a.u8(byte(mask >> 8))
}

func KernelISA(code []byte) (need ISA, declared bool) {
	if len(code) >= ISAMarkerLen && [5]byte(code[:5]) == isaMarker {
		return ISA(code[5]) | ISA(code[6])<<8, true
	}
	return ISATierAVX2, false
}

func (a *Buf) useISA(bit ISA, mnemonic string) {
	if a.declared && a.isa&bit == 0 {
		a.fail("x86: %s needs %s, and this kernel declared only %s; "+
			"declare it (DeclareISA) or do without it", mnemonic, bit, a.isa)
	}
}

func (a *Buf) noVEXInSSE(what string) {
	if a.declared && a.isa&ISAAVX2 == 0 {
		a.fail("x86: %s in a kernel declared %s: a VEX prefix faults "+
			"on a host without AVX (a Goldmont Atom SIGILLs on it whatever the "+
			"vector length)", what, a.isa)
	}
}
