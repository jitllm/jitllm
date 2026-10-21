package cpu

// The constant block every resampling kernel reads (EmitResample,
// EmitResampleH), one copy for all three tiers: the layout is the ABI between
// the emitters and nn.
//
//	0   bias, 1<<(prec-1)                                   (int32)
//	4   255, the clamp                                      (int32)
//	32  the AVX2 vertical pass's byte gather: in each 128-bit half, the low
//	    byte of its four int32 lanes to bytes 0..3, the rest zeroed (32 bytes)
const (
	resampleBiasOff   = 0
	resampleMaxOff    = 4
	resampleGatherOff = 32
	resampleConstLen  = 64
)

// ResampleConsts is the block for precision prec, as 32-bit words.
func ResampleConsts(prec int) []int32 {
	c := make([]int32, resampleConstLen/4)
	c[resampleBiasOff/4] = 1 << (prec - 1)
	c[resampleMaxOff/4] = 255
	// 0x80 in a PSHUFB index zeroes the byte.
	g := [16]byte{0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80}
	for half := 0; half < 2; half++ {
		for w := 0; w < 4; w++ {
			c[(resampleGatherOff+16*half)/4+w] = int32(uint32(g[4*w]) | uint32(g[4*w+1])<<8 |
				uint32(g[4*w+2])<<16 | uint32(g[4*w+3])<<24)
		}
	}
	return c
}

// byteLoops is elemLoops for an x86 kernel over bytes: vec runs over K units
// of unit samples and one over R single samples, the cursors advancing by
// unit and by one byte.
func byteLoops(a *Buf, k, r Reg, cursors []Reg, unit int32, vec, one func()) {
	tail, done := a.Label(), a.Label()
	a.TESTQ(k, k)
	a.JZ(tail)
	lp := a.Label()
	a.Bind(lp)
	vec()
	for _, c := range cursors {
		a.ADDimm(c, unit)
	}
	a.DEC(k)
	a.JNZ(lp)

	a.Bind(tail)
	a.TESTQ(r, r)
	a.JZ(done)
	lt := a.Label()
	a.Bind(lt)
	one()
	for _, c := range cursors {
		a.ADDimm(c, 1)
	}
	a.DEC(r)
	a.JNZ(lt)
	a.Bind(done)
}

// a64ByteLoops is byteLoops' NEON twin.
func a64ByteLoops(a *A64, k, r XReg, cursors []XReg, unit int32, vec, one func()) {
	tail, done := a.Label(), a.Label()
	a.CBZ(k, tail)
	lp := a.Label()
	a.Bind(lp)
	vec()
	for _, c := range cursors {
		a.ADDimm(c, c, unit)
	}
	a.SUBimm(k, k, 1)
	a.CBNZ(k, lp)

	a.Bind(tail)
	a.CBZ(r, done)
	lt := a.Label()
	a.Bind(lt)
	one()
	for _, c := range cursors {
		a.ADDimm(c, c, 1)
	}
	a.SUBimm(r, r, 1)
	a.CBNZ(r, lt)
	a.Bind(done)
}
