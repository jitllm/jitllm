//go:build amd64

package cpu

// EmitRopeTable generates the rotary table: the {cos, sin} pair every rotary
// pair of one position needs. This is the AVX2 tier; sse_ropetab.go and
// ropetab_a64.go are the other two, and ropetab_const.go holds the constant
// block, the position decomposition and the exactness argument all three
// depend on.
//
//	Out     cs, the table: 2*npairs float32, {cos, sin} per pair, written
//	AScale  the per-model plane block (ropetab_const.go's layout)
//	Scr     RopeTabConsts()
//	K       the position
//
// npairs is baked from the model's NRot. One range reduction feeds both
// polynomials on |r| <= pi/4 (no table, branch or call), so sin and cos agree
// on the quadrant: with n the nearest quarter-turn, bit 0 of n swaps the two
// polynomials, bit 1 of n negates sin and bit 1 of n+1 negates cos. The table
// is within one ulp of the float64 one, a floor rather than a defect.
//
// Registers, fixed for the whole kernel:
//
//	RCX  cs        RDX  the plane cursor   RBX  Scr        RAX  the vector count
//	Y0..Y3   the four position digits, as floats
//	Y4  the mod-4 magic   Y5  mscale   Y6  pi/2 hi   Y7  pi/2 lo
//	Y8..Y15  the body's working set
//
// The quadrant derivation, and what was tried against the one-ulp floor:
// docs/engineering-history/cpu-kernels.md, "jit_old/cpu/ropetab.go: EmitRopeTable".
func EmitRopeTable(npairs int) ([]byte, error) {
	const lanes = 8
	if npairs <= 0 {
		return nil, ropeTabShapeErr(npairs)
	}
	stride := int32(4 * RopeTabStride(npairs))
	nv, tail := npairs/lanes, npairs%lanes

	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out    -> cs
	a.MOVLoad(RDX, At(RDI, 24)) // AScale -> the planes
	a.MOVLoad(RBX, At(RDI, 56)) // Scr    -> RopeTabConsts()

	// The position's four base-128 digits, taken apart in the vector unit so
	// nothing crosses to a general register and back. Args.K is an int64 and
	// every position fits its low half.
	a.VPBROADCASTD(Y8, At(RDI, 40))
	a.VPBROADCASTD(Y9, At(RBX, rtDigitOff))
	for d := 0; d < RopeTabDigits; d++ {
		if d > 0 {
			a.VPSRLD(Y8, Y8, RopeTabDigitBits)
		}
		a.VPAND(Y0+Reg(d), Y8, Y9)
		a.VCVTDQ2PS(Y0+Reg(d), Y0+Reg(d))
	}
	a.VBROADCASTSS(Y4, At(RBX, rtMagicOff))
	a.VBROADCASTSS(Y6, At(RBX, rtPio2HiOff))
	a.VBROADCASTSS(Y7, At(RBX, rtPio2LoOff))
	a.VBROADCASTSS(Y5, At(RDX, int32(4*RopeTabScaleOff(npairs))))

	body := func(n int) {
		// s = sum of the four digit-by-head products. Every one of them is
		// EXACT and so is this sum (ropetab_const.go); s is at most 2032.
		a.VMULPSMem(Y8, Y0, At(RDX, 0))
		for d := 1; d < RopeTabDigits; d++ {
			a.VMULPSMem(Y9, Y0+Reg(d), At(RDX, int32(d)*stride))
			a.VADDPS(Y8, Y8, Y9)
		}
		// s mod 4, exactly: s - 4*round(s/4), where the rounding is the magic
		// add at the binade whose ulp is four. The remainder is in [-2,2].
		a.VADDPS(Y9, Y8, Y4)
		a.VSUBPS(Y9, Y9, Y4)
		a.VSUBPS(Y8, Y8, Y9)

		// c = the residual planes' contribution, which is what the head
		// planes' quantization left behind. |c| <= 0.0625.
		a.VMULPSMem(Y10, Y0, At(RDX, int32(RopeTabDigits)*stride))
		for d := 1; d < RopeTabDigits; d++ {
			a.VMULPSMem(Y9, Y0+Reg(d), At(RDX, int32(RopeTabDigits+d)*stride))
			a.VADDPS(Y10, Y10, Y9)
		}

		// The quadrant is chosen from s+c and the remainder taken from s alone,
		// which keeps |w| under a half: f = s-n can exceed a half by |c|, and
		// w = s+c-n cannot. Folding c in before the rounding would round a value
		// of magnitude two and cost three bits of the angle.
		a.VADDPS(Y9, Y8, Y10)
		a.VCVTPS2DQ(Y12, Y9) // n, the nearest quarter-turn
		a.VCVTDQ2PS(Y9, Y12)
		a.VSUBPS(Y8, Y8, Y9) // f = s - n, EXACT, |f| <= 1/2 + |c|

		// r = (f + c)*pi/2 with the large term added last: c*pi/2 + f*pi/2lo is
		// a small quantity rounded at its own scale, and the dominant f*pi/2hi is
		// an exact product inside the last fused add, so the result carries one
		// rounding rather than three. Same instruction count, half the error.
		a.VMULPS(Y9, Y10, Y6)     // c * pi/2 hi
		a.VFMADD231PS(Y9, Y8, Y7) // += f * pi/2 lo
		a.VFMADD231PS(Y9, Y8, Y6) // += f * pi/2 hi, the one rounding that counts
		a.VMULPS(Y10, Y9, Y9)     // z = r*r

		// sin(r) = r + (r*z)*(c3 + z*(c5 + z*c7))
		a.VBROADCASTSS(Y11, At(RBX, rtSinC7Off))
		a.VBROADCASTSS(Y13, At(RBX, rtSinC5Off))
		a.VFMADD213PS(Y11, Y10, Y13)
		a.VBROADCASTSS(Y13, At(RBX, rtSinC3Off))
		a.VFMADD213PS(Y11, Y10, Y13)
		a.VMULPS(Y13, Y9, Y10)
		a.VFMADD213PS(Y11, Y13, Y9)

		// cos(r) = 1 + z*(-1/2 + z*(c4 + z*(c6 + z*c8)))
		a.VBROADCASTSS(Y14, At(RBX, rtCosC8Off))
		for _, off := range []int32{rtCosC6Off, rtCosC4Off, rtNegHalfOff, rtOneOff} {
			a.VBROADCASTSS(Y13, At(RBX, off))
			a.VFMADD213PS(Y14, Y10, Y13)
		}

		// Bit 0 of n swaps the two polynomials; the XOR of the pair, masked,
		// does both halves of that swap with one difference.
		a.VPSLLD(Y15, Y12, 31)
		a.VPSRAD(Y15, Y15, 31)
		a.VPXOR(Y13, Y11, Y14)
		a.VPAND(Y13, Y13, Y15)
		a.VPXOR(Y11, Y11, Y13)
		a.VPXOR(Y14, Y14, Y13)

		// Bit 1 of n negates sin; bit 1 of n+1 negates cos.
		a.VPSLLD(Y15, Y12, 30)
		a.VPANDMem(Y15, Y15, At(RBX, rtSignVecOff))
		a.VPXOR(Y11, Y11, Y15)
		a.VPADDDMem(Y12, Y12, At(RBX, rtOneVecOff))
		a.VPSLLD(Y15, Y12, 30)
		a.VPANDMem(Y15, Y15, At(RBX, rtSignVecOff))
		a.VPXOR(Y14, Y14, Y15)

		a.VMULPS(Y11, Y11, Y5)
		a.VMULPS(Y14, Y14, Y5)

		// Interleave to {cos, sin} per pair and store n of the eight pairs.
		// VSHUFPS gathers the two halves of each 128-bit lane and VPERMILPS
		// puts them in pair order; the two lanes are then four 16-byte stores,
		// because the pairs of one ymm land at four different places in the
		// output and no single 256-bit store spells that.
		a.VSHUFPS(Y13, Y14, Y11, 0x44)
		a.VPERMILPS(Y13, Y13, 0xD8)
		a.VSHUFPS(Y15, Y14, Y11, 0xEE)
		a.VPERMILPS(Y15, Y15, 0xD8)
		// chunk j holds pairs 2j and 2j+1.
		chunk := func(j int) Reg {
			switch j {
			case 0:
				return Y13
			case 1:
				return Y15
			case 2:
				a.VEXTRACTF128(Y9, Y13, 1)
			default:
				a.VEXTRACTF128(Y9, Y15, 1)
			}
			return Y9
		}
		for j := 0; j < n/2; j++ {
			a.VMOVDQUStorex(At(RCX, int32(16*j)), chunk(j))
		}
		// An odd pair count ends in two scalar stores, never a wider one: the
		// engine's rotary tables sit in one contiguous batch buffer, so a 16-byte
		// store of the last chunk would write into the next position's table.
		// The pair always lands in lanes 0 and 1 of its chunk.
		if n%2 != 0 {
			c := chunk(n / 2)
			a.VMOVSSStore(At(RCX, int32(16*(n/2))), c)
			a.VPERMILPS(Y9, c, 0x01)
			a.VMOVSSStore(At(RCX, int32(16*(n/2)+4)), Y9)
		}
	}

	if nv > 0 {
		a.MOVimm32(RAX, int32(nv))
		lp := a.Label()
		a.Bind(lp)
		body(lanes)
		a.ADDimm(RDX, 4*lanes)
		a.ADDimm(RCX, 8*lanes)
		a.DEC(RAX)
		a.JNZ(lp)
	}

	// The ragged tail runs the whole vector body and stores part of it, with no
	// masked store. The plane block is padded to eight lanes and zeroed past
	// the last pair (ropetab_const.go), so idle lanes compute angle zero. That
	// also serves a pair count smaller than the vector (stories260K's four).
	if tail != 0 {
		body(tail)
	}
	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}
