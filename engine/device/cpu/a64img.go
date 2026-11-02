package cpu

import "fmt"

// The NEON instructions the image path's kernels (imgproc_a64.go) added:
// float64 lanes for the resampler's sin/cos table, the narrowing chain and
// the byte and halfword stores that write exactly a pixel's samples, and the
// loads a table lookup and a three-byte pixel need. Every base word came from
// clang-18 --target=aarch64-linux-gnu and is pinned in a64_test.go.

// FADD2d, FSUB2d, FMUL2d and FDIV2d are the float64 forms of the .4s
// arithmetic: two lanes, one rounding each.
func (a *A64) FADD2d(vd, vn, vm VReg) { a.w(0x4E60D400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) FSUB2d(vd, vn, vm VReg) { a.w(0x4EE0D400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) FMUL2d(vd, vn, vm VReg) { a.w(0x6E60DC00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) FDIV2d(vd, vn, vm VReg) { a.w(0x6E60FC00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd)) }

// FRINTN2d rounds two float64 lanes to the nearest integer, ties to even;
// FRINTM2d rounds them down. FRINTM4s is the float32 floor.
func (a *A64) FRINTN2d(vd, vn VReg) { a.w(0x4E618800 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) FRINTM2d(vd, vn VReg) { a.w(0x4E619800 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) FRINTM4s(vd, vn VReg) { a.w(0x4E219800 | uint32(vn)<<5 | uint32(vd)) }

// FCVTL2d widens the low two float32 lanes of vn to two float64; FCVTN2s
// narrows two float64 into the low two float32 lanes, rounding to nearest,
// and zeroes the upper half.
func (a *A64) FCVTL2d(vd, vn VReg) { a.w(0x0E617800 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) FCVTN2s(vd, vn VReg) { a.w(0x0E616800 | uint32(vn)<<5 | uint32(vd)) }

// SHL2d is shl vd.2d, vn.2d, #shift: a float64's exponent field built from an
// integer.
func (a *A64) SHL2d(vd, vn VReg, shift uint8) {
	if shift > 63 {
		panic(fmt.Sprintf("jit: SHL2d shift %d, want 0..63", shift))
	}
	a.w(0x4F005400 | uint32(64+shift)<<16 | uint32(vn)<<5 | uint32(vd))
}

// DUPd2 is dup vd.2d, vn.d[0]: a float64 loaded into lane 0, broadcast.
func (a *A64) DUPd2(vd, vn VReg) { a.w(0x4E080400 | uint32(vn)<<5 | uint32(vd)) }

// DUPs4lane is dup vd.4s, vn.s[lane]: one lane broadcast (DUPs4 is lane 0).
func (a *A64) DUPs4lane(vd, vn VReg, lane uint8) {
	if lane > 3 {
		panic(fmt.Sprintf("jit: DUPs4lane lane %d, want 0..3", lane))
	}
	a.w(0x4E000400 | (uint32(lane)<<3|4)<<16 | uint32(vn)<<5 | uint32(vd))
}

// DUPgp is dup vd.4s, wn: a general register's low word in every lane.
func (a *A64) DUPgp(vd VReg, rn XReg) { a.w(0x4E040C00 | uint32(rn)<<5 | uint32(vd)) }

// REV16_8b swaps the two bytes of each halfword: big-endian 16-bit samples
// to the host's order.
func (a *A64) REV16_8b(vd, vn VReg) { a.w(0x0E201800 | uint32(vn)<<5 | uint32(vd)) }

// SUBreg is sub xd, xn, xm.
func (a *A64) SUBreg(rd, rn, rm XReg) { a.w(0xCB000000 | uint32(rm)<<16 | uint32(rn)<<5 | uint32(rd)) }

// SMAX4s is the signed 32-bit lane maximum, SMIN4s's mirror.
func (a *A64) SMAX4s(vd, vn, vm VReg) { a.w(0x4EA06400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd)) }

// XTN4h narrows four 32-bit lanes to 16 bits and XTN8b eight 16-bit lanes to
// 8, keeping the low half of each: the clamped samples' bytes.
func (a *A64) XTN4h(vd, vn VReg) { a.w(0x0E612800 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) XTN8b(vd, vn VReg) { a.w(0x0E212800 | uint32(vn)<<5 | uint32(vd)) }

// STRb and STRh store lane 0's low byte or halfword: one sample, or two.
func (a *A64) STRb(vt VReg, rn XReg, off int32) {
	a.w(0x3D000000 | scaled(off, 1, "str b") | uint32(rn)<<5 | uint32(vt))
}
func (a *A64) STRh(vt VReg, rn XReg, off int32) {
	a.w(0x7D000000 | scaled(off, 2, "str h") | uint32(rn)<<5 | uint32(vt))
}

// LDRBw is ldrb wt, [xn, #off]: one byte into a general register, zero
// extended through the x view -- an index a table is read at.
func (a *A64) LDRBw(rt, rn XReg, off int32) {
	a.w(0x39400000 | scaled(off, 1, "ldrb") | uint32(rn)<<5 | uint32(rt))
}

// LDRqIdx is ldr qt, [xn, xm]: 16 bytes at a byte offset held in a register,
// for an offset past the scaled immediate's 65520.
func (a *A64) LDRqIdx(vt VReg, rn, rm XReg) {
	a.w(0x3CE06800 | uint32(rm)<<16 | uint32(rn)<<5 | uint32(vt))
}

// LDRsIdx is ldr st, [xn, xm, lsl #2]: the float32 at index xm of a table.
func (a *A64) LDRsIdx(vt VReg, rn, rm XReg) {
	a.w(0xBC607800 | uint32(rm)<<16 | uint32(rn)<<5 | uint32(vt))
}

// LD1b and ST1b load or store one byte lane of vt: ld1 {vt.b}[lane], [xn].
func (a *A64) LD1b(vt VReg, lane uint8, rn XReg) {
	a.w(0x0D400000 | ld1bLane(lane) | uint32(rn)<<5 | uint32(vt))
}
func (a *A64) ST1b(vt VReg, lane uint8, rn XReg) {
	a.w(0x0D000000 | ld1bLane(lane) | uint32(rn)<<5 | uint32(vt))
}

// ld1bLane packs a byte lane into LD1/ST1's Q:S:size fields.
func ld1bLane(lane uint8) uint32 {
	if lane > 15 {
		panic(fmt.Sprintf("jit: byte lane %d, want 0..15", lane))
	}
	l := uint32(lane)
	return (l>>3)<<30 | (l>>2&1)<<12 | (l&3)<<10
}

// FCVTN4h is fcvtn vd.4h, vn.4s: four float32 to binary16 under FPCR's
// rounding (nearest-even as Go runs), the upper half of vd cleared.
func (a *A64) FCVTN4h(vd, vn VReg) { a.w(0x0E216800 | uint32(vn)<<5 | uint32(vd)) }

// CMGT8h is cmgt vd.8h, vn.8h, vm.8h: the signed int16 greater-than.
func (a *A64) CMGT8h(vd, vn, vm VReg) {
	a.w(0x4E603400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}
