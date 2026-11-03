package cpu

import "fmt"

// The instructions the composer's arm64 lowering (composer/lowera64.go) emits
// beside the ones main's emitters already encode: three-register vector
// operations, the scalar and general-register forms of its integer nodes,
// conditional branches, and a load or store of any register size at any
// offset its addressing modes reach.

// A64Access is a load or store of one register of a size.
type A64Access uint8

const (
	A64LdrX A64Access = iota
	A64StrX
	A64LdrW
	A64StrW
	A64LdrQ
	A64StrQ
	A64LdrD
	A64StrD
	A64LdrS
	A64StrS
	A64LdrH
	A64StrH
)

// enc is op's unsigned-offset and unscaled encodings and its size in bytes.
func (op A64Access) enc() (scaled, unscaled, size uint32) {
	switch op {
	case A64LdrX:
		return 0xF9400000, 0xF8400000, 8
	case A64StrX:
		return 0xF9000000, 0xF8000000, 8
	case A64LdrW:
		return 0xB9400000, 0xB8400000, 4
	case A64StrW:
		return 0xB9000000, 0xB8000000, 4
	case A64LdrQ:
		return 0x3DC00000, 0x3CC00000, 16
	case A64StrQ:
		return 0x3D800000, 0x3C800000, 16
	case A64LdrD:
		return 0xFD400000, 0xFC400000, 8
	case A64StrD:
		return 0xFD000000, 0xFC000000, 8
	case A64LdrS:
		return 0xBD400000, 0xBC400000, 4
	case A64StrS:
		return 0xBD000000, 0xBC000000, 4
	case A64LdrH:
		return 0x7D400000, 0x7C400000, 2
	case A64StrH:
		return 0x7D000000, 0x7C000000, 2
	}
	panic(fmt.Sprintf("jit: A64Access %d is not an access", op))
}

// Access is op of register rt at rn+off: the unsigned scaled offset where
// off is a non-negative multiple of the size in reach, else the unscaled
// one. It reports false, emitting nothing, where neither reaches off.
func (a *A64) Access(op A64Access, rt uint8, rn XReg, off int32) bool {
	scaled, unscaled, size := op.enc()
	s := int32(size)
	switch {
	case off >= 0 && off%s == 0 && off/s <= 4095:
		a.w(scaled | uint32(off/s)<<10 | uint32(rn)<<5 | uint32(rt))
	case off >= -256 && off <= 255:
		a.w(unscaled | uint32(off&0x1FF)<<12 | uint32(rn)<<5 | uint32(rt))
	default:
		return false
	}
	return true
}

func (a *A64) v3(op uint32, vd, vn, vm VReg) {
	a.w(op | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

func (a *A64) v2(op uint32, vd, vn VReg) { a.w(op | uint32(vn)<<5 | uint32(vd)) }

func (a *A64) EOR16b(vd, vn, vm VReg)  { a.v3(0x6E201C00, vd, vn, vm) }
func (a *A64) BSL16b(vd, vn, vm VReg)  { a.v3(0x6E601C00, vd, vn, vm) }
func (a *A64) BIF16b(vd, vn, vm VReg)  { a.v3(0x6EE01C00, vd, vn, vm) }
func (a *A64) TRN1s4(vd, vn, vm VReg)  { a.v3(0x4E802800, vd, vn, vm) }
func (a *A64) TRN2s4(vd, vn, vm VReg)  { a.v3(0x4E806800, vd, vn, vm) }
func (a *A64) ZIP1s4(vd, vn, vm VReg)  { a.v3(0x4E803800, vd, vn, vm) }
func (a *A64) ZIP2s4(vd, vn, vm VReg)  { a.v3(0x4E807800, vd, vn, vm) }
func (a *A64) UMAX4s(vd, vn, vm VReg)  { a.v3(0x6EA06400, vd, vn, vm) }
func (a *A64) UMIN4s(vd, vn, vm VReg)  { a.v3(0x6EA06C00, vd, vn, vm) }
func (a *A64) SMINP4s(vd, vn, vm VReg) { a.v3(0x4EA0AC00, vd, vn, vm) }
func (a *A64) FMAXP4s(vd, vn, vm VReg) { a.v3(0x6E20F400, vd, vn, vm) }
func (a *A64) SSHL4s(vd, vn, vm VReg)  { a.v3(0x4EA04400, vd, vn, vm) }
func (a *A64) REV64s4(vd, vn VReg)     { a.v2(0x4EA00800, vd, vn) }
func (a *A64) NEG4s(vd, vn VReg)       { a.v2(0x6EA0B800, vd, vn) }

// FCVTN2h8 is fcvtn2 vd.8h, vn.4s: the upper 64 bits, the lower kept.
func (a *A64) FCVTN2h8(vd, vn VReg) { a.v2(0x4E216800, vd, vn) }

// DUPsW is dup vd.4s, wn.
func (a *A64) DUPsW(vd VReg, rn XReg) { a.w(0x4E040C00 | uint32(rn)<<5 | uint32(vd)) }

// LD1R4s is ld1r {vt.4s}, [xn]: one word in every lane.
func (a *A64) LD1R4s(vt VReg, rn XReg) { a.w(0x4D40C800 | uint32(rn)<<5 | uint32(vt)) }

// LD1h0 is ld1 {vt.h}[0], [xn]: one half into lane 0, the others kept.
func (a *A64) LD1h0(vt VReg, rn XReg) { a.w(0x0D404000 | uint32(rn)<<5 | uint32(vt)) }

// UMOVs0 is mov wd, vn.s[0].
func (a *A64) UMOVs0(rd XReg, vn VReg) { a.w(0x0E043C00 | uint32(vn)<<5 | uint32(rd)) }

// FMOVsW is fmov sd, wn: the word in lane 0, the rest cleared.
func (a *A64) FMOVsW(vd VReg, rn XReg) { a.w(0x1E270000 | uint32(rn)<<5 | uint32(vd)) }

// SCVTFsW and SCVTFsX are scvtf sd, wn and scvtf sd, xn: a signed integer
// converted into lane 0, the rest cleared.
func (a *A64) SCVTFsW(vd VReg, rn XReg) { a.w(0x1E220000 | uint32(rn)<<5 | uint32(vd)) }
func (a *A64) SCVTFsX(vd VReg, rn XReg) { a.w(0x9E220000 | uint32(rn)<<5 | uint32(vd)) }

func (a *A64) x3(op uint32, rd, rn, rm XReg) {
	a.w(op | uint32(rm)<<16 | uint32(rn)<<5 | uint32(rd))
}

// ADDlsl is add xd, xn, xm, lsl #sh.
func (a *A64) ADDlsl(rd, rn, rm XReg, sh uint8) {
	a.w(0x8B000000 | uint32(rm)<<16 | uint32(sh&63)<<10 | uint32(rn)<<5 | uint32(rd))
}

func (a *A64) ANDreg(rd, rn, rm XReg) { a.x3(0x8A000000, rd, rn, rm) }
func (a *A64) ORRreg(rd, rn, rm XReg) { a.x3(0xAA000000, rd, rn, rm) }
func (a *A64) EORreg(rd, rn, rm XReg) { a.x3(0xCA000000, rd, rn, rm) }
func (a *A64) BICreg(rd, rn, rm XReg) { a.x3(0x8A200000, rd, rn, rm) }
func (a *A64) LSLV(rd, rn, rm XReg)   { a.x3(0x9AC02000, rd, rn, rm) }
func (a *A64) LSRV(rd, rn, rm XReg)   { a.x3(0x9AC02400, rd, rn, rm) }

// MUL is mul xd, xn, xm: the low 64 bits of the product.
func (a *A64) MUL(rd, rn, rm XReg) { a.x3(0x9B007C00, rd, rn, rm) }

// UDIV is udiv xd, xn, xm: zero where xm is zero.
func (a *A64) UDIV(rd, rn, rm XReg) { a.x3(0x9AC00800, rd, rn, rm) }

// MVN is mvn xd, xm.
func (a *A64) MVN(rd, rm XReg) { a.x3(0xAA2003E0, rd, 0, rm) }

// NEG is neg xd, xm.
func (a *A64) NEG(rd, rm XReg) { a.x3(0xCB0003E0, rd, 0, rm) }

// MOVw is mov wd, wm: the low 32 bits, the upper cleared.
func (a *A64) MOVw(rd, rm XReg) { a.x3(0x2A0003E0, rd, 0, rm) }

// LSLimm is lsl xd, xn, #sh.
func (a *A64) LSLimm(rd, rn XReg, sh uint8) {
	s := uint32(sh & 63)
	a.w(0xD3400000 | (64-s)%64<<16 | (63-s)<<10 | uint32(rn)<<5 | uint32(rd))
}

// CMPw is cmp wn, wm.
func (a *A64) CMPw(rn, rm XReg) { a.x3(0x6B00001F, 0, rn, rm) }

// A64Cond is a condition code.
type A64Cond uint8

const (
	CondEQ A64Cond = 0
	CondNE A64Cond = 1
	CondHS A64Cond = 2
	CondLO A64Cond = 3
	CondHI A64Cond = 8
	CondLS A64Cond = 9
	CondGE A64Cond = 10
	CondLT A64Cond = 11
	CondGT A64Cond = 12
	CondLE A64Cond = 13
)

// CSEL is csel xd, xn, xm, cond.
func (a *A64) CSEL(rd, rn, rm XReg, c A64Cond) {
	a.w(0x9A800000 | uint32(rm)<<16 | uint32(c)<<12 | uint32(rn)<<5 | uint32(rd))
}

// CSETMw is csetm wd, cond: all ones in the low 32 bits where cond holds,
// else zero, the upper cleared.
func (a *A64) CSETMw(rd XReg, c A64Cond) {
	a.w(0x5A9F03E0 | uint32(c^1)<<12 | uint32(rd))
}

// MOVconst is xd = v for any 64-bit v: a move of the half that sets the
// most of it, by MOVZ or MOVN, then a MOVK for each half it leaves wrong.
func (a *A64) MOVconst(rd XReg, v uint64) {
	ones := 0
	for sh := 0; sh < 4; sh++ {
		if v>>(16*sh)&0xFFFF == 0xFFFF {
			ones++
		}
	}
	inv := ones >= 2
	fill := uint64(0)
	if inv {
		fill = 0xFFFF
	}
	first := true
	for sh := 0; sh < 4; sh++ {
		h := v >> (16 * sh) & 0xFFFF
		if h == fill {
			continue
		}
		switch {
		case first && inv:
			a.w(0x92800000 | uint32(sh)<<21 | uint32(^h&0xFFFF)<<5 | uint32(rd))
		case first:
			a.w(0xD2800000 | uint32(sh)<<21 | uint32(h)<<5 | uint32(rd))
		default:
			a.MOVK(rd, uint16(h), uint8(sh))
		}
		first = false
	}
	switch {
	case first && inv:
		a.w(0x92800000 | uint32(rd))
	case first:
		a.w(0xD2800000 | uint32(rd))
	}
}
