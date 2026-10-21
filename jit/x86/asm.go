// Package x86 is the composer's x86-64 instruction encoder.
package x86

import "fmt"

type Reg uint8

const (
	RAX Reg = iota
	RCX
	RDX
	RBX
	RSP
	RBP
	RSI
	RDI
	R8
	R9
	R10
	R11
	R12
	R13
	R14
	R15
)

const (
	Y0 Reg = iota
	Y1
	Y2
	Y3
	Y4
	Y5
	Y6
	Y7
	Y8
	Y9
	Y10
	Y11
	Y12
	Y13
	Y14
	Y15
)

func (r Reg) lo() byte { return byte(r) & 7 }
func (r Reg) hi() byte { return (byte(r) >> 3) & 1 }

const noIndex Reg = RSP

type Mem struct {
	Base  Reg
	Index Reg
	Scale uint8
	Disp  int32
}

func At(base Reg, disp int32) Mem { return Mem{Base: base, Index: noIndex, Scale: 1, Disp: disp} }

func Idx(base, index Reg, scale uint8, disp int32) Mem {
	return Mem{Base: base, Index: index, Scale: scale, Disp: disp}
}

func (m Mem) hasIndex() bool { return m.Index != noIndex }

func (m Mem) scaleBits() (byte, bool) {
	switch m.Scale {
	case 1:
		return 0, true
	case 2:
		return 1, true
	case 4:
		return 2, true
	case 8:
		return 3, true
	}
	return 0, false
}

type Buf struct {
	b      []byte
	fixup  []fixup
	labels []int

	isa      ISA
	declared bool
	err      error

	regs regUse
}

type fixup struct {
	at   int
	size int
	lbl  int
}

func (a *Buf) Bytes() ([]byte, error) {
	for _, f := range a.fixup {
		if f.lbl >= len(a.labels) || a.labels[f.lbl] < 0 {
			a.fail("x86: a label used and never bound")
		}
	}
	if a.err != nil {
		return nil, a.err
	}
	return a.b, nil
}

func (a *Buf) fail(format string, args ...any) {
	if a.err == nil {
		a.err = fmt.Errorf(format, args...)
	}
}

func (a *Buf) Len() int { return len(a.b) }

func (a *Buf) u8(v byte)    { a.b = append(a.b, v) }
func (a *Buf) u32(v uint32) { a.b = append(a.b, byte(v), byte(v>>8), byte(v>>16), byte(v>>24)) }

func (a *Buf) rex(w bool, r, x, b Reg) {
	v := byte(0x40) | r.hi()<<2 | x.hi()<<1 | b.hi()
	if w {
		v |= 0x08
	}
	if v != 0x40 {
		a.u8(v)
	}
}

const (
	m0F   = 1
	m0F38 = 2
	m0F3A = 3

	pNone = 0
	p66   = 1
	pF3   = 2
	pF2   = 3
)

func (a *Buf) vex(r, v, b, x Reg, l256 bool, pp, mm, w byte) {
	a.noVEXInSSE("a VEX-encoded instruction")
	a.regs.vex = true
	var lb byte
	if l256 {
		lb = 1
	}
	if x.hi() == 0 && b.hi() == 0 && w == 0 && mm == m0F {
		a.u8(0xC5)
		a.u8((^r.hi()&1)<<7 | (^byte(v)&15)<<3 | lb<<2 | pp)
		return
	}
	a.u8(0xC4)
	a.u8((^r.hi()&1)<<7 | (^x.hi()&1)<<6 | (^b.hi()&1)<<5 | mm)
	a.u8(w<<7 | (^byte(v)&15)<<3 | lb<<2 | pp)
}

func (a *Buf) modrmReg(reg, rm Reg) { a.u8(0xC0 | reg.lo()<<3 | rm.lo()) }

func (a *Buf) modrmMem(reg Reg, m Mem) {
	if m.Base == RSP {
		a.regs.stack++
	}
	const rmSIB = 4
	base := m.Base.lo()
	needSIB := m.hasIndex() || base == rmSIB
	forceDisp8 := base == 5
	short := m.Disp >= -128 && m.Disp <= 127

	var mod byte
	switch {
	case m.Disp == 0 && !forceDisp8:
		mod = 0
	case short:
		mod = 1
	default:
		mod = 2
	}

	rm := base
	if needSIB {
		rm = rmSIB
	}
	a.u8(mod<<6 | reg.lo()<<3 | rm)
	if needSIB {
		idx := byte(noIndex.lo()) // 4 == "no index"
		var ss byte
		if m.hasIndex() {
			if m.Index == RSP {
				a.fail("x86: rsp as a SIB index")
			}
			idx = m.Index.lo()
			var ok bool
			if ss, ok = m.scaleBits(); !ok {
				a.fail("x86: a scale of %d", m.Scale)
			}
		}
		a.u8(ss<<6 | idx<<3 | base)
	}
	switch mod {
	case 1:
		a.u8(byte(m.Disp))
	case 2:
		a.u32(uint32(m.Disp))
	}
}

func (a *Buf) vop(op byte, pp, mm, w byte, dst, src1, src2 Reg) {
	a.vopL(op, pp, mm, w, true, dst, src1, src2)
}

func (a *Buf) vopL(op byte, pp, mm, w byte, l256 bool, dst, src1, src2 Reg) {
	a.regs.vec(dst, src1, src2)
	a.vex(dst, src1, src2, noIndex, l256, pp, mm, w)
	a.u8(op)
	a.modrmReg(dst, src2)
}

func (a *Buf) vopMem(op byte, pp, mm, w byte, dst, src1 Reg, m Mem) {
	a.regs.vec(dst, src1)
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(dst, src1, m.Base, idx, true, pp, mm, w)
	a.u8(op)
	a.modrmMem(dst, m)
}

func (a *Buf) VPDPBUSD(dst, src1, src2 Reg) { a.vop(0x50, p66, m0F38, 0, dst, src1, src2) }

func (a *Buf) VPDPBUSDMem(dst, src1 Reg, m Mem) { a.vopMem(0x50, p66, m0F38, 0, dst, src1, m) }

func (a *Buf) VPMADDUBSW(dst, src1, src2 Reg) { a.vop(0x04, p66, m0F38, 0, dst, src1, src2) }

func (a *Buf) VPSIGNB(dst, src1, src2 Reg) { a.vop(0x08, p66, m0F38, 0, dst, src1, src2) }

func (a *Buf) VPCMPEQW(dst, src1, src2 Reg) { a.vop(0x75, p66, m0F, 0, dst, src1, src2) }

func (a *Buf) VPCMPEQD(dst, src1, src2 Reg) { a.vop(0x76, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VPCMPGTD(dst, src1, src2 Reg) { a.vop(0x66, p66, m0F, 0, dst, src1, src2) }

func (a *Buf) VPMADDWD(dst, src1, src2 Reg) { a.vop(0xF5, p66, m0F, 0, dst, src1, src2) }

func (a *Buf) VPMADDWDMem(dst, src1 Reg, m Mem) { a.vopMem(0xF5, p66, m0F, 0, dst, src1, m) }

func (a *Buf) VPABSB(dst, src Reg) { a.vop(0x1C, p66, m0F38, 0, dst, Y0, src) }

func (a *Buf) VPADDD(dst, src1, src2 Reg) { a.vop(0xFE, p66, m0F, 0, dst, src1, src2) }

func (a *Buf) VPADDDMem(dst, src1 Reg, m Mem) { a.vopMem(0xFE, p66, m0F, 0, dst, src1, m) }
func (a *Buf) VPADDW(dst, src1, src2 Reg)     { a.vop(0xFD, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VPSUBB(dst, src1, src2 Reg)     { a.vop(0xF8, p66, m0F, 0, dst, src1, src2) }

func (a *Buf) VPSUBBMem(dst, src1 Reg, m Mem) { a.vopMem(0xF8, p66, m0F, 0, dst, src1, m) }
func (a *Buf) VPAND(dst, src1, src2 Reg)      { a.vop(0xDB, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VPOR(dst, src1, src2 Reg)       { a.vop(0xEB, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VPANDN(dst, src1, src2 Reg)     { a.vop(0xDF, p66, m0F, 0, dst, src1, src2) }

func (a *Buf) VPMINSD(dst, src1, src2 Reg) { a.vop(0x39, p66, m0F38, 0, dst, src1, src2) }
func (a *Buf) VPXOR(dst, src1, src2 Reg)   { a.vop(0xEF, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VADDPS(dst, src1, src2 Reg)  { a.vop(0x58, pNone, m0F, 0, dst, src1, src2) }
func (a *Buf) VMULPS(dst, src1, src2 Reg)  { a.vop(0x59, pNone, m0F, 0, dst, src1, src2) }
func (a *Buf) VPHADDD(dst, src1, src2 Reg) { a.vop(0x02, p66, m0F38, 0, dst, src1, src2) }

func (a *Buf) VFMADD231PS(dst, src1, src2 Reg) { a.vop(0xB8, p66, m0F38, 0, dst, src1, src2) }

func (a *Buf) VFMADD231PSMem(dst, src1 Reg, m Mem) { a.vopMem(0xB8, p66, m0F38, 0, dst, src1, m) }

func (a *Buf) VPSRLW(dst, src Reg, imm byte) { a.vshiftImm(0x71, 2, dst, src, imm) }
func (a *Buf) VPSRLD(dst, src Reg, imm byte) { a.vshiftImm(0x72, 2, dst, src, imm) }
func (a *Buf) VPSLLW(dst, src Reg, imm byte) { a.vshiftImm(0x71, 6, dst, src, imm) }
func (a *Buf) VPSLLD(dst, src Reg, imm byte) { a.vshiftImm(0x72, 6, dst, src, imm) }

func (a *Buf) VPSRAD(dst, src Reg, imm byte) { a.vshiftImm(0x72, 4, dst, src, imm) }

func (a *Buf) vshiftImm(op byte, ext, dst, src Reg, imm byte) {
	a.vex(Reg(ext), dst, src, noIndex, true, p66, m0F, 0)
	a.u8(op)
	a.modrmReg(Reg(ext), src)
	a.u8(imm)
}

func (a *Buf) VMOVDQULoad(dst Reg, m Mem)  { a.vopMem(0x6F, pF3, m0F, 0, dst, Y0, m) }
func (a *Buf) VMOVDQUStore(m Mem, src Reg) { a.vopMem(0x7F, pF3, m0F, 0, src, Y0, m) }

func (a *Buf) VPBROADCASTD(dst Reg, m Mem) { a.vopMem(0x58, p66, m0F38, 0, dst, Y0, m) }
func (a *Buf) VBROADCASTSS(dst Reg, m Mem) { a.vopMem(0x18, p66, m0F38, 0, dst, Y0, m) }

func (a *Buf) VCVTDQ2PS(dst, src Reg) { a.vop(0x5B, pNone, m0F, 0, dst, Y0, src) }

func (a *Buf) VSQRTPS(dst, src Reg)       { a.vop(0x51, pNone, m0F, 0, dst, Y0, src) }
func (a *Buf) VDIVPS(dst, src1, src2 Reg) { a.vop(0x5E, pNone, m0F, 0, dst, src1, src2) }

func (a *Buf) VMAXPS(dst, src1, src2 Reg) { a.vop(0x5F, pNone, m0F, 0, dst, src1, src2) }

func (a *Buf) VMOVAPSReg(dst, src Reg)    { a.vop(0x28, pNone, m0F, 0, dst, Y0, src) }
func (a *Buf) VMINPS(dst, src1, src2 Reg) { a.vop(0x5D, pNone, m0F, 0, dst, src1, src2) }

func (a *Buf) VCVTPS2DQ(dst, src Reg) { a.vop(0x5B, p66, m0F, 0, dst, Y0, src) }

func (a *Buf) VCVTTPS2DQ(dst, src Reg) { a.vop(0x5B, pF3, m0F, 0, dst, Y0, src) }

func (a *Buf) VPACKSSDW(dst, src1, src2 Reg) { a.vop(0x6B, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VPACKSSWB(dst, src1, src2 Reg) { a.vop(0x63, p66, m0F, 0, dst, src1, src2) }

func (a *Buf) VPMULLD(dst, src1, src2 Reg) { a.vop(0x40, p66, m0F38, 0, dst, src1, src2) }

func (a *Buf) VCMPPS(dst, src1, src2 Reg, imm byte) {
	a.vop(0xC2, pNone, m0F, 0, dst, src1, src2)
	a.u8(imm)
}

func (a *Buf) IMUL(dst, src Reg) {
	a.u8(0x48 | byte(dst)>>3<<2 | byte(src)>>3)
	a.u8(0x0F)
	a.u8(0xAF)
	a.modrmReg(dst, src)
}

func (a *Buf) VFMADD213PS(dst, src1, src2 Reg) { a.vop(0xA8, p66, m0F38, 0, dst, src1, src2) }

func (a *Buf) VCVTPH2PS(dst Reg, m Mem) { a.vopMem(0x13, p66, m0F38, 0, dst, Y0, m) }

func (a *Buf) VEXTRACTI128(dst, src Reg, imm byte) {
	a.vex(src, Y0, dst, noIndex, true, p66, m0F3A, 0)
	a.u8(0x39)
	a.modrmReg(src, dst)
	a.u8(imm)
}

func (a *Buf) VBROADCASTI128(dst Reg, m Mem) { a.vopMem(0x5A, p66, m0F38, 0, dst, Y0, m) }

func (a *Buf) VPBLENDD(dst, src1, src2 Reg, imm byte) {
	a.vex(dst, src1, src2, noIndex, true, p66, m0F3A, 0)
	a.u8(0x02)
	a.modrmReg(dst, src2)
	a.u8(imm)
}

func (a *Buf) VCVTPH2PSx(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(dst, Y0, m.Base, idx, false, p66, m0F38, 0)
	a.u8(0x13)
	a.modrmMem(dst, m)
}

func (a *Buf) VBROADCASTSSReg(dst, src Reg) { a.vop(0x18, p66, m0F38, 0, dst, Y0, src) }

func (a *Buf) VEXTRACTF128(dst, src Reg, imm byte) {
	a.vex(src, Y0, dst, noIndex, true, p66, m0F3A, 0)
	a.u8(0x19)
	a.modrmReg(src, dst)
	a.u8(imm)
}

func (a *Buf) VADDPSx(dst, src1, src2 Reg)  { a.vopL(0x58, pNone, m0F, 0, false, dst, src1, src2) }
func (a *Buf) VHADDPSx(dst, src1, src2 Reg) { a.vopL(0x7C, pF2, m0F, 0, false, dst, src1, src2) }

func (a *Buf) VMOVSSLoad(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(dst, Y0, m.Base, idx, false, pF3, m0F, 0)
	a.u8(0x10)
	a.modrmMem(dst, m)
}

func (a *Buf) VMOVSSStore(m Mem, src Reg) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(src, Y0, m.Base, idx, false, pF3, m0F, 0)
	a.u8(0x11)
	a.modrmMem(src, m)
}

func (a *Buf) VSHUFPS(dst, src1, src2 Reg, imm byte) {
	a.vex(dst, src1, src2, noIndex, true, pNone, m0F, 0)
	a.u8(0xC6)
	a.modrmReg(dst, src2)
	a.u8(imm)
}

func (a *Buf) VPSRLVD(dst, src, count Reg) { a.vop(0x45, p66, m0F38, 0, dst, src, count) }

func (a *Buf) VPMOVZXWD(dst Reg, m Mem) { a.vopMem(0x33, p66, m0F38, 0, dst, Y0, m) }

func (a *Buf) VPINSRWLoad(dst, src1 Reg, m Mem, imm byte) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(dst, src1, m.Base, idx, false, p66, m0F, 0)
	a.u8(0xC4)
	a.modrmMem(dst, m)
	a.u8(imm)
}

func (a *Buf) VPINSRBLoad(dst, src1 Reg, m Mem, imm byte) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(dst, src1, m.Base, idx, false, p66, m0F3A, 0)
	a.u8(0x20)
	a.modrmMem(dst, m)
	a.u8(imm)
}

func (a *Buf) VMOVSLDUP(dst, src Reg) { a.vop(0x12, pF3, m0F, 0, dst, Y0, src) }
func (a *Buf) VMOVSHDUP(dst, src Reg) { a.vop(0x16, pF3, m0F, 0, dst, Y0, src) }

func (a *Buf) VADDSUBPS(dst, src1, src2 Reg) { a.vop(0xD0, pF2, m0F, 0, dst, src1, src2) }

func (a *Buf) VFNMADD231PS(dst, src1, src2 Reg) { a.vop(0xBC, p66, m0F38, 0, dst, src1, src2) }

func (a *Buf) VPERMILPS(dst, src Reg, imm byte) {
	a.vex(dst, Y0, src, noIndex, true, p66, m0F3A, 0)
	a.u8(0x04)
	a.modrmReg(dst, src)
	a.u8(imm)
}

func (a *Buf) VPERMQ(dst, src Reg, imm byte) {
	a.vex(dst, Y0, src, noIndex, true, p66, m0F3A, 1)
	a.u8(0x00)
	a.modrmReg(dst, src)
	a.u8(imm)
}

func (a *Buf) VPMOVSXBD(dst Reg, m Mem) { a.vopMem(0x21, p66, m0F38, 0, dst, Y0, m) }

func (a *Buf) VPANDMem(dst, src1 Reg, m Mem) { a.vopMem(0xDB, p66, m0F, 0, dst, src1, m) }

func (a *Buf) VADDPSMem(dst, src1 Reg, m Mem) { a.vopMem(0x58, pNone, m0F, 0, dst, src1, m) }
func (a *Buf) VMULPSMem(dst, src1 Reg, m Mem) { a.vopMem(0x59, pNone, m0F, 0, dst, src1, m) }

func (a *Buf) VSUBPS(dst, src1, src2 Reg) { a.vop(0x5C, pNone, m0F, 0, dst, src1, src2) }

func (a *Buf) VPSUBDMem(dst, src1 Reg, m Mem) { a.vopMem(0xFA, p66, m0F, 0, dst, src1, m) }

func (a *Buf) VPSUBD(dst, src1, src2 Reg) { a.vop(0xFA, p66, m0F, 0, dst, src1, src2) }

func (a *Buf) VPSHUFB(dst, src1, src2 Reg) { a.vop(0x00, p66, m0F38, 0, dst, src1, src2) }

func (a *Buf) VCVTPH2PSReg(dst, src Reg) { a.vop(0x13, p66, m0F38, 0, dst, Y0, src) }

func (a *Buf) VPMOVZXBD(dst Reg, m Mem)  { a.vopMem(0x31, p66, m0F38, 0, dst, Y0, m) }
func (a *Buf) VPMOVZXBDReg(dst, src Reg) { a.vop(0x31, p66, m0F38, 0, dst, Y0, src) }

func (a *Buf) VPUNPCKLDQx(dst, src1, src2 Reg) { a.vopL(0x62, p66, m0F, 0, false, dst, src1, src2) }

func (a *Buf) VMOVQStore(m Mem, src Reg) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(src, Y0, m.Base, idx, false, p66, m0F, 0)
	a.u8(0xD6)
	a.modrmMem(src, m)
}

func (a *Buf) VPEXTRBStore(m Mem, src Reg, imm byte) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(src, Y0, m.Base, idx, false, p66, m0F3A, 0)
	a.u8(0x14)
	a.modrmMem(src, m)
	a.u8(imm)
}

func (a *Buf) VMOVDQUStorex(m Mem, src Reg) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(src, Y0, m.Base, idx, false, pF3, m0F, 0)
	a.u8(0x7F)
	a.modrmMem(src, m)
}

func (a *Buf) VMOVDQULoadx(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(dst, Y0, m.Base, idx, false, pF3, m0F, 0)
	a.u8(0x6F)
	a.modrmMem(dst, m)
}

func (a *Buf) VZEROUPPER() {
	a.noVEXInSSE("VZEROUPPER")
	a.u8(0xC5)
	a.u8(0xF8)
	a.u8(0x77)
}

func (a *Buf) MOVLoad(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(true, dst, idx, m.Base)
	a.u8(0x8B)
	a.modrmMem(dst, m)
}

func (a *Buf) CPUID() { a.u8(0x0F); a.u8(0xA2) }

func (a *Buf) XGETBV() { a.u8(0x0F); a.u8(0x01); a.u8(0xD0) }

func (a *Buf) MOVStore(m Mem, src Reg) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(true, src, idx, m.Base)
	a.u8(0x89)
	a.modrmMem(src, m)
}

func (a *Buf) MOVQ(dst, src Reg) {
	a.rex(true, src, noIndex, dst)
	a.u8(0x89)
	a.modrmReg(src, dst)
}

func (a *Buf) ADDQ(dst, src Reg) {
	a.rex(true, src, noIndex, dst)
	a.u8(0x01)
	a.modrmReg(src, dst)
}

func (a *Buf) ADDQmem(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(true, dst, idx, m.Base)
	a.u8(0x03)
	a.modrmMem(dst, m)
}

func (a *Buf) SUBQ(dst, src Reg) {
	a.rex(true, src, noIndex, dst)
	a.u8(0x29)
	a.modrmReg(src, dst)
}

func (a *Buf) MOVimm32(dst Reg, imm int32) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0xC7)
	a.modrmReg(Y0, dst)
	a.u32(uint32(imm))
}

func (a *Buf) SHRimm(dst Reg, imm byte) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0xC1)
	a.modrmReg(Reg(5), dst)
	a.u8(imm)
}

func (a *Buf) SHLimm(dst Reg, imm byte) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0xC1)
	a.modrmReg(Reg(4), dst)
	a.u8(imm)
}

func (a *Buf) ANDimm8(dst Reg, imm int8) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0x83)
	a.modrmReg(Reg(4), dst)
	a.u8(byte(imm))
}

func (a *Buf) ADDimm(dst Reg, imm int32) {
	a.rex(true, Y0, noIndex, dst)
	if imm >= -128 && imm <= 127 {
		a.u8(0x83)
		a.modrmReg(Y0, dst)
		a.u8(byte(imm))
		return
	}
	a.u8(0x81)
	a.modrmReg(Y0, dst)
	a.u32(uint32(imm))
}

func (a *Buf) MOVimm(dst Reg, imm int64) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0xC7)
	a.modrmReg(Y0, dst)
	a.u32(uint32(int32(imm)))
}

func (a *Buf) SUBimm(dst Reg, imm int32) {
	a.rex(true, Y0, noIndex, dst)
	if imm >= -128 && imm <= 127 {
		a.u8(0x83)
		a.modrmReg(Reg(5), dst)
		a.u8(byte(imm))
		return
	}
	a.u8(0x81)
	a.modrmReg(Reg(5), dst)
	a.u32(uint32(imm))
}

func (a *Buf) DEC(dst Reg) {
	a.rex(false, Y0, noIndex, dst)
	a.u8(0xFF)
	a.modrmReg(Reg(1), dst)
}

func (a *Buf) DECmem(m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(false, Y0, idx, m.Base)
	a.u8(0xFF)
	a.modrmMem(Reg(1), m)
}

func (a *Buf) PREFETCHT0(m Mem) { a.prefetch(1, m) }

func (a *Buf) PREFETCHNTA(m Mem) { a.prefetch(0, m) }

func (a *Buf) PREFETCHT2(m Mem) { a.prefetch(3, m) }

func (a *Buf) prefetch(hint uint8, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(false, Y0, idx, m.Base)
	a.u8(0x0F)
	a.u8(0x18)
	a.modrmMem(Reg(hint), m)
}

func (a *Buf) RET() { a.u8(0xC3) }

func (a *Buf) Label() int {
	a.labels = append(a.labels, -1)
	return len(a.labels) - 1
}

func (a *Buf) Bind(l int) {
	a.labels[l] = len(a.b)
	a.patch()
}

func (a *Buf) JNZ(l int) { a.jcc(0x75, 0x85, l) }

func (a *Buf) JZ(l int) { a.jcc(0x74, 0x84, l) }

func (a *Buf) TESTQ(x, y Reg) {
	a.rex(true, y, noIndex, x)
	a.u8(0x85)
	a.modrmReg(y, x)
}

func (a *Buf) jcc(short, near byte, l int) {
	if tgt := a.labels[l]; tgt >= 0 {
		if d := tgt - (len(a.b) + 2); d >= -128 && d <= 127 {
			a.u8(short)
			a.u8(byte(d))
			return
		}
		if near == 0xE9 {
			a.u8(near)
		} else {
			a.u8(0x0F)
			a.u8(near)
		}
		a.u32(uint32(tgt - (len(a.b) + 4)))
		return
	}
	if near == 0xE9 {
		a.u8(near)
	} else {
		a.u8(0x0F)
		a.u8(near)
	}
	a.fixup = append(a.fixup, fixup{at: len(a.b), size: 4, lbl: l})
	a.u32(0)
}

func (a *Buf) patch() {
	kept := a.fixup[:0]
	for _, f := range a.fixup {
		tgt := a.labels[f.lbl]
		if tgt < 0 {
			kept = append(kept, f)
			continue
		}
		d := uint32(tgt - (f.at + f.size))
		a.b[f.at] = byte(d)
		a.b[f.at+1] = byte(d >> 8)
		a.b[f.at+2] = byte(d >> 16)
		a.b[f.at+3] = byte(d >> 24)
	}
	a.fixup = kept
}

func (a *Buf) VCVTPS2PH(dst, src Reg, l256 bool, imm byte) {
	a.vex(src, Y0, dst, noIndex, l256, p66, m0F3A, 0)
	a.u8(0x1D)
	a.modrmReg(src, dst)
	a.u8(imm)
}

func (a *Buf) VPEXTRWStore(m Mem, src Reg, imm byte) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(src, Y0, m.Base, idx, false, p66, m0F3A, 0)
	a.u8(0x15)
	a.modrmMem(src, m)
	a.u8(imm)
}

func (a *Buf) VPANDx(dst, src1, src2 Reg)  { a.vopL(0xDB, p66, m0F, 0, false, dst, src1, src2) }
func (a *Buf) VPANDNx(dst, src1, src2 Reg) { a.vopL(0xDF, p66, m0F, 0, false, dst, src1, src2) }
func (a *Buf) VPCMPGTWx(dst, src1, src2 Reg) {
	a.vopL(0x65, p66, m0F, 0, false, dst, src1, src2)
}
