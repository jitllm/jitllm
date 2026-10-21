package cpu

// The legacy-SSE instruction set the SSE tier's kernels are written in.
//
// A binary op is (dst, src1, src2) like its V-prefixed twin in asm.go; sse3
// lowers it to the destructive two-operand form, emitting a MOVAPS only when
// dst is not already src1 (sse.go). That lets an AVX2 emitter be ported by
// renaming calls. Unary ops (SQRTPS, PSHUFD, PMOVZXBD, CVTDQ2PS...) are
// (dst, src) because the legacy form is not destructive for them.
//
// dst == src2 on a non-commutative op cannot be served without a scratch
// register, so it panics at emit time. Scalar SS ops are non-commutative here
// because their upper lanes come from src1, and MAXPS/MINPS because they
// return the second operand on a NaN and on a +0/-0 tie. ADDPS, MULPS and the
// like are treated as commutative, which is exact except for which NaN
// payload comes out; no kernel here preserves a NaN payload.
//
// Memory operands exist only where the architecture does not check alignment
// (sseOpMem): unaligned loads and the m32/m64/m16 forms, and no ADDPSMem.
//
// useISA panics when an SSSE3 or SSE4.1 op lands in a buffer that did not
// declare it (isa.go). The encodings are pinned against objdump in
// sse_test.go.
//
// Many of these encoders (and a few in asm.go and a64.go) have no kernel
// calling them. They are kept on purpose, as the instruction vocabulary a
// kernel composer would select from (AGENTS.md, Scope): each one is already
// gated against objdump, so a new kernel can use it without a new encoding to
// verify.

// sseIns is one legacy opcode: mnemonic (for panics), opcode, mandatory
// prefix, map, and the extension that carries it.
type sseIns struct {
	name string
	op   byte
	pp   byte
	mm   byte
	isa  ISA
}

func (a *Buf) bin(i sseIns, commutes bool, dst, src1, src2 Reg) {
	a.useISA(i.isa, i.name)
	a.sse3(i.op, i.pp, i.mm, 0, commutes, dst, src1, src2)
}

func (a *Buf) binImm(i sseIns, commutes bool, dst, src1, src2 Reg, imm byte) {
	a.useISA(i.isa, i.name)
	a.sse3Imm(i.op, i.pp, i.mm, 0, commutes, dst, src1, src2, imm)
}

func (a *Buf) un(i sseIns, dst, src Reg) {
	a.useISA(i.isa, i.name)
	a.sseOp(i.op, i.pp, i.mm, 0, dst, src)
}

func (a *Buf) unImm(i sseIns, dst, src Reg, imm byte) {
	a.useISA(i.isa, i.name)
	a.sseOpImm(i.op, i.pp, i.mm, 0, dst, src, imm)
}

func (a *Buf) mem(i sseIns, reg Reg, m Mem) {
	a.useISA(i.isa, i.name)
	a.sseOpMem(i.op, i.pp, i.mm, 0, reg, m)
}

// binMem is a three-operand op whose second source is memory, for the scalar
// m32 forms: dst = src1 op [m] in lane 0, lanes 1..3 from src1.
func (a *Buf) binMem(i sseIns, dst, src1 Reg, m Mem) {
	a.useISA(i.isa, i.name)
	a.MOVAPS(dst, src1)
	a.sseOpMem(i.op, i.pp, i.mm, 0, dst, m)
}

// ---------------------------------------------------------------------------
// Moves.

var (
	iMOVUPSld  = sseIns{"movups", 0x10, pNone, m0F, ISASSE2}
	iMOVUPSst  = sseIns{"movups", 0x11, pNone, m0F, ISASSE2}
	iMOVDQUld  = sseIns{"movdqu", 0x6F, pF3, m0F, ISASSE2}
	iMOVDQUst  = sseIns{"movdqu", 0x7F, pF3, m0F, ISASSE2}
	iLDDQU     = sseIns{"lddqu", 0xF0, pF2, m0F, ISASSE3}
	iMOVSSld   = sseIns{"movss", 0x10, pF3, m0F, ISASSE2}
	iMOVSSst   = sseIns{"movss", 0x11, pF3, m0F, ISASSE2}
	iMOVDto    = sseIns{"movd", 0x6E, p66, m0F, ISASSE2}
	iMOVDfrom  = sseIns{"movd", 0x7E, p66, m0F, ISASSE2}
	iMOVQld    = sseIns{"movq", 0x7E, pF3, m0F, ISASSE2}
	iMOVQst    = sseIns{"movq", 0xD6, p66, m0F, ISASSE2}
	iMOVHLPS   = sseIns{"movhlps", 0x12, pNone, m0F, ISASSE2}
	iMOVLHPS   = sseIns{"movlhps", 0x16, pNone, m0F, ISASSE2}
	iMOVSLDUP  = sseIns{"movsldup", 0x12, pF3, m0F, ISASSE3}
	iMOVSHDUP  = sseIns{"movshdup", 0x16, pF3, m0F, ISASSE3}
	iMOVDDUP   = sseIns{"movddup", 0x12, pF2, m0F, ISASSE3}
	iMOVMSKPS  = sseIns{"movmskps", 0x50, pNone, m0F, ISASSE2}
	iPMOVMSKB  = sseIns{"pmovmskb", 0xD7, p66, m0F, ISASSE2}
	iPINSRW    = sseIns{"pinsrw", 0xC4, p66, m0F, ISASSE2}
	iPEXTRW    = sseIns{"pextrw", 0xC5, p66, m0F, ISASSE2}
	iPINSRB    = sseIns{"pinsrb", 0x20, p66, m0F3A, ISASSE41}
	iPINSRD    = sseIns{"pinsrd", 0x22, p66, m0F3A, ISASSE41}
	iPEXTRB    = sseIns{"pextrb", 0x14, p66, m0F3A, ISASSE41}
	iPEXTRD    = sseIns{"pextrd", 0x16, p66, m0F3A, ISASSE41}
	iINSERTPS  = sseIns{"insertps", 0x21, p66, m0F3A, ISASSE41}
	iMOVSSmerg = sseIns{"movss", 0x10, pF3, m0F, ISASSE2}
)

// MOVUPSLoad and MOVUPSStore move 16 bytes of floats, unaligned.
func (a *Buf) MOVUPSLoad(dst Reg, m Mem)  { a.mem(iMOVUPSld, dst, m) }
func (a *Buf) MOVUPSStore(m Mem, src Reg) { a.mem(iMOVUPSst, src, m) }

// MOVDQULoad and MOVDQUStore move 16 integer bytes, unaligned. A 32-byte
// AVX2 load becomes two of these.
func (a *Buf) MOVDQULoad(dst Reg, m Mem)  { a.mem(iMOVDQUld, dst, m) }
func (a *Buf) MOVDQUStore(m Mem, src Reg) { a.mem(iMOVDQUst, src, m) }

// LDDQU is the SSE3 unaligned load that may read a whole cache line pair
// rather than split; semantically MOVDQULoad.
func (a *Buf) LDDQU(dst Reg, m Mem) { a.mem(iLDDQU, dst, m) }

// MOVSSLoad reads one float32 into lane 0 and ZEROES lanes 1..3 -- the load
// that lets a vector body run on one tail element. MOVSSStore writes lane 0.
func (a *Buf) MOVSSLoad(dst Reg, m Mem)  { a.mem(iMOVSSld, dst, m) }
func (a *Buf) MOVSSStore(m Mem, src Reg) { a.mem(iMOVSSst, src, m) }

// MOVSS is the register form: lane 0 from src2, lanes 1..3 from src1. It
// merges where the load form zeroes, so XORPS t,t then MOVSS(t, t, x) keeps
// x's lane 0 and clears the rest (VPBLENDD(x, zero, x, 0x01) on AVX2).
func (a *Buf) MOVSS(dst, src1, src2 Reg) { a.bin(iMOVSSmerg, false, dst, src1, src2) }

// MOVDLoad reads 32 bits into lane 0 and zeroes the rest; MOVDStore writes
// lane 0's 32 bits.
func (a *Buf) MOVDLoad(dst Reg, m Mem)  { a.mem(iMOVDto, dst, m) }
func (a *Buf) MOVDStore(m Mem, src Reg) { a.mem(iMOVDfrom, src, m) }

// MOVDToX moves a GP register's low 32 bits into lane 0 of dst, zeroing the
// rest; MOVDFromX moves lane 0 into a GP register (zero-extended to 64).
func (a *Buf) MOVDToX(dst, gp Reg) {
	a.useISA(ISASSE2, "movd")
	a.sseOp(0x6E, p66, m0F, 0, dst, gp)
}
func (a *Buf) MOVDFromX(gp, src Reg) {
	a.useISA(ISASSE2, "movd")
	a.sseOp(0x7E, p66, m0F, 0, src, gp)
}

// MOVQLoad reads 64 bits into the low half and zeroes the high; MOVQStore
// writes the low 64 bits.
func (a *Buf) MOVQLoad(dst Reg, m Mem)  { a.mem(iMOVQld, dst, m) }
func (a *Buf) MOVQStore(m Mem, src Reg) { a.mem(iMOVQst, src, m) }

// MOVQToX and MOVQFromX are the 64-bit GP moves (REX.W 6E / 7E).
func (a *Buf) MOVQToX(dst, gp Reg) {
	a.useISA(ISASSE2, "movq")
	a.sseOp(0x6E, p66, m0F, 1, dst, gp)
}
func (a *Buf) MOVQFromX(gp, src Reg) {
	a.useISA(ISASSE2, "movq")
	a.sseOp(0x7E, p66, m0F, 1, src, gp)
}

// MOVQX copies the low 64 bits of src into dst and zeroes dst's high half.
func (a *Buf) MOVQX(dst, src Reg) { a.un(iMOVQld, dst, src) }

// MOVHLPS: dst.lo64 = src2.hi64, dst.hi64 = src1.hi64. MOVLHPS: dst.lo64 =
// src1.lo64, dst.hi64 = src2.lo64. Register forms only.
func (a *Buf) MOVHLPS(dst, src1, src2 Reg) { a.bin(iMOVHLPS, false, dst, src1, src2) }
func (a *Buf) MOVLHPS(dst, src1, src2 Reg) { a.bin(iMOVLHPS, false, dst, src1, src2) }

// MOVSLDUP and MOVSHDUP duplicate the even and odd lanes; MOVDDUP the low
// 64 bits. On an interleaved {cos, sin} table the first two are the operands a
// consecutive-pair rotation wants.
func (a *Buf) MOVSLDUP(dst, src Reg) { a.un(iMOVSLDUP, dst, src) }
func (a *Buf) MOVSHDUP(dst, src Reg) { a.un(iMOVSHDUP, dst, src) }
func (a *Buf) MOVDDUP(dst, src Reg)  { a.un(iMOVDDUP, dst, src) }

// MOVMSKPS and PMOVMSKB gather lane sign bits into a GP register.
func (a *Buf) MOVMSKPS(gp, src Reg) { a.un(iMOVMSKPS, gp, src) }
func (a *Buf) PMOVMSKB(gp, src Reg) { a.un(iPMOVMSKB, gp, src) }

// PINSRW inserts a GP register's low word into word imm of src1's copy.
// PINSRWLoad reads the word from memory -- the load of exactly one binary16,
// which a 64-bit MOVQ cannot do at the end of a row. With src1 zeroed it is a
// zero-extended u16 in lane 0.
func (a *Buf) PINSRW(dst, src1, gp Reg, imm byte) {
	a.useISA(iPINSRW.isa, iPINSRW.name)
	a.MOVAPS(dst, src1)
	a.sseOpImm(iPINSRW.op, iPINSRW.pp, iPINSRW.mm, 0, dst, gp, imm)
}
func (a *Buf) PINSRWLoad(dst, src1 Reg, m Mem, imm byte) {
	a.useISA(iPINSRW.isa, iPINSRW.name)
	a.MOVAPS(dst, src1)
	a.sseOpMemImm(iPINSRW.op, iPINSRW.pp, iPINSRW.mm, 0, dst, m, imm)
}

// PINSRBLoad inserts ONE byte from memory into lane imm of src1, writing dst.
// With src1 zeroed it is a load of exactly one byte -- which PMOVZXBD, reading
// four or eight, cannot do at the end of a d plane that holds four rows to a
// word. It is MXFP4's E8M0 super-scale, and it is PINSRW's byte twin.
func (a *Buf) PINSRBLoad(dst, src1 Reg, m Mem, imm byte) {
	a.useISA(iPINSRB.isa, iPINSRB.name)
	a.MOVAPS(dst, src1)
	a.sseOpMemImm(iPINSRB.op, iPINSRB.pp, iPINSRB.mm, 0, dst, m, imm)
}

// PEXTRW moves word imm of src into a GP register, zero-extended.
func (a *Buf) PEXTRW(gp, src Reg, imm byte) { a.unImm(iPEXTRW, gp, src, imm) }

// PINSRB/PINSRD insert a byte/dword (SSE4.1); PINSRDLoad reads it from
// memory.
func (a *Buf) PINSRB(dst, src1, gp Reg, imm byte) {
	a.useISA(iPINSRB.isa, iPINSRB.name)
	a.MOVAPS(dst, src1)
	a.sseOpImm(iPINSRB.op, iPINSRB.pp, iPINSRB.mm, 0, dst, gp, imm)
}
func (a *Buf) PINSRD(dst, src1, gp Reg, imm byte) {
	a.useISA(iPINSRD.isa, iPINSRD.name)
	a.MOVAPS(dst, src1)
	a.sseOpImm(iPINSRD.op, iPINSRD.pp, iPINSRD.mm, 0, dst, gp, imm)
}
func (a *Buf) PINSRDLoad(dst, src1 Reg, m Mem, imm byte) {
	a.useISA(iPINSRD.isa, iPINSRD.name)
	a.MOVAPS(dst, src1)
	a.sseOpMemImm(iPINSRD.op, iPINSRD.pp, iPINSRD.mm, 0, dst, m, imm)
}

// PEXTRB/PEXTRD extract a byte/dword into a GP register (SSE4.1); the xmm is
// in ModRM.reg for these, the GP in ModRM.rm. PEXTRDStore writes it to memory.
func (a *Buf) PEXTRB(gp, src Reg, imm byte) {
	a.useISA(iPEXTRB.isa, iPEXTRB.name)
	a.sseOpImm(iPEXTRB.op, iPEXTRB.pp, iPEXTRB.mm, 0, src, gp, imm)
}
func (a *Buf) PEXTRD(gp, src Reg, imm byte) {
	a.useISA(iPEXTRD.isa, iPEXTRD.name)
	a.sseOpImm(iPEXTRD.op, iPEXTRD.pp, iPEXTRD.mm, 0, src, gp, imm)
}
func (a *Buf) PEXTRDStore(m Mem, src Reg, imm byte) {
	a.useISA(iPEXTRD.isa, iPEXTRD.name)
	a.sseOpMemImm(iPEXTRD.op, iPEXTRD.pp, iPEXTRD.mm, 0, src, m, imm)
}

// INSERTPS copies one float of src2 into one lane of src1's copy and zeroes
// the lanes imm's low nibble names (SSE4.1).
func (a *Buf) INSERTPS(dst, src1, src2 Reg, imm byte) {
	a.binImm(iINSERTPS, false, dst, src1, src2, imm)
}

// ---------------------------------------------------------------------------
// Packed float arithmetic. No FMA exists at this tier: every VFMADD231PS in
// an AVX2 emitter becomes MULPS into a temporary plus ADDPS, which rounds
// twice -- so a port is compared with the oracle, not with the AVX2 kernel.

var (
	iADDPS    = sseIns{"addps", 0x58, pNone, m0F, ISASSE2}
	iSUBPS    = sseIns{"subps", 0x5C, pNone, m0F, ISASSE2}
	iMULPS    = sseIns{"mulps", 0x59, pNone, m0F, ISASSE2}
	iDIVPS    = sseIns{"divps", 0x5E, pNone, m0F, ISASSE2}
	iMAXPS    = sseIns{"maxps", 0x5F, pNone, m0F, ISASSE2}
	iMINPS    = sseIns{"minps", 0x5D, pNone, m0F, ISASSE2}
	iANDPS    = sseIns{"andps", 0x54, pNone, m0F, ISASSE2}
	iANDNPS   = sseIns{"andnps", 0x55, pNone, m0F, ISASSE2}
	iORPS     = sseIns{"orps", 0x56, pNone, m0F, ISASSE2}
	iXORPS    = sseIns{"xorps", 0x57, pNone, m0F, ISASSE2}
	iUNPCKLPS = sseIns{"unpcklps", 0x14, pNone, m0F, ISASSE2}
	iUNPCKHPS = sseIns{"unpckhps", 0x15, pNone, m0F, ISASSE2}
	iHADDPS   = sseIns{"haddps", 0x7C, pF2, m0F, ISASSE3}
	iHSUBPS   = sseIns{"hsubps", 0x7D, pF2, m0F, ISASSE3}
	iADDSUBPS = sseIns{"addsubps", 0xD0, pF2, m0F, ISASSE3}
	iCMPPS    = sseIns{"cmpps", 0xC2, pNone, m0F, ISASSE2}
	iSHUFPS   = sseIns{"shufps", 0xC6, pNone, m0F, ISASSE2}
	iBLENDPS  = sseIns{"blendps", 0x0C, p66, m0F3A, ISASSE41}
	iBLENDVPS = sseIns{"blendvps", 0x14, p66, m0F38, ISASSE41}
	iSQRTPS   = sseIns{"sqrtps", 0x51, pNone, m0F, ISASSE2}
	iRCPPS    = sseIns{"rcpps", 0x53, pNone, m0F, ISASSE2}
	iRSQRTPS  = sseIns{"rsqrtps", 0x52, pNone, m0F, ISASSE2}
	iROUNDPS  = sseIns{"roundps", 0x08, p66, m0F3A, ISASSE41}
)

func (a *Buf) ADDPS(dst, src1, src2 Reg)    { a.bin(iADDPS, true, dst, src1, src2) }
func (a *Buf) SUBPS(dst, src1, src2 Reg)    { a.bin(iSUBPS, false, dst, src1, src2) }
func (a *Buf) MULPS(dst, src1, src2 Reg)    { a.bin(iMULPS, true, dst, src1, src2) }
func (a *Buf) DIVPS(dst, src1, src2 Reg)    { a.bin(iDIVPS, false, dst, src1, src2) }
func (a *Buf) MAXPS(dst, src1, src2 Reg)    { a.bin(iMAXPS, false, dst, src1, src2) }
func (a *Buf) MINPS(dst, src1, src2 Reg)    { a.bin(iMINPS, false, dst, src1, src2) }
func (a *Buf) ANDPS(dst, src1, src2 Reg)    { a.bin(iANDPS, true, dst, src1, src2) }
func (a *Buf) ORPS(dst, src1, src2 Reg)     { a.bin(iORPS, true, dst, src1, src2) }
func (a *Buf) XORPS(dst, src1, src2 Reg)    { a.bin(iXORPS, true, dst, src1, src2) }
func (a *Buf) UNPCKLPS(dst, src1, src2 Reg) { a.bin(iUNPCKLPS, false, dst, src1, src2) }
func (a *Buf) UNPCKHPS(dst, src1, src2 Reg) { a.bin(iUNPCKHPS, false, dst, src1, src2) }
func (a *Buf) HADDPS(dst, src1, src2 Reg)   { a.bin(iHADDPS, false, dst, src1, src2) }
func (a *Buf) HSUBPS(dst, src1, src2 Reg)   { a.bin(iHSUBPS, false, dst, src1, src2) }

// ANDNPS is (NOT src1) AND src2.
func (a *Buf) ANDNPS(dst, src1, src2 Reg) { a.bin(iANDNPS, false, dst, src1, src2) }

// ADDSUBPS subtracts in the even lanes and adds in the odd ones -- the sign
// pattern of a 2-D rotation applied to adjacent pairs.
func (a *Buf) ADDSUBPS(dst, src1, src2 Reg) { a.bin(iADDSUBPS, false, dst, src1, src2) }

// CMPPS compares and leaves an all-ones or all-zero mask per lane; imm is the
// predicate (0 EQ, 1 LT, 2 LE, 3 UNORD, 4 NEQ, 5 NLT, 6 NLE, 7 ORD -- the
// legacy form has only these eight).
func (a *Buf) CMPPS(dst, src1, src2 Reg, imm byte) {
	if imm > 7 {
		panic("jit: legacy CMPPS has predicates 0..7 only; 8..31 are VEX")
	}
	a.binImm(iCMPPS, false, dst, src1, src2, imm)
}

// SHUFPS takes lanes 0,1 of the result from src1 and lanes 2,3 from src2, two
// bits of imm per lane: 0x00 broadcasts lane 0 of src1==src2, 0x4E swaps the
// 64-bit halves, 0xB1 swaps adjacent pairs, 0x88/0xDD gather even/odd lanes.
func (a *Buf) SHUFPS(dst, src1, src2 Reg, imm byte) {
	a.binImm(iSHUFPS, false, dst, src1, src2, imm)
}

// BLENDPS takes lane i from src2 where imm bit i is set, else from src1
// (SSE4.1).
func (a *Buf) BLENDPS(dst, src1, src2 Reg, imm byte) {
	a.binImm(iBLENDPS, false, dst, src1, src2, imm)
}

// BLENDVPS takes lane i from src2 where the sign bit of XMM0's lane i is set,
// else from src1 (SSE4.1). The mask is IMPLICITLY XMM0 in the legacy form, so
// dst may not be XMM0 unless src1 is too: the copy into dst would overwrite
// the mask.
func (a *Buf) BLENDVPS(dst, src1, src2 Reg) {
	if dst == XMM0 && src1 != XMM0 {
		panic("jit: BLENDVPS into XMM0 would overwrite its implicit mask")
	}
	a.bin(iBLENDVPS, false, dst, src1, src2)
}

func (a *Buf) SQRTPS(dst, src Reg)  { a.un(iSQRTPS, dst, src) }
func (a *Buf) RCPPS(dst, src Reg)   { a.un(iRCPPS, dst, src) }
func (a *Buf) RSQRTPS(dst, src Reg) { a.un(iRSQRTPS, dst, src) }

// ROUNDPS rounds each lane by imm's mode (0 nearest-even, 1 down, 2 up, 3
// toward zero; bit 3 suppresses the precision exception) -- SSE4.1.
func (a *Buf) ROUNDPS(dst, src Reg, imm byte) { a.unImm(iROUNDPS, dst, src, imm) }

// ---------------------------------------------------------------------------
// Scalar float. Lane 0 is computed; lanes 1..3 come from src1. The Mem forms
// read a 32-bit operand, which the architecture never checks for alignment.

var (
	iADDSS    = sseIns{"addss", 0x58, pF3, m0F, ISASSE2}
	iSUBSS    = sseIns{"subss", 0x5C, pF3, m0F, ISASSE2}
	iMULSS    = sseIns{"mulss", 0x59, pF3, m0F, ISASSE2}
	iDIVSS    = sseIns{"divss", 0x5E, pF3, m0F, ISASSE2}
	iMAXSS    = sseIns{"maxss", 0x5F, pF3, m0F, ISASSE2}
	iMINSS    = sseIns{"minss", 0x5D, pF3, m0F, ISASSE2}
	iSQRTSS   = sseIns{"sqrtss", 0x51, pF3, m0F, ISASSE2}
	iROUNDSS  = sseIns{"roundss", 0x0A, p66, m0F3A, ISASSE41}
	iCVTSS2SD = sseIns{"cvtss2sd", 0x5A, pF3, m0F, ISASSE2}
	iCVTSD2SS = sseIns{"cvtsd2ss", 0x5A, pF2, m0F, ISASSE2}
)

func (a *Buf) ADDSS(dst, src1, src2 Reg)    { a.bin(iADDSS, false, dst, src1, src2) }
func (a *Buf) SUBSS(dst, src1, src2 Reg)    { a.bin(iSUBSS, false, dst, src1, src2) }
func (a *Buf) MULSS(dst, src1, src2 Reg)    { a.bin(iMULSS, false, dst, src1, src2) }
func (a *Buf) DIVSS(dst, src1, src2 Reg)    { a.bin(iDIVSS, false, dst, src1, src2) }
func (a *Buf) MAXSS(dst, src1, src2 Reg)    { a.bin(iMAXSS, false, dst, src1, src2) }
func (a *Buf) MINSS(dst, src1, src2 Reg)    { a.bin(iMINSS, false, dst, src1, src2) }
func (a *Buf) SQRTSS(dst, src1, src2 Reg)   { a.bin(iSQRTSS, false, dst, src1, src2) }
func (a *Buf) CVTSS2SD(dst, src1, src2 Reg) { a.bin(iCVTSS2SD, false, dst, src1, src2) }
func (a *Buf) CVTSD2SS(dst, src1, src2 Reg) { a.bin(iCVTSD2SS, false, dst, src1, src2) }
func (a *Buf) ROUNDSS(dst, src1, src2 Reg, imm byte) {
	a.binImm(iROUNDSS, false, dst, src1, src2, imm)
}

func (a *Buf) ADDSSMem(dst, src1 Reg, m Mem) { a.binMem(iADDSS, dst, src1, m) }
func (a *Buf) SUBSSMem(dst, src1 Reg, m Mem) { a.binMem(iSUBSS, dst, src1, m) }
func (a *Buf) MULSSMem(dst, src1 Reg, m Mem) { a.binMem(iMULSS, dst, src1, m) }
func (a *Buf) DIVSSMem(dst, src1 Reg, m Mem) { a.binMem(iDIVSS, dst, src1, m) }
func (a *Buf) MAXSSMem(dst, src1 Reg, m Mem) { a.binMem(iMAXSS, dst, src1, m) }
func (a *Buf) MINSSMem(dst, src1 Reg, m Mem) { a.binMem(iMINSS, dst, src1, m) }

// ---------------------------------------------------------------------------
// Conversions.

var (
	iCVTDQ2PS  = sseIns{"cvtdq2ps", 0x5B, pNone, m0F, ISASSE2}
	iCVTPS2DQ  = sseIns{"cvtps2dq", 0x5B, p66, m0F, ISASSE2}
	iCVTTPS2DQ = sseIns{"cvttps2dq", 0x5B, pF3, m0F, ISASSE2}
	iCVTPS2PD  = sseIns{"cvtps2pd", 0x5A, pNone, m0F, ISASSE2}
	iCVTPD2PS  = sseIns{"cvtpd2ps", 0x5A, p66, m0F, ISASSE2}
)

// CVTDQ2PS converts int32 lanes to float32.
func (a *Buf) CVTDQ2PS(dst, src Reg) { a.un(iCVTDQ2PS, dst, src) }

// CVTPS2DQ rounds under MXCSR (nearest-even by default) -- the round() a
// range reduction wants, and the same answer VCVTPS2DQ gives.
func (a *Buf) CVTPS2DQ(dst, src Reg) { a.un(iCVTPS2DQ, dst, src) }

// CVTTPS2DQ truncates toward zero.
func (a *Buf) CVTTPS2DQ(dst, src Reg) { a.un(iCVTTPS2DQ, dst, src) }

// CVTPS2PD widens lanes 0,1 to two float64; CVTPD2PS narrows two float64 into
// lanes 0,1 and zeroes 2,3.
func (a *Buf) CVTPS2PD(dst, src Reg) { a.un(iCVTPS2PD, dst, src) }
func (a *Buf) CVTPD2PS(dst, src Reg) { a.un(iCVTPD2PS, dst, src) }

// CVTSI2SS converts a signed 64-bit GP register into lane 0 of a copy of src1
// (REX.W F3 0F 2A).
func (a *Buf) CVTSI2SS(dst, src1, gp Reg) {
	a.useISA(ISASSE2, "cvtsi2ss")
	a.MOVAPS(dst, src1)
	a.sseOp(0x2A, pF3, m0F, 1, dst, gp)
}

// CVTTSS2SI and CVTSS2SI convert lane 0 to a signed 64-bit GP register,
// truncating and rounding under MXCSR respectively.
func (a *Buf) CVTTSS2SI(gp, src Reg) {
	a.useISA(ISASSE2, "cvttss2si")
	a.sseOp(0x2C, pF3, m0F, 1, gp, src)
}
func (a *Buf) CVTSS2SI(gp, src Reg) {
	a.useISA(ISASSE2, "cvtss2si")
	a.sseOp(0x2D, pF3, m0F, 1, gp, src)
}

// ---------------------------------------------------------------------------
// Integer, SSE2.

var (
	iPADDB      = sseIns{"paddb", 0xFC, p66, m0F, ISASSE2}
	iPADDW      = sseIns{"paddw", 0xFD, p66, m0F, ISASSE2}
	iPADDD      = sseIns{"paddd", 0xFE, p66, m0F, ISASSE2}
	iPADDQ      = sseIns{"paddq", 0xD4, p66, m0F, ISASSE2}
	iPSUBB      = sseIns{"psubb", 0xF8, p66, m0F, ISASSE2}
	iPSUBW      = sseIns{"psubw", 0xF9, p66, m0F, ISASSE2}
	iPSUBD      = sseIns{"psubd", 0xFA, p66, m0F, ISASSE2}
	iPSUBQ      = sseIns{"psubq", 0xFB, p66, m0F, ISASSE2}
	iPMULLW     = sseIns{"pmullw", 0xD5, p66, m0F, ISASSE2}
	iPMULHW     = sseIns{"pmulhw", 0xE5, p66, m0F, ISASSE2}
	iPMULHUW    = sseIns{"pmulhuw", 0xE4, p66, m0F, ISASSE2}
	iPMULUDQ    = sseIns{"pmuludq", 0xF4, p66, m0F, ISASSE2}
	iPMADDWD    = sseIns{"pmaddwd", 0xF5, p66, m0F, ISASSE2}
	iPAND       = sseIns{"pand", 0xDB, p66, m0F, ISASSE2}
	iPANDN      = sseIns{"pandn", 0xDF, p66, m0F, ISASSE2}
	iPOR        = sseIns{"por", 0xEB, p66, m0F, ISASSE2}
	iPXOR       = sseIns{"pxor", 0xEF, p66, m0F, ISASSE2}
	iPCMPEQB    = sseIns{"pcmpeqb", 0x74, p66, m0F, ISASSE2}
	iPCMPEQW    = sseIns{"pcmpeqw", 0x75, p66, m0F, ISASSE2}
	iPCMPEQD    = sseIns{"pcmpeqd", 0x76, p66, m0F, ISASSE2}
	iPCMPGTB    = sseIns{"pcmpgtb", 0x64, p66, m0F, ISASSE2}
	iPCMPGTW    = sseIns{"pcmpgtw", 0x65, p66, m0F, ISASSE2}
	iPCMPGTD    = sseIns{"pcmpgtd", 0x66, p66, m0F, ISASSE2}
	iPACKSSWB   = sseIns{"packsswb", 0x63, p66, m0F, ISASSE2}
	iPACKSSDW   = sseIns{"packssdw", 0x6B, p66, m0F, ISASSE2}
	iPACKUSWB   = sseIns{"packuswb", 0x67, p66, m0F, ISASSE2}
	iPUNPCKLBW  = sseIns{"punpcklbw", 0x60, p66, m0F, ISASSE2}
	iPUNPCKLWD  = sseIns{"punpcklwd", 0x61, p66, m0F, ISASSE2}
	iPUNPCKLDQ  = sseIns{"punpckldq", 0x62, p66, m0F, ISASSE2}
	iPUNPCKLQDQ = sseIns{"punpcklqdq", 0x6C, p66, m0F, ISASSE2}
	iPUNPCKHBW  = sseIns{"punpckhbw", 0x68, p66, m0F, ISASSE2}
	iPUNPCKHWD  = sseIns{"punpckhwd", 0x69, p66, m0F, ISASSE2}
	iPUNPCKHDQ  = sseIns{"punpckhdq", 0x6A, p66, m0F, ISASSE2}
	iPUNPCKHQDQ = sseIns{"punpckhqdq", 0x6D, p66, m0F, ISASSE2}
	iPMINUB     = sseIns{"pminub", 0xDA, p66, m0F, ISASSE2}
	iPMAXUB     = sseIns{"pmaxub", 0xDE, p66, m0F, ISASSE2}
	iPMINSW     = sseIns{"pminsw", 0xEA, p66, m0F, ISASSE2}
	iPMAXSW     = sseIns{"pmaxsw", 0xEE, p66, m0F, ISASSE2}
	iPAVGB      = sseIns{"pavgb", 0xE0, p66, m0F, ISASSE2}
	iPAVGW      = sseIns{"pavgw", 0xE3, p66, m0F, ISASSE2}
	iPSADBW     = sseIns{"psadbw", 0xF6, p66, m0F, ISASSE2}
	iPSHUFD     = sseIns{"pshufd", 0x70, p66, m0F, ISASSE2}
	iPSHUFLW    = sseIns{"pshuflw", 0x70, pF2, m0F, ISASSE2}
	iPSHUFHW    = sseIns{"pshufhw", 0x70, pF3, m0F, ISASSE2}
	iPSRLWby    = sseIns{"psrlw", 0xD1, p66, m0F, ISASSE2}
	iPSRLDby    = sseIns{"psrld", 0xD2, p66, m0F, ISASSE2}
	iPSRLQby    = sseIns{"psrlq", 0xD3, p66, m0F, ISASSE2}
	iPSRAWby    = sseIns{"psraw", 0xE1, p66, m0F, ISASSE2}
	iPSRADby    = sseIns{"psrad", 0xE2, p66, m0F, ISASSE2}
	iPSLLWby    = sseIns{"psllw", 0xF1, p66, m0F, ISASSE2}
	iPSLLDby    = sseIns{"pslld", 0xF2, p66, m0F, ISASSE2}
	iPSLLQby    = sseIns{"psllq", 0xF3, p66, m0F, ISASSE2}
)

func (a *Buf) PADDB(dst, src1, src2 Reg)   { a.bin(iPADDB, true, dst, src1, src2) }
func (a *Buf) PADDW(dst, src1, src2 Reg)   { a.bin(iPADDW, true, dst, src1, src2) }
func (a *Buf) PADDD(dst, src1, src2 Reg)   { a.bin(iPADDD, true, dst, src1, src2) }
func (a *Buf) PADDQ(dst, src1, src2 Reg)   { a.bin(iPADDQ, true, dst, src1, src2) }
func (a *Buf) PSUBB(dst, src1, src2 Reg)   { a.bin(iPSUBB, false, dst, src1, src2) }
func (a *Buf) PSUBW(dst, src1, src2 Reg)   { a.bin(iPSUBW, false, dst, src1, src2) }
func (a *Buf) PSUBD(dst, src1, src2 Reg)   { a.bin(iPSUBD, false, dst, src1, src2) }
func (a *Buf) PSUBQ(dst, src1, src2 Reg)   { a.bin(iPSUBQ, false, dst, src1, src2) }
func (a *Buf) PMULLW(dst, src1, src2 Reg)  { a.bin(iPMULLW, true, dst, src1, src2) }
func (a *Buf) PMULHW(dst, src1, src2 Reg)  { a.bin(iPMULHW, true, dst, src1, src2) }
func (a *Buf) PMULHUW(dst, src1, src2 Reg) { a.bin(iPMULHUW, true, dst, src1, src2) }
func (a *Buf) PMULUDQ(dst, src1, src2 Reg) { a.bin(iPMULUDQ, true, dst, src1, src2) }

// PMADDWD is int16 x int16 -> int32, adjacent pairs summed: the second half of
// the pre-VNNI int8 dot (PMADDUBSW, PMADDWD against i16 ones, PADDD).
func (a *Buf) PMADDWD(dst, src1, src2 Reg) { a.bin(iPMADDWD, true, dst, src1, src2) }

func (a *Buf) PAND(dst, src1, src2 Reg) { a.bin(iPAND, true, dst, src1, src2) }
func (a *Buf) POR(dst, src1, src2 Reg)  { a.bin(iPOR, true, dst, src1, src2) }
func (a *Buf) PXOR(dst, src1, src2 Reg) { a.bin(iPXOR, true, dst, src1, src2) }

// PANDN is (NOT src1) AND src2.
func (a *Buf) PANDN(dst, src1, src2 Reg) { a.bin(iPANDN, false, dst, src1, src2) }

func (a *Buf) PCMPEQB(dst, src1, src2 Reg) { a.bin(iPCMPEQB, true, dst, src1, src2) }

// PCMPEQW with one register as both sources materialises -1 in every word --
// the i16 ones vector PMADDWD needs, with no memory (PSRLW 15 turns it to +1).
func (a *Buf) PCMPEQW(dst, src1, src2 Reg) { a.bin(iPCMPEQW, true, dst, src1, src2) }
func (a *Buf) PCMPEQD(dst, src1, src2 Reg) { a.bin(iPCMPEQD, true, dst, src1, src2) }

// PCMPGTB/W/D compare SIGNED lanes: src1 > src2.
func (a *Buf) PCMPGTB(dst, src1, src2 Reg) { a.bin(iPCMPGTB, false, dst, src1, src2) }
func (a *Buf) PCMPGTW(dst, src1, src2 Reg) { a.bin(iPCMPGTW, false, dst, src1, src2) }
func (a *Buf) PCMPGTD(dst, src1, src2 Reg) { a.bin(iPCMPGTD, false, dst, src1, src2) }

// The packs narrow with saturation: src1's lanes into the low half, src2's
// into the high.
func (a *Buf) PACKSSWB(dst, src1, src2 Reg) { a.bin(iPACKSSWB, false, dst, src1, src2) }
func (a *Buf) PACKSSDW(dst, src1, src2 Reg) { a.bin(iPACKSSDW, false, dst, src1, src2) }
func (a *Buf) PACKUSWB(dst, src1, src2 Reg) { a.bin(iPACKUSWB, false, dst, src1, src2) }

// The unpacks interleave src1's lanes with src2's, src1 first. PUNPCKLWD(dst,
// zero, x) is bf16 -> f32 for the low four words (each dword = x_word << 16);
// PUNPCKLWD(dst, x, zero) zero-extends four u16.
func (a *Buf) PUNPCKLBW(dst, src1, src2 Reg)  { a.bin(iPUNPCKLBW, false, dst, src1, src2) }
func (a *Buf) PUNPCKLWD(dst, src1, src2 Reg)  { a.bin(iPUNPCKLWD, false, dst, src1, src2) }
func (a *Buf) PUNPCKLDQ(dst, src1, src2 Reg)  { a.bin(iPUNPCKLDQ, false, dst, src1, src2) }
func (a *Buf) PUNPCKLQDQ(dst, src1, src2 Reg) { a.bin(iPUNPCKLQDQ, false, dst, src1, src2) }
func (a *Buf) PUNPCKHBW(dst, src1, src2 Reg)  { a.bin(iPUNPCKHBW, false, dst, src1, src2) }
func (a *Buf) PUNPCKHWD(dst, src1, src2 Reg)  { a.bin(iPUNPCKHWD, false, dst, src1, src2) }
func (a *Buf) PUNPCKHDQ(dst, src1, src2 Reg)  { a.bin(iPUNPCKHDQ, false, dst, src1, src2) }
func (a *Buf) PUNPCKHQDQ(dst, src1, src2 Reg) { a.bin(iPUNPCKHQDQ, false, dst, src1, src2) }

func (a *Buf) PMINUB(dst, src1, src2 Reg) { a.bin(iPMINUB, true, dst, src1, src2) }
func (a *Buf) PMAXUB(dst, src1, src2 Reg) { a.bin(iPMAXUB, true, dst, src1, src2) }
func (a *Buf) PMINSW(dst, src1, src2 Reg) { a.bin(iPMINSW, true, dst, src1, src2) }
func (a *Buf) PMAXSW(dst, src1, src2 Reg) { a.bin(iPMAXSW, true, dst, src1, src2) }
func (a *Buf) PAVGB(dst, src1, src2 Reg)  { a.bin(iPAVGB, true, dst, src1, src2) }
func (a *Buf) PAVGW(dst, src1, src2 Reg)  { a.bin(iPAVGW, true, dst, src1, src2) }
func (a *Buf) PSADBW(dst, src1, src2 Reg) { a.bin(iPSADBW, true, dst, src1, src2) }

// PSHUFD permutes dwords by imm: 0x00 broadcasts lane 0, 0xB1 swaps adjacent
// pairs, 0x4E swaps the halves. PSHUFLW/PSHUFHW permute the low/high four
// words.
func (a *Buf) PSHUFD(dst, src Reg, imm byte)  { a.unImm(iPSHUFD, dst, src, imm) }
func (a *Buf) PSHUFLW(dst, src Reg, imm byte) { a.unImm(iPSHUFLW, dst, src, imm) }
func (a *Buf) PSHUFHW(dst, src Reg, imm byte) { a.unImm(iPSHUFHW, dst, src, imm) }

// The shifts by a COUNT held in the low 64 bits of an XMM register: uniform
// across lanes, like the immediate forms, but decided at run time.
func (a *Buf) PSRLWBy(dst, src, count Reg) { a.bin(iPSRLWby, false, dst, src, count) }
func (a *Buf) PSRLDBy(dst, src, count Reg) { a.bin(iPSRLDby, false, dst, src, count) }
func (a *Buf) PSRLQBy(dst, src, count Reg) { a.bin(iPSRLQby, false, dst, src, count) }
func (a *Buf) PSRAWBy(dst, src, count Reg) { a.bin(iPSRAWby, false, dst, src, count) }
func (a *Buf) PSRADBy(dst, src, count Reg) { a.bin(iPSRADby, false, dst, src, count) }
func (a *Buf) PSLLWBy(dst, src, count Reg) { a.bin(iPSLLWby, false, dst, src, count) }
func (a *Buf) PSLLDBy(dst, src, count Reg) { a.bin(iPSLLDby, false, dst, src, count) }
func (a *Buf) PSLLQBy(dst, src, count Reg) { a.bin(iPSLLQby, false, dst, src, count) }

// The shifts by an immediate, in the 66 0F 71/72/73 group encoding. The
// register is in ModRM.rm and the shift kind in ModRM.reg, and the legacy form
// is in place, so a shift into a different dst is a MOVAPS then the shift.
func (a *Buf) PSRLW(dst, src Reg, imm byte)  { a.shiftImm("psrlw", 0x71, 2, dst, src, imm) }
func (a *Buf) PSRAW(dst, src Reg, imm byte)  { a.shiftImm("psraw", 0x71, 4, dst, src, imm) }
func (a *Buf) PSLLW(dst, src Reg, imm byte)  { a.shiftImm("psllw", 0x71, 6, dst, src, imm) }
func (a *Buf) PSRLD(dst, src Reg, imm byte)  { a.shiftImm("psrld", 0x72, 2, dst, src, imm) }
func (a *Buf) PSRAD(dst, src Reg, imm byte)  { a.shiftImm("psrad", 0x72, 4, dst, src, imm) }
func (a *Buf) PSLLD(dst, src Reg, imm byte)  { a.shiftImm("pslld", 0x72, 6, dst, src, imm) }
func (a *Buf) PSRLQ(dst, src Reg, imm byte)  { a.shiftImm("psrlq", 0x73, 2, dst, src, imm) }
func (a *Buf) PSLLQ(dst, src Reg, imm byte)  { a.shiftImm("psllq", 0x73, 6, dst, src, imm) }
func (a *Buf) PSRLDQ(dst, src Reg, imm byte) { a.shiftImm("psrldq", 0x73, 3, dst, src, imm) }
func (a *Buf) PSLLDQ(dst, src Reg, imm byte) { a.shiftImm("pslldq", 0x73, 7, dst, src, imm) }

func (a *Buf) shiftImm(name string, op byte, ext Reg, dst, src Reg, imm byte) {
	a.useISA(ISASSE2, name)
	a.MOVAPS(dst, src)
	a.sseOpImm(op, p66, m0F, 0, ext, dst, imm)
}

// ---------------------------------------------------------------------------
// SSSE3.

var (
	iPSHUFB    = sseIns{"pshufb", 0x00, p66, m0F38, ISASSSE3}
	iPHADDW    = sseIns{"phaddw", 0x01, p66, m0F38, ISASSSE3}
	iPHADDD    = sseIns{"phaddd", 0x02, p66, m0F38, ISASSSE3}
	iPMADDUBSW = sseIns{"pmaddubsw", 0x04, p66, m0F38, ISASSSE3}
	iPHSUBW    = sseIns{"phsubw", 0x05, p66, m0F38, ISASSSE3}
	iPHSUBD    = sseIns{"phsubd", 0x06, p66, m0F38, ISASSSE3}
	iPSIGNB    = sseIns{"psignb", 0x08, p66, m0F38, ISASSSE3}
	iPSIGNW    = sseIns{"psignw", 0x09, p66, m0F38, ISASSSE3}
	iPSIGND    = sseIns{"psignd", 0x0A, p66, m0F38, ISASSSE3}
	iPMULHRSW  = sseIns{"pmulhrsw", 0x0B, p66, m0F38, ISASSSE3}
	iPABSB     = sseIns{"pabsb", 0x1C, p66, m0F38, ISASSSE3}
	iPABSW     = sseIns{"pabsw", 0x1D, p66, m0F38, ISASSSE3}
	iPABSD     = sseIns{"pabsd", 0x1E, p66, m0F38, ISASSSE3}
	iPALIGNR   = sseIns{"palignr", 0x0F, p66, m0F3A, ISASSSE3}
)

// PSHUFB permutes the bytes of src1 (the TABLE) by src2 (the indices); an
// index with its top bit set yields zero. The legacy destination IS the
// table, so dst == src2 panics: the copy would destroy the indices.
func (a *Buf) PSHUFB(dst, table, idx Reg) { a.bin(iPSHUFB, false, dst, table, idx) }

func (a *Buf) PHADDW(dst, src1, src2 Reg) { a.bin(iPHADDW, false, dst, src1, src2) }
func (a *Buf) PHADDD(dst, src1, src2 Reg) { a.bin(iPHADDD, false, dst, src1, src2) }
func (a *Buf) PHSUBW(dst, src1, src2 Reg) { a.bin(iPHSUBW, false, dst, src1, src2) }
func (a *Buf) PHSUBD(dst, src1, src2 Reg) { a.bin(iPHSUBD, false, dst, src1, src2) }

// PMADDUBSW is UNSIGNED src1 bytes x SIGNED src2 bytes -> int16, adjacent
// pairs summed WITH SATURATION. The legacy destination is the unsigned
// operand, which it clobbers exactly as DotVEX's VPMADDUBSW(u, u, act) does,
// so PreVNNIPacked's saturation bound applies unchanged.
func (a *Buf) PMADDUBSW(dst, unsigned, signed Reg) {
	a.bin(iPMADDUBSW, false, dst, unsigned, signed)
}

// PSIGNB/W/D multiply src1 by the sign of src2 (zero where src2 is zero). It
// is how a signed payload is read without the +128 centring that overflows
// PMADDUBSW (see prevnni.go's signedFix).
func (a *Buf) PSIGNB(dst, src1, src2 Reg) { a.bin(iPSIGNB, false, dst, src1, src2) }
func (a *Buf) PSIGNW(dst, src1, src2 Reg) { a.bin(iPSIGNW, false, dst, src1, src2) }
func (a *Buf) PSIGND(dst, src1, src2 Reg) { a.bin(iPSIGND, false, dst, src1, src2) }

func (a *Buf) PMULHRSW(dst, src1, src2 Reg) { a.bin(iPMULHRSW, true, dst, src1, src2) }

func (a *Buf) PABSB(dst, src Reg) { a.un(iPABSB, dst, src) }
func (a *Buf) PABSW(dst, src Reg) { a.un(iPABSW, dst, src) }
func (a *Buf) PABSD(dst, src Reg) { a.un(iPABSD, dst, src) }

// PALIGNR concatenates src1:src2 (src1 high) and extracts 16 bytes from byte
// offset imm.
func (a *Buf) PALIGNR(dst, src1, src2 Reg, imm byte) {
	a.binImm(iPALIGNR, false, dst, src1, src2, imm)
}

// ---------------------------------------------------------------------------
// SSE4.1 and SSE4.2.

var (
	iPMOVSXBW = sseIns{"pmovsxbw", 0x20, p66, m0F38, ISASSE41}
	iPMOVSXBD = sseIns{"pmovsxbd", 0x21, p66, m0F38, ISASSE41}
	iPMOVSXBQ = sseIns{"pmovsxbq", 0x22, p66, m0F38, ISASSE41}
	iPMOVSXWD = sseIns{"pmovsxwd", 0x23, p66, m0F38, ISASSE41}
	iPMOVSXWQ = sseIns{"pmovsxwq", 0x24, p66, m0F38, ISASSE41}
	iPMOVSXDQ = sseIns{"pmovsxdq", 0x25, p66, m0F38, ISASSE41}
	iPMOVZXBW = sseIns{"pmovzxbw", 0x30, p66, m0F38, ISASSE41}
	iPMOVZXBD = sseIns{"pmovzxbd", 0x31, p66, m0F38, ISASSE41}
	iPMOVZXBQ = sseIns{"pmovzxbq", 0x32, p66, m0F38, ISASSE41}
	iPMOVZXWD = sseIns{"pmovzxwd", 0x33, p66, m0F38, ISASSE41}
	iPMOVZXWQ = sseIns{"pmovzxwq", 0x34, p66, m0F38, ISASSE41}
	iPMOVZXDQ = sseIns{"pmovzxdq", 0x35, p66, m0F38, ISASSE41}
	iPMULDQ   = sseIns{"pmuldq", 0x28, p66, m0F38, ISASSE41}
	iPCMPEQQ  = sseIns{"pcmpeqq", 0x29, p66, m0F38, ISASSE41}
	iPACKUSDW = sseIns{"packusdw", 0x2B, p66, m0F38, ISASSE41}
	iPMINSB   = sseIns{"pminsb", 0x38, p66, m0F38, ISASSE41}
	iPMINSD   = sseIns{"pminsd", 0x39, p66, m0F38, ISASSE41}
	iPMINUW   = sseIns{"pminuw", 0x3A, p66, m0F38, ISASSE41}
	iPMINUD   = sseIns{"pminud", 0x3B, p66, m0F38, ISASSE41}
	iPMAXSB   = sseIns{"pmaxsb", 0x3C, p66, m0F38, ISASSE41}
	iPMAXSD   = sseIns{"pmaxsd", 0x3D, p66, m0F38, ISASSE41}
	iPMAXUW   = sseIns{"pmaxuw", 0x3E, p66, m0F38, ISASSE41}
	iPMAXUD   = sseIns{"pmaxud", 0x3F, p66, m0F38, ISASSE41}
	iPMULLD   = sseIns{"pmulld", 0x40, p66, m0F38, ISASSE41}
	iPTEST    = sseIns{"ptest", 0x17, p66, m0F38, ISASSE41}
	iPBLENDVB = sseIns{"pblendvb", 0x10, p66, m0F38, ISASSE41}
	iPBLENDW  = sseIns{"pblendw", 0x0E, p66, m0F3A, ISASSE41}
	iPCMPGTQ  = sseIns{"pcmpgtq", 0x37, p66, m0F38, ISASSE42}
)

// The sign- and zero-extensions, register and memory forms. The memory form
// reads only what it widens -- 4 bytes for BD, 8 for BW/WD, 2 for BQ -- so it
// never reads past a tail, and the register form spreads the low bytes of a
// register already shifted into place (the replacement for VPSRLVD in the
// secondary-plane unpack).
func (a *Buf) PMOVSXBW(dst, src Reg)       { a.un(iPMOVSXBW, dst, src) }
func (a *Buf) PMOVSXBD(dst, src Reg)       { a.un(iPMOVSXBD, dst, src) }
func (a *Buf) PMOVSXBQ(dst, src Reg)       { a.un(iPMOVSXBQ, dst, src) }
func (a *Buf) PMOVSXWD(dst, src Reg)       { a.un(iPMOVSXWD, dst, src) }
func (a *Buf) PMOVSXWQ(dst, src Reg)       { a.un(iPMOVSXWQ, dst, src) }
func (a *Buf) PMOVSXDQ(dst, src Reg)       { a.un(iPMOVSXDQ, dst, src) }
func (a *Buf) PMOVZXBW(dst, src Reg)       { a.un(iPMOVZXBW, dst, src) }
func (a *Buf) PMOVZXBD(dst, src Reg)       { a.un(iPMOVZXBD, dst, src) }
func (a *Buf) PMOVZXBQ(dst, src Reg)       { a.un(iPMOVZXBQ, dst, src) }
func (a *Buf) PMOVZXWD(dst, src Reg)       { a.un(iPMOVZXWD, dst, src) }
func (a *Buf) PMOVZXWQ(dst, src Reg)       { a.un(iPMOVZXWQ, dst, src) }
func (a *Buf) PMOVZXDQ(dst, src Reg)       { a.un(iPMOVZXDQ, dst, src) }
func (a *Buf) PMOVSXBWLoad(dst Reg, m Mem) { a.mem(iPMOVSXBW, dst, m) }
func (a *Buf) PMOVSXBDLoad(dst Reg, m Mem) { a.mem(iPMOVSXBD, dst, m) }
func (a *Buf) PMOVSXBQLoad(dst Reg, m Mem) { a.mem(iPMOVSXBQ, dst, m) }
func (a *Buf) PMOVSXWDLoad(dst Reg, m Mem) { a.mem(iPMOVSXWD, dst, m) }
func (a *Buf) PMOVSXWQLoad(dst Reg, m Mem) { a.mem(iPMOVSXWQ, dst, m) }
func (a *Buf) PMOVSXDQLoad(dst Reg, m Mem) { a.mem(iPMOVSXDQ, dst, m) }
func (a *Buf) PMOVZXBWLoad(dst Reg, m Mem) { a.mem(iPMOVZXBW, dst, m) }
func (a *Buf) PMOVZXBDLoad(dst Reg, m Mem) { a.mem(iPMOVZXBD, dst, m) }
func (a *Buf) PMOVZXBQLoad(dst Reg, m Mem) { a.mem(iPMOVZXBQ, dst, m) }
func (a *Buf) PMOVZXWDLoad(dst Reg, m Mem) { a.mem(iPMOVZXWD, dst, m) }
func (a *Buf) PMOVZXWQLoad(dst Reg, m Mem) { a.mem(iPMOVZXWQ, dst, m) }
func (a *Buf) PMOVZXDQLoad(dst Reg, m Mem) { a.mem(iPMOVZXDQ, dst, m) }

func (a *Buf) PMULLD(dst, src1, src2 Reg)   { a.bin(iPMULLD, true, dst, src1, src2) }
func (a *Buf) PMULDQ(dst, src1, src2 Reg)   { a.bin(iPMULDQ, true, dst, src1, src2) }
func (a *Buf) PCMPEQQ(dst, src1, src2 Reg)  { a.bin(iPCMPEQQ, true, dst, src1, src2) }
func (a *Buf) PACKUSDW(dst, src1, src2 Reg) { a.bin(iPACKUSDW, false, dst, src1, src2) }
func (a *Buf) PMINSB(dst, src1, src2 Reg)   { a.bin(iPMINSB, true, dst, src1, src2) }
func (a *Buf) PMINSD(dst, src1, src2 Reg)   { a.bin(iPMINSD, true, dst, src1, src2) }
func (a *Buf) PMINUW(dst, src1, src2 Reg)   { a.bin(iPMINUW, true, dst, src1, src2) }
func (a *Buf) PMINUD(dst, src1, src2 Reg)   { a.bin(iPMINUD, true, dst, src1, src2) }
func (a *Buf) PMAXSB(dst, src1, src2 Reg)   { a.bin(iPMAXSB, true, dst, src1, src2) }
func (a *Buf) PMAXSD(dst, src1, src2 Reg)   { a.bin(iPMAXSD, true, dst, src1, src2) }
func (a *Buf) PMAXUW(dst, src1, src2 Reg)   { a.bin(iPMAXUW, true, dst, src1, src2) }
func (a *Buf) PMAXUD(dst, src1, src2 Reg)   { a.bin(iPMAXUD, true, dst, src1, src2) }

// PCMPGTQ compares signed 64-bit lanes (SSE4.2).
func (a *Buf) PCMPGTQ(dst, src1, src2 Reg) { a.bin(iPCMPGTQ, false, dst, src1, src2) }

// PTEST sets ZF when (x AND y) is zero and CF when (NOT x AND y) is zero; it
// writes no register.
func (a *Buf) PTEST(x, y Reg) { a.un(iPTEST, x, y) }

// PBLENDVB takes byte i from src2 where XMM0's byte i has its top bit set,
// else from src1 (SSE4.1). XMM0 is the implicit mask; see BLENDVPS.
func (a *Buf) PBLENDVB(dst, src1, src2 Reg) {
	if dst == XMM0 && src1 != XMM0 {
		panic("jit: PBLENDVB into XMM0 would overwrite its implicit mask")
	}
	a.bin(iPBLENDVB, false, dst, src1, src2)
}

// PBLENDW takes word i from src2 where imm bit i is set, else from src1.
func (a *Buf) PBLENDW(dst, src1, src2 Reg, imm byte) {
	a.binImm(iPBLENDW, false, dst, src1, src2, imm)
}

// POPCNT counts the set bits of a 64-bit GP register (F3 REX.W 0F B8).
func (a *Buf) POPCNT(dst, src Reg) {
	a.useISA(ISAPOPCNT, "popcnt")
	a.sseOp(0xB8, pF3, m0F, 1, dst, src)
}

// ---------------------------------------------------------------------------
// Packed double, and the narrow stores. The image path's float64 work (the
// resampler's sin/cos table, imgproc_sse.go) is two lanes a register here,
// and the byte stores write exactly the bytes a pixel owns.

var (
	iADDPD     = sseIns{"addpd", 0x58, p66, m0F, ISASSE2}
	iSUBPD     = sseIns{"subpd", 0x5C, p66, m0F, ISASSE2}
	iMULPD     = sseIns{"mulpd", 0x59, p66, m0F, ISASSE2}
	iDIVPD     = sseIns{"divpd", 0x5E, p66, m0F, ISASSE2}
	iROUNDPD   = sseIns{"roundpd", 0x09, p66, m0F3A, ISASSE41}
	iPEXTRWmem = sseIns{"pextrw", 0x15, p66, m0F3A, ISASSE41}
)

func (a *Buf) ADDPD(dst, src1, src2 Reg) { a.bin(iADDPD, true, dst, src1, src2) }
func (a *Buf) SUBPD(dst, src1, src2 Reg) { a.bin(iSUBPD, false, dst, src1, src2) }
func (a *Buf) MULPD(dst, src1, src2 Reg) { a.bin(iMULPD, true, dst, src1, src2) }
func (a *Buf) DIVPD(dst, src1, src2 Reg) { a.bin(iDIVPD, false, dst, src1, src2) }

// ROUNDPD is ROUNDPS on two float64 lanes (SSE4.1): imm 0 nearest-even, 1
// down.
func (a *Buf) ROUNDPD(dst, src Reg, imm byte) { a.unImm(iROUNDPD, dst, src, imm) }

// MOVDDUPLoad reads one float64 and writes it to both lanes (SSE3): a double
// broadcast from a constant block.
func (a *Buf) MOVDDUPLoad(dst Reg, m Mem) { a.mem(iMOVDDUP, dst, m) }

// PEXTRBStore and PEXTRWStore write byte or word imm of src to memory
// (SSE4.1's memory forms): one sample, or two, and nothing past them.
func (a *Buf) PEXTRBStore(m Mem, src Reg, imm byte) {
	a.useISA(iPEXTRB.isa, iPEXTRB.name)
	a.sseOpMemImm(iPEXTRB.op, iPEXTRB.pp, iPEXTRB.mm, 0, src, m, imm)
}
func (a *Buf) PEXTRWStore(m Mem, src Reg, imm byte) {
	a.useISA(iPEXTRWmem.isa, iPEXTRWmem.name)
	a.sseOpMemImm(iPEXTRWmem.op, iPEXTRWmem.pp, iPEXTRWmem.mm, 0, src, m, imm)
}

// MOVZXBLoad is movzx r32, byte [mem]: one byte into a GP register, the upper
// bits cleared -- an index a table is read at.
func (a *Buf) MOVZXBLoad(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(false, dst, idx, m.Base)
	a.u8(0x0F)
	a.u8(0xB6)
	a.modrmMem(dst, m)
}
