package cpu

// Legacy SSE encoding, for hosts that have no AVX at all.
//
// This cannot be VEX at L=128: a Goldmont Atom faults on a VEX prefix
// whatever the L bit says, so a 128-bit tier needs the legacy form (optional
// 66/F3/F2 prefix, optional REX, the 0F/0F38/0F3A map, opcode, modrm).
//
// The real difference is arity: VEX is three-operand, legacy SSE is
// two-operand and destructive. The emitters stay in the three-operand form,
// and the gap is closed here by materialising a MOVAPS when dst is not already
// src1.
//
// The prefix and map fields are the ones vex() takes (VEX packs them into
// bits, this unpacks them into bytes), so every op definition describes both
// encodings with no second table.

// ssePrefix is the legacy byte for each pp value; pp0 emits nothing.
var ssePrefix = [4]byte{0, 0x66, 0xF3, 0xF2}

// XMM0..XMM15 name the vector registers in legacy-SSE code. They are the same
// numbers as Y0..Y15 (and as the GP registers), exactly as in the hardware;
// the separate names only say "this is 128-bit legacy code" at the call site.
// X0..X15 is taken by the A64 encoder's general registers.
const (
	XMM0 Reg = iota
	XMM1
	XMM2
	XMM3
	XMM4
	XMM5
	XMM6
	XMM7
	XMM8
	XMM9
	XMM10
	XMM11
	XMM12
	XMM13
	XMM14
	XMM15
)

// sseOp emits one legacy-SSE instruction in register-register form:
//
//	[66|F3|F2] [REX] 0F [38|3A] opcode modrm(dst, src)
//
// w selects REX.W. It is the operand-size bit, not a width: no SSE vector op
// here needs it, and it is carried so the op table can stay shared with vex().
func (a *Buf) sseOp(op byte, pp, mm, w byte, dst, src Reg) {
	a.regs.vec(dst)
	if p := ssePrefix[pp]; p != 0 {
		a.u8(p)
	}
	// REX is emitted only when a high register or W demands it: an unnecessary
	// REX is legal but changes the bytes, and these are pinned against objdump.
	if r := rexFor(dst, src, w); r != 0 {
		a.u8(r)
	}
	a.sseEsc(op, mm)
	a.u8(0xC0 | byte(dst&7)<<3 | byte(src&7))
}

// rexFor returns the REX byte for a reg-reg SSE op, or 0 when none is needed.
// dst is ModRM.reg (REX.R) and src is ModRM.rm (REX.B).
func rexFor(dst, src Reg, w byte) byte {
	var r byte
	if w != 0 {
		r |= 0x08
	}
	if dst.hi() != 0 {
		r |= 0x04
	}
	if src.hi() != 0 {
		r |= 0x01
	}
	if r == 0 {
		return 0
	}
	return 0x40 | r
}

// sseEsc emits the 0F [38|3A] escape and the opcode.
func (a *Buf) sseEsc(op, mm byte) {
	a.u8(0x0F)
	switch mm {
	case m0F38:
		a.u8(0x38)
	case m0F3A:
		a.u8(0x3A)
	}
	a.u8(op)
}

// sseOpImm is sseOp followed by an imm8: the SHUFPS/PSHUFD/CMPPS/BLENDPS
// shape.
func (a *Buf) sseOpImm(op byte, pp, mm, w byte, dst, src Reg, imm byte) {
	a.sseOp(op, pp, mm, w, dst, src)
	a.u8(imm)
}

// sseOpMem emits one legacy-SSE instruction with a memory operand:
//
//	[66|F3|F2] [REX] 0F [38|3A] opcode modrm/sib/disp(reg, m)
//
// Only the alignment-free forms reach it. A legacy packed op with an m128
// operand faults on an address that is not 16-byte aligned (VEX never did),
// so there is no ADDPSMem or PANDMem here: memory reaches the SSE tier
// through MOVUPS/MOVDQU/LDDQU, the m32/m64/m16 forms (MOVSS, MOVD, MOVQ,
// PINSRW, PINSRD, PMOVZX/PMOVSX, scalar ADDSS/MULSS...), and PREFETCH.
func (a *Buf) sseOpMem(op byte, pp, mm, w byte, reg Reg, m Mem) {
	if p := ssePrefix[pp]; p != 0 {
		a.u8(p)
	}
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(w != 0, reg, idx, m.Base)
	a.sseEsc(op, mm)
	a.modrmMem(reg, m)
}

// sseOpMemImm is sseOpMem followed by an imm8 (PINSRW, PINSRD, PEXTRD to
// memory).
func (a *Buf) sseOpMemImm(op byte, pp, mm, w byte, reg Reg, m Mem, imm byte) {
	a.sseOpMem(op, pp, mm, w, reg, m)
	a.u8(imm)
}

// sse3Imm is sse3 for an op that also takes an imm8 (SHUFPS, CMPPS, BLENDPS,
// PALIGNR, PBLENDW, INSERTPS, ROUNDSS). The copy rules are sse3's exactly.
func (a *Buf) sse3Imm(op byte, pp, mm, w byte, commutes bool, dst, src1, src2 Reg, imm byte) {
	switch {
	case dst == src1:
	case dst == src2 && commutes:
		src2 = src1
	case dst == src2:
		panic("jit: sse3: dst aliases src2 on a non-commutative op; " +
			"this needs a scratch register the encoder does not own")
	default:
		a.MOVAPS(dst, src1)
	}
	a.sseOpImm(op, pp, mm, w, dst, src2, imm)
}

// MOVAPS is movaps dst, src -- the register copy the two-operand form needs.
//
// APS rather than APD or DQA because it is the shortest (no 66 prefix); all
// three copy 16 bytes at the same latency on every part that runs this tier.
//
// It is register-to-register only: MOVAPS with memory #GPs on a misaligned
// address, and MOVUPSLoad/MOVUPSStore are the memory forms.
func (a *Buf) MOVAPS(dst, src Reg) {
	if dst == src {
		return
	}
	a.useISA(ISASSE2, "movaps")
	a.sseOp(0x28, 0, m0F, 0, dst, src)
}

// sse3 renders a THREE-operand op into the two-operand form, materialising the
// copy only when it is needed.
//
// dst == src1 is the common case (VADDPS(acc, acc, x)) and emits no copy.
// When dst == src2 and the op commutes, the operands swap; otherwise one
// MOVAPS is correct for any op.
func (a *Buf) sse3(op byte, pp, mm, w byte, commutes bool, dst, src1, src2 Reg) {
	switch {
	case dst == src1:
		a.sseOp(op, pp, mm, w, dst, src2)
	case dst == src2 && commutes:
		a.sseOp(op, pp, mm, w, dst, src1)
	case dst == src2:
		// dst == src2 on a non-commutative op: MOVAPS(dst, src1) would
		// overwrite src2 and compute src1 op src1. Serving it needs a
		// scratch register this encoder has no authority to take, so it
		// panics at emit time rather than emit a wrong answer.
		panic("jit: sse3: dst aliases src2 on a non-commutative op; " +
			"this needs a scratch register the encoder does not own")
	default:
		a.MOVAPS(dst, src1)
		a.sseOp(op, pp, mm, w, dst, src2)
	}
}
