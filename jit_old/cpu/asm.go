// Package cpu emits x86-64 and arm64 machine code at load time.
//
// Not a JIT in the tiering sense: every kernel shape a model needs is known
// before the first token, so all of them are emitted eagerly and never
// revisited. There is no deopt, no hotness counter, and no on-disk code cache.
//
// jitllm owns its encoder because it has to: Go's assembler (and GNU as by
// default) emits only the EVEX encoding of VPDPBUSD, which is AVX-512-VNNI and
// SIGILLs on a CPU with AVX-VNNI but no AVX-512. Only the VEX form runs there.
// Owning the bytes makes the instruction set "what the CPU supports" rather
// than "what the toolchain supports".
package cpu

import "fmt"

// Reg is a register number, 0-15. The same numbering serves GP and YMM; which
// file it names is decided by the instruction, exactly as in the hardware.
type Reg uint8

// General-purpose registers.
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

// Vector registers. Y0..Y15 alias the GP numbering by design.
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

// noIndex marks an addressing mode with no index register. RSP's encoding is
// stolen for "no index" by the hardware, and RSP genuinely cannot be an index,
// so the two facts cancel.
const noIndex Reg = RSP

// Mem is [base + index*scale + disp]. Scale must be 1, 2, 4 or 8.
type Mem struct {
	Base  Reg
	Index Reg
	Scale uint8
	Disp  int32
}

// At is [base + disp].
func At(base Reg, disp int32) Mem { return Mem{Base: base, Index: noIndex, Scale: 1, Disp: disp} }

// Idx is [base + index*scale + disp].
func Idx(base, index Reg, scale uint8, disp int32) Mem {
	return Mem{Base: base, Index: index, Scale: scale, Disp: disp}
}

func (m Mem) hasIndex() bool { return m.Index != noIndex }

func (m Mem) scaleBits() byte {
	switch m.Scale {
	case 1:
		return 0
	case 2:
		return 1
	case 4:
		return 2
	case 8:
		return 3
	}
	panic(fmt.Sprintf("jit: bad scale %d, want 1, 2, 4 or 8", m.Scale))
}

// Buf accumulates emitted bytes.
type Buf struct {
	b      []byte
	fixup  []fixup
	labels []int

	// isa and declared are the kernel's in-band ISA declaration (isa.go).
	// They change no byte of an undeclared buffer; they arm emit-time checks
	// in a declared one.
	isa      ISA
	declared bool

	// regs is what this kernel asked of the register file and the stack,
	// counted as it was emitted (regreport.go).
	regs regUse
}

type fixup struct {
	at   int // offset of the rel32/rel8 field
	size int // 1 or 4
	lbl  int
}

// Bytes returns the emitted code. Panics if a label was used but never bound —
// a dangling jump is a silent infinite loop or a jump into data, so it must not
// be possible to get bytes out.
func (a *Buf) Bytes() []byte {
	for _, f := range a.fixup {
		if f.lbl >= len(a.labels) || a.labels[f.lbl] < 0 {
			panic("jit: label used but never bound")
		}
	}
	noteEmit(a.regs, len(a.b), a.regs.vex)
	return a.b
}

// Len is the current offset, which is what a label binds to.
func (a *Buf) Len() int { return len(a.b) }

func (a *Buf) u8(v byte)    { a.b = append(a.b, v) }
func (a *Buf) u32(v uint32) { a.b = append(a.b, byte(v), byte(v>>8), byte(v>>16), byte(v>>24)) }

// ---------------------------------------------------------------------------
// Prefixes.
//
// REX and VEX are alternatives, never both: VEX carries its own R/X/B bits, so
// modrm takes no prefix argument and is called after whichever prefix the
// instruction needs.

// rex emits a REX prefix if one is required, or always when w is set.
// r is ModRM.reg, x is the SIB index, b is ModRM.rm or the base.
func (a *Buf) rex(w bool, r, x, b Reg) {
	v := byte(0x40) | r.hi()<<2 | x.hi()<<1 | b.hi()
	if w {
		v |= 0x08
	}
	if v != 0x40 {
		a.u8(v)
	}
}

// VEX map and prefix selectors.
const (
	m0F   = 1 // 0x0F
	m0F38 = 2 // 0x0F38
	m0F3A = 3 // 0x0F3A

	pNone = 0
	p66   = 1
	pF3   = 2
	pF2   = 3
)

// vex emits a VEX prefix.
//
//	r  ModRM.reg
//	v  the vvvv source operand (use Y0 when the instruction has none — the field
//	   is inverted, so an unused vvvv must be encoded as 1111, i.e. register 0)
//	b  ModRM.rm, or the memory base
//	x  the SIB index, or noIndex
//
// R, X, B and vvvv are all stored inverted. Getting one backwards does not
// fault: it silently names a different register, which only a disassembly diff
// finds.
func (a *Buf) vex(r, v, b, x Reg, l256 bool, pp, mm, w byte) {
	a.noVEXInSSE("a VEX-encoded instruction")
	a.regs.vex = true
	var lb byte
	if l256 {
		lb = 1
	}
	// The 2-byte form can only express X=0, B=0, W=0 and the 0F map.
	if x.hi() == 0 && b.hi() == 0 && w == 0 && mm == m0F {
		a.u8(0xC5)
		a.u8((^r.hi()&1)<<7 | (^byte(v)&15)<<3 | lb<<2 | pp)
		return
	}
	a.u8(0xC4)
	a.u8((^r.hi()&1)<<7 | (^x.hi()&1)<<6 | (^b.hi()&1)<<5 | mm)
	a.u8(w<<7 | (^byte(v)&15)<<3 | lb<<2 | pp)
}

// ---------------------------------------------------------------------------
// ModRM / SIB.

// modrmReg emits ModRM for a register-register form.
func (a *Buf) modrmReg(reg, rm Reg) { a.u8(0xC0 | reg.lo()<<3 | rm.lo()) }

// modrmMem emits ModRM, SIB and displacement for a memory operand.
//
// Four hardware quirks, none of which is optional and each of which is a wrong
// address rather than an error if missed:
//
//   - RBP and R13 cannot use mod=00: that encoding means RIP-relative. They must
//     carry an explicit disp8 of zero.
//   - RSP and R12 always need a SIB byte, even with no index, because their rm
//     encoding is what selects "there is a SIB byte".
//   - RSP cannot be an index at all; its index encoding means "no index".
//   - disp8 when it fits, disp32 otherwise. Purely a size win, but it is most of
//     the code size in an unrolled kernel.
func (a *Buf) modrmMem(reg Reg, m Mem) {
	if m.Base == RSP {
		a.regs.stack++
	}
	const rmSIB = 4
	base := m.Base.lo()
	needSIB := m.hasIndex() || base == rmSIB // rsp or r12
	forceDisp8 := base == 5                  // rbp or r13
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
				panic("jit: rsp cannot be a SIB index register")
			}
			idx = m.Index.lo()
			ss = m.scaleBits()
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

// ---------------------------------------------------------------------------
// Vector instructions.
//
// Every encoding below is transcribed from the system assembler's output, not
// from memory, and jit_old/cpu/asm_test.go re-checks each one against objdump.

// vop emits a three-operand 256-bit VEX instruction: dst = op(src1, src2).
func (a *Buf) vop(op byte, pp, mm, w byte, dst, src1, src2 Reg) {
	a.vopL(op, pp, mm, w, true, dst, src1, src2)
}

// vopL is vop with an explicit width. The 128-bit forms exist for the tail of a
// horizontal reduction, where the upper lane has already been folded in and
// operating on it would just burn power.
func (a *Buf) vopL(op byte, pp, mm, w byte, l256 bool, dst, src1, src2 Reg) {
	a.regs.vec(dst, src1, src2)
	a.vex(dst, src1, src2, noIndex, l256, pp, mm, w)
	a.u8(op)
	a.modrmReg(dst, src2)
}

// vopMem emits a three-operand VEX instruction whose second source is memory.
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

// VPDPBUSD is the AVX-VNNI int8 dot product: dst += sum of four
// uint8 x int8 products per 32-bit lane. The reason this package exists.
//
// Note the operand asymmetry, which decides the whole weight layout: the FIRST
// source is treated as UNSIGNED bytes and the second as SIGNED. Quantized
// weights are signed and activations can be made unsigned, or vice versa, but
// one side must be biased and the choice is not free.
func (a *Buf) VPDPBUSD(dst, src1, src2 Reg) { a.vop(0x50, p66, m0F38, 0, dst, src1, src2) }

// VPDPBUSDMem is VPDPBUSD reading its second source straight from memory, which
// is how weights are consumed: no separate load, no register pressure.
func (a *Buf) VPDPBUSDMem(dst, src1 Reg, m Mem) { a.vopMem(0x50, p66, m0F38, 0, dst, src1, m) }

// VPMADDUBSW is the pre-VNNI fallback: uint8 x int8 -> int16, adjacent pairs
// summed, with saturation. Followed by VPMADDWD against an i16 all-ones vector
// and a VPADDD it is exactly VPDPBUSD; jit_old/cpu/prevnni.go is the one place
// that sequence is written.
//
// The saturation bound is per format and per plane (worst adjacent pair is
// 127*(u0+u1) against 32767). The nibble and hi-plane formats fit, Q6_K the
// tightest at 37% of the range; Q8_0's +128-centred byte saturates, so
// prevnni.go computes it as |w| x (a*sign(w)) through VPSIGNB instead
// (2*128*127 = 32,512 fits). PreVNNIPacked re-derives the bound at emit time.
func (a *Buf) VPMADDUBSW(dst, src1, src2 Reg) { a.vop(0x04, p66, m0F38, 0, dst, src1, src2) }

// VPSIGNB applies the sign of each byte of src2 to the corresponding byte of
// src1: dst = src1 * sign(src2), and 0 where src2 is 0. SSSE3/AVX2 baseline.
//
// It is how the pre-VNNI path reads a signed payload without the +128 centring
// that overflows VPMADDUBSW: |w| is at most 128 (VPSIGNB of -128 by itself is
// -128, which as an unsigned byte is 128 and pairs with a negated activation,
// so the product is still exact), and the activation carries the sign instead.
func (a *Buf) VPSIGNB(dst, src1, src2 Reg) { a.vop(0x08, p66, m0F38, 0, dst, src1, src2) }

// VPCMPEQW compares int16 lanes for equality, writing all-ones where they are
// equal. With one register as both sources it materialises -1 in every int16
// lane in one instruction and no memory, which is how the i16 ones vector
// VPMADDWD needs is built where no register can be spared to hold a pointer to
// it (see prevnni.go's ones16).
func (a *Buf) VPCMPEQW(dst, src1, src2 Reg) { a.vop(0x75, p66, m0F, 0, dst, src1, src2) }

// VPCMPEQD and VPCMPGTD are the int32 equality and signed greater-than (the
// VEX twins of SSE2's PCMPEQD and PCMPGTD).
//
// The router's top-k compares expert ids in integer, not float: an id read as
// float32 is a denormal, and under MXCSR's FTZ/DAZ (process state this package
// does not own) every id reads as zero and every comparison ties.
func (a *Buf) VPCMPEQD(dst, src1, src2 Reg) { a.vop(0x76, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VPCMPGTD(dst, src1, src2 Reg) { a.vop(0x66, p66, m0F, 0, dst, src1, src2) }

// VPMADDWD is int16 x int16 -> int32, pairwise added. Follows VPMADDUBSW to
// widen out of the saturating range.
func (a *Buf) VPMADDWD(dst, src1, src2 Reg) { a.vop(0xF5, p66, m0F, 0, dst, src1, src2) }

// VPMADDWDMem reads VPMADDWD's second source from memory. The pre-VNNI prefill
// GEMM widens its int16 partial sums against the ones vector PackedScratch
// holds at offset 32, and at eight stationary weight registers there is no
// register left to keep that vector in.
func (a *Buf) VPMADDWDMem(dst, src1 Reg, m Mem) { a.vopMem(0xF5, p66, m0F, 0, dst, src1, m) }

// VPABSB is the byte-wise absolute value. The pre-VNNI GEMM holds a SIGNED
// payload in its weight registers and needs |w| for VPMADDUBSW's unsigned
// operand once per token; the payload itself is what VPSIGNB takes the sign
// from, so it cannot be replaced by |w| in place.
func (a *Buf) VPABSB(dst, src Reg) { a.vop(0x1C, p66, m0F38, 0, dst, Y0, src) }

func (a *Buf) VPADDD(dst, src1, src2 Reg) { a.vop(0xFE, p66, m0F, 0, dst, src1, src2) }

// VPADDDMem is VPADDD with the addend in memory, which is how an accumulator
// that lives in Scratch is read and written in two instructions instead of
// three.
func (a *Buf) VPADDDMem(dst, src1 Reg, m Mem) { a.vopMem(0xFE, p66, m0F, 0, dst, src1, m) }
func (a *Buf) VPADDW(dst, src1, src2 Reg)     { a.vop(0xFD, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VPSUBB(dst, src1, src2 Reg)     { a.vop(0xF8, p66, m0F, 0, dst, src1, src2) }

func (a *Buf) VPSUBBMem(dst, src1 Reg, m Mem) { a.vopMem(0xF8, p66, m0F, 0, dst, src1, m) }
func (a *Buf) VPAND(dst, src1, src2 Reg)      { a.vop(0xDB, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VPOR(dst, src1, src2 Reg)       { a.vop(0xEB, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VPANDN(dst, src1, src2 Reg)     { a.vop(0xDF, p66, m0F, 0, dst, src1, src2) }

// VPMINSD is signed 32-bit lane minimum (SSE4.1's PMINSD, VEX-encoded). The
// argmax fold needs it to pick the LOWEST index among the lanes that tie for
// the maximum, which is the rule model.Greedy's `v > best` keeps.
func (a *Buf) VPMINSD(dst, src1, src2 Reg) { a.vop(0x39, p66, m0F38, 0, dst, src1, src2) }
func (a *Buf) VPXOR(dst, src1, src2 Reg)   { a.vop(0xEF, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VADDPS(dst, src1, src2 Reg)  { a.vop(0x58, pNone, m0F, 0, dst, src1, src2) }
func (a *Buf) VMULPS(dst, src1, src2 Reg)  { a.vop(0x59, pNone, m0F, 0, dst, src1, src2) }
func (a *Buf) VPHADDD(dst, src1, src2 Reg) { a.vop(0x02, p66, m0F38, 0, dst, src1, src2) }

// VFMADD231PS is dst += src1 * src2, single rounding.
func (a *Buf) VFMADD231PS(dst, src1, src2 Reg) { a.vop(0xB8, p66, m0F38, 0, dst, src1, src2) }

// VFMADD231PSMem is the same fused multiply-add reading its second source from
// memory. Attention's dot product wants it: the query vector is reused for
// every cached position, so it stays L1-hot and costs nothing to re-read, where
// pinning it in registers would cap the head dimension at what the register
// file holds.
func (a *Buf) VFMADD231PSMem(dst, src1 Reg, m Mem) { a.vopMem(0xB8, p66, m0F38, 0, dst, src1, m) }

// VPSRLW / VPSRLD shift by an immediate. These use the /2 group encoding, where
// ModRM.reg is an opcode extension and the DESTINATION travels in vvvv — the one
// family where the operands are not where they look like they should be.
func (a *Buf) VPSRLW(dst, src Reg, imm byte) { a.vshiftImm(0x71, 2, dst, src, imm) }
func (a *Buf) VPSRLD(dst, src Reg, imm byte) { a.vshiftImm(0x72, 2, dst, src, imm) }
func (a *Buf) VPSLLW(dst, src Reg, imm byte) { a.vshiftImm(0x71, 6, dst, src, imm) }
func (a *Buf) VPSLLD(dst, src Reg, imm byte) { a.vshiftImm(0x72, 6, dst, src, imm) }

// VPSRAD is the arithmetic right shift of the same family (/4 where VPSRLD is
// /2 and VPSLLD is /6). Shifting a bit into the sign and back down by 31 turns
// it into a whole-lane mask: the branchless select the rotary table's quadrant
// fixup needs, without a broadcast constant.
func (a *Buf) VPSRAD(dst, src Reg, imm byte) { a.vshiftImm(0x72, 4, dst, src, imm) }

func (a *Buf) vshiftImm(op byte, ext, dst, src Reg, imm byte) {
	a.vex(Reg(ext), dst, src, noIndex, true, p66, m0F, 0)
	a.u8(op)
	a.modrmReg(Reg(ext), src)
	a.u8(imm)
}

// VMOVDQU loads and stores 256 unaligned bits. Unaligned by default: quantized
// block sizes are 18, 110 and 210 bytes, so nothing in a GGUF is 32-byte aligned
// and insisting on alignment would mean copying every weight.
func (a *Buf) VMOVDQULoad(dst Reg, m Mem)  { a.vopMem(0x6F, pF3, m0F, 0, dst, Y0, m) }
func (a *Buf) VMOVDQUStore(m Mem, src Reg) { a.vopMem(0x7F, pF3, m0F, 0, src, Y0, m) }

// VPBROADCASTD splats one 32-bit value across all eight lanes — how a single
// activation value reaches every lane of a weight vector.
func (a *Buf) VPBROADCASTD(dst Reg, m Mem) { a.vopMem(0x58, p66, m0F38, 0, dst, Y0, m) }
func (a *Buf) VBROADCASTSS(dst Reg, m Mem) { a.vopMem(0x18, p66, m0F38, 0, dst, Y0, m) }

// VCVTDQ2PS converts int32 lanes to float32. The last step of an integer dot
// product, before the scale is applied.
func (a *Buf) VCVTDQ2PS(dst, src Reg) { a.vop(0x5B, pNone, m0F, 0, dst, Y0, src) }

// VSQRTPS and VDIVPS are what a norm needs and a matvec never did: an RMSNorm
// ends in 1/sqrt(mean+eps), which is one scalar per vector of work. Two-operand
// VEX forms, so vvvv is unused and Y0 stands in for it exactly as VCVTDQ2PS
// does above.
func (a *Buf) VSQRTPS(dst, src Reg)       { a.vop(0x51, pNone, m0F, 0, dst, Y0, src) }
func (a *Buf) VDIVPS(dst, src1, src2 Reg) { a.vop(0x5E, pNone, m0F, 0, dst, src1, src2) }

// The exp building blocks. A softmax and a SiLU both reduce to exp, and exp on
// AVX2 is: range-reduce with a round-to-int, evaluate a polynomial by Horner,
// then scale by 2^n built directly in the exponent field. That needs a rounding
// convert, a Horner-shaped FMA (213: dst = src1*dst + src2, where 231 accumulates
// instead), and max/min for the reduction and the clamp.
func (a *Buf) VMAXPS(dst, src1, src2 Reg) { a.vop(0x5F, pNone, m0F, 0, dst, src1, src2) }

// VMOVAPS between registers, which Horner needs to seed its accumulator from a
// constant that has to survive the loop.
func (a *Buf) VMOVAPSReg(dst, src Reg)    { a.vop(0x28, pNone, m0F, 0, dst, Y0, src) }
func (a *Buf) VMINPS(dst, src1, src2 Reg) { a.vop(0x5D, pNone, m0F, 0, dst, src1, src2) }

// VCVTPS2DQ rounds to nearest under the default MXCSR, which is exactly the
// round() a range reduction wants -- VCVTTPS2DQ truncates and would bias it.
func (a *Buf) VCVTPS2DQ(dst, src Reg) { a.vop(0x5B, p66, m0F, 0, dst, Y0, src) }

// VCVTTPS2DQ converts float32 to int32 with truncation toward zero, which is
// what a round-half-away-from-zero needs once the sign-carrying 0.5 has been
// added. VCVTPS2DQ rounds to nearest-even and would give a different integer on
// every exact .5 -- a different quantization, not a different rounding mode.
// Encoding verified against clang: c5 fe 5b c1.
func (a *Buf) VCVTTPS2DQ(dst, src Reg) { a.vop(0x5B, pF3, m0F, 0, dst, Y0, src) }

// VPACKSSDW and VPACKSSWB narrow int32 to int16 to int8 with signed saturation,
// which performs the packer's [-128,127] clamp for free. Both work per 128-bit
// lane: packing a vector with itself puts the low four results in the low dword
// of lane 0 and the high four in the low dword of lane 1, which is exactly the
// two dwords the packed layout wants. c5 f5 6b c2 / c5 f5 63 c2.
func (a *Buf) VPACKSSDW(dst, src1, src2 Reg) { a.vop(0x6B, p66, m0F, 0, dst, src1, src2) }
func (a *Buf) VPACKSSWB(dst, src1, src2 Reg) { a.vop(0x63, p66, m0F, 0, dst, src1, src2) }

// VPMULLD is the 32-bit integer multiply, low half. c4 e2 75 40 c2. It keeps
// the packer's bias * acc in a vector register instead of a round trip through
// a GPR.
func (a *Buf) VPMULLD(dst, src1, src2 Reg) { a.vop(0x40, p66, m0F38, 0, dst, src1, src2) }

// VCMPPS compares and leaves an all-ones or all-zero mask per lane. imm 4 is
// NEQ_UQ. c5 f4 c2 c2 04.
func (a *Buf) VCMPPS(dst, src1, src2 Reg, imm byte) {
	a.vop(0xC2, pNone, m0F, 0, dst, src1, src2)
	a.u8(imm)
}

// IMUL is the 64-bit signed multiply, REX.W 0F AF /r: 48 0f af c2.
func (a *Buf) IMUL(dst, src Reg) {
	a.u8(0x48 | byte(dst)>>3<<2 | byte(src)>>3)
	a.u8(0x0F)
	a.u8(0xAF)
	a.modrmReg(dst, src)
}

// VFMADD213PS is dst = src1*dst + src2, the shape Horner needs.
func (a *Buf) VFMADD213PS(dst, src1, src2 Reg) { a.vop(0xA8, p66, m0F38, 0, dst, src1, src2) }

// VCVTPH2PS converts eight f16 values to f32 — the quantization scales, which
// every block carries as an f16.
func (a *Buf) VCVTPH2PS(dst Reg, m Mem) { a.vopMem(0x13, p66, m0F38, 0, dst, Y0, m) }

// VEXTRACTI128 pulls the upper or lower 128 bits out, for horizontal reduction.
func (a *Buf) VEXTRACTI128(dst, src Reg, imm byte) {
	a.vex(src, Y0, dst, noIndex, true, p66, m0F3A, 0)
	a.u8(0x39)
	a.modrmReg(src, dst)
	a.u8(imm)
}

// VBROADCASTI128 loads 16 bytes into both 128-bit lanes. This is what makes a
// Q4_0 nibble unpack cheap: one 16-byte load covers all 32 weights, and the low
// and high nibbles are then separated per lane rather than by a shuffle.
func (a *Buf) VBROADCASTI128(dst Reg, m Mem) { a.vopMem(0x5A, p66, m0F38, 0, dst, Y0, m) }

// VPBLENDD selects dwords from src1 or src2 by an immediate bitmask, one bit per
// dword. Used with 0xF0 to take the lower lane from one register and the upper
// lane from another.
func (a *Buf) VPBLENDD(dst, src1, src2 Reg, imm byte) {
	a.vex(dst, src1, src2, noIndex, true, p66, m0F3A, 0)
	a.u8(0x02)
	a.modrmReg(dst, src2)
	a.u8(imm)
}

// VCVTPH2PSx converts four f16 to f32 into an xmm. Quantization scales are f16
// and there is only one per block, so the 128-bit form is enough — it reads 8
// bytes, of which two are the scale and the rest are discarded.
func (a *Buf) VCVTPH2PSx(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(dst, Y0, m.Base, idx, false, p66, m0F38, 0)
	a.u8(0x13)
	a.modrmMem(dst, m)
}

// VBROADCASTSSReg splats lane 0 of an xmm across a ymm — how a scale that was
// just converted from f16 reaches every lane.
func (a *Buf) VBROADCASTSSReg(dst, src Reg) { a.vop(0x18, p66, m0F38, 0, dst, Y0, src) }

// VEXTRACTF128 moves the upper 128 bits down, the first step of a horizontal sum.
func (a *Buf) VEXTRACTF128(dst, src Reg, imm byte) {
	a.vex(src, Y0, dst, noIndex, true, p66, m0F3A, 0)
	a.u8(0x19)
	a.modrmReg(src, dst)
	a.u8(imm)
}

// VADDPSx and VHADDPSx are the 128-bit tail of the reduction.
func (a *Buf) VADDPSx(dst, src1, src2 Reg)  { a.vopL(0x58, pNone, m0F, 0, false, dst, src1, src2) }
func (a *Buf) VHADDPSx(dst, src1, src2 Reg) { a.vopL(0x7C, pF2, m0F, 0, false, dst, src1, src2) }

// VMOVSSLoad reads one float32 into lane 0 and ZEROES the rest of the
// register, which is what lets a vector body run on a single tail element
// without reading past it.
func (a *Buf) VMOVSSLoad(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(dst, Y0, m.Base, idx, false, pF3, m0F, 0)
	a.u8(0x10)
	a.modrmMem(dst, m)
}

// VMOVSSStore writes lane 0 as a single float32 — the kernel's only output.
func (a *Buf) VMOVSSStore(m Mem, src Reg) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(src, Y0, m.Base, idx, false, pF3, m0F, 0)
	a.u8(0x11)
	a.modrmMem(src, m)
}

// VSHUFPS selects two floats from src1 and two from src2 within each 128-bit
// lane, by an immediate. With 0x88 it gathers the EVEN lanes of both, which is
// how the interleaved {scale, bias} activation pairs get split apart.
func (a *Buf) VSHUFPS(dst, src1, src2 Reg, imm byte) {
	a.vex(dst, src1, src2, noIndex, true, pNone, m0F, 0)
	a.u8(0xC6)
	a.modrmReg(dst, src2)
	a.u8(imm)
}

// VPSRLVD shifts each dword of src right by the matching dword of count --
// per-lane shifts, which is how a secondary plane's bits, scattered at a
// different offset for every element, come out in one instruction.
func (a *Buf) VPSRLVD(dst, src, count Reg) { a.vop(0x45, p66, m0F38, 0, dst, src, count) }

// VPMOVZXWD zero-extends eight 16-bit words to eight dwords. Shifted left by
// sixteen, a word that is a bfloat16 becomes the float32 it abbreviates.
func (a *Buf) VPMOVZXWD(dst Reg, m Mem) { a.vopMem(0x33, p66, m0F38, 0, dst, Y0, m) }

// VPINSRWLoad inserts one 16-bit word from memory into lane imm of src1's
// low 128 bits, writing dst and zeroing its upper half. With src1 zeroed it is
// a load of exactly one binary16 -- which VCVTPH2PS, reading four or eight,
// cannot do at the end of a row.
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

// VPINSRBLoad inserts one byte from memory into lane imm of src1's low 128
// bits, writing dst and zeroing its upper half. With src1 zeroed it is a load of
// exactly one byte: MXFP4's super-scale, where wider loads would read past the
// last row of the d plane. Encoding VEX.128.66.0F3A.W0 20 /r ib.
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

// VMOVSLDUP and VMOVSHDUP duplicate the even and the odd lanes:
// [a0 a0 a2 a2 ...] and [a1 a1 a3 a3 ...]. On an interleaved {cos, sin} table
// they are the two operands a consecutive-pair rotation wants.
func (a *Buf) VMOVSLDUP(dst, src Reg) { a.vop(0x12, pF3, m0F, 0, dst, Y0, src) }
func (a *Buf) VMOVSHDUP(dst, src Reg) { a.vop(0x16, pF3, m0F, 0, dst, Y0, src) }

// VADDSUBPS subtracts in the even lanes and adds in the odd ones -- exactly
// the sign pattern of a 2-D rotation applied to adjacent pairs.
func (a *Buf) VADDSUBPS(dst, src1, src2 Reg) { a.vop(0xD0, pF2, m0F, 0, dst, src1, src2) }

// VFNMADD231PS is dst -= src1*src2.
func (a *Buf) VFNMADD231PS(dst, src1, src2 Reg) { a.vop(0xBC, p66, m0F38, 0, dst, src1, src2) }

// VPERMILPS permutes lanes WITHIN each 128-bit half by an immediate; 0xB1
// swaps every adjacent pair.
func (a *Buf) VPERMILPS(dst, src Reg, imm byte) {
	a.vex(dst, Y0, src, noIndex, true, p66, m0F3A, 0)
	a.u8(0x04)
	a.modrmReg(dst, src)
	a.u8(imm)
}

// VPERMQ permutes 64-bit lanes across the whole 256-bit register by an
// immediate -- the only cheap way to move data BETWEEN the two 128-bit halves,
// which VSHUFPS cannot do.
func (a *Buf) VPERMQ(dst, src Reg, imm byte) {
	a.vex(dst, Y0, src, noIndex, true, p66, m0F3A, 1)
	a.u8(0x00)
	a.modrmReg(dst, src)
	a.u8(imm)
}

// VPMOVSXBD sign-extends eight bytes to eight int32 lanes. Q6_K's sub-block
// scales are plain signed int8; zero-extending them would silently turn every
// negative scale into a large positive one.
func (a *Buf) VPMOVSXBD(dst Reg, m Mem) { a.vopMem(0x21, p66, m0F38, 0, dst, Y0, m) }

// VPANDMem is VPAND with a memory source, so a mask constant can live in the
// kernel's constant block instead of occupying a register. The k-quants need
// more distinct masks than the 16-register file has room for.
func (a *Buf) VPANDMem(dst, src1 Reg, m Mem) { a.vopMem(0xDB, p66, m0F, 0, dst, src1, m) }

// The float arithmetic forms that read their second source from memory. The
// prefill GEMM's fold needs both: the per-token activation scales and the
// bias correction are vectors in the packed activation buffer, not registers,
// and at a 12-accumulator tile there is no register left to stage them in.
func (a *Buf) VADDPSMem(dst, src1 Reg, m Mem) { a.vopMem(0x58, pNone, m0F, 0, dst, src1, m) }
func (a *Buf) VMULPSMem(dst, src1 Reg, m Mem) { a.vopMem(0x59, pNone, m0F, 0, dst, src1, m) }

func (a *Buf) VSUBPS(dst, src1, src2 Reg) { a.vop(0x5C, pNone, m0F, 0, dst, src1, src2) }

// VPSUBD subtracts packed int32 lanes, reading the subtrahend from memory.
//
// The prefill GEMM removes its unsigned-weight bias with this rather than in
// float, for correctness: the accumulator holds sum((w+128)*a), where the
// 128*sum(a) term dominates, so subtracting after conversion is catastrophic
// cancellation. Both terms are exact integers, so int32 is exact.
func (a *Buf) VPSUBDMem(dst, src1 Reg, m Mem) { a.vopMem(0xFA, p66, m0F, 0, dst, src1, m) }

// VPSUBD is the register form of the same 32-bit integer subtract.
func (a *Buf) VPSUBD(dst, src1, src2 Reg) { a.vop(0xFA, p66, m0F, 0, dst, src1, src2) }

// VPSHUFB permutes bytes within each 128-bit lane by an index vector. The
// k-quants pack their 6-bit sub-block scales across 12 bytes in an order no
// arithmetic can reach, so gathering them takes a shuffle.
func (a *Buf) VPSHUFB(dst, src1, src2 Reg) { a.vop(0x00, p66, m0F38, 0, dst, src1, src2) }

// VCVTPH2PSReg converts the eight f16 in src's low 128 bits to eight f32. The
// register form serves the packed layout, where eight rows' scales arrive
// strided and must be gathered into a vector before conversion.
func (a *Buf) VCVTPH2PSReg(dst, src Reg) { a.vop(0x13, p66, m0F38, 0, dst, Y0, src) }

// VPMOVZXBD zero-extends eight bytes to eight int32 lanes — how a packed 6-bit
// scale becomes something VCVTDQ2PS can turn into a float.
func (a *Buf) VPMOVZXBD(dst Reg, m Mem)  { a.vopMem(0x31, p66, m0F38, 0, dst, Y0, m) }
func (a *Buf) VPMOVZXBDReg(dst, src Reg) { a.vop(0x31, p66, m0F38, 0, dst, Y0, src) }

// VPUNPCKLDQx interleaves the low dwords of two XMM registers (VEX.128):
// [a0 b0 a1 b1]. The image resampler joins its two lanes' packed bytes with it.
func (a *Buf) VPUNPCKLDQx(dst, src1, src2 Reg) { a.vopL(0x62, p66, m0F, 0, false, dst, src1, src2) }

// VMOVQStore writes the low 64 bits of an XMM register: eight packed samples.
func (a *Buf) VMOVQStore(m Mem, src Reg) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(src, Y0, m.Base, idx, false, p66, m0F, 0)
	a.u8(0xD6)
	a.modrmMem(src, m)
}

// VPEXTRBStore writes byte imm of an XMM register to memory: one sample and
// nothing past it.
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

// VMOVDQUStorex is the 128-bit store, for staging 16 bytes in the red zone.
func (a *Buf) VMOVDQUStorex(m Mem, src Reg) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(src, Y0, m.Base, idx, false, pF3, m0F, 0)
	a.u8(0x7F)
	a.modrmMem(src, m)
}

// VMOVDQULoadx is the 128-bit load, for a block header that is only 16 bytes.
func (a *Buf) VMOVDQULoadx(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.vex(dst, Y0, m.Base, idx, false, pF3, m0F, 0)
	a.u8(0x6F)
	a.modrmMem(dst, m)
}

// VZEROUPPER must run before returning to Go. Leaving the upper halves of the
// YMM registers dirty imposes a state-transition penalty on every subsequent SSE
// instruction in the process, a slowdown with no local symptom.
func (a *Buf) VZEROUPPER() {
	a.noVEXInSSE("VZEROUPPER")
	a.u8(0xC5)
	a.u8(0xF8)
	a.u8(0x77)
}

// ---------------------------------------------------------------------------
// Scalar instructions, enough to drive a loop.

// MOVLoad is mov r64, [mem].
func (a *Buf) MOVLoad(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(true, dst, idx, m.Base)
	a.u8(0x8B)
	a.modrmMem(dst, m)
}

// CPUID is 0F A2: leaf in EAX, sub-leaf in ECX, results in EAX/EBX/ECX/EDX.
// jitllm probes its own CPU with its own JIT, which avoids depending on
// golang.org/x/sys/cpu (internal/cpu is not importable).
func (a *Buf) CPUID() { a.u8(0x0F); a.u8(0xA2) }

// XGETBV reads extended control register ECX into EDX:EAX.
//
// CPUID says the silicon has a unit; XCR0 says the OS saves its state on a
// context switch. With `noxsave`, an old hypervisor or a VM that hides the
// state, the CPUID bit is set and the first AVX instruction still raises #UD.
func (a *Buf) XGETBV() { a.u8(0x0F); a.u8(0x01); a.u8(0xD0) }

// MOVStore is mov [mem], r64.
func (a *Buf) MOVStore(m Mem, src Reg) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(true, src, idx, m.Base)
	a.u8(0x89)
	a.modrmMem(src, m)
}

// MOVQ is mov r64, r64.
func (a *Buf) MOVQ(dst, src Reg) {
	a.rex(true, src, noIndex, dst)
	a.u8(0x89)
	a.modrmReg(src, dst)
}

// ADDQ is add r64, r64.
func (a *Buf) ADDQ(dst, src Reg) {
	a.rex(true, src, noIndex, dst)
	a.u8(0x01)
	a.modrmReg(src, dst)
}

// ADDQmem is add r64, [mem]. It exists so a kernel with no spare register can
// still fold a runtime stride into a cursor: the narrow d plane's super-block
// stride is a second stride the packed kernels have nowhere to hold, and one
// instruction reading it from Args beats spilling a register to make room.
func (a *Buf) ADDQmem(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(true, dst, idx, m.Base)
	a.u8(0x03)
	a.modrmMem(dst, m)
}

// SUBQ is sub r64, r64.
func (a *Buf) SUBQ(dst, src Reg) {
	a.rex(true, src, noIndex, dst)
	a.u8(0x29)
	a.modrmReg(src, dst)
}

// MOVimm32 is mov r64, imm32 (sign-extended).
func (a *Buf) MOVimm32(dst Reg, imm int32) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0xC7)
	a.modrmReg(Y0, dst)
	a.u32(uint32(imm))
}

// SHRimm is shr r64, imm8 -- a logical right shift by a constant.
func (a *Buf) SHRimm(dst Reg, imm byte) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0xC1)
	a.modrmReg(Reg(5), dst) // /5 = SHR
	a.u8(imm)
}

// SHLimm is the mirror of SHRimm: same C1 encoding, /4 instead of /5. It lets a
// kernel halve a row offset for the narrow scale plane without spending a
// register.
func (a *Buf) SHLimm(dst Reg, imm byte) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0xC1)
	a.modrmReg(Reg(4), dst) // /4 = SHL
	a.u8(imm)
}

// ANDimm8 is and r64, imm8 (sign-extended) -- how a row kernel splits a
// runtime element count into its vector and scalar remainders.
func (a *Buf) ANDimm8(dst Reg, imm int8) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0x83)
	a.modrmReg(Reg(4), dst) // /4 = AND
	a.u8(byte(imm))
}

// ADDimm is add r64, imm32, with the imm8 form when it fits.
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

// MOVimm is mov r64, imm32 (sign-extended). The GEMM bakes its shape, so the
// only runtime counter it needs to load is the block count.
func (a *Buf) MOVimm(dst Reg, imm int64) {
	a.rex(true, Y0, noIndex, dst)
	a.u8(0xC7)
	a.modrmReg(Y0, dst)
	a.u32(uint32(int32(imm)))
}

// SUBimm is sub r64, imm32.
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

// DEC is dec r32, which sets ZF for the loop branch.
func (a *Buf) DEC(dst Reg) {
	a.rex(false, Y0, noIndex, dst)
	a.u8(0xFF)
	a.modrmReg(Reg(1), dst)
}

// DECmem is dec dword [m]. The prefill GEMM keeps its super-block counter in
// the Args block rather than in a register: its token loop needs one more
// general-purpose register than the fourteen a kernel may touch, and the
// super-block loop runs once per 256 elements, where a memory decrement is
// free.
func (a *Buf) DECmem(m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(false, Y0, idx, m.Base)
	a.u8(0xFF)
	a.modrmMem(Reg(1), m)
}

// PREFETCHT0 asks for a line into every cache level: 0F 18 /1, no operand size,
// no destination register. It serves the explicit weight-stream prefetch
// (EmitOpts.Prefetch).
//
// The reg field selects the hint: 0 NTA, 1 T0, 2 T1, 3 T2. REX.W must be clear
// -- there is no 64-bit form -- but REX.B/X are still needed to reach r8-r15.
func (a *Buf) PREFETCHT0(m Mem) { a.prefetch(1, m) }

// PREFETCHNTA loads without polluting the caches, for a stream read once.
func (a *Buf) PREFETCHNTA(m Mem) { a.prefetch(0, m) }

// PREFETCHT2 asks for a line into the outer levels (0F 18 /3): the fused
// region's lookahead, whose lines another core reads.
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

// ---------------------------------------------------------------------------
// Labels.

// Label allocates an unbound label.
func (a *Buf) Label() int {
	a.labels = append(a.labels, -1)
	return len(a.labels) - 1
}

// Bind fixes a label at the current offset.
func (a *Buf) Bind(l int) {
	a.labels[l] = len(a.b)
	a.patch()
}

// JNZ jumps to l when ZF is clear. Emits the short form for a backward jump that
// fits, which is every kernel loop.
func (a *Buf) JNZ(l int) { a.jcc(0x75, 0x85, l) }

// JZ jumps to l when ZF is set -- the skip past a loop whose count is zero.
func (a *Buf) JZ(l int) { a.jcc(0x74, 0x84, l) }

// TESTQ is test r64, r64: ZF set when their AND is zero, which with one
// register twice is "is it zero" without changing it.
func (a *Buf) TESTQ(x, y Reg) {
	a.rex(true, y, noIndex, x)
	a.u8(0x85)
	a.modrmReg(y, x)
}

func (a *Buf) jcc(short, near byte, l int) {
	if tgt := a.labels[l]; tgt >= 0 {
		// Backward: the distance is known, so take the 2-byte form when it fits.
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
	// Forward: the target is unknown, so always reserve the near form.
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

// VCVTPS2PH narrows float32 lanes of src to binary16 in dst (eight to an XMM
// with l256, four without) under the rounding control imm; imm 0 is
// round-to-nearest-even whatever MXCSR says. F16C.
func (a *Buf) VCVTPS2PH(dst, src Reg, l256 bool, imm byte) {
	a.vex(src, Y0, dst, noIndex, l256, p66, m0F3A, 0)
	a.u8(0x1D)
	a.modrmReg(src, dst)
	a.u8(imm)
}

// VPEXTRWStore writes word imm of an XMM register to memory: one binary16 and
// nothing past it.
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

// VPANDx, VPANDNx and VPCMPGTWx are the 128-bit VPAND, VPANDN and the signed
// int16 greater-than, for eight binary16 lanes.
func (a *Buf) VPANDx(dst, src1, src2 Reg)  { a.vopL(0xDB, p66, m0F, 0, false, dst, src1, src2) }
func (a *Buf) VPANDNx(dst, src1, src2 Reg) { a.vopL(0xDF, p66, m0F, 0, false, dst, src1, src2) }
func (a *Buf) VPCMPGTWx(dst, src1, src2 Reg) {
	a.vopL(0x65, p66, m0F, 0, false, dst, src1, src2)
}
