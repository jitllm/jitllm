package cpu

import "fmt"

// A64 emits ARMv8-A (AArch64) machine code, the arm64 sibling of Buf.
//
// Every A64 instruction is exactly 32 bits, 4-byte aligned, little-endian: no
// prefixes, no ModRM/SIB, no variable length, and one branch form, so code
// size is exactly predictable and a fixup only rewrites an immediate field.
// Register fields sit at fixed positions (Rd/Rt [4:0], Rn [9:5], Rm [20:16])
// and are five bits wide, so v16-v31 need no escape bit.
//
// The file is a64.go, not asm_arm64.go, deliberately: Go applies an implicit
// GOARCH constraint to *_arm64.go, and emitting arm64 bytes (and the byte-exact
// encoding gate) must work on any host. Only executing them is arm64-only.
//
// Every base word was produced by clang-18 --target=aarch64-linux-gnu with
// .arch armv8.2-a+dotprod+fp16 and read back with llvm-objdump-18; the tests
// re-check each one.

// XReg is a general-purpose register, x0-x30 plus the context-dependent 31.
type XReg uint8

// General-purpose registers.
//
// Registers a generated kernel must never write:
//
//	x28  Go's goroutine pointer (REGG); corrupting it breaks the runtime far
//	     from the cause.
//	x18  the platform register, reserved by macOS. Go does not use it either.
//	x27  REGTMP, the assembler's scratch.
//	x29  the frame pointer, which tracebacks and profilers walk.
//	x30  the link register, i.e. the return address of a leaf kernel.
//	sp   must stay 16-byte aligned and correct.
//
// Everything else is free: Go's arm64 ABIInternal has no callee-saved
// registers, so the arm64 trampoline saves nothing and kernels perform no
// pushes.
const (
	X0 XReg = iota
	X1
	X2
	X3
	X4
	X5
	X6
	X7
	X8
	X9
	X10
	X11
	X12
	X13
	X14
	X15
	X16
	X17
	X18
	X19
	X20
	X21
	X22
	X23
	X24
	X25
	X26
	X27
	X28
	X29
	X30
)

// XZR and SP are both register number 31; the instruction class decides which:
// SP in a load/store base and in ADD/SUB-immediate, the zero register
// elsewhere. Naming both makes the intent explicit at the call site.
const (
	XZR XReg = 31
	SP  XReg = 31
)

// VReg is a SIMD/FP register, v0-v31. Twice the file the AVX2 side has, which
// is why the arm64 kernels never spill and never bake a row stride.
type VReg uint8

// Vector registers.
const (
	V0 VReg = iota
	V1
	V2
	V3
	V4
	V5
	V6
	V7
	V8
	V9
	V10
	V11
	V12
	V13
	V14
	V15
	V16
	V17
	V18
	V19
	V20
	V21
	V22
	V23
	V24
	V25
	V26
	V27
	V28
	V29
	V30
	V31
)

// A64 accumulates emitted instruction words.
type A64 struct {
	b      []byte
	fixup  []a64fixup
	labels []int

	// dotT is the two scratch vectors SDOT and SDOTelem are widened through on
	// a chip without FEAT_DotProd (sdotemu.go), and dotEmu whether they are:
	// both set by DotScratch, which every kernel emitting SDOT calls. An A64
	// that never called it encodes SDOT as written, which keeps the encoding
	// tests host-pure; a kernel that forgot is caught by
	// TestNoDotProdKernelsEmitNoSDOT and, on a chip without the feature, by
	// SIGILL.
	dotT   [2]VReg
	dotEmu bool
	// dotPend is the accumulator a DotChain is open on and dotLinks how many
	// links it has taken; the chain's int16 sums live in dotT until its last
	// link folds them into dotPend.
	dotPend  VReg
	dotLinks int
}

// a64fixup is a branch whose target was not yet bound. bits is the width of the
// immediate field: 19 for CBZ/CBNZ (+/-1 MB), 26 for B (+/-128 MB). Both hold a
// signed displacement in INSTRUCTIONS, not bytes.
type a64fixup struct {
	at   int // byte offset of the instruction word
	bits uint
	lbl  int
}

// w appends one instruction word, little-endian: 0x4E829420 becomes 20 94 82 4E.
func (a *A64) w(v uint32) {
	a.b = append(a.b, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
}

// Bytes returns the emitted code. Panics if a label was used but never bound —
// a dangling branch is a silent infinite loop or a branch into data, so it must
// not be possible to get bytes out.
func (a *A64) Bytes() []byte {
	for _, f := range a.fixup {
		if f.lbl >= len(a.labels) || a.labels[f.lbl] < 0 {
			panic("jit: label used but never bound")
		}
	}
	noteEmitA64(a.b)
	return a.b
}

// ---------------------------------------------------------------------------
// Immediate encodings.
//
// Every scaled load/store form divides its byte offset by the access size (16
// for Q, 8 for X, 4 for S, 2 for H). A misaligned offset does not fault: an
// emitter computing (off/16)<<10 would silently address the wrong byte. So
// scaled panics on a misaligned or out-of-range offset, and callers that need
// an arbitrary byte offset must ask for LDUR by name.

// scaled encodes a 12-bit unsigned offset field, refusing anything the form
// cannot express.
func scaled(off int32, size int32, what string) uint32 {
	if off < 0 || off%size != 0 || off/size > 4095 {
		panic(fmt.Sprintf("jit: %s offset %d must be a non-negative multiple of %d, at most %d",
			what, off, size, 4095*size))
	}
	return uint32(off/size) << 10
}

// unscaled encodes the signed 9-bit field LDUR/STUR use, which takes ANY byte
// offset in [-256, 255]. This is what makes a Q4_0 block header addressable:
// its weights sit at +2 and +18, neither of which is a multiple of 16.
func unscaled(off int32, what string) uint32 {
	if off < -256 || off > 255 {
		panic(fmt.Sprintf("jit: %s offset %d is outside the imm9 range [-256,255]", what, off))
	}
	return uint32(off&0x1FF) << 12
}

// imm12 encodes an ADD/SUB immediate. No lsl #12 form: nothing here needs it,
// and offering it invites a caller to pass a value that silently loses its low
// twelve bits.
func imm12(v int32, what string) uint32 {
	if v < 0 || v > 4095 {
		panic(fmt.Sprintf("jit: %s immediate %d is outside [0,4095]", what, v))
	}
	return uint32(v) << 10
}

// ---------------------------------------------------------------------------
// Scalar instructions, enough to drive a loop.

// LDRx is ldr xt, [xn, #off]. off is scaled by 8, so 0..32760 in steps of 8.
func (a *A64) LDRx(rt, rn XReg, off int32) {
	a.w(0xF9400000 | scaled(off, 8, "ldr x") | uint32(rn)<<5 | uint32(rt))
}

// STRx is str xt, [xn, #off].
func (a *A64) STRx(rt, rn XReg, off int32) {
	a.w(0xF9000000 | scaled(off, 8, "str x") | uint32(rn)<<5 | uint32(rt))
}

// ADDimm is add xd, xn, #imm.
func (a *A64) ADDimm(rd, rn XReg, imm int32) {
	a.w(0x91000000 | imm12(imm, "add") | uint32(rn)<<5 | uint32(rd))
}

// ADDreg is add xd, xn, xm. The immediate form is only 12 bits, and a GEMM row
// stride reaches 17408 bytes, so a row offset has to come from a register.
func (a *A64) ADDreg(rd, rn, rm XReg) {
	a.w(0x8B000000 | uint32(rm)<<16 | uint32(rn)<<5 | uint32(rd))
}

// ANDlow is and rd, rn, #(1<<bits)-1: keep the low bits of a count, which is
// how a row kernel splits a runtime length into its remainders.
func (a *A64) ANDlow(rd, rn XReg, bits uint32) {
	if bits < 1 || bits > 63 {
		panic("a64: ANDlow: width out of range")
	}
	a.w(0x92400000 | (bits-1)<<10 | uint32(rn)<<5 | uint32(rd))
}

// LSRimm is lsr xd, xn, #sh, which arm64 encodes as UBFM xd, xn, #sh, #63.
func (a *A64) LSRimm(rd, rn XReg, sh uint32) {
	if sh > 63 {
		panic("a64: LSRimm: shift out of range")
	}
	a.w(0xD340_0000 | sh<<16 | 63<<10 | uint32(rn)<<5 | uint32(rd))
}

// SUBimm is sub xd, xn, #imm. Flags are not set: loops branch on CBNZ, which
// tests the register itself.
func (a *A64) SUBimm(rd, rn XReg, imm int32) {
	a.w(0xD1000000 | imm12(imm, "sub") | uint32(rn)<<5 | uint32(rd))
}

// MOVimm materialises a 64-bit immediate: one MOVZ when it fits a single
// 16-bit field at any of the four positions (8.0f is 0x4100 << 16, one
// instruction), otherwise MOVZ then a MOVK per non-zero field.
func (a *A64) MOVimm(rd XReg, imm int64) {
	if imm < 0 {
		panic(fmt.Sprintf("jit: MOVimm %d is negative", imm))
	}
	for sh := 0; sh < 4; sh++ {
		if v := uint64(imm) >> (16 * sh); v <= 0xFFFF && v<<(16*sh) == uint64(imm) {
			a.w(0xD2800000 | uint32(sh)<<21 | uint32(v)<<5 | uint32(rd))
			return
		}
	}
	// Needs more than one field: MOVZ then MOVK per non-zero field, at most
	// four instructions, once per kernel.
	first := true
	for sh := 0; sh < 4; sh++ {
		v := (uint64(imm) >> (16 * sh)) & 0xFFFF
		if v == 0 && !(first && sh == 3) {
			continue
		}
		if first {
			a.w(0xD2800000 | uint32(sh)<<21 | uint32(v)<<5 | uint32(rd))
			first = false
			continue
		}
		a.MOVK(rd, uint16(v), uint8(sh))
	}
	if first {
		// imm is zero and the loop above skipped every field.
		a.w(0xD2800000 | uint32(rd))
	}
}

// MOVK is movk xd, #imm16, lsl #(16*shift): overwrite one 16-bit field and
// leave the rest. It is what lets MOVimm reach a constant needing more than one
// field. Pinned by TestA64ShiftEncodingsMatchClang.
func (a *A64) MOVK(rd XReg, imm16 uint16, shift uint8) {
	if shift > 3 {
		panic(fmt.Sprintf("jit: MOVK shift %d, want 0..3", shift))
	}
	a.w(0xF2800000 | uint32(shift)<<21 | uint32(imm16)<<5 | uint32(rd))
}

// MOVreg is mov xd, xm, which the hardware spells orr xd, xzr, xm. Register 31
// in the Rn position means the zero register here, not SP — the class-dependent
// meaning of 31 is exactly why this is a named method rather than a raw ORR.
func (a *A64) MOVreg(rd, rm XReg) {
	a.w(0xAA000000 | uint32(rm)<<16 | uint32(XZR)<<5 | uint32(rd))
}

// RET returns through x30, the link register.
func (a *A64) RET() { a.w(0xD65F03C0) }

// ---------------------------------------------------------------------------
// Vector load/store.

// LD1x4Post is ld1 {vt.16b, vt+1, vt+2, vt+3}, [xn], #64: four consecutive
// vector registers in one instruction, with the base advanced by 64 after.
// The register quad must be consecutive and wrap at v31; the caller owns that.
func (a *A64) LD1x4Post(vt VReg, rn XReg) {
	a.w(0x4CDF2000 | uint32(rn)<<5 | uint32(vt))
}

// LD1x4 is the same without the post-increment.
func (a *A64) LD1x4(vt VReg, rn XReg) {
	a.w(0x4C402000 | uint32(rn)<<5 | uint32(vt))
}

// LD2 is ld2 {vt.4s, vt+1.4s}, [xn]: eight floats DEINTERLEAVED, the even
// ones into vt and the odd into vt+1. On a {cos, sin} table that is the two
// operands a rotation wants; on a head it splits every adjacent pair.
// LD2Post advances xn by the 32 bytes it read; ST2 and ST2Post interleave
// back.
func (a *A64) LD2(vt VReg, rn XReg)     { a.w(0x4C408800 | uint32(rn)<<5 | uint32(vt)) }
func (a *A64) LD2Post(vt VReg, rn XReg) { a.w(0x4CDF8800 | uint32(rn)<<5 | uint32(vt)) }
func (a *A64) ST2(vt VReg, rn XReg)     { a.w(0x4C008800 | uint32(rn)<<5 | uint32(vt)) }
func (a *A64) ST2Post(vt VReg, rn XReg) { a.w(0x4C9F8800 | uint32(rn)<<5 | uint32(vt)) }

// STRd is str dt, [xn, #off]: the low 8 bytes. off is scaled by 8.
func (a *A64) STRd(vt VReg, rn XReg, off int32) {
	a.w(0xFD000000 | scaled(off, 8, "str d") | uint32(rn)<<5 | uint32(vt))
}

// USHL4s is ushl vd.4s, vn.4s, vm.4s: each lane shifted left by the matching
// lane of vm, RIGHT when that count is negative -- NEON's only variable shift.
func (a *A64) USHL4s(vd, vn, vm VReg) {
	a.w(0x6EA04400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// USHL16b is ushl vd.16b, vn.16b, vm.16b: the byte form of the variable shift.
// It differs from USHL4s only in the size field (23:22), and a wrong size is a
// valid instruction that shifts 32-bit lanes instead: a wrong number, no fault.
//
// It serves Q5_0's fifth-bit plane: after TBL fans the qh bytes out, lane j
// needs bit (j%8), and a shift vector of {4,3,2,1,0,-1,-2,-3} lands every
// lane's own bit at position 4 in one instruction.
func (a *A64) USHL16b(vd, vn, vm VReg) {
	a.w(0x6E204400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// LDRq is ldr qt, [xn, #off]: 16 bytes. off is scaled by 16 (0..65520).
func (a *A64) LDRq(vt VReg, rn XReg, off int32) {
	a.w(0x3DC00000 | scaled(off, 16, "ldr q") | uint32(rn)<<5 | uint32(vt))
}

// LDPq is ldp qt1, qt2, [xn, #off]: thirty-two bytes in one instruction.
//
// Unlike LD1 multi-register, which is microcoded on Apple silicon (four LD1x4
// decode to about the uops of sixteen LDRs), LDP is a single uop loading two
// registers.
//
// off is scaled by 16 and the immediate is a signed 7-bit field, so the range is
// -1024..1008 -- much tighter than LDR q's unsigned 0..65520, and it is checked.
func (a *A64) LDPq(vt1, vt2 VReg, rn XReg, off int32) {
	if off%16 != 0 || off < -1024 || off > 1008 {
		panic(fmt.Sprintf("ldp q: offset %d not a multiple of 16 in -1024..1008", off))
	}
	imm7 := uint32(off/16) & 0x7F
	a.w(0xAD400000 | imm7<<15 | uint32(vt2)<<10 | uint32(rn)<<5 | uint32(vt1))
}

// STRq is str qt, [xn, #off].
func (a *A64) STRq(vt VReg, rn XReg, off int32) {
	a.w(0x3D800000 | scaled(off, 16, "str q") | uint32(rn)<<5 | uint32(vt))
}

// LDURq is ldur qt, [xn, #off]: 16 bytes at any byte offset in [-256,255].
// This is how quantized weights are read: a Q4_0 block is 18 bytes and a Q8_0
// block 34, so the payload is never 16-byte aligned.
func (a *A64) LDURq(vt VReg, rn XReg, off int32) {
	a.w(0x3CC00000 | unscaled(off, "ldur q") | uint32(rn)<<5 | uint32(vt))
}

// LDRh is ldr ht, [xn, #off]: 2 bytes into lane 0, upper bits zeroed. The block
// scale d_w is an f16 at the head of every block. off is scaled by 2.
func (a *A64) LDRh(vt VReg, rn XReg, off int32) {
	a.w(0x7D400000 | scaled(off, 2, "ldr h") | uint32(rn)<<5 | uint32(vt))
}

// LDRb is ldr bt, [xn, #off]: one byte into lane 0, upper bits zeroed. off is
// scaled by 1. MXFP4's d plane holds four rows to a word, so a wider load on a
// tensor's last row would read past the plane.
func (a *A64) LDRb(vt VReg, rn XReg, off int32) {
	a.w(0x3D400000 | scaled(off, 1, "ldr b") | uint32(rn)<<5 | uint32(vt))
}

// LDRs is ldr st, [xn, #off]: 4 bytes into lane 0. Landing a float32 in lane 0
// is exactly what the by-element FMLA wants, so this replaces amd64's
// VBROADCASTSS with no broadcast at all. off is scaled by 4.
func (a *A64) LDRs(vt VReg, rn XReg, off int32) {
	a.w(0xBD400000 | scaled(off, 4, "ldr s") | uint32(rn)<<5 | uint32(vt))
}

// STRs is str st, [xn, #off]: lane 0 as one float32, the kernel's only output.
func (a *A64) STRs(vt VReg, rn XReg, off int32) {
	a.w(0xBD000000 | scaled(off, 4, "str s") | uint32(rn)<<5 | uint32(vt))
}

// ---------------------------------------------------------------------------
// Vector arithmetic.

// MOVIzero is movi vd.4s, #0. Zeroing an accumulator, where amd64 needs a VPXOR
// against itself.
func (a *A64) MOVIzero(vd VReg) { a.w(0x4F000400 | uint32(vd)) }

// MOVI16b is movi vd.16b, #imm8: one byte replicated across all sixteen lanes.
// The imm8 is split across the word: high three bits at [18:16], low five at
// [9:5]. It lets masks like 0x0F live as instructions rather than constants.
func (a *A64) MOVI16b(vd VReg, imm8 byte) {
	a.w(0x4F00E400 | uint32(imm8>>5)<<16 | uint32(imm8&31)<<5 | uint32(vd))
}

// SDOT is sdot vd.4s, vn.16b, vm.16b: the ARMv8.2 DotProd int8 dot product, and
// the arm64 analogue of VPDPBUSD. Four int8 x int8 products accumulate into each
// of four int32 lanes, so one 32-element quant block takes two of these.
//
// SDOT is signed x signed, where VPDPBUSD is unsigned x signed: amd64's bias
// machinery (XOR 0x80, unsigned nibbles, the -sum(q)*biasC correction) does
// not apply here, and porting it would double-bias every weight. USDOT has
// VPDPBUSD's semantics but needs ARMv8.6. Use SDOT, and do not mix the two.
func (a *A64) SDOT(vd, vn, vm VReg) {
	if a.dotEmu {
		a.sdotWide(vd, vn, vm, false, 0)
		return
	}
	a.w(0x4E809400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SDOTelem is sdot vd.4s, vn.16b, vm.4b[idx]: the by-element dot product. The
// four weight bytes are selected by an immediate index and broadcast by the
// instruction itself, so sixteen weight bytes loaded once serve four k-groups
// with no broadcast and no bias correction.
//
// The index is split the same way FMLAelem's is -- L at bit 21, H at bit 11 --
// and getting it wrong silently selects a different k-group.
func (a *A64) SDOTelem(vd, vn, vm VReg, idx uint8) {
	if idx > 3 {
		panic(fmt.Sprintf("jit: SDOTelem index %d, want 0..3", idx))
	}
	if a.dotEmu {
		a.sdotWide(vd, vn, vm, true, idx)
		return
	}
	a.w(0x4F80E000 | uint32(idx&1)<<21 | uint32(idx>>1)<<11 |
		uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FMUL4s is fmul vd.4s, vn.4s, vm.4s: four float32 lanes, used to fold the
// per-token activation scales in one go.
func (a *A64) FMUL4s(vd, vn, vm VReg) {
	a.w(0x6E20DC00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FMLA4s is fmla vd.4s, vn.4s, vm.4s: fused multiply-add across four lanes.
func (a *A64) FMLA4s(vd, vn, vm VReg) {
	a.w(0x4E20CC00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// AND16b is and vd.16b, vn.16b, vm.16b.
func (a *A64) AND16b(vd, vn, vm VReg) {
	a.w(0x4E201C00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SUB16b is sub vd.16b, vn.16b, vm.16b, per-byte and wrapping.
func (a *A64) SUB16b(vd, vn, vm VReg) {
	a.w(0x6E208400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// USHR16b is ushr vd.16b, vn.16b, #shift: unsigned per-byte right shift.
//
// The immediate is inverted: the field holds (16 - shift), where SHL holds
// (esize + shift). Getting it backwards is a valid instruction with a silently
// wrong shift.
func (a *A64) USHR16b(vd, vn VReg, shift uint8) {
	if shift < 1 || shift > 8 {
		panic(fmt.Sprintf("jit: USHR16b shift %d, want 1..8", shift))
	}
	a.w(0x6F000400 | uint32(16-shift)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SHL16b is shl vd.16b, vn.16b, #shift, the mirror of USHR16b. For 8-bit
// elements a left shift encodes immh:immb as 8+shift while a right shift
// encodes 16-shift; getting that backwards shifts by the wrong amount.
func (a *A64) SHL16b(vd, vn VReg, shift uint8) {
	if shift > 7 {
		panic(fmt.Sprintf("jit: SHL16b shift %d, want 0..7", shift))
	}
	a.w(0x4F005400 | uint32(8+shift)<<16 | uint32(vn)<<5 | uint32(vd))
}

// MOVvec is the register-to-register vector move, which A64 spells as
// `orr vd.16b, vn.16b, vn.16b`. There is no separate MOV encoding.
func (a *A64) MOVvec(vd, vn VReg) { a.ORR16b(vd, vn, vn) }

// ORR16b is orr vd.16b, vn.16b, vm.16b. AND is size=00 and ORR is size=10, so
// the two differ only in bits 23:22.
func (a *A64) ORR16b(vd, vn, vm VReg) {
	a.w(0x4EA01C00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// BIT16b is bit vd.16b, vn.16b, vm.16b -- Bitwise Insert if True:
//
//	vd = (vn & vm) | (vd &^ vm)
//
// It is the AND/BIC/ORR/EOR family and differs only in U (bit 29) and size
// (bits 23:22): BIT is U=1 size=10. It folds a two-plane unpack's AND + ORR
// (Q5_K/Q6_K: primary | hi<<bits) into one vector instruction.
//
// A wrong size field is a valid instruction that inserts under the wrong
// mask, so the encoding is pinned against llvm-mc -triple=aarch64:
//
//	bit v7.16b,  v2.16b,  v4.16b   -> [0x47,0x1c,0xa4,0x6e]  0x6EA41C47
//	bit v0.16b,  v1.16b,  v5.16b   -> [0x20,0x1c,0xa5,0x6e]  0x6EA51C20
//	bit v31.16b, v30.16b, v29.16b  -> [0xdf,0x1f,0xbd,0x6e]  0x6EBD1FDF
//
// TestA64BITMatchesClang re-derives all three from this base.
func (a *A64) BIT16b(vd, vn, vm VReg) {
	a.w(0x6EA01C00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// BIC16b is bic vd.16b, vn.16b, vm.16b: vn AND NOT vm. AND is size=00, BIC is
// size=01 and ORR is size=10, so all three differ only in bits 23:22.
//
// It serves Q3_K's zero point, qs + 4*hbit - 4: 4 &^ hbit_mask is the amount to
// subtract, one instruction where AND, ORR and SUB took three.
func (a *A64) BIC16b(vd, vn, vm VReg) {
	a.w(0x4E601C00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// TBL is tbl vd.16b, {vn.16b}, vm.16b: byte permutation by a table of indices,
// the arm64 analogue of VPSHUFB. It differs in one useful way -- an index at or
// above 16 yields ZERO rather than being masked -- which the k-quant scale
// unpacks rely on to blank lanes they do not want.
func (a *A64) TBL(vd, vn, vm VReg) {
	a.w(0x4E000000 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// INSs is ins vd.s[to], vn.s[from]: move one 32-bit lane, which is how a
// dword-granularity blend is spelled here. AVX2 has VPBLENDD with an immediate
// mask; NEON has no such thing, so a two-dword blend is two of these.
func (a *A64) INSs(vd VReg, to uint8, vn VReg, from uint8) {
	if to > 3 || from > 3 {
		panic(fmt.Sprintf("jit: INSs lanes %d,%d, want 0..3", to, from))
	}
	imm5 := uint32(to)<<3 | 4
	a.w(0x6E000400 | imm5<<16 | uint32(from)<<13 | uint32(vn)<<5 | uint32(vd))
}

// The widening chain. A quantized scale arrives as one byte and has to become a
// float32 lane, which on arm64 is two widenings and a convert -- byte to
// halfword, halfword to word, word to float. SXTL sign-extends and UXTL zero-
// extends; the "2" forms take the HIGH half of the source register, so a
// 16-byte vector becomes four 4-lane float vectors as SXTL/SXTL2 then SXTL/SXTL2
// again on each.
func (a *A64) SXTL8h(vd, vn VReg)   { a.w(0x0F08A400 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) SXTL2_8h(vd, vn VReg) { a.w(0x4F08A400 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) SXTL4s(vd, vn VReg)   { a.w(0x0F10A400 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) SXTL2_4s(vd, vn VReg) { a.w(0x4F10A400 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) UXTL8h(vd, vn VReg)   { a.w(0x2F08A400 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) UXTL2_8h(vd, vn VReg) { a.w(0x6F08A400 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) UXTL4s(vd, vn VReg)   { a.w(0x2F10A400 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) UXTL2_4s(vd, vn VReg) { a.w(0x6F10A400 | uint32(vn)<<5 | uint32(vd)) }

// UCVTF4s is ucvtf vd.4s, vn.4s: UNSIGNED int32 lanes to float32, for the
// k-quant scales that are unsigned 6-bit values rather than signed bytes.
func (a *A64) UCVTF4s(vd, vn VReg) { a.w(0x6E21D800 | uint32(vn)<<5 | uint32(vd)) }

// SCVTF4s is scvtf vd.4s, vn.4s: signed int32 lanes to float32. The last step of
// an integer dot product, before the scale is applied.
func (a *A64) SCVTF4s(vd, vn VReg) {
	a.w(0x4E21D800 | uint32(vn)<<5 | uint32(vd))
}

// FCVTsh is fcvt sd, hn: one f16 to f32, scalar. Every quantized block carries
// its scale as an f16 and there is exactly one per block, so the vector FCVTL
// (which widens four at a time) would have nothing to do with the other three.
func (a *A64) FCVTsh(vd, vn VReg) {
	a.w(0x1EE24000 | uint32(vn)<<5 | uint32(vd))
}

// The packer's instructions. Encodings verified against clang, two registers
// apart each so a base-vs-field mistake cannot hide: FABS4s reads 0x4EA0F820 for
// v0<-v1 and 0x4EA0FAB4 for v20<-v21.
//
// FCVTZS4s truncates toward zero, which is what round-half-away needs once the
// sign-carrying 0.5 is added.
//
// SQXTN4h and SQXTN8b narrow int32 -> int16 -> int8 with signed saturation,
// performing the packer's [-128,127] clamp for free.
func (a *A64) FABS4s(vd, vn VReg)   { a.w(0x4EA0F800 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) FCVTZS4s(vd, vn VReg) { a.w(0x4EA1B800 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) SQXTN4h(vd, vn VReg)  { a.w(0x0E614800 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) SQXTN8b(vd, vn VReg)  { a.w(0x0E214800 | uint32(vn)<<5 | uint32(vd)) }

// ADDV reduces four int32 lanes to one, in a single instruction where amd64
// needs an extract and two horizontal adds.
func (a *A64) ADDV(vd, vn VReg) { a.w(0x4EB1B800 | uint32(vn)<<5 | uint32(vd)) }

// FCMEQzero sets all-ones lanes where the operand IS zero; the packer wants the
// complement, which BIC16b gives without a second compare.
func (a *A64) FCMEQzero(vd, vn VReg) { a.w(0x4EA0D800 | uint32(vn)<<5 | uint32(vd)) }

// FCMGT4s and FCMEQ4s are the register-to-register float compares, all-ones in
// every lane where the predicate holds.
//
// FCMGT is > and is false for a NaN on either side, which is why the argmax
// uses a compare rather than FMAX: model.Greedy is `if v > best`, and ARM's
// FMAX propagates a NaN into the running maximum.
//
// The two differ only in U (bit 29) and size bit 23. Pinned against llvm-mc
// -show-encoding:
//
//	fcmgt v9.4s,  v7.4s,  v0.4s   -> [0xe9,0xe4,0xa0,0x6e]  0x6EA0E4E9
//	fcmgt v31.4s, v30.4s, v29.4s  -> [0xdf,0xe7,0xbd,0x6e]  0x6EBDE7DF
//	fcmeq v12.4s, v0.4s,  v11.4s  -> [0x0c,0xe4,0x2b,0x4e]  0x4E2BE40C
//	fcmeq v31.4s, v30.4s, v29.4s  -> [0xdf,0xe7,0x3d,0x4e]  0x4E3DE7DF
func (a *A64) FCMGT4s(vd, vn, vm VReg) {
	a.w(0x6EA0E400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

func (a *A64) FCMEQ4s(vd, vn, vm VReg) {
	a.w(0x4E20E400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FCMGE4s is ordered greater-or-equal, false for a NaN on either side. FCMEQ
// (U=0, sz=0) 0x4E20E400, FCMGE (U=1, sz=0) 0x6E20E400 and FCMGT (U=1, sz=1)
// 0x6EA0E400 differ in exactly those two bits.
//
// The sampler's top-p test is acc >= TopP, and !(acc < TopP) is not the same
// predicate: it is true for a NaN.
//
//	fcmge v9.4s,  v7.4s,  v0.4s   -> [0xe9,0xe4,0x20,0x6e]  0x6E20E4E9
//	fcmge v31.4s, v30.4s, v29.4s  -> [0xdf,0xe7,0x3d,0x6e]  0x6E3DE7DF
func (a *A64) FCMGE4s(vd, vn, vm VReg) {
	a.w(0x6E20E400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SMIN4s and SMINV are the signed integer minimum, lanewise and across a
// vector. The float twins are wrong here: the argmax folds an index, and the
// 0x7FFFFFFF sentinel read as a float is a NaN.
//
//	smin  v15.4s, v15.4s, v16.4s  -> [0xef,0x6d,0xb0,0x4e]  0x4EB06DEF
//	smin  v31.4s, v30.4s, v29.4s  -> [0xdf,0x6f,0xbd,0x4e]  0x4EBD6FDF
//	sminv s15, v15.4s             -> [0xef,0xa9,0xb1,0x4e]  0x4EB1A9EF
//	sminv s31, v30.4s             -> [0xdf,0xab,0xb1,0x4e]  0x4EB1ABDF
func (a *A64) SMIN4s(vd, vn, vm VReg) {
	a.w(0x4EA06C00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

func (a *A64) SMINV(vd, vn VReg) { a.w(0x4EB1A800 | uint32(vn)<<5 | uint32(vd)) }

// CMGT4s and CMEQ4s are the integer compares (signed greater-than, equality)
// over 32-bit lanes; they differ from the float twins only in the opcode field.
//
// The router's top-k compares expert ids and must not do it in float: an id
// read as float32 is a denormal, and with FPCR.FZ set (process state this
// package does not own) every id reads as zero and every comparison ties.
//
//	cmgt v9.4s,  v7.4s,  v0.4s   -> [0xe9,0x34,0xa0,0x4e]  0x4EA034E9
//	cmgt v31.4s, v30.4s, v29.4s  -> [0xdf,0x37,0xbd,0x4e]  0x4EBD37DF
//	cmeq v12.4s, v0.4s,  v11.4s  -> [0x0c,0x8c,0xab,0x6e]  0x6EAB8C0C
//	cmeq v31.4s, v30.4s, v29.4s  -> [0xdf,0x8f,0xbd,0x6e]  0x6EBD8FDF
func (a *A64) CMGT4s(vd, vn, vm VReg) {
	a.w(0x4EA03400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

func (a *A64) CMEQ4s(vd, vn, vm VReg) {
	a.w(0x6EA08C00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

func (a *A64) MUL4s(vd, vn, vm VReg) {
	a.w(0x4EA09C00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// MLA4s is mla vd.4s, vn.4s, vm.4s: vd += vn * vm on int32 lanes, the
// integer fold's scale-and-accumulate. Encoding from clang-18 (0x4EA39441 for
// `mla v1.4s, v2.4s, v3.4s`).
func (a *A64) MLA4s(vd, vn, vm VReg) {
	a.w(0x4EA09400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// LDRd loads 64 bits into a d register: four f16, which FCVTL then widens to a
// full 4S vector. Encoding verified against clang (0xFD400444 for
// `ldr d4, [x2, #8]`); the offset is scaled by 8 and caps at 32760.
func (a *A64) LDRd(vt VReg, rn XReg, off int32) {
	a.w(0xFD400000 | uint32(off/8)<<10 | uint32(rn)<<5 | uint32(vt))
}

// SHLL and SHLL2 widen four 16-bit lanes (the low or the high half) to four
// 32-bit lanes shifted left by sixteen: a bfloat16 becomes its float32.
func (a *A64) SHLL(vd, vn VReg)  { a.w(0x2E613800 | uint32(vn)<<5 | uint32(vd)) }
func (a *A64) SHLL2(vd, vn VReg) { a.w(0x6E613800 | uint32(vn)<<5 | uint32(vd)) }

// FCVTL is fcvtl vd.4s, vn.4h: the low four f16 lanes widened to f32. FCVTL2
// takes the high four of the same register, so one LDRq of eight halves gives
// two 4S vectors. A wrong Rn field silently widens a different register.
func (a *A64) FCVTL(vd, vn VReg) { a.w(0x0E217800 | uint32(vn)<<5 | uint32(vd)) }

// UZP1h8 and UZP2h8 are uzp1/uzp2 vd.8h, vn.8h, vm.8h: the even and the odd
// 16-bit lanes of vn:vm. With vn == vm the low four lanes are vn's evens (or
// odds), which is how four `lo | hi<<16` words split into four lo halves and
// four hi halves in one instruction each. Encodings from llvm-mc.
func (a *A64) UZP1h8(vd, vn, vm VReg) {
	a.w(0x4E401800 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}
func (a *A64) UZP2h8(vd, vn, vm VReg) {
	a.w(0x4E405800 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// PRFM is prfm pldl1keep, [xn, #off]: a load hint for the line at xn+off that
// retires without waiting for it. Encoding from llvm-mc.
func (a *A64) PRFM(rn XReg, off int32) {
	a.w(0xF9800000 | scaled(off, 8, "prfm") | uint32(rn)<<5)
}

// prfmPLDL2KEEP is PRFM's prfop for a load hint into L2 (Rt = 0b00010).
const prfmPLDL2KEEP = 2

// PRFMop is PRFM with the prfop in Rt: prfm <op>, [xn, #off].
func (a *A64) PRFMop(op uint32, rn XReg, off int32) {
	a.w(0xF9800000 | scaled(off, 8, "prfm") | uint32(rn)<<5 | op)
}

func (a *A64) FCVTL2(vd, vn VReg) { a.w(0x4E217800 | uint32(vn)<<5 | uint32(vd)) }

// FMULs is fmul sd, sn, sm: scalar float32, used to combine d_w with d_x.
func (a *A64) FMULs(vd, vn, vm VReg) {
	a.w(0x1E200800 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FMLAelem is fmla vd.4s, vn.4s, vm.s[idx]: dst += src * lane idx of vm,
// broadcast, single rounding. The by-element form is why arm64 needs no
// broadcast instruction for a scale.
//
// The element index is split: L at bit 21, H at bit 11; a wrong one silently
// selects a different lane. Rm is the full 5-bit field [20:16] (the manual's
// 4-bit Rm plus M bit), so masking it to four bits would drop v16-v31.
func (a *A64) FMLAelem(vd, vn, vm VReg, idx uint8) {
	if idx > 3 {
		panic(fmt.Sprintf("jit: FMLAelem index %d, want 0..3", idx))
	}
	a.w(0x4F801000 | uint32(idx&1)<<21 | uint32(idx>>1)<<11 |
		uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// MLAelem is mla vd.4s, vn.4s, vm.s[idx]: the integer twin of FMLAelem. A
// k-quant sub-block's dot and its 6-bit scale are both integers, so the inner
// sum stays exact in int32 and d applies once. It differs from FMLA only in U
// (bit 29) and the opcode field (0000 against 0001).
func (a *A64) MLAelem(vd, vn, vm VReg, idx uint8) { a.fmaElem(0x6F800000, vd, vn, vm, idx) }

// ADD4s is add vd.4s, vn.4s, vm.4s: the integer twin of FADD4s, which merges the
// two integer accumulator chains before the one conversion to float.
func (a *A64) ADD4s(vd, vn, vm VReg) {
	a.w(0x4EA08400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FMULelem is fmul vd.4s, vn.4s, vm.s[idx], and FMLSelem is the subtracting
// twin of FMLAelem. All three share FMLAelem's split index encoding and differ
// only in bits 15:12: 0001 for FMLA, 0101 for FMLS, 1001 for FMUL.
func (a *A64) FMULelem(vd, vn, vm VReg, idx uint8) { a.fmaElem(0x4F809000, vd, vn, vm, idx) }
func (a *A64) FMLSelem(vd, vn, vm VReg, idx uint8) { a.fmaElem(0x4F805000, vd, vn, vm, idx) }

func (a *A64) fmaElem(base uint32, vd, vn, vm VReg, idx uint8) {
	if idx > 3 {
		panic(fmt.Sprintf("jit: by-element index %d, want 0..3", idx))
	}
	a.w(base | uint32(idx&1)<<21 | uint32(idx>>1)<<11 |
		uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FADD4s is fadd vd.4s, vn.4s, vm.4s: lane-wise float add, which is what merges
// several accumulator chains back into one before the horizontal sum.
func (a *A64) FADD4s(vd, vn, vm VReg) {
	a.w(0x4E20D400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// The float ops the elementwise kernels need and the matvec never did:
// subtraction, min/max, divide, square root, a rounding convert and a lane
// shift. Every encoding came from llvm-mc -show-encoding and is pinned in
// a64_test.go.

// FMLS4s is fmls vd.4s, vn.4s, vm.4s: vd -= vn*vm, fused.
func (a *A64) FMLS4s(vd, vn, vm VReg) {
	a.w(0x4EA0CC00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FSUB4s is fsub vd.4s, vn.4s, vm.4s.
func (a *A64) FSUB4s(vd, vn, vm VReg) {
	a.w(0x4EA0D400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FMAX4s and FMIN4s: lanewise max and min, for the softmax reduction and for
// exp's clamps.
func (a *A64) FMAX4s(vd, vn, vm VReg) {
	a.w(0x4E20F400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

func (a *A64) FMIN4s(vd, vn, vm VReg) {
	a.w(0x4EA0F400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FDIV4s is fdiv vd.4s, vn.4s, vm.4s.
func (a *A64) FDIV4s(vd, vn, vm VReg) {
	a.w(0x6E20FC00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FSQRT4s is fsqrt vd.4s, vn.4s.
func (a *A64) FSQRT4s(vd, vn VReg) {
	a.w(0x6EA1F800 | uint32(vn)<<5 | uint32(vd))
}

// FNEG4s is fneg vd.4s, vn.4s.
func (a *A64) FNEG4s(vd, vn VReg) {
	a.w(0x6EA0F800 | uint32(vn)<<5 | uint32(vd))
}

// FCVTNS4s is fcvtns vd.4s, vn.4s: float to int, round to nearest. exp
// range-reduces with n = round(x*log2e); truncation would bias every reduced
// argument into a systematic error.
func (a *A64) FCVTNS4s(vd, vn VReg) {
	a.w(0x4E21A800 | uint32(vn)<<5 | uint32(vd))
}

// MOVI4s is movi vd.4s, #imm8: imm8 in the low byte of each 32-bit lane, zero
// above (MOVI16b would give 0x20202020 where 32 is wanted). The imm8 is split:
// abc in bits 18:16, defgh in bits 9:5. Pinned by
// TestA64ShiftEncodingsMatchClang.
func (a *A64) MOVI4s(vd VReg, imm8 byte) {
	a.w(0x4F000400 | uint32(imm8>>5)<<16 | uint32(imm8&0x1F)<<5 | uint32(vd))
}

// SUB4s is sub vd.4s, vn.4s, vm.4s: per-32-bit-lane subtract. SUB16b borrows
// within a byte and does not propagate, so it cannot take a sub-block scale
// below its offset -- which is exactly the Q3_K case.
func (a *A64) SUB4s(vd, vn, vm VReg) {
	a.w(0x6EA08400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// USHR4s is ushr vd.4s, vn.4s, #shift: unsigned right shift of each 32-bit
// lane, for extracting packed sub-block scale fields (USHR16b cannot cross a
// byte). The immediate is (2*esize - shift), so its width picks the element
// size; pinned by TestA64ShiftEncodingsMatchClang.
func (a *A64) USHR4s(vd, vn VReg, shift uint8) {
	if shift < 1 || shift > 32 {
		panic(fmt.Sprintf("jit: USHR4s shift %d, want 1..32", shift))
	}
	a.w(0x6F000400 | uint32(64-shift)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SSHR4s is sshr vd.4s, vn.4s, #shift: USHR4s with the sign shifted in, the
// U bit clear. The image resampler's fixed-point rounding shift.
func (a *A64) SSHR4s(vd, vn VReg, shift uint8) {
	if shift < 1 || shift > 32 {
		panic(fmt.Sprintf("jit: SSHR4s shift %d, want 1..32", shift))
	}
	a.w(0x4F000400 | uint32(64-shift)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SHL4s is shl vd.4s, vn.4s, #shift: the lane shift that builds 2^n straight in
// the f32 exponent field.
func (a *A64) SHL4s(vd, vn VReg, shift uint8) {
	if shift > 31 {
		panic(fmt.Sprintf("jit: SHL4s shift %d, want 0..31", shift))
	}
	a.w(0x4F005400 | uint32(32+shift)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FMAXV is fmaxv sd, vn.4s: the horizontal maximum, in one instruction where
// amd64 needs a shuffle ladder.
func (a *A64) FMAXV(vd, vn VReg) {
	a.w(0x6E30F800 | uint32(vn)<<5 | uint32(vd))
}

// DUPs4 is dup vd.4s, vn.s[0]: splat lane 0 across the vector, which is how a
// reduced scalar (a norm's scale, a softmax's max or sum) gets back into a form
// the next pass can multiply by. amd64 needs VBROADCASTSSReg for the same thing.
func (a *A64) DUPs4(vd, vn VReg) {
	a.w(0x4E040400 | uint32(vn)<<5 | uint32(vd))
}

// FADDP4s is faddp vd.4s, vn.4s, vm.4s: pairwise float add.
func (a *A64) FADDP4s(vd, vn, vm VReg) {
	a.w(0x6E20D400 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// FADDPs is faddp sd, vn.2s: adds the low two lanes into a scalar.
//
// FADDP4s twice, then this, is the cheapest correct float32 horizontal sum.
// There is no FADDV in base ARMv8 (only SVE), and ADDV is integer only:
// pointed at a float accumulator it assembles fine and produces garbage.
func (a *A64) FADDPs(vd, vn VReg) {
	a.w(0x7E30D800 | uint32(vn)<<5 | uint32(vd))
}

// ---------------------------------------------------------------------------
// Labels and branches. There is no short-vs-near choice, so a forward branch
// reserves the four bytes it will occupy and a fixup rewrites the immediate.

// Label allocates an unbound label.
func (a *A64) Label() int {
	a.labels = append(a.labels, -1)
	return len(a.labels) - 1
}

// Bind fixes a label at the current offset.
func (a *A64) Bind(l int) {
	a.labels[l] = len(a.b)
	a.patch()
}

// CBNZ is cbnz xt, l — branch when xt is non-zero. Every kernel loop ends in
// one: SUB then CBNZ tests the counter itself, so unlike amd64's DEC/JNZ there
// is no flag register in the dependency chain.
func (a *A64) CBNZ(rt XReg, l int) { a.branch(0xB5000000|uint32(rt), 19, l) }

// CBZ is cbz xt, l — branch when xt is zero.
func (a *A64) CBZ(rt XReg, l int) { a.branch(0xB4000000|uint32(rt), 19, l) }

func (a *A64) branch(word uint32, bits uint, l int) {
	at := len(a.b)
	a.w(word)
	if tgt := a.labels[l]; tgt >= 0 {
		a.setDisp(at, bits, tgt)
		return
	}
	a.fixup = append(a.fixup, a64fixup{at: at, bits: bits, lbl: l})
}

// setDisp writes a branch displacement in INSTRUCTIONS into an already-emitted
// word. The imm19 field sits at [23:5] and imm26 at [25:0].
func (a *A64) setDisp(at int, bits uint, tgt int) {
	d := (tgt - at) / 4
	lim := 1 << (bits - 1)
	if d < -lim || d >= lim {
		panic(fmt.Sprintf("jit: branch displacement %d instructions does not fit in imm%d", d, bits))
	}
	mask := uint32(1)<<bits - 1
	shift := uint(5)
	if bits == 26 {
		shift = 0
	}
	w := uint32(a.b[at]) | uint32(a.b[at+1])<<8 | uint32(a.b[at+2])<<16 | uint32(a.b[at+3])<<24
	w &^= mask << shift
	w |= (uint32(d) & mask) << shift
	a.b[at] = byte(w)
	a.b[at+1] = byte(w >> 8)
	a.b[at+2] = byte(w >> 16)
	a.b[at+3] = byte(w >> 24)
}

func (a *A64) patch() {
	kept := a.fixup[:0]
	for _, f := range a.fixup {
		tgt := a.labels[f.lbl]
		if tgt < 0 {
			kept = append(kept, f)
			continue
		}
		a.setDisp(f.at, f.bits, tgt)
	}
	a.fixup = kept
}
