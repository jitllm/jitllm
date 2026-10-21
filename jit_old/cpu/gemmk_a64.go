// No build tag and no arch suffix on the filename: emitters are host-pure, so
// arm64 kernels can be generated and disassembled on any host; only executing
// them needs arm64. (EmitGEMM, which collides with an amd64 name, stays in
// gemm_arm64.go.)

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// emitGEMMA64K generates the arm64 prefill kernel for the k-quants.
//
// SDOT is signed x signed, so unlike the x86 kernels there is no bias
// correction: the zero point is subtracted from the weight bytes before the
// dot product.
//
//	Q6_K  q is 0..63, value = d*sc*(q-32)  -> SUB #32, no correction at all
//	Q3_K  u is 0..7,  value = d*sc*(u-4)   -> SUB #4,  no correction at all
//	Q4_K  q is 0..15, value = d*sc*q - dmin*m  -> used as-is; only the MIN term
//	Q5_K  q is 0..31, ... likewise             -> needs sum(a), which ASum is
//
// So AHalfSum is never read here. The by-element SDOT lets one unpacked 16-byte
// weight register serve four k-groups through an immediate index.
func emitGEMMA64K(t quant.Type, mr, nr, nb, window int) ([]byte, error) {
	var (
		bb, qsBase, scBase, dOff int32
		sub                      int32 // elements per weight scale
		zero                     byte  // zero point subtracted from the bytes
		hasMin                   bool
	)
	switch t {
	case quant.Q4_K:
		bb, qsBase, scBase, dOff, sub, hasMin = 144, 16, 4, 0, 32, true
	case quant.Q5_K:
		bb, qsBase, scBase, dOff, sub, hasMin = 176, 48, 4, 0, 32, true
	case quant.Q6_K:
		bb, qsBase, scBase, dOff, sub, zero = 210, 0, 192, 208, 16, 32
	case quant.Q3_K:
		bb, qsBase, scBase, dOff, sub, zero = 110, 32, 96, 108, 16, 4
	default:
		return nil, fmt.Errorf("jit: EmitGEMM: %s is not a k-quant", t)
	}
	if mr < 1 || nr < 1 || nb < 1 {
		return nil, fmt.Errorf("jit: EmitGEMM: tile %dx%d nb=%d out of range", mr, nr, nb)
	}
	rowStr := int32(nb) * bb
	if rowStr > 0xFFFF {
		return nil, fmt.Errorf("jit: EmitGEMM: row stride %d exceeds 65535", rowStr)
	}
	nsub := 256 / sub      // 8 or 16 scales per super-block
	kgPer := sub / 4       // k-groups per scale
	wregs := int(sub / 16) // weight registers per row per sub-block
	tok := int32(8 * nr)
	aKG := tok * 4
	vpk := int(2 * nr) // SDOT vectors per k-group: four tokens each
	outRow := tok * 4

	// V register budget: float accumulators, int32 accumulators, one weight
	// register per row per 16 bytes, the activations of one k-group, and four
	// temporaries. 32 registers total.
	//
	// At window 256 the activation scale is constant across a super-block, so
	// the fold loads it once per super-block instead of once per (sub-block,
	// row, vector); see dareg for why it costs no registers. At window 32 the
	// scales differ every two sub-blocks and the per-sub-block load stays.
	wideAct := window >= 256
	// The whole fold goes integer when there is no min term. At window 256 one
	// d_a and one d cover the super-block, so
	//
	//	out = d * d_a * SUM_s sc[s] * dot_s
	//
	// and every factor inside the sum is an integer (the zero point is already
	// removed from the weight bytes, so dot_s is the bracket). MLA is a 32-bit
	// multiply, so there is no int16 width bound as on amd64; the worst case,
	// Q6_K, is 16*32*127 * 127 * 16 = 132M against int32's 2.1B.
	//
	// Q4_K/Q5_K are excluded: their min term would need the min scales as int32
	// too, and emitA64KScales' intSc converts only the main ones because decode
	// needs the min ones float with dmin folded in. Widening that flag's meaning
	// would apply dmin twice or not at all.
	intFold := wideAct && !hasMin
	// The min-carrying formats get the other half of the idea: at window 256
	// d_a factors out of all sixteen sub-blocks and applies once, deleting an
	// FMUL from both the main and the min term. It borrows intFold's structure:
	// facc becomes the per-super-block total and the row total lives in the
	// scratch slot intFold already reserves.
	daHoist := wideAct && hasMin
	nacc := mr * vpk
	need := 2*nacc + mr*wregs + vpk + 4
	// Whether dareg reserves registers depends on the format: reserving keeps
	// the activation scales loaded from the top of the super-block; not
	// reserving hands those vectors to the unpack's constant pool and loads the
	// scales in the epilogue. Reserve exactly when the pool can still hold every
	// immediate the unpack asks for, since anything the pool cannot hold is
	// materialised per use inside the sub-block loop.
	kneed := map[byte]bool{}
	for _, imm := range a64KUnpackKonsts(t, zero) {
		kneed[imm] = true
	}
	daHold := wideAct && need+vpk+len(kneed) <= 32
	if daHold {
		need += vpk
	}
	if need > 32 {
		return nil, fmt.Errorf("jit: EmitGEMM: tile %dx%d needs %d vectors, have 32", mr, nr, need)
	}
	// The unpack's masks go in whatever the tile did not use. Falls back to
	// materialising per use when the tile leaves nothing spare, so a tile that
	// does not fit the pool is slower, never wrong. The budget is the same at
	// both windows, which TestA64GEMMWindowShrinksTheFold relies on.
	kpool := make([]VReg, 0, 3)
	for r := VReg(31); r > VReg(need-1) && len(kpool) < 3; r-- {
		kpool = append(kpool, r)
	}
	inPool := func(r VReg) bool {
		for _, k := range kpool {
			if k == r {
				return true
			}
		}
		return false
	}
	// Under intFold these registers hold the int32 running total for the current
	// super-block and the float accumulator moves to Scratch, read and written
	// once per super-block, so no third register accumulator set is needed.
	facc := func(m, v int) VReg { return VReg(m*vpk + v) }
	faccOff := int32(mr) * int32(nsub) * 8 // after the per-row scale tiles
	iacc := func(m, v int) VReg { return VReg(nacc + m*vpk + v) }
	wreg := func(m, w int) VReg { return VReg(2*nacc + m*wregs + w) }
	areg := func(v int) VReg { return VReg(2*nacc + mr*wregs + v) }
	t0 := VReg(2*nacc + mr*wregs + vpk)
	t1, t2, t3 := t0+1, t0+2, t0+3
	// dareg holds the super-block's activation scales when they are uniform.
	//
	// It need not reserve anything, because it is read only in the epilogue:
	// wideAct implies intFold or daHoist (exhaustive over hasMin), and both apply
	// d_a once after the sub-block loop, when the integer accumulators are dead.
	// So when the constant pool wants those vectors more (daHold false) the
	// scales live in iacc and load in the epilogue instead.
	dareg := func(v int) VReg {
		if daHold {
			return VReg(need-vpk) + VReg(v)
		}
		return iacc(0, v)
	}

	rowPtr := []XReg{X7, X11, X12, X13, X14, X15, X16, X17}
	if mr > len(rowPtr) {
		return nil, fmt.Errorf("jit: EmitGEMM: tile height %d exceeds %d row cursors", mr, len(rowPtr))
	}

	var a A64
	// The unpack asks for its masks through these; a value already in the pool
	// is reused, and anything past the pool falls back to a scratch register
	// materialised at the point of use.
	//
	// The set is derived from the format (a64KUnpackKonsts, kept beside
	// emitA64KUnpackK) because the MOVIs must precede the row loop.
	konsts := map[byte]VReg{}
	// The order is kept (a slice, not a map) so emission is a deterministic
	// function of its inputs; anything that compares emitted code depends on it.
	korder := make([]byte, 0, 3)
	for _, imm := range a64KUnpackKonsts(t, zero) {
		if len(konsts) >= len(kpool) {
			break
		}
		if _, ok := konsts[imm]; !ok {
			konsts[imm] = kpool[len(konsts)]
			korder = append(korder, imm)
		}
	}
	hoist := func(scratch VReg) a64Konst {
		return func(imm byte) VReg {
			if r, ok := konsts[imm]; ok {
				return r
			}
			a.MOVI16b(scratch, imm)
			return scratch
		}
	}
	kLo, kHi := hoist(t0), hoist(t2)
	// t0..t3 are temporaries of the unpack before the dots and of the fold
	// after them, so a widened SDOT (sdotemu.go) may clobber two of them.
	a.DotScratch(t1, t3)
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 8)  // W
	a.LDRx(X3, X0, 16) // A
	a.LDRx(X4, X0, 24) // AScale
	a.LDRx(X5, X0, 32) // row groups
	a.LDRx(X6, X0, 56) // Scr
	a.LDRx(X19, X0, 72)
	a.LDRx(X20, X0, 96) // ASum, for the min term
	for _, imm := range korder {
		a.MOVI16b(konsts[imm], imm)
	}
	a.MOVimm(X8, int64(rowStr))
	_ = X19

	row := a.Label()
	a.Bind(row)
	a.MOVreg(X9, X3)
	a.MOVreg(X10, X4)
	a.MOVreg(X21, X20)
	a.MOVimm(X22, int64(nb))
	a.MOVreg(rowPtr[0], X2)
	for m := 1; m < mr; m++ {
		a.ADDreg(rowPtr[m], rowPtr[m-1], X8)
	}
	if intFold || daHoist {
		a.MOVIzero(t0)
		for i := 0; i < nacc; i++ {
			a.STRq(t0, X19, faccOff+int32(i)*16)
		}
	} else {
		for m := 0; m < mr; m++ {
			for v := 0; v < vpk; v++ {
				a.MOVIzero(facc(m, v))
			}
		}
	}

	blk := a.Label()
	a.Bind(blk)
	// Scales, once per super-block per row, into Scratch as floats. The integer
	// accumulators are dead here -- they are zeroed per sub-block below -- so
	// they join the temporary pool, which is what makes room for the 6-bit
	// unpacks' table lookups and blends.
	tmps := []VReg{t0, t1, t2, t3}
	for m := 0; m < mr; m++ {
		for v := 0; v < vpk; v++ {
			tmps = append(tmps, iacc(m, v))
		}
	}
	// Not the pool: tmps is handed to emitA64KScales, which may write any of
	// it, and the konsts were MOVI'd once before the row loop and must survive
	// every super-block.
	for r := VReg(need); r < 32; r++ {
		if !inPool(r) {
			tmps = append(tmps, r)
		}
	}
	if len(tmps) < 10 {
		return nil, fmt.Errorf("jit: EmitGEMM: tile %dx%d leaves %d temporaries, need 10 for the scale unpack", mr, nr, len(tmps))
	}
	for m := 0; m < mr; m++ {
		a.emitA64KScales(t, rowPtr[m], X19, int32(m)*int32(nsub)*8, int32(nsub), scBase, dOff, X6, tmps, intFold, nil)
	}

	if daHold {
		// Held across the sub-block loop in its own registers; every
		// sub-block's scale slot holds the same value at window 256.
		for v := 0; v < vpk; v++ {
			a.LDRq(dareg(v), X10, int32(v)*16)
		}
	}
	if intFold || daHoist {
		for m := 0; m < mr; m++ {
			for v := 0; v < vpk; v++ {
				// intFold: the int32 running total for this super-block.
				// daHoist: the float total BEFORE d_a, same scope.
				a.MOVIzero(facc(m, v))
			}
		}
	}
	for s := int32(0); s < int32(nsub); s++ {
		for m := 0; m < mr; m++ {
			for v := 0; v < vpk; v++ {
				a.MOVIzero(iacc(m, v))
			}
		}
		for m := 0; m < mr; m++ {
			a.emitA64KUnpackK(t, rowPtr[m], qsBase, s, sub, zero, wreg(m, 0), t1, kLo, kHi)
		}
		for g := int32(0); g < kgPer; g++ {
			for v := 0; v < vpk; v++ {
				a.LDRq(areg(v), X9, (s*kgPer+g)*aKG+int32(v)*16)
			}
			for m := 0; m < mr; m++ {
				w := wreg(m, int(g/4))
				for v := 0; v < vpk; v++ {
					a.SDOTelem(iacc(m, v), areg(v), w, uint8(g%4))
				}
			}
		}
		// Fold. Under intFold this is one instruction per (sub-block, row,
		// vector) against three: `MLA itot.4s, iacc.4s, sc[s]` with the scale
		// still an integer.
		if intFold {
			for m := 0; m < mr; m++ {
				a.LDRs(t0, X19, int32(m)*int32(nsub)*8+s*4) // sc[s], int32, lane 0
				for v := 0; v < vpk; v++ {
					a.MLAelem(facc(m, v), iacc(m, v), t0, 0)
				}
			}
			continue
		}
		// Fold: scale by d*sc[s] and the four per-token activation scales.
		for m := 0; m < mr; m++ {
			a.LDRs(t0, X19, int32(m)*int32(nsub)*8+s*4) // d*sc[s], lane 0
			for v := 0; v < vpk; v++ {
				a.SCVTF4s(iacc(m, v), iacc(m, v))
				if daHoist {
					// d_a is uniform over the super-block and applies once, in
					// the epilogue.
					a.FMLAelem(facc(m, v), iacc(m, v), t0, 0)
					continue
				}
				// Only reachable at a narrow window: wideAct takes one of the
				// two hoists above, which are exhaustive over hasMin.
				a.LDRq(t1, X10, (s/(32/sub))*(tok*4)+int32(v)*16) // four tokens' d_a
				a.FMUL4s(t1, t1, iacc(m, v))
				a.FMLAelem(facc(m, v), t1, t0, 0)
			}
			if hasMin {
				a.LDRs(t2, X19, int32(m)*int32(nsub)*8+int32(nsub)*4+s*4) // dmin*m[s]
				for v := 0; v < vpk; v++ {
					a.LDRq(t1, X21, s*(tok*4)+int32(v)*16) // sum(a), int32
					a.SCVTF4s(t1, t1)
					if daHoist {
						a.FMLSelem(facc(m, v), t1, t2, 0) // ... SUBTRACTED
						continue
					}
					a.LDRq(t3, X10, (s/(32/sub))*(tok*4)+int32(v)*16)
					a.FMUL4s(t1, t1, t3)
					a.FMLSelem(facc(m, v), t1, t2, 0) // ... SUBTRACTED
				}
			}
		}
	}

	// The one float pass of the integer fold, once per (row, vector) per
	// super-block: d from the block header, d_a from the hoisted activation
	// scale, and the running float accumulator read and written in Scratch.
	if (intFold || daHoist) && !daHold {
		// One load per vector for the whole super-block, into accumulators that
		// are dead until the next sub-block zeroes them. X10 has not advanced
		// yet, so it still addresses this super-block's scales.
		for v := 0; v < vpk; v++ {
			a.LDRq(dareg(v), X10, int32(v)*16)
		}
	}
	if intFold {
		for m := 0; m < mr; m++ {
			a.LDRh(t2, rowPtr[m], dOff)
			a.FCVTsh(t2, t2)
			for v := 0; v < vpk; v++ {
				a.SCVTF4s(facc(m, v), facc(m, v))
				a.FMUL4s(facc(m, v), facc(m, v), dareg(v))
				a.LDRq(t1, X19, faccOff+int32(m*vpk+v)*16)
				a.FMLAelem(t1, facc(m, v), t2, 0)
				a.STRq(t1, X19, faccOff+int32(m*vpk+v)*16)
			}
		}
	}
	// daHoist's epilogue: d is already inside d*sc[s], so this is only the
	// activation scale and the running total, once per (row, vector) per
	// super-block.
	if daHoist {
		for m := 0; m < mr; m++ {
			for v := 0; v < vpk; v++ {
				a.LDRq(t1, X19, faccOff+int32(m*vpk+v)*16)
				a.FMLA4s(t1, facc(m, v), dareg(v))
				a.STRq(t1, X19, faccOff+int32(m*vpk+v)*16)
			}
		}
	}

	for m := 0; m < mr; m++ {
		a.ADDimm(rowPtr[m], rowPtr[m], bb)
	}
	// ADD's immediate is twelve bits. One super-block of activations is
	// tok*4*64 bytes, which reaches 4096 at a 16-token tile and overflows it --
	// as a panic at emit time rather than a wrong address.
	a.addBig(X9, aKG*64, X23)
	a.addBig(X10, tok*4*8, X23)
	a.addBig(X21, tok*4*int32(nsub), X23)
	a.SUBimm(X22, X22, 1)
	a.CBNZ(X22, blk)

	for m := 0; m < mr; m++ {
		for v := 0; v < vpk; v++ {
			if intFold || daHoist {
				a.LDRq(t0, X19, faccOff+int32(m*vpk+v)*16)
				a.STRq(t0, X1, int32(m)*outRow+int32(v)*16)
				continue
			}
			a.STRq(facc(m, v), X1, int32(m)*outRow+int32(v)*16)
		}
	}
	for m := 0; m < mr; m++ {
		a.ADDreg(X2, X2, X8)
	}
	a.addBig(X1, int32(mr)*outRow, X23)
	a.SUBimm(X5, X5, 1)
	a.CBNZ(X5, row)

	a.RET()
	return a.Bytes(), nil
}

// a64Konst supplies a register holding an 8-bit immediate broadcast across 16
// lanes. The GEMM hoists what fits its pool and the decode kernel hoists them
// out of every loop: the values are loop-invariant, and on an issue-bound core
// rebuilding them per use is a real share of the unpack.
type a64Konst func(imm byte) VReg

// a64KUnpackKonsts is every immediate emitA64KUnpackK asks for, per format.
// It lives next to that function because the two must agree; a value missing
// here is materialised per use, which is slower rather than wrong.
func a64KUnpackKonsts(t quant.Type, zero byte) []byte {
	switch t {
	case quant.Q6_K:
		return []byte{0x0F, 0x30, zero}
	case quant.Q3_K:
		return []byte{0x03, 0x04, zero}
	case quant.Q4_K:
		return []byte{0x0F, zero}
	case quant.Q5_K:
		return []byte{0x0F, 0x10, zero}
	}
	return nil
}

// inlineKonst is the GEMM's behaviour: materialise into scratch at each use.
func inlineKonst(a *A64, scratch VReg) a64Konst {
	return func(imm byte) VReg {
		a.MOVI16b(scratch, imm)
		return scratch
	}
}

// emitA64KUnpack lays `sub` weight elements of one row's sub-block s into
// consecutive registers starting at wdst, as signed bytes with the format's zero
// point already removed, so SDOT can consume them directly.
func (a *A64) emitA64KUnpack(t quant.Type, w XReg, qsBase, s, sub int32, zero byte, wdst, t0, t1, t2 VReg) {
	a.emitA64KUnpackK(t, w, qsBase, s, sub, zero, wdst, t1, inlineKonst(a, t0), inlineKonst(a, t2))
}

func (a *A64) emitA64KUnpackK(t quant.Type, w XReg, qsBase, s, sub int32, zero byte, wdst, t1 VReg, kLo, kHi a64Konst) {
	switch t {
	case quant.Q6_K:
		// Sixteen consecutive elements share a quarter, so they share a nibble
		// half and a qh shift, and land on sixteen consecutive bytes of each.
		e := s * 16
		grp, ep := e/128, e%128
		q, l := ep/32, ep%32
		a.LDURq(wdst, w, grp*64+(q&1)*32+l)
		if q >= 2 {
			a.USHR16b(wdst, wdst, 4)
		}
		a.AND16b(wdst, wdst, kLo(0x0F))
		a.LDURq(t1, w, 128+grp*32+l)
		if sh := q * 2; sh <= 4 {
			a.SHL16b(t1, t1, byte(4-sh))
		} else {
			a.USHR16b(t1, t1, byte(sh-4))
		}
		a.AND16b(t1, t1, kHi(0x30))
		a.ORR16b(wdst, wdst, t1)
	case quant.Q3_K:
		// The high mask is inverted (a set bit means no -4) and hmask does not
		// re-base for the second 128-element group, exactly as on x86.
		e := s * 16
		grp, ep := e/128, e%128
		j, bs := ep/32, ((ep%32)/16)*16
		a.LDURq(wdst, w, qsBase+grp*32+bs)
		if sh := byte(2 * j); sh != 0 {
			a.USHR16b(wdst, wdst, sh)
		}
		a.AND16b(wdst, wdst, kLo(0x03))
		a.LDURq(t1, w, bs) // hmask at offset 0, never re-based
		if hb := grp*4 + j; hb <= 2 {
			a.SHL16b(t1, t1, byte(2-hb))
		} else {
			a.USHR16b(t1, t1, byte(hb-2))
		}
		a.AND16b(t1, t1, kHi(0x04))
		a.ORR16b(wdst, wdst, t1)
	default: // Q4_K, Q5_K: 32 elements from 32 bytes, one nibble each
		for h := int32(0); h < 2; h++ {
			d := wdst + VReg(h)
			a.LDURq(d, w, qsBase+(s/2)*32+h*16)
			if s%2 == 1 {
				a.USHR16b(d, d, 4)
			}
			a.AND16b(d, d, kLo(0x0F))
			if t == quant.Q5_K {
				// qh does NOT advance per group: the same 32 bytes serve all
				// four, one bit per sub-block.
				a.LDURq(t1, w, 16+h*16)
				if s <= 4 {
					a.SHL16b(t1, t1, byte(4-s))
				} else {
					a.USHR16b(t1, t1, byte(s-4))
				}
				a.AND16b(t1, t1, kHi(0x10))
				a.ORR16b(d, d, t1)
			}
		}
		return // no zero point: Q4_K/Q5_K quants are used as-is, with a min term
	}
	if zero != 0 {
		a.SUB16b(wdst, wdst, kLo(zero))
	}
}

// emitA64KScales writes one row's per-sub-block scales into Scratch as floats:
// nsub of d*sc[i], then for the affine formats nsub more of dmin*m[i].
//
// NEON has TBL for the packed 6-bit scale gather but no dword blend, so a blend
// is spelled as individual lane moves. TBL's out-of-range index yields zero,
// which the Q4_K donor table relies on.
//
// intSc stores the main scales as int32 without d, so Sum_s sc[s]*dot_s stays
// exact and d applies once; the min scales carry dmin and stay float. Prefill
// passes false.
//
// scDst keeps the sixteen int32 scales in registers instead of Scratch, saving
// a store and four reloads per super-block. Only the 16-sub-block formats
// (Q3_K, Q6_K) have the four spare vectors for it.
func (a *A64) emitA64KScales(t quant.Type, w, scratch XReg, off, nsub, scBase, dOff int32, konst XReg, tmp []VReg, intSc bool, scDst []VReg) {
	// Every temporary is named once and never aliased: reusing a tmp under two
	// names lets the second write destroy the first operand.
	d, x, u, v := tmp[0], tmp[1], tmp[2], tmp[3]
	res, sh := tmp[4], tmp[5]
	m0F, m3F, m30, w0 := tmp[6], tmp[7], tmp[8], tmp[9]

	a.LDRh(d, w, dOff)
	a.FCVTsh(d, d)

	// widen16 turns sixteen bytes into four float vectors, scaled by d.
	// dstFor names the register a widened quarter should be built in, so the
	// widen writes straight there with no store or move.
	dstFor := func(at int32) (VReg, bool) {
		if scDst != nil && at >= off && at < off+nsub*4 {
			if i := (at - off) / 16; int(i) < len(scDst) {
				return scDst[i], true
			}
		}
		return 0, false
	}
	widen16 := func(src VReg, at int32, signed bool) {
		half := func(hi bool) {
			if signed {
				if hi {
					a.SXTL2_8h(w0, src)
				} else {
					a.SXTL8h(w0, src)
				}
			} else if hi {
				a.UXTL2_8h(w0, src)
			} else {
				a.UXTL8h(w0, src)
			}
			for _, top := range []bool{false, true} {
				n := int32(0)
				if hi {
					n += 32
				}
				if top {
					n += 16
				}
				out, inReg := dstFor(at + n)
				if !inReg {
					out = sh
				}
				if signed {
					if top {
						a.SXTL2_4s(out, w0)
					} else {
						a.SXTL4s(out, w0)
					}
					if !intSc {
						a.SCVTF4s(out, out)
					}
				} else {
					if top {
						a.UXTL2_4s(out, w0)
					} else {
						a.UXTL4s(out, w0)
					}
					if !intSc {
						a.UCVTF4s(out, out)
					}
				}
				if !intSc {
					a.FMULelem(out, out, d, 0)
				}
				if !inReg {
					a.STRq(sh, scratch, at+n)
				}
			}
		}
		half(false)
		half(true)
	}

	switch t {
	case quant.Q6_K:
		// Sixteen plain signed bytes: no unpacking, just widening.
		a.LDURq(x, w, scBase)
		widen16(x, off, true)

	case quant.Q3_K:
		// Sixteen 6-bit scales across twelve bytes, biased by 32, in a third
		// packing arrangement. Same recipe as emitQ3KMatVec: TBL replaces
		// VPSHUFB, and a dword blend becomes individual lane moves.
		a.LDURq(x, w, scBase)
		a.LDURq(res, konst, 0)
		a.TBL(u, x, res) // A
		a.LDURq(res, konst, 32)
		a.TBL(v, x, res) // B, the high-bit donors
		a.MOVI16b(m0F, 0x0F)
		a.MOVI16b(m30, 0x30)
		a.AND16b(res, u, m0F)
		a.USHR16b(sh, u, 4)
		a.AND16b(sh, sh, m0F)
		a.INSs(res, 2, sh, 2) // dwords 2 and 3 take the shifted copy
		a.INSs(res, 3, sh, 3)
		// Each dword wants a different shift of the same donors, and
		// ((x >> s) & 3) << 4 collapses to one shift masked with 0x30.
		a.SHL16b(x, v, 4) // s=0
		a.SHL16b(sh, v, 2)
		a.INSs(x, 1, sh, 1) // s=2
		a.INSs(x, 2, v, 2)  // s=4 needs no shift
		a.USHR16b(sh, v, 2)
		a.INSs(x, 3, sh, 3) // s=6
		a.AND16b(x, x, m30)
		a.ORR16b(res, res, x)
		a.MOVI16b(sh, 0x20)
		a.SUB16b(res, res, sh) // bias by 32 as BYTES, so the widen is signed
		widen16(res, off, true)

	default: // Q4_K, Q5_K: eight unsigned (sc, m) pairs
		a.LDURq(x, w, scBase)
		a.LDURq(res, konst, 0)
		a.TBL(u, x, res) // [b0-3 | b8-11 | b4-7 | b8-11]
		a.LDURq(res, konst, 32)
		a.TBL(v, x, res) // [_ | b0-3 | _ | b4-7]
		a.MOVI16b(m0F, 0x0F)
		a.MOVI16b(m3F, 0x3F)
		a.MOVI16b(m30, 0x30)
		a.USHR16b(x, v, 2)
		a.AND16b(x, x, m30) // P = (donor >> 6) << 4
		a.AND16b(res, u, m3F)
		a.AND16b(sh, u, m0F)
		a.ORR16b(sh, sh, x) // sc[4..7]
		a.INSs(res, 1, sh, 1)
		a.USHR16b(sh, u, 4)
		a.AND16b(sh, sh, m0F)
		a.ORR16b(sh, sh, x) // m[4..7]
		a.INSs(res, 3, sh, 3)
		// res holds sc[0..7] in bytes 0..7 and m[0..7] in bytes 8..15. The two
		// halves take different scales, so they are widened separately.
		a.UXTL8h(w0, res)
		a.UXTL4s(sh, w0)
		if !intSc {
			a.UCVTF4s(sh, sh)
			a.FMULelem(sh, sh, d, 0)
		}
		a.STRq(sh, scratch, off)
		a.UXTL2_4s(sh, w0)
		if !intSc {
			a.UCVTF4s(sh, sh)
			a.FMULelem(sh, sh, d, 0)
		}
		a.STRq(sh, scratch, off+16)
		a.LDRh(d, w, dOff+2) // dmin
		a.FCVTsh(d, d)
		a.UXTL2_8h(w0, res)
		a.UXTL4s(sh, w0)
		a.UCVTF4s(sh, sh)
		a.FMULelem(sh, sh, d, 0)
		a.STRq(sh, scratch, off+nsub*4)
		a.UXTL2_4s(sh, w0)
		a.UCVTF4s(sh, sh)
		a.FMULelem(sh, sh, d, 0)
		a.STRq(sh, scratch, off+nsub*4+16)
	}
}

// addBig advances a pointer by an offset that may not fit ADD's twelve-bit
// immediate, staging it through a scratch register when it does not.
func (a *A64) addBig(rd XReg, imm int32, tmp XReg) {
	if imm >= 0 && imm <= 4095 {
		a.ADDimm(rd, rd, imm)
		return
	}
	a.MOVimm(tmp, int64(imm))
	a.ADDreg(rd, rd, tmp)
}
