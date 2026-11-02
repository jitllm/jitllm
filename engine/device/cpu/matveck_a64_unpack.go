package cpu

import "github.com/jitllm/jitllm/format/quant"

// The decode kernel's own weight unpack, written so that every load and every
// constant can be hoisted out of the sub-block loop.
//
// It is separate from the GEMM's emitA64KUnpack on purpose: that one shifts its
// loaded register in place, which is shorter for single use; decode never
// writes a loaded register, so the same sixteen bytes serve several sub-blocks
// (Q3_K's sixteen sub-blocks touch only six distinct offsets).
//
// ld returns a register holding the sixteen bytes at an offset, and must not be
// written to by the caller. konst returns a register holding an 8-bit immediate
// broadcast across sixteen lanes. Both may fall back to re-emitting.
func (a *A64) emitA64KDecodeUnpack(t quant.Type, qsBase, s int32, zero byte,
	wdst, t1 VReg, ld func(int32) VReg, konst func(byte) VReg) {

	// shiftInto puts src >> sh (or src itself) into dst, then masks it.
	andShifted := func(dst, src VReg, sh byte, mask byte) {
		if sh == 0 {
			a.AND16b(dst, src, konst(mask))
			return
		}
		a.USHR16b(dst, src, sh)
		a.AND16b(dst, dst, konst(mask))
	}

	switch t {
	case quant.Q6_K:
		e := s * 16
		grp, ep := e/128, e%128
		q, l := ep/32, ep%32
		lo := ld(grp*64 + (q&1)*32 + l)
		sh := byte(0)
		if q >= 2 {
			sh = 4
		}
		andShifted(wdst, lo, sh, 0x0F)
		hi := ld(128 + grp*32 + l)
		if s := q * 2; s <= 4 {
			a.SHL16b(t1, hi, byte(4-s))
		} else {
			a.USHR16b(t1, hi, byte(s-4))
		}
		a.AND16b(t1, t1, konst(0x30))
		a.ORR16b(wdst, wdst, t1)
	case quant.Q3_K:
		// The high mask is inverted and hmask does not re-base for the second
		// 128-element group, exactly as on x86.
		e := s * 16
		grp, ep := e/128, e%128
		j, bs := ep/32, ((ep%32)/16)*16
		andShifted(wdst, ld(qsBase+grp*32+bs), byte(2*j), 0x03)
		hm := ld(bs) // hmask at offset 0, never re-based
		if hb := grp*4 + j; hb <= 2 {
			a.SHL16b(t1, hm, byte(2-hb))
		} else {
			a.USHR16b(t1, hm, byte(hb-2))
		}
		// The high bit and the zero point collapse into one subtract: the value
		// is qs + 4*hbit - 4, i.e. qs minus 4 exactly when hbit is clear, and
		// BIC computes 4 &^ shifted in one instruction (which also puts the
		// inverted polarity in the instruction). This consumes the zero point,
		// so the shared SUB below must not run: the return is load-bearing.
		a.BIC16b(t1, konst(0x04), t1)
		a.SUB16b(wdst, wdst, t1)
		return
	default: // Q4_K, Q5_K: 32 elements from 32 bytes, one nibble each
		for h := int32(0); h < 2; h++ {
			d := wdst + VReg(h)
			sh := byte(0)
			if s%2 == 1 {
				sh = 4
			}
			andShifted(d, ld(qsBase+(s/2)*32+h*16), sh, 0x0F)
			if t == quant.Q5_K {
				// qh does not advance per group: the same 32 bytes serve all
				// four, one bit per sub-block.
				qh := ld(16 + h*16)
				if s <= 4 {
					a.SHL16b(t1, qh, byte(4-s))
				} else {
					a.USHR16b(t1, qh, byte(s-4))
				}
				a.AND16b(t1, t1, konst(0x10))
				a.ORR16b(d, d, t1)
			}
		}
		return // no zero point: Q4_K/Q5_K quants are used as-is, with a min term
	}
	if zero != 0 {
		a.SUB16b(wdst, wdst, konst(zero))
	}
}
