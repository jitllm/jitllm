package cpu

import "github.com/jitllm/jitllm/jit_old/gpu/kernels"

// The host readers of container v27's SC stream (kernels.ScStream: Q4_K and
// Q5_K's sixteen 6-bit scale/min fields in three words a super-block). Every
// host kernel unrolls a super-block's eight sub-blocks, so a field's word and
// bit are emit-time constants and a field inside one word costs the same two
// shifts a byte did. What changes is the cursor: it walks three words a
// super-block instead of four, and two fields straddle into the NEXT word.

// scCursorWord is the stream word the SC cursor holds while sub-block sn runs:
// the one its scale starts in.
func scCursorWord(sn int) int { w, _, _ := kernels.ScField(sn, false); return w }

// scStep reports whether the SC cursor steps one plane stride after sub-block
// sn of a super-block's eight: when the next sub-block's scale starts in the
// next word, and after the last (onto the next super-block's first word).
func scStep(sn int) bool { return sn == 7 || scCursorWord(sn+1) != scCursorWord(sn) }

// scOps is the three lane-wise u32 operations a tier reads the stream with.
type scOps[R comparable] struct {
	shl, shr func(dst, src R, n byte)
	or       func(dst, a, b R)
}

// scField emits dst = field (sn, min) of the stream as u32 lanes. cur holds the
// cursor's word (scCursorWord(sn)); next loads the following word into its
// argument. tmp is written only when the field straddles, and must differ
// from dst; dst may be cur.
func scField[R comparable](o scOps[R], dst, cur, tmp R, sn int, min bool, next func(R)) {
	w, b, hi := kernels.ScField(sn, min)
	switch rel := w - scCursorWord(sn); {
	case rel == 0 && hi == 0:
		o.shl(dst, cur, byte(26-b))
		o.shr(dst, dst, 26)
	case rel == 0:
		if dst == tmp {
			panic("cpu: scField: a straddling field needs a scratch other than its destination")
		}
		o.shr(dst, cur, byte(b))
		next(tmp)
		o.shl(tmp, tmp, byte(32-hi))
		o.shr(tmp, tmp, 26)
		o.or(dst, dst, tmp)
	default:
		next(dst)
		o.shl(dst, dst, byte(26-b))
		o.shr(dst, dst, 26)
	}
}

// scNeedsNext reports whether field (sn, min) reads the word after the
// cursor's.
func scNeedsNext(sn int, min bool) bool {
	w, _, hi := kernels.ScField(sn, min)
	return hi > 0 || w != scCursorWord(sn)
}

// avxSC is scOps on AVX2.
func avxSC(a *Buf) scOps[Reg] {
	return scOps[Reg]{shl: a.VPSLLD, shr: a.VPSRLD, or: a.VPOR}
}

// sseSC is scOps on the SSE tier.
func sseSC(a *Buf) scOps[Reg] { return scOps[Reg]{shl: a.PSLLD, shr: a.PSRLD, or: a.POR} }

// a64SC is scOps on NEON.
func a64SC(a *A64) scOps[VReg] {
	return scOps[VReg]{shl: a.SHL4s, shr: a.USHR4s, or: a.ORR16b}
}
