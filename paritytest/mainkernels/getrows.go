package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// GetRowsPacked gathers ntok rows of a packed weight into float32:
// out[t*k + e] = scale * W[ids[t]][e], with W in the device layout the matvecs
// read ([sub-block][word][row], see PackWeights). It is the token embedding
// lookup of a model whose output projection is its embedding (tied), done on
// the card that already holds the projection.
//
// In this layout one row's words sit nrows*4 bytes apart, so a host lookup is
// a cache miss per word. Here thread (t, s) decodes sub-block s of row ids[t],
// with lanes covering neighbouring rows' words, as ggml's k_get_rows does.
//
// The arithmetic is oracle.RowPacked's term for term: value = scale*q - bias,
// the planes combined as primary | hi<<4, then times the model's embedding
// scale, in the host's order. Parameters: pQS, pD, pSC (the resident weight's
// three buffers), pIds (u32 rows; nrows or more is a padding row of zeros),
// pOut. A launch may cover fewer than ntok rows: ntok*(k/sub) threads at most
// in groups of 128; the grid is clamped to its last item, which only
// re-stores the same values.
func GetRowsPacked(q Quant, nrows, k, ntok int, scale float32) (*ir.Kernel, error) {
	qi := q.info()
	switch {
	case qi.float > 0:
		return nil, fmt.Errorf("kernels: GetRowsPacked: %v is a float format", q)
	case qi.codes != nil:
		return nil, fmt.Errorf("kernels: GetRowsPacked: %v has a code table", q)
	case DeviceWhyNot(q) != "":
		return nil, fmt.Errorf("kernels: GetRowsPacked: %s", DeviceWhyNot(q))
	case qi.blockE == 0 || k <= 0 || k%qi.blockE != 0:
		return nil, fmt.Errorf("kernels: GetRowsPacked: k=%d is not a multiple of %v's %d", k, q, qi.blockE)
	case nrows <= 0 || ntok <= 0:
		return nil, fmt.Errorf("kernels: GetRowsPacked: nrows=%d ntok=%d", nrows, ntok)
	case qi.bits != 2 && qi.bits != 4 && qi.bits != 8:
		return nil, fmt.Errorf("kernels: GetRowsPacked: %v is %d bits", q, qi.bits)
	}
	nsub := k / qi.sub
	pw := qi.sub * qi.bits / 32
	words := pw + qi.sub*qi.hi/32

	b := ir.New(fmt.Sprintf("getrows_%v", q), [3]int{128, 1, 1})
	pQS := b.Param("pQS", ir.U32)
	pD := b.Param("pD", ir.U32)
	pSC := b.Param("pSC", ir.U32)
	pIds := b.Param("pIds", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	f := b.ConstF32

	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	t := b.Min(ir.U32, tid, c(int64(ntok*nsub-1)))
	tok := b.Div(ir.U32, t, c(int64(nsub)))
	si := b.Rem(ir.U32, t, c(int64(nsub)))
	// An id past the vocabulary is a padding row and gathers zeros: a prompt
	// chunk's scratch has zeroed surplus rows, and the caller pads the ids with
	// nrows. The load is clamped to the last row; the Select discards it.
	id := b.Load(ir.U32, pIds, tok, 0)
	valid := b.Lt(ir.U32, id, c(int64(nrows)))
	r := b.Min(ir.U32, id, c(int64(nrows-1)))
	super := b.Div(ir.U32, si, c(int64(qi.perSuper)))

	// The super-scale, through kernels.DIndex's arithmetic (shift and mask:
	// DSlots is a power of two).
	var dw ir.Value
	if n := DSlots(q); n > 1 {
		idx := b.Add(ir.U32, b.Mul(ir.U32, super, c(int64((nrows+n-1)/n))), b.Shr(ir.U32, r, c(int64(log2(n)))))
		w := b.Load(ir.U32, pD, idx, 0)
		slot := b.And(ir.U32, r, c(int64(n-1)))
		dw = b.Shr(ir.U32, w, b.Mul(ir.U32, slot, c(int64(32/n))))
	} else {
		dw = b.Load(ir.U32, pD, b.Add(ir.U32, b.Mul(ir.U32, super, c(int64(nrows))), r), 0)
	}
	var d ir.Value
	switch {
	case qi.e8m0:
		d = b.Bitcast(ir.F32, b.Shl(ir.U32, b.And(ir.U32, dw, c(0xFF)), c(23)))
	case qi.d32:
		d = b.Bitcast(ir.F32, dw) // TQ2F: the word is the scale
	default:
		d = b.CvtF16H(dw)
	}
	sc := d
	var bias ir.Value
	if ScStream(q) {
		// Container v27's stream (scstream.go).
		lo, hi, bit := scStreamAt(b, si)
		ld := func(w ir.Value) ir.Value {
			return b.Load(ir.U32, pSC, b.Add(ir.U32, b.Mul(ir.U32, w, c(int64(nrows))), r), 0)
		}
		sb, mb := scStreamPair(b, ld(lo), ld(hi), bit)
		sc = b.Mul(ir.F32, d, b.Add(ir.F32, b.CvtF32(sb), f(float32(qi.scOff))))
		bias = b.Mul(ir.F32, b.CvtF16H(b.Shr(ir.U32, dw, c(16))), b.CvtF32(mb))
	} else if qi.perSuper > 1 {
		perWord, stride := 4, 8
		if qi.biasArray {
			perWord, stride = 2, 16
		}
		sw := b.Load(ir.U32, pSC, b.Add(ir.U32,
			b.Mul(ir.U32, b.Shr(ir.U32, si, c(int64(log2(perWord)))), c(int64(nrows))), r), 0)
		sh := b.Mul(ir.U32, b.And(ir.U32, si, c(int64(perWord-1))), c(int64(stride)))
		sb := b.And(ir.U32, b.Shr(ir.U32, sw, sh), c(0xFF))
		sc = b.Mul(ir.F32, d, b.Add(ir.F32, b.CvtF32(sb), f(float32(qi.scOff))))
		if qi.biasArray {
			dmin := b.CvtF16H(b.Shr(ir.U32, dw, c(16)))
			mb := b.And(ir.U32, b.Shr(ir.U32, sw, b.Add(ir.U32, sh, c(8))), c(0xFF))
			bias = b.Mul(ir.F32, dmin, b.CvtF32(mb))
		}
	} else if qi.biasArray {
		bias = b.CvtF16H(b.Shr(ir.U32, dw, c(16))) // Q5_1: the minimum alone (MinInD)
	}
	if !qi.biasArray {
		bias = b.Mul(ir.F32, sc, f(qi.biasK))
	}

	base := b.Add(ir.U32, b.Mul(ir.U32, si, c(int64(words*nrows))), r)
	e0 := b.Add(ir.U32, b.Mul(ir.U32, tok, c(int64(k))), b.Mul(ir.U32, si, c(int64(qi.sub))))
	emb := f(scale)
	put := func(l int, qv ir.Value) {
		v := b.Sub(ir.F32, b.Mul(ir.F32, sc, qv), bias)
		if scale != 1 {
			v = b.Mul(ir.F32, v, emb)
		}
		b.Store(pOut, e0, b.Select(ir.F32, valid, v, f(0)), int64(l))
	}
	if qi.bits == 8 {
		for w := 0; w < pw; w++ {
			v := b.Load(ir.U32, pQS, base, int64(w*nrows))
			for bb := 0; bb < 4; bb++ {
				by := b.And(ir.U32, b.Shr(ir.U32, v, c(int64(8*bb))), c(0xFF))
				qv := b.CvtF32(by)
				if qi.signedQ {
					// int8: subtract 256 where the top bit is set.
					qv = b.Sub(ir.F32, qv, b.Mul(ir.F32, f(256), b.CvtF32(b.Shr(ir.U32, by, c(7)))))
				}
				put(w*4+bb, qv)
			}
		}
		return b.Done(), nil
	}
	if qi.bits == 2 {
		// TQ2: byte p's field fi is element p + fi*run (packSub).
		run := qi.sub / 4
		for w := 0; w < pw; w++ {
			v := b.Load(ir.U32, pQS, base, int64(w*nrows))
			for bb := 0; bb < 4; bb++ {
				for fi := 0; fi < 4; fi++ {
					q := b.And(ir.U32, b.Shr(ir.U32, v, c(int64(8*bb+2*fi))), c(3))
					put(w*4+bb+fi*run, b.CvtF32(q))
				}
			}
		}
		return b.Done(), nil
	}
	// The secondary plane: element l's code is (word >> (8*(l%4) + hi*((l/4)%lanes)))
	// & (1<<hi - 1), word (pw + (l/4)/lanes) of the sub-block. See packSub.
	lanes := 1
	if qi.hi > 0 {
		lanes = 8 / qi.hi
	}
	hiWords := map[int]ir.Value{}
	hiOf := func(l int) ir.Value {
		g := l / 4
		wi := pw + g/lanes
		hwv, ok := hiWords[wi]
		if !ok {
			hwv = b.Load(ir.U32, pQS, base, int64(wi*nrows))
			hiWords[wi] = hwv
		}
		sh := 8*(l%4) + qi.hi*(g%lanes)
		return b.And(ir.U32, b.Shr(ir.U32, hwv, c(int64(sh))), c(int64(1<<qi.hi-1)))
	}
	h := qi.sub / 2
	for w := 0; w < pw; w++ {
		v := b.Load(ir.U32, pQS, base, int64(w*nrows))
		for bb := 0; bb < 4; bb++ {
			by := b.Shr(ir.U32, v, c(int64(8*bb)))
			l := w*4 + bb
			q0 := b.And(ir.U32, by, c(0xF))
			q1 := b.And(ir.U32, b.Shr(ir.U32, by, c(4)), c(0xF))
			if qi.hi > 0 {
				q0 = b.Add(ir.U32, q0, b.Shl(ir.U32, hiOf(l), c(4)))
				q1 = b.Add(ir.U32, q1, b.Shl(ir.U32, hiOf(l+h), c(4)))
			}
			put(l, b.CvtF32(q0))
			put(l+h, b.CvtF32(q1))
		}
	}
	return b.Done(), nil
}

// GetRowsThreads is how many threads GetRowsPacked launches.
func GetRowsThreads(q Quant, k, ntok int) int { return ntok * (k / q.info().sub) }

// CopySpan copies n floats from src at a runtime offset: dst[i] = src[off[0]+i].
// It moves one row of a chunk's residual into a one-row scratch -- the last
// row, which is the only one a prefill's output projection reads -- without a
// kernel per row index. Launch n threads in groups of 128; the grid is clamped.
func CopySpan(n int) (*ir.Kernel, error) {
	if n <= 0 {
		return nil, fmt.Errorf("kernels: CopySpan: n=%d", n)
	}
	b := ir.New("copyspan", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pDst := b.Param("pDst", ir.F32)
	pOff := b.Param("pOff", ir.U32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, int64(n-1)))
	off := b.Load(ir.U32, pOff, b.Const(ir.U32, 0), 0)
	b.Store(pDst, i, b.Load(ir.F32, pSrc, b.Add(ir.U32, off, i), 0), 0)
	return b.Done(), nil
}
