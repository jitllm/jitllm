package oldkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// SubDotEnv is what a matvec's sub-block arithmetic (SubDot) reads of the
// launch around it: the shape (format, k, rows and tokens a thread, the
// gate's second stream, the operand's slots), the operand planes, the
// weights' addressing (the stride between a row's words, the scale planes'
// row bases, a narrow d plane's row select and an expert's base) and the
// constants and policies the loop shares (centering, the dot instruction).
// The matvec former that owns the loop nest fills it; SubDot is the op's
// arithmetic, the former the loop.
type SubDotEnv struct {
	S                             MatVecShape
	PA, PAX, PX                   ir.Value
	Mask, Four, ZeroF, One, NegBK ir.Value
	Center                        func(ir.Value) ir.Value
	Dot4                          func(a, b, c ir.Value) ir.Value
	Centered                      bool
	HW, PW, AWords, NB, FW, FSub  int
	BankRows, WSel, WRowD, WRowSC ir.Value
	WRows                         int
	DExpAdd                       func(ir.Value) ir.Value
	TileOff                       func(int) int64
	SlotAct                       bool
	AxS, AxU                      ir.Value
	SumOff                        int64
}

// SubDot is a weight format's arithmetic over one sub-block of a matvec's k
// loop, for the iteration whose payload, operand and sub-block indices are
// wIdx, aIdx and subIdx: the returned stream emits one weight stream's
// payload against the operand -- unpacking, centering, the integer or float
// dot, the super-block and sub-block scales, the bias against the operand's
// sums -- into accs (float, a phi each) and iaccs (integer), and returns the
// payload index past the sub-block. The gate's second stream calls it again
// with its own planes and accumulators, sharing the operand's loads.
func SubDot(b *ir.Builder, e *SubDotEnv, wIdx, aIdx, subIdx ir.Value) func(pQS, pD, pSC ir.Value, accs, iaccs []ir.Value) ir.Value {
	s, qi := e.S, infoOf(e.S.T)
	pA, pAX, pX := e.PA, e.PAX, e.PX
	mask, four, zeroF, one, negBiasK := e.Mask, e.Four, e.ZeroF, e.One, e.NegBK
	center, dot4, centered := e.Center, e.Dot4, e.Centered
	hw, pw, awords, nb, fw, fsub := e.HW, e.PW, e.AWords, e.NB, e.FW, e.FSub
	bankRows, wsel, wrowD, wrowSC, wrows := e.BankRows, e.WSel, e.WRowD, e.WRowSC, e.WRows
	dExpAdd, tileOff := e.DExpAdd, e.TileOff
	slotAct, axS, axU, sumOff := e.SlotAct, e.AxS, e.AxU, e.SumOff
	// aTok is the element stride between one token's activations and the next,
	// in the u32 words Load counts. It is a compile-time constant, so a token
	// offset is an IMMEDIATE on the load rather than address arithmetic.
	aTok := s.K / 4
	// The activation word is loaded once and reused by every row of the tile.
	// The memo emits on first use, so at Rowt == 1 the load sits exactly where it
	// always did and the decode kernels are byte-identical.
	var amemo map[int64]ir.Value
	aload := func(off int64) ir.Value {
		if s.Rowt == 1 && !s.Gate2 {
			return b.Load(ir.U32, pA, aIdx, off)
		}
		if v, ok := amemo[off]; ok {
			return v
		}
		v := b.Load(ir.U32, pA, aIdx, off)
		amemo[off] = v
		return v
	}
	// The tile's rows share everything on the activation side (scale, per-16
	// sums and their addressing), so that half is emitted on the first pass and
	// reused (by the second weight stream too, Gate2), which also keeps Rowt == 1 emitting its original sequence.
	var das []ir.Value
	var aBlk, axIdx ir.Value
	var sumAFor func(int) ir.Value
	// The per-16 sums are per token too, so a tile row past the first reuses
	// the first one's loads rather than issuing its own.
	sumMemo := map[int]ir.Value{}
	sumOf := func(t int) ir.Value {
		if v, ok := sumMemo[t]; ok {
			return v
		}
		v := sumAFor(t)
		sumMemo[t] = v
		return v
	}
	// stream emits one weight stream's sums into accs: the payload, scales and
	// minima at pQS, pD and pSC against the shared activation. It returns the
	// payload index past this sub-block. Gate2 runs it twice, the gate's weights
	// beside up's, so both rows are read in one pass over the operand.
	stream := func(pQS, pD, pSC ir.Value, accs, iaccs []ir.Value) ir.Value {
		w := wIdx
		// The secondary plane is loaded first because it must be combined before
		// center(): center is affine per byte, so centering the planes separately
		// would subtract the offset twice. packSub's pairing puts each hi code in
		// the same byte lane as its nibble, so the combine is one Xor.
		var hiw [][]ir.Value
		var hiMask ir.Value
		lanes := 0
		if hw > 0 {
			lanes = 8 / qi.hi
			hiMask = b.Const(ir.U32, int64(rep4(uint32(1<<uint(qi.hi)-1))))
			hiw = make([][]ir.Value, s.Rowt)
			wh := b.Add(ir.U32, w, b.Mul(ir.U32, b.Const(ir.U32, int64(pw)), bankRows))
			for hwi := 0; hwi < hw; hwi++ {
				for j := 0; j < s.Rowt; j++ {
					hiw[j] = append(hiw[j], b.Load(ir.U32, pQS, wh, tileOff(j)))
				}
				wh = b.Add(ir.U32, wh, bankRows)
			}
		}
		// hiOf is element group gi's secondary code, already shifted into the
		// nibble position it occupies in the full weight.
		hiOf := func(j, gi int) ir.Value {
			v := hiw[j][gi/lanes]
			if sh := qi.hi * (gi % lanes); sh > 0 {
				v = b.Shr(ir.U32, v, b.Const(ir.U32, int64(sh)))
			}
			return b.Shl(ir.U32, b.And(ir.U32, v, hiMask), b.Const(ir.U32, int64(qi.bits)))
		}
		for u := 0; u < pw; u++ {
			if amemo == nil || !s.Gate2 {
				amemo = map[int64]ir.Value{}
			}
			for j := 0; j < s.Rowt; j++ {
				x := b.Load(ir.U32, pQS, w, tileOff(j))
				if qi.bits == 4 {
					// Low nibbles are the first half of the sub-block and high nibbles the
					// second, not interleaved, so word u feeds activation words u and
					// u + awords/2.
					loQ := b.And(ir.U32, x, mask)
					hiQ := b.And(ir.U32, b.Shr(ir.U32, x, four), mask)
					if qi.codes != nil {
						loQ, hiQ = decodeE2M1(b, loQ), decodeE2M1(b, hiQ)
					}
					if hw > 0 {
						// Xor, not Or: the planes occupy disjoint bits, and the IR
						// has no Or.
						loQ = b.Xor(ir.U32, loQ, hiOf(j, u))
						hiQ = b.Xor(ir.U32, hiQ, hiOf(j, u+pw))
					}
					lo := center(loQ)
					hi := center(hiQ)
					for t := 0; t < s.Tok; t++ {
						o := int64(t * aTok)
						n := j*s.Tok + t
						iaccs[n] = dot4(lo, aload(int64(u)+o), iaccs[n])
						iaccs[n] = dot4(hi, aload(int64(u+awords/2)+o), iaccs[n])
					}
				} else if qi.bits == 2 {
					// TQ2's four 2-bit fields: field fi of word u is the elements
					// a field-run on, activation word u + fi*pw (packSub).
					m3 := b.Const(ir.U32, 0x03030303)
					for fi := 0; fi < 4; fi++ {
						q := x
						if fi > 0 {
							q = b.Shr(ir.U32, x, b.Const(ir.U32, int64(2*fi)))
						}
						qc := center(b.And(ir.U32, q, m3))
						for t := 0; t < s.Tok; t++ {
							n := j*s.Tok + t
							iaccs[n] = dot4(qc, aload(int64(u+fi*pw)+int64(t*aTok)), iaccs[n])
						}
					}
				} else {
					xc := center(x)
					for t := 0; t < s.Tok; t++ {
						n := j*s.Tok + t
						iaccs[n] = dot4(xc, aload(int64(u)+int64(t*aTok)), iaccs[n])
					}
				}
			}
			w = b.Add(ir.U32, w, bankRows)
		}
		for hwi := 0; hwi < hw; hwi++ {
			w = b.Add(ir.U32, w, bankRows) // the secondary words were read above
		}
		// fdots are a float format's per-(row, token) dot products over this
		// sub-block: the raw weight times the float activation.
		var fdots []ir.Value
		if qi.float > 0 {
			fdots = make([]ir.Value, s.Rowt*s.Tok)
			for n := range fdots {
				fdots[n] = zeroF
			}
			// aIdx counts int8 activation words, four elements each.
			xi := b.Mul(ir.U32, aIdx, four)
			// weight is element i of tile row j's sub-block.
			weight := func(j, i int) ir.Value {
				off := int64(j*fw + i*qi.float/4)
				x := b.Load(ir.U32, pQS, w, off)
				switch {
				case qi.float == 4:
					return b.Bitcast(ir.F32, x)
				case qi.bf16 && i%2 == 0:
					return b.Bitcast(ir.F32, b.Shl(ir.U32, x, b.Const(ir.U32, 16)))
				case qi.bf16:
					return b.Bitcast(ir.F32, b.And(ir.U32, x, b.Const(ir.U32, 0xFFFF0000)))
				case i%2 == 0:
					return b.CvtF16H(x)
				default:
					return b.CvtF16H(b.Shr(ir.U32, x, b.Const(ir.U32, 16)))
				}
			}
			for u := 0; u < awords; u++ {
				acts := make([][4]ir.Value, s.Tok)
				for t := range acts {
					for i := range acts[t] {
						acts[t][i] = b.Load(ir.F32, pX, xi, int64(4*u+i+t*s.K))
					}
				}
				for j := 0; j < s.Rowt; j++ {
					for i := 0; i < 4; i++ {
						wv := weight(j, 4*u+i)
						for t := 0; t < s.Tok; t++ {
							n := j*s.Tok + t
							fdots[n] = b.Fma(wv, acts[t][i], fdots[n])
						}
					}
				}
			}
			w = b.Add(ir.U32, w, b.Const(ir.U32, int64(fsub)))
		}

		// value = d_a * (scale*dot - bias*sum(a)).
		//
		// The scale is rebuilt from a super-block f16 and a per-sub-block integer,
		// which is both exact and smaller than a folded f32 -- see qtab.
		super := subIdx
		if qi.perSuper > 1 {
			super = b.Shr(ir.U32, subIdx, b.Const(ir.U32, int64(log2(qi.perSuper))))
		}
		for j := 0; j < s.Rowt; j++ {
			if qi.float > 0 {
				// No scale on either side: the dot is the value.
				for t := 0; t < s.Tok; t++ {
					n := j*s.Tok + t
					b.SetPhi(accs[n], b.Add(ir.F32, accs[n], fdots[n]))
				}
				continue
			}
			// One d word per super-block: d in the low half, dmin in the high (zero
			// without a minimum array). On a narrow format several rows share a word
			// (kernels.DIndex pairs rows), and the slot is a runtime shift here. MXFP4
			// packs four E8M0 bytes, and a byte shifted left 23 is the f32 scale.
			var dw ir.Value
			var d ir.Value
			if n := DSlots(s.T); n > 1 {
				rowj := b.Add(ir.U32, wsel, b.Const(ir.U32, tileOff(j)))
				idx := dExpAdd(b.Add(ir.U32,
					b.Mul(ir.U32, super, b.Const(ir.U32, int64((wrows+n-1)/n))),
					b.Shr(ir.U32, rowj, b.Const(ir.U32, int64(log2(n))))))
				w := b.Load(ir.U32, pD, idx, 0)
				// Shift and mask, not Div and Rem: DSlots is a power of two, and this
				// form keeps the two-slot IR identical to the lowertest goldens.
				slot := b.And(ir.U32, rowj, b.Const(ir.U32, int64(n-1)))
				dw = b.Shr(ir.U32, w, b.Mul(ir.U32, slot, b.Const(ir.U32, int64(32/n))))
			} else {
				dw = b.Load(ir.U32, pD, b.Add(ir.U32, wrowD, b.Mul(ir.U32, super, bankRows)), tileOff(j))
			}
			switch {
			case qi.e8m0:
				d = b.Bitcast(ir.F32, b.Shl(ir.U32,
					b.And(ir.U32, dw, b.Const(ir.U32, 0xFF)), b.Const(ir.U32, 23)))
			case qi.d32:
				d = b.Bitcast(ir.F32, dw) // TQ2F: the word is the scale
			default:
				d = b.CvtF16H(dw)
			}
			var dmin ir.Value
			if qi.biasArray {
				dmin = b.CvtF16H(b.Shr(ir.U32, dw, b.Const(ir.U32, 16)))
			}
			scale := d
			var mInt ir.Value
			if ScStream(s.T) {
				// Container v27's 96-bit stream (scstream.go).
				lo, hi, bit := scStreamAt(b, subIdx)
				ld := func(w ir.Value) ir.Value {
					return b.Load(ir.U32, pSC, b.Add(ir.U32, wrowSC, b.Mul(ir.U32, w, bankRows)), tileOff(j))
				}
				sc, m := scStreamPair(b, ld(lo), ld(hi), bit)
				scale, mInt = b.Mul(ir.F32, d, b.CvtF32(sc)), m
			} else if qi.perSuper > 1 {
				// Integer scales packed into bytes: two per sub-block when the format
				// carries a minimum, one otherwise.
				perWord := 4
				if qi.biasArray {
					perWord = 2
				}
				wi := b.Shr(ir.U32, subIdx, b.Const(ir.U32, int64(log2(perWord))))
				sw := b.Load(ir.U32, pSC, b.Add(ir.U32, wrowSC, b.Mul(ir.U32, wi, bankRows)), tileOff(j))
				lane := b.And(ir.U32, subIdx, b.Const(ir.U32, int64(perWord-1)))
				stride := 8
				if qi.biasArray {
					stride = 16
				}
				base := b.Mul(ir.U32, lane, b.Const(ir.U32, int64(stride)))
				sc := b.And(ir.U32, b.Shr(ir.U32, sw, base), b.Const(ir.U32, 0xFF))
				scI := sc
				if qi.scOff != 0 {
					scI = b.Sub(ir.I32, sc, b.Const(ir.I32, int64(-qi.scOff)))
				}
				scale = b.Mul(ir.F32, d, b.CvtF32(scI))
				if qi.biasArray {
					m := b.And(ir.U32, b.Shr(ir.U32, sw, b.Add(ir.U32, base, b.Const(ir.U32, 8))),
						b.Const(ir.U32, 0xFF))
					mInt = m
				}
			}
			// The activation scale is per 32 elements; a 16-element sub-block shares one
			// with its neighbour, which is what the divide by 32/sub selects.
			if das == nil {
				aBlk = subIdx
				if qi.sub == 16 {
					aBlk = b.Shr(ir.U32, subIdx, one)
				}
				// Shift first, then add the base: adding the slot base before halving
				// lands on another slot's scale.
				axIdx = subIdx
				if slotAct {
					aBlk = b.Add(ir.U32, aBlk, axS)
					axIdx = b.Add(ir.U32, subIdx, axU)
				}
				// Each token column has its own activation scale and sums, one nb and
				// one K/16 apart. Emitted once each rather than per use, which keeps the
				// Tok == 1 kernel byte-identical.
				das = make([]ir.Value, s.Tok)
				for t := range das {
					das[t] = b.Load(ir.F32, pAX, aBlk, int64(t*nb))
				}
				// sum(a) over this sub-block, from the per-16 array. Only loaded where
				// the bias term survives (not Q8_0, whose zero point is 0).
				sumAFor = func(t int) ir.Value {
					o := int64(t * (s.K / 16))
					if qi.sub == 16 {
						return b.Load(ir.F32, pAX, axIdx, sumOff+o)
					}
					i0 := b.Mul(ir.U32, subIdx, b.Const(ir.U32, 2))
					if slotAct {
						i0 = b.Add(ir.U32, i0, axU)
					}
					return b.Add(ir.F32, b.Load(ir.F32, pAX, i0, sumOff+o),
						b.Load(ir.F32, pAX, i0, sumOff+o+1))
				}
			}
			// The scale is factored out of the bias term for a constant zero point:
			//
			//	scale*dot - (biasK*scale)*sum   ==   scale*(dot - biasK*sum)
			//
			// dot and biasK*sum are integers below 2^24 for every shipped format, so the
			// subtraction is exact and the multiply is the only rounding. It is neutral
			// on Metal (MSL's default fast math already reassociates) and strictly less
			// IR on PTX and SPIR-V. Q4_K and Q5_K keep the expanded form: their bias is
			// an independent array value, with nothing to factor.
			for t := 0; t < s.Tok; t++ {
				var v ir.Value
				switch {
				case qi.biasArray:
					bias := dmin // Q5_1: no sc plane and m is one (MinInD)
					if qi.perSuper > 1 {
						bias = b.Mul(ir.F32, dmin, b.CvtF32(mInt))
					}
					v = b.Sub(ir.F32, b.Mul(ir.F32, b.CvtF32(iaccs[j*s.Tok+t]), scale), b.Mul(ir.F32, sumOf(t), bias))
				case qi.biasK == 0 || centered:
					// centered: the weights already carry -biasK, so the integer
					// dot is the corrected dot and sumOf(t) is never referenced.
					v = b.Mul(ir.F32, b.CvtF32(iaccs[j*s.Tok+t]), scale)
				default:
					v = b.Mul(ir.F32, scale, b.Fma(negBiasK, sumOf(t), b.CvtF32(iaccs[j*s.Tok+t])))
				}
				n := j*s.Tok + t
				b.SetPhi(accs[n], b.Fma(v, das[t], accs[n]))
			}
		}
		return w
	}
	return stream
}

// NewSubDotEnv is the SubDotEnv of a dense matvec over s (one expert, one
// token a thread, no pick) in b, with the sizes, constants and policies its
// sub-block arithmetic shares, derived from the format as MatVec derives
// them: the sub-block (a float format's narrowed one, FloatSub), payload and
// operand words a sub-block, the centering policy at one token a thread, the
// dot instruction. The caller sets the operand planes and the weights'
// addressing (PA, PAX, PX, BankRows, WSel, WRowD, WRowSC, WRows), which are
// its loop nest's. It returns the sub-block count a row walks.
func NewSubDotEnv(b *ir.Builder, s MatVecShape) (*SubDotEnv, int, error) {
	e, n, err := PlanSubDot(s)
	if err != nil {
		return nil, 0, err
	}
	e.EmitConsts(b, b.Const(ir.U32, 1))
	return e, n, nil
}

// PlanSubDot is NewSubDotEnv with nothing emitted: every field a shape
// decides, the constants left for EmitConsts, so a loop nest can emit them
// where its index arithmetic leaves room. It also returns the sub-blocks a row.
func PlanSubDot(s MatVecShape) (*SubDotEnv, int, error) {
	qi := infoOf(s.T)
	if why := DeviceWhyNot(s.T); why != "" {
		return nil, 0, fmt.Errorf("kernels: SubDot: %s", why)
	}
	if s.PickOf > 0 || s.Gate2 || s.In.On || s.NTok > 1 && (s.Experts > 1 || s.Slots > 1) {
		return nil, 0, fmt.Errorf("kernels: SubDot: NewSubDotEnv takes a dense chunk or a bank's slots at one token")
	}
	// A chunk of NTok tokens is NTok operand slots (MatVec's own reading of
	// it): each token's vector at K/4 words, its scales nb apart, its sums
	// after every slot's scales; Tok of them a thread.
	ntok := max(s.NTok, 1)
	s.Tok = max(s.Tok, 1)
	if ntok == 1 {
		s.Tok = 1
	}
	if ntok%s.Tok != 0 {
		return nil, 0, fmt.Errorf("kernels: SubDot: %d tokens at %d a thread", ntok, s.Tok)
	}
	s.Rowt = max(s.Rowt, 1)
	sub := qi.sub
	if s.Sub == 0 {
		s.Sub = FloatSub(s.T, s.K)
	}
	if s.Sub > 0 {
		if qi.float == 0 || s.Sub%4 != 0 || s.Sub > qi.sub || s.K%s.Sub != 0 {
			return nil, 0, fmt.Errorf("kernels: SubDot: a sub-block of %d over %v at K=%d", s.Sub, s.T, s.K)
		}
		sub = s.Sub
	}
	if s.K <= 0 || (s.Sub == 0 && s.K%qi.blockE != 0) || s.Rows <= 0 {
		return nil, 0, fmt.Errorf("kernels: SubDot: K=%d, Rows=%d over %v", s.K, s.Rows, s.T)
	}
	nb := s.K / 32
	e := &SubDotEnv{S: s,
		PW: sub * qi.bits / 32, HW: sub * qi.hi / 32, AWords: sub / 4, NB: nb, SumOff: int64(nb),
		DExpAdd: func(v ir.Value) ir.Value { return v }, TileOff: func(j int) int64 { return int64(j) }}
	// Each slot its own operand vector: a chunk's tokens, or a bank's slots
	// that read their own (SlotAct); the sums after every slot's scales.
	if slots := max(s.Slots, 1); ntok > 1 || s.SlotAct && slots > 1 {
		n := max(ntok, slots)
		e.SlotAct, e.SumOff = true, int64(n*nb)
	}
	if qi.float > 0 {
		e.FW, e.FSub = s.K*qi.float/4, sub*qi.float/4
	}
	e.Centered = !s.Center.NoCenter && qi.biasK != 0 && !qi.biasArray && s.Tok >= s.Center.Tokens()
	e.Center = func(v ir.Value) ir.Value { return v }
	return e, s.K / sub, nil
}

// EmitConsts emits e's constants in the order MatVec does -- the nibble
// mask, four, zero, the negated bias, then the centring pair -- with one the
// caller's own, and picks the format's dot.
func (e *SubDotEnv) EmitConsts(b *ir.Builder, one ir.Value) {
	qi := infoOf(e.S.T)
	e.One = one
	e.Mask, e.Four, e.ZeroF = b.Const(ir.U32, 0x0F0F0F0F), b.Const(ir.U32, 4), b.ConstF32(0)
	e.NegBK = b.ConstF32(-qi.biasK)
	e.Dot4 = b.Dot4Bounded
	if qi.signedQ {
		e.Dot4 = b.Dot4
	}
	if e.Centered {
		c := (0x80 - int64(qi.biasK)) & 0xFF
		cAdd, cXor := b.Const(ir.U32, c|c<<8|c<<16|c<<24), b.Const(ir.U32, int64(0x80808080))
		e.Center = func(v ir.Value) ir.Value { return b.Xor(ir.U32, b.Add(ir.U32, v, cAdd), cXor) }
	}
}
