//go:build amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// EmitPackedMatVecFused runs every word of a sub-block in one pass over the
// rows, so the integer accumulator never leaves a register. In the wide
// kernel's separate pass per word, two thirds of the memory operations were
// the accumulator's load and store (see AGENTS.md's top-down counters and
// docs/engineering-history/cpu-kernels.md).
//
// Holding every word's activation broadcast at once costs 2*words registers.
// They are found by precomputing the super-block scale and its minimum as f32
// into scratch once per super-block: the f16 gather costs four instructions
// and two shuffle-mask registers, a precomputed f32 one load and none.
//
//	per 16 rows of a Q4_K sub-block   wide   fused
//	  instructions                     120      94
//	  memory operations                 34      18
//
// Args:
//
//	Out      [rows] f32, ACCUMULATED into; the caller zeroes it
//	W/PD/PSC the three spans, at this call's first row
//	Rows     how many PackedFusedGroup-row groups this call serves
//	RowStr   the layout's row stride in bytes
//	Scratch  [rows] f32, the precomputed super-block scale
//	Q32      [rows] f32, its minimum -- only for a format that has one
//
// EmitPackedMatVecFusedWin has no amd64 form: VPDPBUSD sums unscaled bytes,
// so an integer fold would need a VPMULLD per sub-block (packedgemm.go).
func EmitPackedMatVecFusedWin(t quant.Type, _ int) ([]byte, error) {
	return nil, fmt.Errorf("jit: EmitPackedMatVecFusedWin: %s: no amd64 integer form", t)
}

func EmitPackedMatVecFused(t quant.Type, dk DotKind) ([]byte, error) {
	return EmitPackedMatVecFusedAhead(t, dk, 0)
}

// FusedAheadWords selects EmitPackedMatVecFusedAhead's word-ahead prefetch:
// eight payload words ahead in the same row group, rather than a distance in
// row groups.
const FusedAheadWords = 64

// EmitPackedMatVecFusedAhead is the fused kernel with a PREFETCHT0 per payload
// word, `ahead` row groups in front of the one being read; 0 emits none. The
// right distance depends on the host's prefetcher, so nn measures it on real
// tokens (engine/nn/tune.go's prefetch duel).
func EmitPackedMatVecFusedAhead(t quant.Type, dk DotKind, ahead int) ([]byte, error) {
	// FusedAheadWords is a different prefetch, not a longer one. A distance in
	// row groups runs off the end of the call's row range at the last groups of
	// each sub-block, so the next sub-block's first lines are demand misses.
	// Eight words ahead in the same row group is exactly the set of lines this
	// call reads next.
	wordAhead := ahead == FusedAheadWords
	const who = "EmitPackedMatVecFused"
	f, err := packedFormatOf(t, who)
	if err != nil {
		return nil, err
	}
	// The saturation refusal, before anything is emitted; see PreVNNIPacked.
	if dk == DotVEX {
		if err := PreVNNIPacked(t); err != nil {
			return nil, err
		}
	}
	sub, bits, biasK, biasArray := f.sub, f.bits, f.biasK, f.biasArray
	perSuper, scOff := f.perSuper, f.scOff
	signed, minD, hi, codes := f.signed, f.minD, f.hi, f.codes
	// The payload is pw primary words then hw secondary ones. The hi code
	// reaches the same accumulator as the primary because its mask is
	// pre-shifted by 4 (see PackedScratch).
	pw, hw := f.pw, f.hw
	halfSub := f.halfSub
	const vec = 32
	group := PackedFusedGroupOf(t)
	grp := group / 8

	// The register budget is checked: a spilling tile has measured as a large
	// loss, and the activation set grows with the payload width.
	nact := pw
	if bits < 8 {
		nact = f.fields * pw // one broadcast a field a word
	}
	need := 3 + grp + 3 + nact // mask, da, bscale + accumulators + pay/tmp/t2
	if scOff != 0 {
		need++
	}
	if hi != 0 {
		need++ // the pre-shifted secondary mask
	}
	if codes {
		need++ // the code table
	}
	if need > 16 {
		return nil, fmt.Errorf("jit: EmitPackedMatVecFused: %s needs %d vector registers, have 16",
			t, need)
	}

	var a Buf
	next := Y0
	take := func() Reg { r := next; next++; return r }
	mask, da, bscale := take(), take(), take()
	iac := make([]Reg, grp)
	for i := range iac {
		iac[i] = take()
	}
	act := make([]Reg, nact)
	for i := range act {
		act[i] = take()
	}
	pay, tmp, t2 := take(), take(), take()
	// A straddling SC field reads the next word with no register to hold its
	// address (R14 is g): the cursor steps one plane stride and back around the
	// load (scstream.go).
	stream := f.stream
	nextSC := func(off int32) func(Reg) {
		return func(r Reg) {
			a.ADDQ(R9, R13)
			a.VMOVDQULoad(r, Idx(R9, R15, 1, off))
			a.SUBQ(R9, R13)
		}
	}
	// The i16 ones VPMADDWD needs costs no register; the budget above lands at
	// exactly 16 for Q4_0, Q8_0, Q4_K and Q5_K.
	//   bits == 8: the pre-VNNI path loads neither the nibble mask nor the 0x80
	//     sign flip (VPSIGNB replaces the centring), so `mask` holds the ones
	//     vector for the whole kernel.
	//   bits == 4: `mask` is live, but t2 is an epilogue temporary untouched by
	//     the payload loop, so ones is rebuilt into it once per row group.
	dot := dotEmitter{kind: dk, flip: mask}
	if dk == DotVEX {
		dot.ones = mask
		if bits < 8 {
			dot.ones = t2
		}
	}
	scOffV := Y0
	if scOff != 0 {
		scOffV = take()
	}
	hiMask := Y0
	if hi != 0 {
		hiMask = take()
	}
	lut := Y0
	if codes {
		lut = take()
	}

	// R10 is AHalf or the minimum scratch, never both (packedArgs): that is
	// what closes the GPR budget at thirteen with R14 unused.
	packedArgs(&a, &f)
	switch {
	case bits < 8:
		a.VMOVDQULoad(mask, At(RBX, 0))
		fieldMask(&a, mask, tmp, bits)
	case dk == DotVEX:
		// The ones vector, built rather than loaded; see ones16.
		ones16(&a, mask)
	case signed:
		a.VMOVDQULoad(mask, At(RBX, 192))
	}
	packedConsts(&a, &f, scOffV, hiMask, lut)
	// The two shuffle masks are needed only by the precompute pass, which runs
	// while the activation registers are still free.
	shufLo, shufHi := act[0], act[1]

	packedCursors(&a, &f)
	outer := a.Label()
	a.Bind(outer)

	// --- precompute this super-block's scales, once ------------------------
	packedScalePass(&a, &f, t, group, grp, shufLo, shufHi, pay, tmp, t2, true)

	// pl is the payload word at off in the current row group, and prefetch
	// issues the software prefetch this kernel was emitted with; pfd is set
	// per row group below.
	var pfd int32
	pl := func(off int32) Mem {
		if wordAhead {
			return At(RSI, off)
		}
		return Idx(RSI, R15, 1, off)
	}
	prefetch := func() {
		switch {
		case wordAhead:
			a.PREFETCHT0(Idx(RSI, R13, 8, 0))
		case pfd > 0:
			a.PREFETCHT0(Idx(RSI, R15, 1, pfd))
		}
	}

	// --- the sub-blocks ----------------------------------------------------
	for sn := 0; sn < perSuper; sn++ {
		pairOff := f.pairOff(sn)
		sh := f.scShift(sn)
		for w := 0; w < pw; w++ {
			a.VPBROADCASTD(act[w], At(R11, int32(sub*sn+4*w)))
			if bits == 4 {
				a.VPBROADCASTD(act[pw+w], At(R11, int32(sub*sn+halfSub+4*w)))
			} else if bits < 8 {
				// Field fi of word w multiplies the four activations a
				// field-run on (packSub).
				for fi := 1; fi < f.fields; fi++ {
					a.VPBROADCASTD(act[fi*pw+w], At(R11, int32(sub*sn+fi*f.run+4*w)))
				}
			}
		}
		a.VBROADCASTSS(da, At(R12, pairOff))
		a.MOVLoad(RBX, At(RDI, 56)) // the row loop clobbered it
		switch {
		case biasArray:
			a.VBROADCASTSS(bscale, At(R12, pairOff+4))
			a.VMOVDQULoad(tmp, At(RBX, 128)) // -8
			a.VMULPS(bscale, bscale, tmp)
			a.VMULPS(bscale, bscale, da)
		case f.sub16:
			a.VBROADCASTSS(bscale, At(R10, int32(4*sn)))
			a.VMOVDQULoad(tmp, At(RBX, 288)) // +4
			a.VMULPS(bscale, bscale, tmp)
		case biasK != 0 || signed:
			a.VBROADCASTSS(bscale, At(R12, pairOff+4))
			a.VMOVDQULoad(tmp, At(RBX, 160)) // +8
			a.VMULPS(bscale, bscale, tmp)
		}
		packedRowLoop(&a, group, func() {
			// RSI is the payload cursor and then the scratch base: the cursor
			// dies when the last word has issued, and the scratch is only read
			// in the epilogue.
			a.MOVQ(RSI, RDX) // reset every row group
			if wordAhead {
				a.ADDQ(RSI, R15) // the cursor carries the row offset; see below
			}
			for g := 0; g < grp; g++ {
				a.VPXOR(iac[g], iac[g], iac[g])
			}
			if dk == DotVEX && bits < 8 {
				ones16(&a, t2) // dead until the epilogue; see the note above
			}
			// A word's lines are contiguous across row groups, so the line
			// `ahead` groups on is a constant displacement. The hardware
			// prefetcher sees each word as a short run and re-trains every
			// sub-block, which matters on high-latency hosts (two-socket
			// Broadwell); see docs/engineering-history/cpu-kernels.md.
			pfd = 0
			if !wordAhead {
				pfd = int32(ahead * grp * vec)
			}
			for w := 0; w < pw; w++ {
				prefetch()
				for g := 0; g < grp; g++ {
					off := int32(g * vec)
					a.VMOVDQULoad(pay, pl(off))
					if bits == 4 {
						a.VPAND(tmp, pay, mask)
						if codes {
							a.VPSHUFB(tmp, lut, tmp)
						}
						dot.unsigned(&a, iac[g], tmp, act[w])
						a.VPSRLD(tmp, pay, 4)
						a.VPAND(tmp, tmp, mask)
						if codes {
							a.VPSHUFB(tmp, lut, tmp)
						}
						dot.unsigned(&a, iac[g], tmp, act[pw+w])
					} else if bits < 8 {
						for fi := 0; fi < f.fields; fi++ {
							fieldOf(&a, tmp, pay, mask, bits, fi)
							dot.unsigned(&a, iac[g], tmp, act[fi*pw+w])
						}
					} else if signed {
						dot.signed(&a, iac[g], pay, act[w], tmp)
					} else {
						dot.unsigned(&a, iac[g], pay, act[w])
					}
				}
				a.ADDQ(RSI, R13)
			}
			// The secondary plane, after the cursor has walked the primary
			// one, so RSI already points at hi word 0; each field is a single
			// shift (packedFormat.hiFields).
			for hwi := 0; hwi < hw; hwi++ {
				prefetch()
				for g := 0; g < grp; g++ {
					off := int32(g * vec)
					a.VMOVDQULoad(pay, pl(off))
					f.hiFields(hwi, func(half, w, s int) {
						hiField(&a, tmp, pay, hiMask, s)
						dot.unsigned(&a, iac[g], tmp, act[half*pw+w])
					})
				}
				a.ADDQ(RSI, R13)
			}
			// The pre-VNNI signed payload accumulated dot(w, a); put it back on
			// VPDPBUSD's dot(w+128, a) so the epilogue below -- and its rounding
			// -- is byte for byte the kernel a VNNI host runs.
			if signed {
				dot.signedFix(&a, iac, At(R12, pairOff+4), t2)
			}
			a.MOVLoad(RSI, At(RDI, 72)) // the scratch base, for the epilogue
			for g := 0; g < grp; g++ {
				off := int32(g * vec)
				if stream {
					// Container v27's stream: the scale into t2 before d takes
					// tmp, which a straddling field borrows.
					a.VMOVDQULoad(pay, Idx(R9, R15, 1, off))
					scField(avxSC(&a), t2, pay, tmp, sn, false, nextSC(off))
					if scOff != 0 {
						a.VPSUBD(t2, t2, scOffV)
					}
					a.VMOVDQULoad(tmp, Idx(RSI, R15, 1, off)) // d, precomputed f32
					a.VCVTDQ2PS(t2, t2)
					a.VMULPS(tmp, tmp, t2)
				} else if perSuper > 1 {
					a.VMOVDQULoad(tmp, Idx(RSI, R15, 1, off)) // d, precomputed f32
					a.VMOVDQULoad(pay, Idx(R9, R15, 1, off))  // the scale word
					scByte(&a, t2, pay, sh)
					if scOff != 0 {
						a.VPSUBD(t2, t2, scOffV)
					}
					a.VCVTDQ2PS(t2, t2)
					a.VMULPS(tmp, tmp, t2)
				} else {
					a.VMOVDQULoad(tmp, Idx(RSI, R15, 1, off)) // d, precomputed f32
				}
				a.VMULPS(tmp, tmp, da) // scale * d_a
				a.VCVTDQ2PS(iac[g], iac[g])
				a.VMOVDQULoad(t2, Idx(RCX, R15, 1, off))
				a.VFMADD231PS(t2, iac[g], tmp)
				if stream {
					// The minimum into pay first: a straddling one borrows tmp.
					scField(avxSC(&a), pay, pay, tmp, sn, true, nextSC(off))
					a.VCVTDQ2PS(pay, pay)
					a.VMOVDQULoad(tmp, Idx(R10, R15, 1, off)) // dmin, precomputed
					a.VMULPS(tmp, tmp, pay)
					a.VMULPS(tmp, tmp, bscale)
					a.VSUBPS(t2, t2, tmp)
				} else if biasArray {
					a.VMOVDQULoad(tmp, Idx(R10, R15, 1, off)) // dmin, precomputed
					if !minD {                                // Q5_1's m is one: no sc word to read it from
						scMinByte(&a, pay, pay, sh)
						a.VCVTDQ2PS(pay, pay)
						a.VMULPS(tmp, tmp, pay)
					}
					a.VMULPS(tmp, tmp, bscale)
					a.VSUBPS(t2, t2, tmp)
				} else {
					a.VMULPS(tmp, tmp, bscale)
					a.VADDPS(t2, t2, tmp)
				}
				a.VMOVDQUStore(Idx(RCX, R15, 1, off), t2)
			}
		})
		packedSubStep(&a, &f, sn)
	}
	packedSuperStep(&a, &f, true)
	a.DEC(RAX)
	a.JNZ(outer)
	return packedCode(&a, who, t)
}
