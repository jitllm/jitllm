//go:build amd64

package cpu

import (
	"encoding/binary"
	"fmt"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// PackedRows is the main tile, and PackedTail the one that finishes a row count
// the main tile does not divide. Both come off the same emitter.
//
// A jlm container stores element u of consecutive rows adjacently, so a load is
// one payload word for many rows and each 32-bit lane accumulates its own row's
// dot. VPDPBUSD reduces within a lane, so there is no horizontal sum: the
// accumulators are the results. A tile of 64 rows makes each stride step
// deliver 256 contiguous bytes, where 8 rows used half a cache line per load
// and left the stride (nrows*4) to defeat the prefetcher.
const PackedRows = 64

// PackedTail is the remainder tile.
const PackedTail = 8

// PackedScratchBytes is the constant block a packed kernel reads through Scr.
const PackedScratchBytes = 416

// PackedScratch fills the constant block. It is loop-invariant and identical
// for every kernel, so it is built once per JIT rather than baked into code.
func PackedScratch(b []byte) error {
	if len(b) < PackedScratchBytes {
		return fmt.Errorf("jit: PackedScratch: %d bytes, need %d", len(b), PackedScratchBytes)
	}
	for i := 0; i < 32; i++ {
		b[i] = 0x0F // nibble mask
	}
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint16(b[32+2*i:], 1) // i16 ones, for VPMADDWD
	}
	// Two byte-gathers, because a scale word holds two f16. VPSHUFB permutes
	// within each 128-bit lane, pulling the low (or high) u16 of each of four
	// u32 into that lane's low eight bytes; a VPERMQ then joins the two lanes.
	lo := []byte{0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80}
	hi := []byte{2, 3, 6, 7, 10, 11, 14, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80}
	copy(b[64:], lo)
	copy(b[80:], lo)
	copy(b[96:], hi)
	copy(b[112:], hi)
	// -8 and +8 undo a factor the row-major kernels bake in: QuantizeQ8 stores
	// -sum(a)*BiasC/8 for their eight-lane horizontal sum, and this kernel has
	// none. A format with its own minimum array subtracts dmin*m*sum(a); one
	// with a constant bias adds BiasC*scale*sum(a), the sign already in the pair.
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(b[128+4*i:], 0xC1000000) // -8.0f
	}
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(b[160+4*i:], 0x41000000) // +8.0f
	}
	for i := 0; i < 32; i++ {
		b[192+i] = 0x80 // the sign flip that makes a signed payload unsigned
	}
	// -scOff as int32, so the stored sub-block scale can be corrected before it
	// is converted: Q3_K keeps 0..63 for -32..31, Q6_K 0..255 for -128..127.
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(b[224+4*i:], 32)
		binary.LittleEndian.PutUint32(b[256+4*i:], 128)
	}
	// +4 for the sixteen-wide formats, whose QuantizeHalfSums stores
	// -sum(a)*BiasC/4.
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(b[288+4*i:], 0x40800000) // +4.0f
	}
	// The secondary plane's masks, pre-shifted into place. A weight is primary |
	// hi<<bits, and 3<<4 = 48 still fits the unsigned byte VPDPBUSD wants, so
	// the two planes share one accumulator; masking with 0x30 (or 0x10) folds
	// the multiply by 16 into the mask.
	for i := 0; i < 32; i++ {
		b[320+i] = 0x30 // hi == 2, i.e. Q6_K
		b[352+i] = 0x10 // hi == 1
	}
	putCodes(b)
	return nil
}

// hiMaskSlot is where PackedScratch keeps the pre-shifted mask for a plane of
// this width.
func hiMaskSlot(hi int) int32 {
	if hi == 1 {
		return 352
	}
	return 320
}

// PackedSupported reports whether a generated kernel exists for this type's
// device layout. PackWeights normalises every format to two payload widths and
// two flat scale arrays, so one emitter serves them all.
func PackedSupported(t quant.Type) bool {
	_, ok := kernels.QuantOf(t)
	return ok
}

// PackedFusedSupported reports whether the fused packed matvec exists for t on
// this host (its tier and dot sequence). It differs from PackedSupported per
// architecture, and the batched mixture dispatch (nn.MatVecPackedMulti,
// MatVecPackedGather) depends on it, so gates must ask rather than assume.
func PackedFusedSupported(t quant.Type) bool {
	// Ask the host's tier and dot sequence: on a pre-VNNI box the fused kernel
	// exists in its DotVEX form, and on an SSE host it is the SSE tier's.
	_, err := Native().PackedFused(t)
	return err == nil
}

// EmitPackedMatVec generates out[r] = dot(row r, x) over the DEVICE layout.
//
// Args, with the packed additions:
//
//	Out     [rows] f32, written not added
//	W       the QS payload span -- transposed, so row r's words are r*4 apart
//	A       int8 activations, 32 per group
//	AScale  {scale, negSum} pairs, one per 32-element group
//	Rows    how many 8-row TILES this call serves
//	RowStr  the layout's row stride in bytes -- the tensor's FULL row count x4
//	K       super-block PAIRS per row (see the outer loop)
//	Scr     PackedScratch
//	PD/PSC  the D and SC spans
//
// RowStr is the tensor's full width, not the call's: every index in the
// layout is `... * nrows + r`, and the wrong width gives plausible garbage.
//
// Register assignment, fixed by hand (16 YMM, no spills):
//
//	Y0  f32 accumulator, 8 rows      Y8  0x0F mask / sign flip
//	Y1  payload word                 Y9  i32 dot accumulator
//	Y2  low nibbles / payload        Y10 d      (8 rows, f32)
//	Y3  high nibbles                 Y11 dmin   (8 rows, f32)
//	Y4  activations, low half        Y12 scale  (8 rows, f32)
//	Y5  activations, high half       Y13 bias   (8 rows, f32)
//	Y6  correction temp              Y14 scale word / temp
//	Y7  bias-correction constant     Y15 temp
//
// GP: RDI Args, kept and re-read once per tile; RCX out, RDX payload cursor, R8
// d cursor, R9 sc cursor, R10 half-sum cursor, RSI tile byte offset, RBX
// constants, RAX k counter, R11 activation cursor, R12 pair cursor, R13 row
// stride in bytes, R15 tile counter. R14 is never touched: Go's register ABI
// pins it to the current goroutine.
func EmitPackedMatVec(t quant.Type, rows int, dk DotKind) ([]byte, error) {
	if rows < 8 || rows%8 != 0 || rows > 64 {
		return nil, fmt.Errorf("jit: EmitPackedMatVec: %d rows per tile (want 8..64, a multiple of 8)", rows)
	}
	q, ok := kernels.QuantOf(t)
	if !ok {
		return nil, fmt.Errorf("jit: EmitPackedMatVec: %s has no device layout", t)
	}
	// The saturation check is a refusal, not a clamp: VPMADDUBSW sums adjacent
	// byte products into int16, and a format that overflows it would give
	// slightly wrong dots. PreVNNIPacked re-derives the bound from the masks.
	if dk == DotVEX {
		if err := PreVNNIPacked(t); err != nil {
			return nil, err
		}
	}
	sub, bits, biasK, biasArray := kernels.Layout(q)
	perSuper, scOff := kernels.ScaleLayout(q)
	signed := kernels.SignedPayload(q)
	codes, err := hasCodes(q)
	if err != nil {
		return nil, err
	}
	// pw primary words then hw secondary ones; see qinfo.hi and packSub.
	hiW := kernels.HiPlane(q)
	pw, hw := sub*bits/32, sub*hiW/32
	halfSub := sub / 2
	const vec = 32     // bytes in a YMM, and 8 rows of the layout
	groups := rows / 8 // YMM-wide row groups in one tile

	// One d word per super-block for every format, so there is no parity
	// to unroll: see kernels.PackedWords.
	const supers = 1
	// A narrow format packs two rows into one d word, so its plane is an f16
	// array at a 2-byte row stride: eight rows are 16 bytes that VCVTPH2PS reads
	// straight from memory. The cost is only in the addressing (R9 and DStr).
	narrow := NarrowD(t)
	e8 := E8M0D(t)
	d32 := kernels.D32(q)
	minD := kernels.MinInD(q)
	perWord, stride := 4, 8
	if biasArray {
		perWord, stride = 2, 16
	}
	stream := kernels.ScStream(q)

	// Only the integer accumulators are held across the k loop, which is what
	// buys the 64-row tile; the f32 accumulators live in Out and are
	// read-modify-written once per sub-block.
	iac := []Reg{Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7}[:groups]
	const (
		pay  = Y8
		tmp  = Y9
		actL = Y10
		actH = Y11
		mask = Y12
		corr = Y13
		t0   = Y14
		t1   = Y15
	)

	// The ones vector VPMADDWD needs, without a seventeenth register. For a
	// 4-bit format `mask` is live throughout, so ones goes in t0, an epilogue
	// temporary rebuilt once per sub-block. An 8-bit format on the pre-VNNI path
	// needs neither the nibble mask nor the sign flip (VPSIGNB replaces the
	// centring), so `mask` itself holds it.
	dot := dotEmitter{kind: dk, flip: mask}
	if dk == DotVEX {
		dot.ones = mask
		if bits < 8 {
			dot.ones = t0
		}
	}

	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out, and the f32 accumulators
	a.MOVLoad(R15, At(RDI, 32)) // Rows -> TILES this call serves
	a.MOVLoad(R13, At(RDI, 48)) // RowStr -> the layout's row stride, in bytes
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	a.SUBQ(RSI, RSI)            // tile byte offset into the three weight spans
	if narrow {
		// R9 is the SC cursor for every other format and there is no SC plane
		// here, so it carries the d plane's own tile offset (DRowBytes per row).
		a.SUBQ(R9, R9)
	}
	switch {
	case bits < 8:
		a.VMOVDQULoad(mask, At(RBX, 0))
		fieldMask(&a, mask, t0, bits)
	case dk == DotVEX:
		// The ones vector, built rather than loaded; see ones16.
		ones16(&a, mask)
	case signed:
		a.VMOVDQULoad(mask, At(RBX, 192))
	}
	switch {
	case biasArray:
		a.VMOVDQULoad(corr, At(RBX, 128)) // -8
	case sub == Q8Block/2:
		a.VMOVDQULoad(corr, At(RBX, 288)) // +4
	default:
		a.VMOVDQULoad(corr, At(RBX, 160)) // +8
	}

	tileL := a.Label()
	a.Bind(tileL)
	a.MOVLoad(RDX, At(RDI, 8))
	a.MOVLoad(R8, At(RDI, 144))
	a.ADDQ(RDX, RSI)
	if narrow {
		a.ADDQ(R8, R9)
	} else {
		a.MOVLoad(R9, At(RDI, 152))
		a.ADDQ(R8, RSI)
		a.ADDQ(R9, RSI)
	}
	a.MOVLoad(R11, At(RDI, 16))
	a.MOVLoad(R12, At(RDI, 24))
	a.MOVLoad(R10, At(RDI, 64))
	a.MOVLoad(RAX, At(RDI, 40))
	// Out holds the running sums, so it starts at zero rather than being
	// written once at the end.
	a.VPXOR(t0, t0, t0)
	for g := 0; g < groups; g++ {
		a.VMOVDQUStore(At(RCX, int32(g*vec)), t0)
	}

	// f16x8 gathers the low (or high) u16 of eight strided u32 into eight f32.
	f16x8 := func(dst, src Reg, hi bool) {
		off := int32(64)
		if hi {
			off = 96
		}
		a.VMOVDQULoad(tmp, At(RBX, off))
		a.VPSHUFB(dst, src, tmp)
		a.VPERMQ(dst, dst, 0xD8)
		a.VCVTPH2PSReg(dst, dst)
	}

	outer := a.Label()
	a.Bind(outer)

	for sp := 0; sp < supers; sp++ {
		// One d word per super-block for every format, so the low half is
		// always d and the high always dmin (zero where there is none).
		const hi = false
		_ = sp
		for sn := 0; sn < perSuper; sn++ {
			// Hoisted above the payload loop because signedFix needs it there;
			// it is a constant of sn either way.
			pairOff := int32(sub * sn / Q8Block * 8)
			for g := 0; g < groups; g++ {
				a.VPXOR(iac[g], iac[g], iac[g])
			}
			if dk == DotVEX && bits < 8 {
				ones16(&a, t0) // dead until the epilogue; see the note above
			}
			// The code table rides in t1, which is an epilogue temporary and
			// untouched by a payload loop with no secondary plane -- and a
			// format with a table has none.
			if codes {
				a.VMOVDQULoad(t1, At(RBX, codesSlot))
			}
			for w := 0; w < pw; w++ {
				// One prefetch per stride step: the hardware prefetcher works
				// within a step, but cannot see the jump of nrows*4 to the next.
				a.PREFETCHT0(Idx(RDX, R13, 8, 0))
				if bits == 4 {
					a.VPBROADCASTD(actL, At(R11, int32(sub*sn+4*w)))
					a.VPBROADCASTD(actH, At(R11, int32(sub*sn+halfSub+4*w)))
					for g := 0; g < groups; g++ {
						a.VMOVDQULoad(pay, At(RDX, int32(g*vec)))
						a.VPAND(tmp, pay, mask)
						if codes {
							a.VPSHUFB(tmp, t1, tmp)
						}
						dot.unsigned(&a, iac[g], tmp, actL)
						a.VPSRLD(tmp, pay, 4)
						a.VPAND(tmp, tmp, mask)
						if codes {
							a.VPSHUFB(tmp, t1, tmp)
						}
						dot.unsigned(&a, iac[g], tmp, actH)
					}
				} else if bits < 8 {
					// Narrower fields (TQ2's 2-bit codes): field f of a byte is
					// the element a field-run on, so its activations are the
					// four at f*run + 4w. Two broadcasts at a time, as above.
					fields, run := 8/bits, sub/(8/bits)
					for f0 := 0; f0 < fields; f0 += 2 {
						a.VPBROADCASTD(actL, At(R11, int32(sub*sn+f0*run+4*w)))
						a.VPBROADCASTD(actH, At(R11, int32(sub*sn+(f0+1)*run+4*w)))
						for g := 0; g < groups; g++ {
							// Reloaded per pair: the four activations would take
							// two more registers than the tile has.
							a.VMOVDQULoad(pay, At(RDX, int32(g*vec)))
							fieldOf(&a, tmp, pay, mask, bits, f0)
							dot.unsigned(&a, iac[g], tmp, actL)
							fieldOf(&a, tmp, pay, mask, bits, f0+1)
							dot.unsigned(&a, iac[g], tmp, actH)
						}
					}
				} else {
					a.VPBROADCASTD(actL, At(R11, int32(sub*sn+4*w)))
					for g := 0; g < groups; g++ {
						a.VMOVDQULoad(pay, At(RDX, int32(g*vec)))
						if signed {
							dot.signed(&a, iac[g], pay, actL, tmp)
						} else {
							dot.unsigned(&a, iac[g], pay, actL)
						}
					}
				}
				a.ADDQ(RDX, R13)
			}
			// The secondary plane. RDX has walked the primary words, so it
			// points at hi word 0. The pre-shifted mask is loaded into t1 (an
			// epilogue temporary) once per sub-block.
			if hw > 0 {
				lanes := 8 / hiW
				for hwi := 0; hwi < hw; hwi++ {
					a.VMOVDQULoad(t1, At(RBX, hiMaskSlot(hiW)))
					for g := 0; g < groups; g++ {
						a.VMOVDQULoad(pay, At(RDX, int32(g*vec)))
						for half := 0; half < 2; half++ {
							for w := 0; w < pw; w++ {
								gi := w + half*pw
								if gi/lanes != hwi {
									continue
								}
								a.VPBROADCASTD(actL, At(R11, int32(sub*sn+half*halfSub+4*w)))
								switch sh := hiW * (gi % lanes); {
								case sh < 4:
									a.VPSLLD(tmp, pay, byte(4-sh))
									a.VPAND(tmp, tmp, t1)
								case sh == 4:
									a.VPAND(tmp, pay, t1)
								default:
									a.VPSRLD(tmp, pay, byte(sh-4))
									a.VPAND(tmp, tmp, t1)
								}
								dot.unsigned(&a, iac[g], tmp, actL)
							}
						}
					}
					a.ADDQ(RDX, R13)
				}
			}

			// Put a pre-VNNI signed accumulator back on VPDPBUSD's scale, so
			// the epilogue below is byte for byte the VNNI kernel's.
			if signed {
				dot.signedFix(&a, iac, At(R12, pairOff+4), t1)
			}
			sh := byte((sn % perWord) * stride)
			for g := 0; g < groups; g++ {
				off := int32(g * vec)
				// The scales are contiguous across rows too, so a group's eight
				// come off one 32-byte load at the same offset as its payload.
				if e8 {
					// MXFP4's super-scale is a biased E8M0 exponent: the byte
					// shifted left 23 is the f32, exactly. See kernels.DSlots.
					a.VPMOVZXBD(t1, At(R8, int32(g*vec/4)))
					a.VPSLLD(t1, t1, 23)
				} else if narrow {
					a.VCVTPH2PS(t1, At(R8, int32(g*vec/2)))
				} else if d32 {
					// TQ2_F32: the word is the scale, a row a word.
					a.VMOVDQULoad(t1, At(R8, off))
				} else {
					a.VMOVDQULoad(t0, At(R8, off))
					f16x8(t1, t0, hi) // d
				}
				scale := t1
				nextSC := func(r Reg) { a.VMOVDQULoad(r, Idx(R9, R13, 1, off)) }
				if stream {
					// Container v27's stream (scstream.go); actH is free until
					// Out is read below.
					a.VMOVDQULoad(pay, At(R9, off))
					scField(avxSC(&a), tmp, pay, actH, sn, false, nextSC)
					if scOff != 0 {
						a.VMOVDQULoad(actL, At(RBX, 224+int32(scOffSlot(scOff))))
						a.VPSUBD(tmp, tmp, actL)
					}
					a.VCVTDQ2PS(tmp, tmp)
					a.VMULPS(scale, scale, tmp)
				} else if perSuper > 1 {
					a.VMOVDQULoad(pay, At(R9, off))
					if 24-sh > 0 {
						a.VPSLLD(tmp, pay, 24-sh)
					} else {
						a.VMOVAPSReg(tmp, pay)
					}
					a.VPSRLD(tmp, tmp, 24)
					if scOff != 0 {
						a.VMOVDQULoad(actL, At(RBX, 224+int32(scOffSlot(scOff))))
						a.VPSUBD(tmp, tmp, actL)
					}
					a.VCVTDQ2PS(tmp, tmp)
					a.VMULPS(scale, scale, tmp)
				}
				a.VBROADCASTSS(actL, At(R12, pairOff)) // d_a
				a.VMULPS(scale, scale, actL)
				a.VCVTDQ2PS(tmp, iac[g])
				// out[g] += dot * scale * d_a
				a.VMOVDQULoad(actH, At(RCX, off))
				a.VFMADD231PS(actH, tmp, scale)
				switch {
				case biasArray:
					// bias = dmin*m, and it carries d_a like the scale does.
					// Q5_1 has no sc plane and m is one: dmin alone.
					f16x8(tmp, t0, true)
					if stream {
						// pay is dead once read, so it is the straddle's scratch.
						scField(avxSC(&a), t0, pay, pay, sn, true, nextSC)
						a.VCVTDQ2PS(t0, t0)
						a.VMULPS(tmp, tmp, t0)
					} else if !minD {
						a.VPSLLD(t0, pay, 16-sh)
						a.VPSRLD(t0, t0, 24)
						a.VCVTDQ2PS(t0, t0)
						a.VMULPS(tmp, tmp, t0)
					}
					a.VMULPS(tmp, tmp, actL) // * d_a
					a.VBROADCASTSS(t0, At(R12, pairOff+4))
					a.VMULPS(tmp, tmp, t0)
					a.VMULPS(tmp, tmp, corr)
					a.VSUBPS(actH, actH, tmp)
				case sub == Q8Block/2:
					a.VBROADCASTSS(tmp, At(R10, int32(4*sn)))
					a.VMULPS(tmp, tmp, scale)
					a.VMULPS(tmp, tmp, corr)
					a.VADDPS(actH, actH, tmp)
				case biasK != 0 || signed:
					a.VBROADCASTSS(tmp, At(R12, pairOff+4))
					a.VMULPS(tmp, tmp, scale)
					a.VMULPS(tmp, tmp, corr)
					a.VADDPS(actH, actH, tmp)
				}
				a.VMOVDQUStore(At(RCX, off), actH)
			}
			if stream && scStep(sn) || !stream && perSuper > 1 && sn%perWord == perWord-1 {
				a.ADDQ(R9, R13)
			}
		}
		a.ADDimm(R11, int32(perSuper*sub))
		a.ADDimm(R12, int32(perSuper*sub/Q8Block*8))
		if sub == Q8Block/2 {
			a.ADDimm(R10, int32(perSuper*4))
		}
		if narrow {
			// Half the payload's stride, rounded up to a whole word; DStr
			// carries it because no register is free to hold a second stride.
			a.ADDQmem(R8, At(RDI, 160))
		} else {
			a.ADDQ(R8, R13)
			if perSuper == 1 {
				a.ADDQ(R9, R13)
			}
		}
	}
	a.DEC(RAX)
	a.JNZ(outer)

	a.ADDimm(RCX, int32(4*rows))
	a.ADDimm(RSI, int32(4*rows))
	if narrow {
		// The d plane's own tile offset: DRowBytes per row, not four.
		a.ADDimm(R9, int32(DRowBytes(t)*rows))
	}
	a.DEC(R15)
	a.JNZ(tileL)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// PackedOuterElems is how many elements of a row one outer iteration consumes,
// so a caller can turn k into the K the kernel wants.
func PackedOuterElems(t quant.Type) int {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return 0
	}
	sub, _, _, _ := kernels.Layout(q)
	perSuper, _ := kernels.ScaleLayout(q)
	// One super-block per outer step for every format since container v2
	// stores one d word per super-block.
	return sub * perSuper
}

// scOffSlot picks the constant vector holding -scOff for this format.
func scOffSlot(scOff int) int {
	if scOff == -32 {
		return 0
	}
	return 32 // -128
}

// packedMatVecPF is the primary table's PackedMatVec; the arm64 prefetch
// distance has no amd64 form.
func packedMatVecPF(t quant.Type, rows, _ int) ([]byte, error) {
	return EmitPackedMatVec(t, rows, HostDotKind())
}
