//go:build amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// PackedTiledTokens is the widest token tile any format gets. A format's own
// tile is MaxTiledTokens, which is smaller when its constants take registers.
const PackedTiledTokens = 4

// tiledBudget returns the register demand of a tok-wide tile for this format,
// and the pieces that make it up.
//
// The fused matvec hoists every activation broadcast and arrives at exactly
// sixteen; a tile cannot (tok x pw broadcasts is thirty-two), so it holds the
// weight instead and pays with tok accumulators per row group. What is left
// differs by format:
//
//	Q8_0         8-bit payload, one pay register        3 + 3*tok   tok 4
//	Q5_K         4-bit + a secondary plane, group 8       see PackedFusedGroupOf
//	Q4_0, Q4_K   4-bit, pay must survive the hi phase   4 + 3*tok   tok 4
//	Q6_K         4-bit + a secondary plane's mask       5 + 3*tok   tok 3
//	Q3_K         4-bit + the scale offset vector        5 + 3*tok   tok 3
func tiledBudget(q kernels.Quant) (need, payR, tok int) {
	_, bits, _, _ := kernels.Layout(q)
	_, scOff := kernels.ScaleLayout(q)
	grp := packedFusedGroupK(q) / 8
	payR = 1
	if bits < 8 {
		// The lo and hi nibbles are two phases over the SAME payload word, and
		// the activation registers are reused between them, so the payload has
		// to survive both for every row group.
		payR = grp
	}
	base := 2 + payR // the mask and one temporary
	if scOff != 0 {
		base++
	}
	if kernels.HiPlane(q) != 0 {
		base++
	}
	if kernels.Codes(q) != nil {
		base++ // the code table, held like the masks for the whole kernel
	}
	for tok = PackedTiledTokens; tok >= 1; tok-- {
		if need = base + grp*tok + tok; need <= 16 {
			return need, payR, tok
		}
	}
	return need, payR, 0
}

// MaxTiledTokens is how many tokens this format's tiled kernel covers, or 0
// when it has none.
func MaxTiledTokens(t quant.Type) int {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return 0
	}
	if !PackedSupported(t) {
		return 0
	}
	_, _, tok := tiledBudget(q)
	return tok
}

// MaxTiledTokensNative is MaxTiledTokens for a caller that is about to execute.
// EmitPackedMatMulTiled refuses under DotVEX, while MaxTiledTokens is
// host-pure and answers for the register budget alone; the CPU question lives
// here, as with Supported and SupportedNative.
func MaxTiledTokensNative(t quant.Type) int {
	if HostDotKind() == DotVEX {
		return 0
	}
	return MaxTiledTokens(t)
}

// EmitPackedMatMulTiled runs tok activations against one pass over the weight,
// in the container's layout: one payload word load feeds tok dots, where
// MatVecPacked reads the whole weight once per token. The broadcasts move
// inside the row loop to pay for it.
//
// k and nrows are baked so the token strides are immediates and cost no
// register, the granularity nn.AddShape already emits at.
//
// Args, beyond EmitPackedMatVecFused's: Out is [tok][rows], A is [tok][k] and
// AScale is [tok][2*nb], with AHalf [tok][k/16] for the 16-wide formats.
func EmitPackedMatMulTiled(t quant.Type, k, nrows, tok int, dk DotKind) ([]byte, error) {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return nil, fmt.Errorf("jit: EmitPackedMatMulTiled: %s has no device layout", t)
	}
	// The pre-VNNI path declines this kernel. tiledBudget lands at exactly 16
	// registers for tok=4 and nothing is free across the payload loop for the
	// ones vector, so a VEX form needs a narrower tile: a different kernel,
	// owed rather than faked. Declining costs the pre-VNNI host token tiling
	// only: nn.MatMulPacked falls to the fused per-token kernel and the answer
	// is identical (model.TestEveryModelRunsPreVNNI).
	if dk == DotVEX {
		return nil, fmt.Errorf("jit: EmitPackedMatMulTiled: %s has no pre-VNNI form; "+
			"the per-token fused kernel serves this host", t)
	}
	maxTok := MaxTiledTokens(t)
	if maxTok < 2 {
		return nil, fmt.Errorf("jit: EmitPackedMatMulTiled: %s has no tiled kernel", t)
	}
	if tok < 2 || tok > maxTok {
		return nil, fmt.Errorf("jit: EmitPackedMatMulTiled: %s tok=%d out of range 2..%d", t, tok, maxTok)
	}
	const who = "EmitPackedMatMulTiled"
	f, err := packedFormatK(q, t)
	if err != nil {
		return nil, err
	}
	sub, bits, biasK, biasArray := f.sub, f.bits, f.biasK, f.biasArray
	perSuper, scOff := f.perSuper, f.scOff
	signed, minD, hi, codes := f.signed, f.minD, f.hi, f.codes
	pw, hw := f.pw, f.hw
	halfSub := f.halfSub
	const vec = 32
	grp := PackedFusedGroupOf(t) / 8

	group := PackedFusedGroupOf(t)
	if k <= 0 || k%(sub*perSuper) != 0 || nrows <= 0 || nrows%group != 0 {
		return nil, fmt.Errorf("jit: EmitPackedMatMulTiled: k=%d nrows=%d", k, nrows)
	}
	need, payR, _ := tiledBudget(q)
	need = need - grp*maxTok - maxTok + grp*tok + tok
	if need > 16 {
		return nil, fmt.Errorf("jit: EmitPackedMatMulTiled: %s tok=%d needs %d registers, have 16", t, tok, need)
	}

	nb := k / Q8Block
	aTok := int32(k)         // A is int8[tok][k]
	asTok := int32(8 * nb)   // AScale is f32[tok][2*nb]
	oTok := int32(4 * nrows) // Out is f32[tok][nrows]
	ahTok := int32(4 * (k / 16))

	var a Buf
	next := Y0
	take := func() Reg { r := next; next++; return r }
	mask := take()
	iac := make([][]Reg, grp)
	for g := range iac {
		iac[g] = make([]Reg, tok)
		for i := range iac[g] {
			iac[g][i] = take()
		}
	}
	payv := make([]Reg, payR)
	for i := range payv {
		payv[i] = take()
	}
	act := make([]Reg, tok)
	for i := range act {
		act[i] = take()
	}
	tmp := take()
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
	// The epilogue runs when every activation and payload register is dead, so
	// they are its scratch. Six are needed: the row scale, the scale word, the
	// per-token product, the output, the correction and one spare. A narrower
	// tile leaves more free, and those join the pool too.
	//
	// mask, scOffV, hiMask and lut are not in the pool: they are loaded once
	// above the outer loop and read by every super-block, so borrowing one breaks
	// every row group after the first while staying finite and plausible.
	pool := append(append([]Reg{tmp}, act...), payv...)
	for int(next) <= int(Y15) {
		pool = append(pool, take())
	}
	if len(pool) < 6 {
		return nil, fmt.Errorf("jit: EmitPackedMatMulTiled: %s tok=%d leaves %d epilogue registers, need 6",
			t, tok, len(pool))
	}
	eD, eSC, eS, eO, eC, eT := pool[0], pool[1], pool[2], pool[3], pool[4], pool[5]

	packedArgs(&a, &f)
	if bits < 8 {
		a.VMOVDQULoad(mask, At(RBX, 0))
		fieldMask(&a, mask, tmp, bits)
	} else if signed {
		a.VMOVDQULoad(mask, At(RBX, 192))
	}
	packedConsts(&a, &f, scOffV, hiMask, lut)

	rowLoop := func(body func()) { packedRowLoop(&a, group, body) }

	// R9 is the SC cursor, and a narrow format has no SC plane, so there it
	// carries the d plane's own row offset (DRowBytes per row).
	stream := f.stream
	packedCursors(&a, &f)

	outer := a.Label()
	a.Bind(outer)

	// --- this super-block's row scales, f16 -> f32 into Scratch ------------
	// One temporary for both shuffles: the stores are sequential, and a second
	// register would collide with the epilogue pool on formats whose pool is
	// only seven deep (Q5_K at tok=4).
	shufLo, shufHi := act[0], act[1]
	packedScalePass(&a, &f, t, group, grp, shufLo, shufHi, payv[0], tmp, tmp, false)

	// --- the sub-blocks ---------------------------------------------------
	for sn := 0; sn < perSuper; sn++ {
		pairOff := f.pairOff(sn)
		sh := f.scShift(sn)
		// The correction constant is loaded in the epilogue, not hoisted: above
		// the row loop it would sit in a register the broadcasts clobber (the
		// epilogue pool), and loading it from RBX inside the body clobbers the row
		// counter. RSI, the payload cursor, is dead once the last word has issued
		// and is reloaded with the scratch base a line later anyway.
		corr := int32(-1)
		switch {
		case biasArray:
			corr = 128 // -8
		case sub == Q8Block/2:
			corr = 288 // +4
		case biasK != 0 || signed:
			corr = 160 // +8
		}
		rowLoop(func() {
			a.MOVQ(RSI, RDX)
			for g := 0; g < grp; g++ {
				for i := 0; i < tok; i++ {
					a.VPXOR(iac[g][i], iac[g][i], iac[g][i])
				}
			}
			for w := 0; w < pw; w++ {
				if bits < 8 {
					for g := 0; g < grp; g++ {
						a.VMOVDQULoad(payv[g], Idx(RSI, R15, 1, int32(g*vec)))
					}
					// One phase a field: lo nibbles then hi, or TQ2's four
					// 2-bit fields -- the activation registers are reused
					// between the phases, which is why the payload has to be
					// the thing that is held.
					for fi := 0; fi < f.fields; fi++ {
						off := int32(sub*sn + fi*f.run + 4*w)
						for i := 0; i < tok; i++ {
							a.VPBROADCASTD(act[i], At(R11, off+int32(i)*aTok))
						}
						for g := 0; g < grp; g++ {
							fieldOf(&a, tmp, payv[g], mask, bits, fi)
							if codes {
								a.VPSHUFB(tmp, lut, tmp)
							}
							for i := 0; i < tok; i++ {
								a.VPDPBUSD(iac[g][i], tmp, act[i])
							}
						}
					}
				} else {
					for i := 0; i < tok; i++ {
						a.VPBROADCASTD(act[i], At(R11, int32(sub*sn+4*w)+int32(i)*aTok))
					}
					for g := 0; g < grp; g++ {
						a.VMOVDQULoad(payv[0], Idx(RSI, R15, 1, int32(g*vec)))
						if signed {
							a.VPXOR(payv[0], payv[0], mask)
						}
						for i := 0; i < tok; i++ {
							a.VPDPBUSD(iac[g][i], payv[0], act[i])
						}
					}
				}
				a.ADDQ(RSI, R13)
			}
			// --- the secondary plane ---
			for hwi := 0; hwi < hw; hwi++ {
				for g := 0; g < grp; g++ {
					a.VMOVDQULoad(payv[g%payR], Idx(RSI, R15, 1, int32(g*vec)))
					f.hiFields(hwi, func(half, w, s int) {
						hiField(&a, tmp, payv[g%payR], hiMask, s)
						base := int32(sub*sn + 4*w)
						if half == 1 {
							base = int32(sub*sn + halfSub + 4*w)
						}
						for i := 0; i < tok; i++ {
							a.VPBROADCASTD(act[i], At(R11, base+int32(i)*aTok))
							a.VPDPBUSD(iac[g][i], tmp, act[i])
						}
					})
				}
				a.ADDQ(RSI, R13)
			}
			// --- epilogue ---
			if corr >= 0 {
				a.MOVLoad(RSI, At(RDI, 56))
				a.VMOVDQULoad(eC, At(RSI, corr))
			}
			a.MOVLoad(RSI, At(RDI, 72))
			for g := 0; g < grp; g++ {
				off := int32(g * vec)
				a.VMOVDQULoad(eD, Idx(RSI, R15, 1, off)) // d_row, f32
				if stream {
					// Container v27's stream (scstream.go): the scale into eD
					// and the minimum, as f32, into eSC -- once a row group
					// rather than once a token. eT and eS are free until the
					// token loop, and a straddling field borrows eS. The cursor
					// steps a plane stride and back for the next word: no
					// register is free to hold its address.
					next := func(r Reg) {
						a.ADDQ(R9, R13)
						a.VMOVDQULoad(r, Idx(R9, R15, 1, off))
						a.SUBQ(R9, R13)
					}
					a.VMOVDQULoad(eSC, Idx(R9, R15, 1, off))
					scField(avxSC(&a), eT, eSC, eS, sn, false, next)
					if scOff != 0 {
						a.VPSUBD(eT, eT, scOffV)
					}
					a.VCVTDQ2PS(eT, eT)
					a.VMULPS(eD, eD, eT) // d_row * sub-block scale
					scField(avxSC(&a), eSC, eSC, eS, sn, true, next)
					a.VCVTDQ2PS(eSC, eSC)
				} else if perSuper > 1 {
					a.VMOVDQULoad(eSC, Idx(R9, R15, 1, off))
					scByte(&a, eT, eSC, sh)
					if scOff != 0 {
						a.VPSUBD(eT, eT, scOffV)
					}
					a.VCVTDQ2PS(eT, eT)
					a.VMULPS(eD, eD, eT) // d_row * sub-block scale
				}
				for i := 0; i < tok; i++ {
					// The activation scale is needed twice and separately: the
					// FMA wants d_row * sub-scale * d_act, while a biasArray
					// format's correction wants d_act alone (its bscale is
					// AScale[+4] * -8 * d_act and the dmin term carries its own
					// row scale).
					a.VCVTDQ2PS(iac[g][i], iac[g][i])
					a.VBROADCASTSS(eS, At(R12, pairOff+int32(i)*asTok)) // d_act
					a.VMULPS(eT, eS, eD)                                // * d_row * sub-scale
					a.VMOVDQULoad(eO, Idx(RCX, R15, 1, off+int32(i)*oTok))
					a.VFMADD231PS(eO, iac[g][i], eT)
					switch {
					case biasArray:
						a.VBROADCASTSS(eT, At(R12, pairOff+4+int32(i)*asTok))
						a.VMULPS(eT, eT, eC) // * -8
						a.VMULPS(eT, eT, eS) // * d_act  -> bscale
						// eS is free from here: d_act is folded into eT.
						a.VMOVDQULoad(eS, Idx(R10, R15, 1, off)) // dmin, precomputed
						if stream {
							a.VMULPS(eS, eS, eSC) // * m, extracted above
						} else if !minD { // Q5_1's m is one, and it has no sc word
							scMinByte(&a, eSC, eSC, sh)
							a.VCVTDQ2PS(eSC, eSC)
							a.VMULPS(eS, eS, eSC)
							a.VMOVDQULoad(eSC, Idx(R9, R15, 1, off)) // restore the word
						}
						a.VMULPS(eS, eS, eT)
						a.VSUBPS(eO, eO, eS)
					case sub == Q8Block/2:
						a.VBROADCASTSS(eS, At(R10, int32(4*sn)+int32(i)*ahTok))
						a.VMULPS(eS, eS, eC) // * +4
						a.VMULPS(eS, eS, eT)
						a.VADDPS(eO, eO, eS)
					case biasK != 0 || signed:
						a.VBROADCASTSS(eS, At(R12, pairOff+4+int32(i)*asTok))
						a.VMULPS(eS, eS, eC) // * +8
						a.VMULPS(eS, eS, eT)
						a.VADDPS(eO, eO, eS)
					}
					a.VMOVDQUStore(Idx(RCX, R15, 1, off+int32(i)*oTok), eO)
				}
			}
		})
		packedSubStep(&a, &f, sn)
	}
	packedSuperStep(&a, &f, true)
	a.DEC(RAX)
	a.JNZ(outer)
	return packedCode(&a, who, t)
}
