//go:build amd64

package cpu

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// GEMMPad is the bytes EmitPackedMatMulStationary expects after each token's
// row of every per-token stream (A, AScale, AHalf and Out).
//
// The token loop touches the same offset of every token's row back to back,
// and every model shape puts those rows a multiple of 4 KiB apart, so past
// eight tokens each access evicts the line the next token wants (the unpadded
// kernel fell several-fold as the call widened). One line of padding per token puts consecutive tokens in
// consecutive L1 sets.
const GEMMPad = 64

// StationaryIntAcc reports whether EmitPackedMatMulStationary accumulates t's
// super-blocks in integers at this activation window and dot sequence -- the
// form that is bounded against the fused kernel rather than equal to it.
func StationaryIntAcc(t quant.Type, win int, dk DotKind) bool {
	q, ok := kernels.QuantOf(t)
	if !ok || dk != DotVEX {
		return false
	}
	sub, _, _, biasArray := kernels.Layout(q)
	perSuper, _ := kernels.ScaleLayout(q)
	return perSuper > 1 && win >= sub*perSuper && (biasArray || sub == Q8Block/2)
}

// EmitPackedMatMulStationary is the prefill GEMM in the container layout: one
// sub-block of eight rows is decoded into registers once and then dotted
// against every token of the call, the tokens looping innermost at run time.
//
// The token tile is a loop count, not a register count. EmitPackedMatMulTiled
// holds tok accumulators and broadcasts per row group, which caps it at four
// tokens (and zero pre-VNNI). Here the weight stays in registers (at most
// eight YMM), with one accumulator and one broadcast, so a call covers
// Args.Cols tokens and a prefill reads its weights n/Cols times. This is
// llama.cpp's shape (ggml_gemm_q4_K_8x8_q8_K in ggml-cpu/arch/x86/repack.cpp).
//
// The pre-VNNI dot accumulates in int16 (VPADDW) and widens once with
// VPMADDWD, under a bound the emitter tracks from the format's maximum
// (PreVNNIPacked's derivation): where the running bound would pass 32767 the
// partial sum is widened into int32 first. A Q4_K sub-block (eight products
// of 2*15*127) fits whole.
//
// The two planes of a 4+hi format are merged (OR) before the dot, decoded
// once per row group. The integer sums are exact and the float epilogue is
// the fused kernel's operation for operation, so the result is bit-identical
// to the fused per-token kernel (nn.TestMatMulPackedMatchesTheRowLoopExactly),
// except on the StationaryIntAcc path.
//
// k and nrows are baked. Every per-token stream is padded by GEMMPad bytes.
// Args:
//
//	Out      [tok][nrows] f32 at this call's first row, ACCUMULATED into
//	W/PD/PSC the three spans at this call's first row
//	A        [tok][k] int8;  AScale [tok][2*nb];  AHalf [tok][k/16] (16-wide)
//	         -- each token's row followed by GEMMPad bytes, Out's too
//	Rows     eight-row groups this call serves
//	Cols     tokens this call serves, >= 1
//	Scratch  16 f32 of kernel scratch: a row group's super-block scales and minimums
//
// It writes five Args fields it does not read (OutStr, ASum, AHalfSum, Out2
// and Q2) as spill slots, because the token loop needs one more GPR than a
// kernel may touch (R14 is g; RBP is not saved by the trampoline). Args is the
// caller's per-call copy and none of those fields is read back.
func EmitPackedMatMulStationary(t quant.Type, k, nrows, win int, dk DotKind) ([]byte, error) {
	const who = "EmitPackedMatMulStationary"
	f, err := packedFormatOf(t, who)
	if err != nil {
		return nil, err
	}
	if dk == DotVEX {
		if err := PreVNNIPacked(t); err != nil {
			return nil, err
		}
	}
	q := f.q
	sub, bits, biasK, biasArray := f.sub, f.bits, f.biasK, f.biasArray
	perSuper, scOff := f.perSuper, f.scOff
	signed, hi, codes := f.signed, f.hi, f.codes
	if bits == 8 && !signed {
		return nil, fmt.Errorf("jit: EmitPackedMatMulStationary: %s: an unsigned 8-bit payload has no form here", t)
	}
	pw, hw := f.pw, f.hw
	const group = 8 // rows per YMM: one lane a row
	if k <= 0 || k%(sub*perSuper) != 0 || nrows <= 0 || nrows%group != 0 {
		return nil, fmt.Errorf("jit: EmitPackedMatMulStationary: k=%d nrows=%d", k, nrows)
	}
	nW := pw
	if bits < 8 {
		nW = f.fields * pw // a weight register a field a word
	}
	if nW > 8 {
		return nil, fmt.Errorf("jit: EmitPackedMatMulStationary: %s holds %d weight registers, the budget is 8", t, nW)
	}
	// The per-product int16 bound: VPMADDUBSW sums two products of an
	// unsigned byte (at most umax) and a signed one (|a| <= 127).
	umax := 0
	switch {
	case signed:
		umax = 128 // |w| through VPSIGNB/VPABSB; see dotEmitter.signed
	case codes:
		for _, c := range kernels.Codes(q) {
			umax = max(umax, int(c))
		}
	default:
		umax = 1<<bits - 1
	}
	if hi != 0 {
		umax += ((1 << hi) - 1) << 4
	}
	prod := 2 * umax * 127
	const i16Max = 32767
	if dk == DotVEX && prod > i16Max {
		return nil, fmt.Errorf("jit: EmitPackedMatMulStationary: %s's product reaches %d, over int16", t, prod)
	}

	nb := k / Q8Block
	aTok := int32(k + GEMMPad)
	asTok := int32(8*nb + GEMMPad)
	oTok := int32(4*nrows + GEMMPad)
	ahTok := int32(4*(k/16) + GEMMPad)
	sub16 := f.sub16

	var args Args
	off := func(p unsafe.Pointer) int32 {
		return int32(uintptr(p) - uintptr(unsafe.Pointer(&args)))
	}
	var (
		oOut     = off(unsafe.Pointer(&args.Out))
		oW       = off(unsafe.Pointer(&args.W))
		oA       = off(unsafe.Pointer(&args.A))
		oAScale  = off(unsafe.Pointer(&args.AScale))
		oRows    = off(unsafe.Pointer(&args.Rows))
		oK       = off(unsafe.Pointer(&args.K))
		oRowStr  = off(unsafe.Pointer(&args.RowStr))
		oScr     = off(unsafe.Pointer(&args.Scr))
		oAHalf   = off(unsafe.Pointer(&args.AHalf))
		oScratch = off(unsafe.Pointer(&args.Scratch))
		oCols    = off(unsafe.Pointer(&args.Cols))
		oPD      = off(unsafe.Pointer(&args.PD))
		oPSC     = off(unsafe.Pointer(&args.PSC))
		oDStr    = off(unsafe.Pointer(&args.DStr))
		// The spill slots; see the comment above.
		sK = off(unsafe.Pointer(&args.OutStr))
		sA = off(unsafe.Pointer(&args.ASum))
		sS = off(unsafe.Pointer(&args.AHalfSum))
		sO = off(unsafe.Pointer(&args.Out2))
		sH = off(unsafe.Pointer(&args.Q2))
	)

	var a Buf
	next := Y0
	take := func() Reg { r := next; next++; return r }
	W := make([]Reg, nW)
	for i := range W {
		W[i] = take()
	}
	pool := [3]Reg{take(), take(), take()}
	eD, eM, eT, eO := take(), take(), take(), take()
	// A k-quant under a 256-wide activation window accumulates its super-block
	// in integers, as llama.cpp's Q8_K path does (ggml_gemm_q4_K_8x8_q8_K):
	// d_act is one number for the super-block, so VPMADDWD applies the
	// sub-block scale to the int16 partials and the float epilogue runs once per
	// super-block. The minimum (or 16-wide bias) accumulates in f32.
	//
	// Pre-VNNI only: VPDPBUSD sums unscaled bytes, so a VNNI host keeps the float
	// epilogue and bit equality with the fused kernel. The integer path is not
	// bit-identical (its sum is exact where the fused kernel rounds per
	// sub-block), which the gate states and bounds.
	intAcc := StationaryIntAcc(t, win, dk)
	var dR, d4R Reg
	if intAcc && !biasArray {
		dR, d4R = take(), take()
	}
	if next > Y15+1 {
		return nil, fmt.Errorf("jit: EmitPackedMatMulStationary: %s needs %d vector registers", t, next)
	}

	narrow, e8 := f.narrow, f.e8
	a.MOVLoad(RCX, At(RDI, oOut))
	a.MOVLoad(R13, At(RDI, oRowStr))
	a.MOVLoad(RBX, At(RDI, oRows))
	a.SUBQ(R15, R15)
	if intAcc && biasArray {
		a.MOVLoad(R10, At(RDI, oScratch)) // the scale base; see the final fold
	}

	// Row group outermost, all of k inside it: a row group's T outputs (T x 32
	// bytes) are the whole working set of the accumulation, so the output block
	// stays in L1 however many rows the call covers.
	rowL := a.Label()
	a.Bind(rowL)
	a.MOVLoad(R11, At(RDI, oA))
	a.MOVLoad(R12, At(RDI, oAScale))
	if sub16 {
		a.MOVLoad(R10, At(RDI, oAHalf))
	}
	a.MOVLoad(RDX, At(RDI, oW))
	a.MOVLoad(R8, At(RDI, oPD))
	if narrow {
		// R9 is the d plane's ROW offset: two bytes a row for an f16, one for
		// an E8M0 byte, against R15's four.
		a.MOVQ(R9, R15)
		if e8 {
			a.SHRimm(R9, 2)
		} else {
			a.SHRimm(R9, 1)
		}
	} else {
		a.MOVLoad(R9, At(RDI, oPSC))
	}
	a.MOVLoad(RAX, At(RDI, oK))
	a.MOVStore(At(RDI, sK), RAX)

	outer := a.Label()
	a.Bind(outer)

	// --- this super-block's eight row scales, f16 -> f32 into Scratch -----
	// (and the minimum into Scratch+32): the fused kernel's precompute, for
	// one row group.
	{
		shufLo, shufHi, tmp, pay := pool[0], pool[1], pool[2], eT
		a.MOVLoad(RSI, At(RDI, oScr))
		a.VMOVDQULoad(shufLo, At(RSI, 64))
		if biasArray {
			a.VMOVDQULoad(shufHi, At(RSI, 96))
		}
		a.MOVLoad(RSI, At(RDI, oScratch))
		switch {
		case e8:
			a.VPMOVZXBD(tmp, Idx(R8, R9, 1, 0))
			a.VPSLLD(tmp, tmp, 23)
		case narrow:
			a.VCVTPH2PS(tmp, Idx(R8, R9, 1, 0))
		case f.d32:
			a.VMOVDQULoad(tmp, Idx(R8, R15, 1, 0))
		default:
			a.VMOVDQULoad(pay, Idx(R8, R15, 1, 0))
			a.VPSHUFB(tmp, pay, shufLo)
			a.VPERMQ(tmp, tmp, 0xD8)
			a.VCVTPH2PSReg(tmp, tmp)
		}
		a.VMOVDQUStore(At(RSI, 0), tmp)
		if intAcc && !biasArray {
			a.VMOVAPSReg(dR, tmp)
			a.VADDPS(d4R, tmp, tmp)
			a.VADDPS(d4R, d4R, d4R) // 4*d, exactly
		}
		if biasArray {
			a.VPSHUFB(tmp, pay, shufHi)
			a.VPERMQ(tmp, tmp, 0xD8)
			a.VCVTPH2PSReg(tmp, tmp)
			if intAcc {
				// 8*dmin, exactly: the fused kernel's bscale is pair4*(-8),
				// and folding the eight here leaves the final fold one FMA.
				a.VADDPS(tmp, tmp, tmp)
				a.VADDPS(tmp, tmp, tmp)
				a.VADDPS(tmp, tmp, tmp)
			}
			a.VMOVDQUStore(At(RSI, 32), tmp)
		}
	}
	if narrow {
		a.ADDQmem(R8, At(RDI, oDStr))
	} else {
		a.ADDQ(R8, R13)
	}

	// actOff is the activation byte a weight register's four lanes multiply.
	actOff := func(sn, j int) int32 {
		if bits < 8 && j >= pw {
			// W[fi*pw+w] is field fi of word w: the activations a field-run on.
			return int32(sub*sn + (j/pw)*f.run + 4*(j%pw))
		}
		return int32(sub*sn + 4*j)
	}

	// Container v27's SC stream (scstream.go): a straddling field's next word
	// is one plane stride on, reached by stepping the cursor there and back --
	// no register is free to hold the address.
	stream := f.stream
	nextSC := func(r Reg) {
		a.ADDQ(R9, R13)
		a.VMOVDQULoad(r, Idx(R9, R15, 1, 0))
		a.SUBQ(R9, R13)
	}
	for sn := 0; sn < perSuper; sn++ {
		pairOff := f.pairOff(sn)
		sh := f.scShift(sn)

		// --- decode this row group's sub-block into W, once -------------
		mask, aux, dtmp, dpay := pool[0], pool[1], pool[2], eT
		a.MOVLoad(RSI, At(RDI, oScr))
		switch {
		case bits < 8:
			a.VMOVDQULoad(mask, At(RSI, 0))
			fieldMask(&a, mask, dtmp, bits)
		case signed && dk == DotVNNI:
			a.VMOVDQULoad(mask, At(RSI, 192)) // the 0x80 flip
		}
		switch {
		case codes:
			a.VMOVDQULoad(aux, At(RSI, codesSlot))
		case hi != 0:
			a.VMOVDQULoad(aux, At(RSI, hiMaskSlot(hi)))
		}
		if scOff != 0 {
			a.VMOVDQULoad(eO, At(RSI, 224+int32(scOffSlot(scOff))))
		}
		// RSI walks this row group's word lines, RowStr apart.
		//
		// Eight words ahead of each is prefetched: consecutive words are RowStr
		// apart (8 KiB on a 2048-row projection), a stride no hardware prefetcher
		// follows, and eight words is enough tokens of work to cover DRAM latency.
		a.MOVQ(RSI, RDX)
		a.ADDQ(RSI, R15)
		for w := 0; w < pw; w++ {
			a.PREFETCHT0(Idx(RSI, R13, 8, 0))
			if bits == 4 {
				a.VMOVDQULoad(dpay, At(RSI, 0))
				a.VPAND(W[w], dpay, mask)
				a.VPSRLD(W[pw+w], dpay, 4)
				a.VPAND(W[pw+w], W[pw+w], mask)
				if codes {
					a.VPSHUFB(W[w], aux, W[w])
					a.VPSHUFB(W[pw+w], aux, W[pw+w])
				}
			} else if bits < 8 {
				a.VMOVDQULoad(dpay, At(RSI, 0))
				for fi := 0; fi < f.fields; fi++ {
					fieldOf(&a, W[fi*pw+w], dpay, mask, bits, fi)
				}
			} else if dk == DotVNNI {
				a.VMOVDQULoad(dpay, At(RSI, 0))
				a.VPXOR(W[w], dpay, mask)
			} else {
				a.VMOVDQULoad(W[w], At(RSI, 0))
			}
			a.ADDQ(RSI, R13)
		}
		for hwi := 0; hwi < hw; hwi++ {
			a.PREFETCHT0(Idx(RSI, R13, 8, 0))
			a.VMOVDQULoad(dpay, At(RSI, 0))
			f.hiFields(hwi, func(half, w, s int) {
				hiField(&a, dtmp, dpay, aux, s)
				a.VPOR(W[half*pw+w], W[half*pw+w], dtmp)
			})
			a.ADDQ(RSI, R13)
		}

		// --- the token-invariant half of the epilogue --------------------
		if intAcc {
			// eD becomes S, the sub-block scale as an int16 in both halves of
			// each lane (VPMADDWD's second operand), and eM the per-row f32
			// the bias accumulates against: the minimum m for a format with
			// one, the (signed) scale itself for a 16-wide format.
			scw := pool[0]
			a.VMOVDQULoad(scw, Idx(R9, R15, 1, 0))
			scScale(&a, &f, eT, scw, pool[1], sn, nextSC)
			if scOff != 0 {
				a.VPSUBD(eT, eT, eO)
			}
			a.VPSLLD(eD, eT, 16)
			a.VPSRLD(pool[1], eD, 16)
			a.VPOR(eD, eD, pool[1])
			if stream {
				scField(avxSC(&a), eM, scw, pool[1], sn, true, nextSC)
				a.VCVTDQ2PS(eM, eM)
			} else if biasArray {
				scMinByte(&a, eM, scw, sh)
				a.VCVTDQ2PS(eM, eM)
			} else {
				a.VCVTDQ2PS(eM, eT)
			}
		}
		if !intAcc {
			a.MOVLoad(RSI, At(RDI, oScratch))
			a.VMOVDQULoad(eD, At(RSI, 0)) // d_row, f32
			if perSuper > 1 {
				scw := pool[0]
				a.VMOVDQULoad(scw, Idx(R9, R15, 1, 0))
				scScale(&a, &f, eT, scw, pool[1], sn, nextSC)
				if scOff != 0 {
					a.VPSUBD(eT, eT, eO)
				}
				a.VCVTDQ2PS(eT, eT)
				a.VMULPS(eD, eD, eT) // d_row * sub-block scale
				if stream {
					a.VMOVDQULoad(eM, At(RSI, 32)) // dmin
					scField(avxSC(&a), scw, scw, pool[1], sn, true, nextSC)
					a.VCVTDQ2PS(scw, scw)
					a.VMULPS(eM, eM, scw) // dmin * m
				} else if biasArray {
					a.VMOVDQULoad(eM, At(RSI, 32)) // dmin
					scMinByte(&a, scw, scw, sh)
					a.VCVTDQ2PS(scw, scw)
					a.VMULPS(eM, eM, scw) // dmin * m
				}
			} else if biasArray {
				// Q5_1 (kernels.MinInD): no sc plane, so m is one and the
				// term is dmin alone. Unloaded, eM held whatever the last
				// row group left in it.
				a.VMOVDQULoad(eM, At(RSI, 32))
			}
		}

		// --- the tokens -------------------------------------------------
		a.MOVStore(At(RDI, sA), R11)
		a.MOVStore(At(RDI, sS), R12)
		a.MOVStore(At(RDI, sO), RCX)
		if sub16 {
			a.MOVStore(At(RDI, sH), R10)
		}
		a.MOVLoad(RAX, At(RDI, oCols))
		if intAcc {
			// Scratch+64 on: per token, the int32 sum [32 B] and the f32 bias
			// sum [32 B] carried across the super-block's sub-blocks.
			a.MOVLoad(RSI, At(RDI, oScratch))
			a.ADDimm(RSI, 64)
		} else {
			a.MOVLoad(RSI, At(RDI, oScr))
		}
		widen := func(r Reg) {
			if intAcc {
				a.VPMADDWD(r, r, eD) // * the sub-block scale
			} else {
				a.VPMADDWDMem(r, r, At(RSI, 32)) // i16 ones
			}
		}
		tokL := a.Label()
		a.Bind(tokL)

		// Roles within one token's body. Every token starts from the same
		// assignment, so renaming at emit time is free.
		b, a16, a32 := pool[0], pool[1], pool[2]
		have16, have32 := false, false
		bound := 0
		flush := func() {
			widen(a16)
			if have32 {
				a.VPADDD(a32, a32, a16)
			} else {
				a16, a32 = a32, a16
				have32 = true
			}
			have16, bound = false, 0
		}
		// Two chains where no partial sum has to be widened: one accumulator
		// makes a token's sum a dependent chain of nW adds (or VPDPBUSDs), which
		// stalls the scheduler before the next token's independent work. Even
		// and odd words into two registers halve the chain, and the third pool
		// register is free exactly when no flush is needed.
		twoChains := nW >= 2 && (dk == DotVNNI || nW*prod <= i16Max)
		for j := 0; j < nW && twoChains; j++ {
			a.VPBROADCASTD(b, At(R11, actOff(sn, j)))
			if dk == DotVNNI {
				switch j {
				case 0:
					a.VPXOR(a32, a32, a32)
					a.VPDPBUSD(a32, W[j], b)
				case 1:
					a.VPXOR(a16, a16, a16)
					a.VPDPBUSD(a16, W[j], b)
				default:
					if j%2 == 0 {
						a.VPDPBUSD(a32, W[j], b)
					} else {
						a.VPDPBUSD(a16, W[j], b)
					}
				}
				continue
			}
			if signed {
				a.VPSIGNB(b, b, W[j])
				a.VPABSB(eT, W[j])
				a.VPMADDUBSW(b, eT, b)
			} else {
				a.VPMADDUBSW(b, W[j], b)
			}
			switch j {
			case 0:
				b, a16 = a16, b
			case 1:
				b, a32 = a32, b
			default:
				if j%2 == 0 {
					a.VPADDW(a16, a16, b)
				} else {
					a.VPADDW(a32, a32, b)
				}
			}
		}
		if twoChains {
			if dk == DotVNNI {
				a.VPADDD(a32, a32, a16)
			} else {
				a.VPADDW(a16, a16, a32)
				widen(a16)
				a16, a32 = a32, a16
			}
			have32 = true
		}
		for j := 0; j < nW && !twoChains; j++ {
			a.VPBROADCASTD(b, At(R11, actOff(sn, j)))
			if dk == DotVNNI {
				if !have32 {
					a.VPXOR(a32, a32, a32)
					have32 = true
				}
				a.VPDPBUSD(a32, W[j], b)
				continue
			}
			if signed {
				a.VPSIGNB(b, b, W[j]) // a * sign(w)
				a.VPABSB(eT, W[j])    // |w| <= 128
				a.VPMADDUBSW(b, eT, b)
			} else {
				a.VPMADDUBSW(b, W[j], b)
			}
			if have16 && bound+prod > i16Max {
				flush()
			}
			if !have16 {
				b, a16 = a16, b
				have16, bound = true, prod
			} else {
				a.VPADDW(a16, a16, b)
				bound += prod
			}
		}
		if have16 {
			flush()
		}
		iac := a32
		if dk == DotVEX && signed {
			// dotEmitter.signedFix: back onto VPDPBUSD's dot(w+128, a) so the
			// epilogue below is the fused kernel's byte for byte.
			a.VBROADCASTSS(eT, At(R12, pairOff+4))
			a.VCVTPS2DQ(eT, eT)
			a.VPSLLD(eT, eT, 3)
			a.VPSUBD(iac, iac, eT)
		}
		// b, a16 and a32 are a permutation of the pool, so with the sum in
		// a32 the other two are free for the epilogue.
		x, eS := a16, b
		if intAcc {
			last := sn == perSuper-1
			if sn > 0 {
				a.VPADDDMem(iac, iac, At(RSI, 0))
			}
			if !last {
				a.VMOVDQUStore(At(RSI, 0), iac)
			}
			if biasArray {
				a.VBROADCASTSS(eS, At(R12, pairOff+4)) // -sum/8
			} else {
				a.VBROADCASTSS(eS, At(R10, int32(4*sn))) // -sum16*BiasC/4
			}
			if sn == 0 {
				a.VMULPS(x, eM, eS)
			} else {
				a.VMOVDQULoad(x, At(RSI, 32))
				a.VFMADD231PS(x, eM, eS)
			}
			if !last {
				a.VMOVDQUStore(At(RSI, 32), x)
			} else {
				// The super-block's one float fold:
				//   out += d*da * sum(sc*isum)  +  8*dmin*da * sum(m*pair4)
				//   out += d*da * sum(sc*isum)  +  4*d*da * sum(sc*ahalf)
				a.VCVTDQ2PS(iac, iac)
				a.VBROADCASTSS(eS, At(R12, pairOff)) // d_act, one per super-block
				a.VMOVDQULoad(eO, Idx(RCX, R15, 1, 0))
				if biasArray {
					a.VMULPSMem(eT, eS, At(R10, 0))
					a.VFMADD231PS(eO, iac, eT)
					a.VMULPSMem(eT, eS, At(R10, 32))
					a.VFMADD231PS(eO, x, eT)
				} else {
					a.VMULPS(eT, eS, dR)
					a.VFMADD231PS(eO, iac, eT)
					a.VMULPS(eT, eS, d4R)
					a.VFMADD231PS(eO, x, eT)
				}
				a.VMOVDQUStore(Idx(RCX, R15, 1, 0), eO)
			}
			a.ADDimm(RSI, 64)
		}
		if !intAcc {
			a.VCVTDQ2PS(iac, iac)
			a.VBROADCASTSS(eS, At(R12, pairOff)) // d_act
			a.VMULPS(eT, eD, eS)                 // (d_row * sub-scale) * d_act
			a.VMOVDQULoad(eO, Idx(RCX, R15, 1, 0))
			a.VFMADD231PS(eO, iac, eT)
			switch {
			case biasArray:
				a.VBROADCASTSS(x, At(R12, pairOff+4))
				a.VMULPSMem(x, x, At(RSI, 128)) // * -8
				a.VMULPS(x, x, eS)              // * d_act -> bscale
				a.VMULPS(x, eM, x)              // (dmin * m) * bscale
				a.VSUBPS(eO, eO, x)
			case sub16:
				a.VBROADCASTSS(x, At(R10, int32(4*sn)))
				a.VMULPSMem(x, x, At(RSI, 288)) // * +4
				a.VMULPS(x, x, eT)
				a.VADDPS(eO, eO, x)
			case biasK != 0 || signed:
				a.VBROADCASTSS(x, At(R12, pairOff+4))
				a.VMULPSMem(x, x, At(RSI, 160)) // * +8
				a.VMULPS(x, x, eT)
				a.VADDPS(eO, eO, x)
			}
			a.VMOVDQUStore(Idx(RCX, R15, 1, 0), eO)
		}

		a.ADDimm(R11, aTok)
		a.ADDimm(R12, asTok)
		a.ADDimm(RCX, oTok)
		if sub16 {
			a.ADDimm(R10, ahTok)
		}
		a.DEC(RAX)
		a.JNZ(tokL)

		a.MOVLoad(R11, At(RDI, sA))
		a.MOVLoad(R12, At(RDI, sS))
		a.MOVLoad(RCX, At(RDI, sO))
		if sub16 {
			a.MOVLoad(R10, At(RDI, sH))
		}

		packedSubStep(&a, &f, sn)
	}
	// A narrow format's R9 is the d plane's row offset, rebuilt per row group.
	packedSuperStep(&a, &f, !narrow)
	a.DECmem(At(RDI, sK))
	a.JNZ(outer)

	a.ADDimm(R15, group*4)
	a.DEC(RBX)
	a.JNZ(rowL)
	return packedCode(&a, who, t)
}
