//go:build arm64 || amd64

package cpu

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// StationaryIntAccA64 reports whether EmitA64PackedMatMulStationary sums t's
// super-blocks in integers at this activation window: a k-quant whose whole
// super-block shares one d_act (win >= sub*perSuper) and whose correction is a
// per-sub-block term (a minimum, or the 16-wide half sums). MLA makes the
// sub-block scale one instruction on NEON, so unlike amd64 there is no dot
// sequence for which the float form is cheaper.
func StationaryIntAccA64(t quant.Type, win int) bool {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return false
	}
	sub, _, _, biasArray := kernels.Layout(q)
	perSuper, _ := kernels.ScaleLayout(q)
	return perSuper > 1 && win >= sub*perSuper && (biasArray || sub == Q8Block/2)
}

// EmitA64PackedMatMulStationary is EmitPackedMatMulStationary's arm64 form,
// under the same Args contract and the same padded per-token streams (GEMMPad):
// a row group of eight rows is decoded into registers once per sub-block and
// dotted against every token of the call, the tokens looping innermost.
//
// The token tile (EmitA64PackedMatMulTiled) decodes a payload word once per
// four tokens and pays a float epilogue per token per sub-block. Here the
// decode is paid once per call, and under StationaryIntAccA64 the sub-block
// scale is one MLA into an int32 per-token sum, so Out is touched once per
// super-block (llama.cpp's ggml_gemm_q4_K_8x4_q8_K shape). Where the integer
// fold does not apply the epilogue is the tiled kernel's, so
// nn.TestPackedGEMMMatchesTheRowLoopExactly holds it to ==; the integer form
// is held to amd64's NMSE bound.
//
// Registers: the decoded weights take 2*nW (at most 16) V registers, the
// per-sub-block scale invariants four, the token body at most nine. Args:
//
//	Out      [tok][nrows] f32 at this call's first row, ACCUMULATED into
//	W/PD/PSC the three spans at this call's first row
//	A        [tok][k] int8; AScale [tok][2*nb]; AHalf [tok][k/16] -- each
//	         token's row followed by GEMMPad bytes, Out's too
//	Rows     eight-row groups this call serves
//	K        super-blocks
//	Cols     tokens this call serves, >= 1
//	Scratch  64 bytes of row scales, then 64 bytes per token of accumulators
func EmitA64PackedMatMulStationary(t quant.Type, k, nrows, win int) ([]byte, error) {
	f, err := packedFormatOf(t, "EmitA64PackedMatMulStationary")
	if err != nil {
		return nil, err
	}
	sub, bits, biasArray := f.sub, f.bits, f.biasArray
	perSuper, scOff := f.perSuper, f.scOff
	signed, minD, hi, codes := f.signed, f.minD, f.hi, f.codes
	if bits == 8 && !signed {
		return nil, fmt.Errorf("jit: EmitA64PackedMatMulStationary: %s: an unsigned 8-bit payload has no form here", t)
	}
	if codes && hi != 0 {
		return nil, fmt.Errorf("jit: EmitA64PackedMatMulStationary: %s: a code table with a high plane has no form here", t)
	}
	pw, hw := f.pw, f.hw
	nW := pw
	if bits < 8 {
		nW = f.fields * pw // W[fi*pw+w] is field fi of word w, chunk fi*pw+w
	}
	const group = 8
	if k <= 0 || k%(sub*perSuper) != 0 || nrows <= 0 || nrows%group != 0 {
		return nil, fmt.Errorf("jit: EmitA64PackedMatMulStationary: k=%d nrows=%d", k, nrows)
	}
	nact := sub / 16
	intAcc := StationaryIntAccA64(t, win)
	narrow, e8, sub16 := f.narrow, f.e8, f.sub16
	corrK := f.corrK()

	next := VReg(0)
	take := func() VReg { r := next; next++; return r }
	W := [2][]VReg{make([]VReg, nW), make([]VReg, nW)}
	for h := range W {
		for j := range W[h] {
			W[h][j] = take()
		}
	}
	// Per sub-block, per half: the scale (int for intAcc, d*sc otherwise) and
	// the bias factor (m or the signed scale as f32; dmin*m otherwise).
	inv := [2][2]VReg{{take(), take()}, {take(), take()}}
	vK := VReg(0)
	if !intAcc && corrK != 0 {
		vK = take()
	}
	// The token body. act, acc and six more; the decode's temporaries reuse
	// them, since nothing token-shaped is live while a sub-block decodes.
	act := make([]VReg, nact)
	for i := range act {
		act[i] = take()
	}
	acc := [2]VReg{take(), take()}
	r := [6]VReg{take(), take(), take(), take(), take(), take()}
	if int(next) > 32 {
		return nil, fmt.Errorf("jit: EmitA64PackedMatMulStationary: %s needs %d vector registers", t, next)
	}

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
	)

	nb := k / Q8Block
	var a A64
	// r[0] and r[1] are written after the token's dots and never read before
	// them, so a widened SDOT (sdotemu.go) may clobber them.
	a.DotScratch(r[0], r[1])
	stream := f.stream
	// X6 row offset (bytes, 4 a row), X5 row groups left, X7 RowStr, X23
	// Scratch, X19..X22 the per-token strides, X25 DStr.
	a.LDRx(X5, X0, oRows)
	a.LDRx(X7, X0, oRowStr)
	a.LDRx(X23, X0, oScratch)
	a.MOVimm(X19, int64(k+GEMMPad))
	a.MOVimm(X20, int64(8*nb+GEMMPad))
	a.MOVimm(X21, int64(4*nrows+GEMMPad))
	a.MOVimm(X22, int64(4*(k/16)+GEMMPad))
	if narrow {
		a.LDRx(X25, X0, oDStr)
	}
	a.MOVimm(X6, 0)
	if vK != 0 { // v0 is always a weight register, so 0 is "none"
		a64CorrConst(&a, vK, corrK)
	}

	rowL := a.Label()
	a.Bind(rowL)
	a.LDRx(X1, X0, oOut)
	a.ADDreg(X1, X1, X6)
	a.LDRx(X2, X0, oW)
	a.ADDreg(X2, X2, X6)
	a.LDRx(X3, X0, oA)
	a.LDRx(X4, X0, oAScale)
	if sub16 {
		a.LDRx(X16, X0, oAHalf)
	}
	a.LDRx(X8, X0, oPD)
	switch {
	case e8:
		a.LSRimm(X24, X6, 2)
		a.ADDreg(X8, X8, X24)
	case narrow:
		a.LSRimm(X24, X6, 1)
		a.ADDreg(X8, X8, X24)
	default:
		a.ADDreg(X8, X8, X6)
	}
	if !narrow {
		a.LDRx(X15, X0, oPSC)
		a.ADDreg(X15, X15, X6)
	}
	a.LDRx(X9, X0, oK)

	outer := a.Label()
	a.Bind(outer)

	// --- this super-block's eight row scales into Scratch[0,64): d at 0,
	// the minimum (x8 under intAcc) or 4*d (a 16-wide intAcc) at 32.
	{
		t0, t1, t2 := r[0], r[1], r[2]
		for h := 0; h < 2; h++ {
			switch {
			case e8:
				a.LDRs(t0, X8, int32(h*4))
				a.UXTL8h(t0, t0)
				a.UXTL4s(t0, t0)
				a.SHL4s(t0, t0, 23)
			case narrow:
				a.LDRd(t0, X8, int32(h*8))
				a.FCVTL(t0, t0)
			case f.d32:
				a.LDRq(t0, X8, int32(h*16)) // TQ2_F32: four rows' f32 scales
			default:
				a.LDRq(t1, X8, int32(h*16))
				if biasArray {
					a.UZP2h8(t2, t1, t1)
					a.FCVTL(t2, t2)
					if intAcc {
						a.FADD4s(t2, t2, t2)
						a.FADD4s(t2, t2, t2)
						a.FADD4s(t2, t2, t2) // 8*dmin, exactly
					}
					a.STRq(t2, X23, int32(32+h*16))
				}
				a.UZP1h8(t0, t1, t1)
				a.FCVTL(t0, t0)
			}
			a.STRq(t0, X23, int32(h*16))
			if intAcc && sub16 {
				a.FADD4s(t0, t0, t0)
				a.FADD4s(t0, t0, t0) // 4*d, exactly
				a.STRq(t0, X23, int32(32+h*16))
			}
		}
	}
	if narrow {
		a.ADDreg(X8, X8, X25)
	} else {
		a.ADDreg(X8, X8, X7)
	}

	for sn := 0; sn < perSuper; sn++ {
		pairOff := f.pairOff(sn)
		sh := f.scShift(sn)
		last := sn == perSuper-1

		// --- decode this sub-block of eight rows into W, once -------------
		vMask, vAux, pay, sec, tmp := r[0], r[1], r[2], r[3], r[4]
		if bits < 8 {
			a.MOVI16b(vMask, byte(1<<bits-1))
		}
		switch {
		case codes:
			a.LDRx(X24, X0, oScr)
			a.LDRq(vAux, X24, codesSlot)
		case hi != 0:
			a.MOVI16b(vAux, byte(((1<<hi)-1)<<bits))
		}
		a.MOVreg(X24, X2)
		for w := 0; w < pw; w++ {
			for h := 0; h < 2; h++ {
				if bits == 8 {
					a.LDRq(W[h][w], X24, int32(h*16))
					continue
				}
				if bits == 2 {
					a.LDRq(pay, X24, int32(h*16))
					for fi := 0; fi < f.fields; fi++ {
						if fi == 0 {
							a.AND16b(W[h][w], pay, vMask)
						} else {
							a.USHR16b(W[h][fi*pw+w], pay, byte(2*fi))
							a.AND16b(W[h][fi*pw+w], W[h][fi*pw+w], vMask)
						}
					}
					continue
				}
				a.LDRq(pay, X24, int32(h*16))
				a.AND16b(W[h][w], pay, vMask)
				a.USHR16b(W[h][pw+w], pay, 4) // per byte: the top nibble is zero
				if codes {
					a.TBL(W[h][w], vAux, W[h][w])
					a.TBL(W[h][pw+w], vAux, W[h][pw+w])
				}
			}
			a.ADDreg(X24, X24, X7)
		}
		for hwi := 0; hwi < hw; hwi++ {
			lanes := lanesOfA64(hi)
			for h := 0; h < 2; h++ {
				a.LDRq(sec, X24, int32(h*16))
				for gi := 0; gi < nW; gi++ {
					if gi/lanes != hwi {
						continue
					}
					switch s := hi * (gi % lanes); {
					case s < bits:
						a.SHL16b(tmp, sec, byte(bits-s))
						a.BIT16b(W[h][gi], tmp, vAux)
					case s == bits:
						a.BIT16b(W[h][gi], sec, vAux)
					default:
						a.USHR16b(tmp, sec, byte(s-bits))
						a.BIT16b(W[h][gi], tmp, vAux)
					}
				}
			}
			a.ADDreg(X24, X24, X7)
		}
		a.MOVreg(X2, X24)

		// --- the token-invariant half of the epilogue ----------------------
		for h := 0; h < 2; h++ {
			scw, t0 := r[0], r[1]
			sc, bf := inv[h][0], inv[h][1]
			// Container v27's SC stream (scstream.go): X24, the decode's
			// cursor and free here, addresses a straddling field's next word.
			next := func(r VReg) {
				a.ADDreg(X24, X15, X7)
				a.LDRq(r, X24, int32(h*16))
			}
			if intAcc {
				a.LDRq(scw, X15, int32(h*16))
				if stream {
					scField(a64SC(&a), sc, scw, t0, sn, false, next)
				} else {
					a64SCByte(&a, sc, scw, sh)
				}
				if scOff != 0 {
					a.MOVI4s(t0, byte(-scOff))
					a.SUB4s(sc, sc, t0)
				}
				if stream {
					scField(a64SC(&a), bf, scw, scw, sn, true, next)
					a.SCVTF4s(bf, bf)
				} else if biasArray {
					a64SCMinByte(&a, bf, scw, sh)
					a.SCVTF4s(bf, bf)
				} else {
					a.SCVTF4s(bf, sc)
				}
				continue
			}
			a.LDRq(sc, X23, int32(h*16)) // d_row
			if stream {
				a.LDRq(scw, X15, int32(h*16))
				scField(a64SC(&a), t0, scw, r[2], sn, false, next)
				if scOff != 0 {
					a.MOVI4s(r[2], byte(-scOff))
					a.SUB4s(t0, t0, r[2])
				}
				a.SCVTF4s(t0, t0)
				a.FMUL4s(sc, sc, t0) // d_row * sub-block scale
			} else if perSuper > 1 {
				a.LDRq(scw, X15, int32(h*16))
				a64SCByte(&a, t0, scw, sh)
				if scOff != 0 {
					a.MOVI4s(r[2], byte(-scOff))
					a.SUB4s(t0, t0, r[2])
				}
				a.SCVTF4s(t0, t0)
				a.FMUL4s(sc, sc, t0) // d_row * sub-block scale
			}
			if biasArray {
				a.LDRq(bf, X23, int32(32+h*16)) // dmin
				if stream {
					a.LDRq(scw, X15, int32(h*16))
					scField(a64SC(&a), t0, scw, scw, sn, true, next)
					a.SCVTF4s(t0, t0)
					a.FMUL4s(bf, bf, t0) // dmin * m
				} else if !minD {
					a.LDRq(scw, X15, int32(h*16))
					a64SCMinByte(&a, t0, scw, sh)
					a.SCVTF4s(t0, t0)
					a.FMUL4s(bf, bf, t0) // dmin * m
				}
			}
		}

		// --- the tokens ----------------------------------------------------
		a.LDRx(X10, X0, oCols)
		a.MOVreg(X11, X3)
		a.MOVreg(X12, X4)
		a.MOVreg(X13, X1)
		if sub16 {
			a.MOVreg(X14, X16)
		}
		a.ADDimm(X17, X23, 64)
		tokL := a.Label()
		a.Bind(tokL)
		if nact == 2 {
			a.LDPq(act[0], act[1], X11, int32(sub*sn))
		} else {
			a.LDRq(act[0], X11, int32(sub*sn))
		}
		for h := 0; h < 2; h++ {
			a.MOVIzero(acc[h])
			for j := 0; j < nW; j++ {
				a.SDOTelem(acc[h], W[h][j], act[j/4], uint8(j%4))
			}
		}
		bias := r[4]
		switch {
		case intAcc && biasArray:
			a.LDRs(bias, X12, pairOff+4) // -sum/8
		case intAcc:
			a.LDRs(bias, X14, int32(4*sn)) // the 16-wide half sum
		}
		if intAcc {
			S := [2]VReg{r[0], r[1]}
			B := [2]VReg{r[2], r[3]}
			if sn > 0 {
				a.LDPq(S[0], S[1], X17, 0)
				a.LDPq(B[0], B[1], X17, 32)
			}
			for h := 0; h < 2; h++ {
				if sn == 0 {
					a.MUL4s(S[h], acc[h], inv[h][0])
					a.FMULelem(B[h], inv[h][1], bias, 0)
				} else {
					a.MLA4s(S[h], acc[h], inv[h][0])
					a.FMLAelem(B[h], inv[h][1], bias, 0)
				}
			}
			if !last {
				a.STRq(S[0], X17, 0)
				a.STRq(S[1], X17, 16)
				a.STRq(B[0], X17, 32)
				a.STRq(B[1], X17, 48)
			} else {
				// The super-block's one float fold:
				//   out += d*da * sum(sc*isum) + 8*dmin*da * sum(m*pair4)
				//   out += d*da * sum(sc*isum) + 4*d*da * sum(sc*ahalf)
				// acc and act are dead after the dot and the MLA, and bias
				// after its FMLA; a 16-wide format has one act register.
				da, tt, O := r[5], [2]VReg{act[0], bias}, acc
				a.LDRs(da, X12, pairOff) // d_act, one per super-block
				a.LDPq(O[0], O[1], X13, 0)
				a.LDPq(tt[0], tt[1], X23, 0)
				for h := 0; h < 2; h++ {
					a.SCVTF4s(S[h], S[h])
					a.FMULelem(tt[h], tt[h], da, 0)
					a.FMLA4s(O[h], S[h], tt[h])
				}
				a.LDPq(tt[0], tt[1], X23, 32)
				for h := 0; h < 2; h++ {
					a.FMULelem(tt[h], tt[h], da, 0)
					a.FMLA4s(O[h], B[h], tt[h])
				}
				a.STRq(O[0], X13, 0)
				a.STRq(O[1], X13, 16)
			}
		} else {
			// The tiled kernel's epilogue, operation for operation.
			da, O, t2, s := r[5], [2]VReg{r[0], r[1]}, [2]VReg{r[2], r[3]}, r[4]
			a.LDRs(da, X12, pairOff)
			a.LDPq(O[0], O[1], X13, 0)
			for h := 0; h < 2; h++ {
				a.SCVTF4s(acc[h], acc[h])
				a.FMULelem(t2[h], inv[h][0], da, 0) // d_row * sub-scale * d_act
				a.FMLA4s(O[h], acc[h], t2[h])
			}
			switch {
			case corrK == 0:
			case biasArray:
				a.LDRs(s, X12, pairOff+4)
				a.FMUL4s(s, s, da) // bscale carries d_act
				a.FMUL4s(s, s, vK)
				for h := 0; h < 2; h++ {
					a.FMULelem(acc[h], inv[h][1], s, 0)
					a.FADD4s(O[h], O[h], acc[h])
				}
			case sub16:
				a.LDRs(s, X14, int32(4*sn))
				a.FMUL4s(s, s, vK)
				for h := 0; h < 2; h++ {
					a.FMULelem(acc[h], t2[h], s, 0)
					a.FADD4s(O[h], O[h], acc[h])
				}
			default:
				a.LDRs(s, X12, pairOff+4)
				a.FMUL4s(s, s, vK)
				for h := 0; h < 2; h++ {
					a.FMULelem(acc[h], t2[h], s, 0)
					a.FADD4s(O[h], O[h], acc[h])
				}
			}
			a.STRq(O[0], X13, 0)
			a.STRq(O[1], X13, 16)
		}
		a.ADDreg(X11, X11, X19)
		a.ADDreg(X12, X12, X20)
		a.ADDreg(X13, X13, X21)
		if sub16 {
			a.ADDreg(X14, X14, X22)
		}
		a.ADDimm(X17, X17, 64)
		a.SUBimm(X10, X10, 1)
		a.CBNZ(X10, tokL)

		if f.scWordDone(sn) {
			a.ADDreg(X15, X15, X7)
		}
	}
	a.ADDimm(X3, X3, int32(perSuper*sub))
	a.ADDimm(X4, X4, int32(perSuper*sub/Q8Block*8))
	if sub16 {
		a.ADDimm(X16, X16, int32(perSuper*4))
	}
	if perSuper == 1 && !narrow {
		a.ADDreg(X15, X15, X7)
	}
	a.SUBimm(X9, X9, 1)
	a.CBNZ(X9, outer)

	a.ADDimm(X6, X6, group*4)
	a.SUBimm(X5, X5, 1)
	a.CBNZ(X5, rowL)
	a.RET()
	return a.Bytes(), nil
}
