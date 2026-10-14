//go:build arm64 || amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// SupportedA64Packed reports whether the arm64 packed matvec covers a format.
// Every entry is gated against kernels.MatVecPacked on Apple hardware before
// it ships.
func SupportedA64Packed(t quant.Type) bool {
	switch t {
	case quant.Q8_0, quant.Q4_0, quant.Q5_0, quant.Q5_1, quant.Q4_K, quant.Q5_K, quant.Q3_K, quant.Q6_K, quant.MXFP4, quant.TQ2_0, quant.TQ2_F32:
		return true
	}
	return false
}

// EmitA64PackedMatVecPF is the container layout's decode matvec for arm64:
// out[r] += d_row[r] * d_act * dot(row r, activations), over `rows` rows per
// iteration and Args.Rows iterations.
//
// The packer writes qs[(si*words+w)*nrows + r], so four consecutive rows' word
// w is one 128-bit load and SDOT's four int32 lanes each accumulate one row,
// with no horizontal reduction. SDOTelem takes the activation four bytes at a
// time, so one LDRq of the activation serves four payload words. SDOT is
// signed by signed (VPDPBUSD is unsigned by signed), so Q8_0 needs no XOR
// centring and no +8 correction term.
//
//	x0  *Args on entry     v0..v(rows/4-1)  row accumulators
//	v16 payload            v17,v18          activation words
//	v19 d_act splatted     v20..v22         epilogue temporaries
//
// pf is the PRFM distance in bytes ahead of each payload line
// (EmitOpts.A64Prefetch): 0 takes the default, one tile (rows*4 bytes), and a
// negative value emits none.
func EmitA64PackedMatVecPF(t quant.Type, rows, pf int) ([]byte, error) {
	return emitA64Packed(t, rows, false, 0, false, pf)
}

// EmitA64PackedMatVecFused is EmitA64PackedMatVec with the loops the other way
// round, amd64's fused shape: row groups of `rows` outside (Args.Rows of them),
// every super-block of Args.K inside, and the float accumulators in registers
// for the whole walk. Out is read once at a group's head and written once at
// its tail, where the tiled kernel loads and stores it every sub-block (two of
// every ten vector memory operations on Q8_0). The sums are the same in the
// same order, so the two are bit-identical (nn.TestFusedMatchesTheTiledKernel).
//
// ahead is the software prefetch, amd64's two forms (EmitPackedMatVecFusedAhead):
// FusedAheadWords hints the payload a8Words planes ahead in the same rows (X23
// walks the planes beside X12), any other positive value hints the same word
// `ahead` row groups on, and 0 emits none.
func EmitA64PackedMatVecFused(t quant.Type, rows, ahead int) ([]byte, error) {
	return emitA64Packed(t, rows, true, ahead, false, 0)
}

// a8Words is the word-ahead form's distance, amd64's eight.
const a8Words = 8

func emitA64Packed(t quant.Type, rows int, fused bool, ahead int, intFold bool, a64pf int) ([]byte, error) {
	if !SupportedA64Packed(t) {
		return nil, fmt.Errorf("jit: EmitA64PackedMatVec: %s has no arm64 packed kernel", t)
	}
	q, ok := kernels.QuantOf(t)
	if !ok {
		return nil, fmt.Errorf("jit: EmitA64PackedMatVec: %s has no device layout", t)
	}
	if rows <= 0 || rows%4 != 0 || rows > 64 {
		return nil, fmt.Errorf("jit: EmitA64PackedMatVec: rows=%d, want a multiple of 4 up to 64", rows)
	}
	f, err := packedFormatK(q, t)
	if err != nil {
		return nil, err
	}
	sub, bits, biasArray := f.sub, f.bits, f.biasArray
	perSuper, scOff := f.perSuper, f.scOff
	minD, hi, codes := f.minD, f.hi, f.codes
	pw, hw := f.pw, f.hw
	grp := rows / 4
	// The activation span of one sub-block is `sub` bytes: 2*pw 32-bit words
	// for a 4-bit payload (lo nibbles then hi), pw for an 8-bit one.
	elems := sub / 4
	nact := (elems + 3) / 4

	// v0..v(grp-1) accumulators, then the fixed set.
	vPay := VReg(grp)
	vAct := func(i int) VReg { return VReg(grp + 1 + i) }
	vMask := VReg(grp + 1 + nact)
	vHi := vMask + 1
	vDA := vHi + 1
	t0, t1, t2, t3 := vDA+1, vDA+2, vDA+3, vDA+4
	vK := t3 + 1
	// The code table, held for the whole kernel; TBL needs it in a register.
	vLut, last := vK+1, vK
	if codes {
		last = vLut
	}
	// Three more for the epilogue's loop invariants: the sub-block's correction
	// factor, the row group's packed scale word (read once), and Q3_K/Q6_K's
	// scale offset.
	vBias, vSC, vOff := last+1, last+2, last+3
	last = vOff
	// The fused kernel's float accumulators, one per four rows, held across
	// every super-block.
	vF := func(g int) VReg { return vOff + 1 + VReg(g) }
	if fused {
		last = vF(grp - 1)
	}
	// The integer fold's per-group sums: vS the int32 sum of sc*dot over the
	// super-block, vB the f32 sum of m*bias (or sc*half-sum).
	vS := func(g int) VReg { return vF(grp-1) + 1 + VReg(g) }
	vB := func(g int) VReg { return vS(grp-1) + 1 + VReg(g) }
	if intFold {
		last = vB(grp - 1)
	}
	if int(last) > 31 {
		return nil, fmt.Errorf("jit: EmitA64PackedMatVec: %s rows=%d needs v%d, have 32", t, rows, last)
	}
	// Merging the planes costs no register: the payload loop's only scratch is
	// t0, and t1, t2, t3 and vDA are epilogue-local, so the secondary payload and
	// its shifted field borrow two of them. That is why the combine fits at
	// rows=64, where grp is 16 and the fixed set already reaches v25.
	vSec, vSecSh := t1, t2

	var a A64
	// t3 and vDA are epilogue-local too, and both are written there before they
	// are read, so a widened SDOT (sdotemu.go) may clobber them in the payload
	// loop.
	a.DotScratch(t3, vDA)
	// Container v27's SC stream (scstream.go): X24, free in this kernel,
	// addresses a straddling field's next word, one plane stride on.
	stream := f.stream
	stepSC := f.scWordDone
	nextSC := func(g int) func(VReg) {
		return func(r VReg) {
			a.ADDreg(X24, X17, X7)
			a.LDRq(r, X24, int32(g*16))
		}
	}
	a64PackedArgs(&a)
	// A narrow format packs two rows into one d word (kernels.DIndex), so its
	// plane is an f16 array at a 2-byte row stride: four rows are one LDRd plus
	// one FCVTL. The second stride and tile offset get their own registers.
	narrow, e8 := f.narrow, f.e8
	if narrow {
		a.LDRx(X25, X0, 160) // DStr
	}
	a.LDRx(X16, X0, 64) // AHalf
	// The two planes merge into one dot, exactly. A weight is `primary |
	// hi<<bits`, and since the fields are disjoint the OR is an addition:
	//
	//	sum(p_i*a_i) + sum((h_i<<bits)*a_i) == sum((p_i | h_i<<bits)*a_i)
	//
	// so one SDOT per element chunk serves both planes (Q6_K goes from 0.50 SDOT
	// per element to Q4_K's 0.25). This is arm64-only: VPDPBUSD takes unsigned
	// bytes, so amd64 adds the planes as two products into one accumulator for
	// free and must not change. SDOT is signed by signed and the combined byte
	// (0..63 for Q6_K, 0..31 for Q5_K) fits int8. The bias correction reads only
	// the activation sums and scales, so it does not move.
	//
	// Gated on hw == 1: one secondary word per (sub-block, row group) is
	// addressable from a single cursor beside X12, and the tiled twin has no
	// spare GPR for more. Q5_K and Q6_K are both hw == 1; a future format with
	// hw > 1 keeps the two-pass path below.
	combine := hi != 0 && hw == 1 && bits == 4
	// Without FEAT_DotProd the payload loop chains its dots (DotChain): the two
	// a row group takes per word -- the low and the high nibble, or the two
	// halves of a merged plane -- share one ADDP and one SADALP, and each
	// element's broadcast is taken once a word, before the row-group loop,
	// rather than once a dot. The broadcasts live in bA and bB: t1 and t2 when
	// the merged plane does not hold them, else the two registers past the
	// kernel's last, when there are two. A kernel with no room for them keeps
	// the per-SDOT widening, and the secondary plane's two-pass loop always
	// does. The chain is two links, inside DotChainCap for every weight here:
	// a nibble (15), a code (MXFP4's are at most 12), a merged plane (31 or 63).
	bA, bB := t1, t2
	chain := a.DotEmulating()
	if chain && combine {
		bA, bB = last+1, last+2
		chain = int(bB) <= 31
	}
	pair := false // the two nibbles' dots in one chain
	if chain && bits < 8 {
		wmax := 1<<bits - 1
		if combine {
			wmax = 1<<(bits+hi) - 1
		} else if codes {
			wmax = 0
			for _, c := range kernels.Codes(q) {
				if v := int(int8(c)); v > wmax {
					wmax = v
				} else if -v > wmax {
					wmax = -v
				}
			}
		}
		pair = DotChainCap(wmax) >= 2
	}
	// bcast takes element e of the activation into r for the row-group loop.
	bcast := func(r VReg, e int) { a.DUPs4lane(r, vAct(e/4), uint8(e%4)) }
	a64PackedConsts(&a, &f, vMask, vHi, vLut)
	// The correction's constant factor, which amd64 folds into bscale. A
	// signed payload needs none: SDOT is signed by signed, so there is no
	// amd64-style XOR to undo.
	corrK := f.corrK()
	if corrK != 0 {
		a64CorrConst(&a, vK, corrK)
	}
	if perSuper > 1 && scOff != 0 {
		a.MOVI4s(vOff, byte(-scOff))
	}

	// The activation, its scales and its per-16 sums: walked in place by the
	// tiled kernel, whose bases move per super-block; the fused kernel walks a
	// copy per group and resets it at the next.
	rA, rAS, rAH := X3, X4, X16
	words := fused && ahead == FusedAheadWords
	if fused {
		rA, rAS, rAH = X19, X20, X21
	}
	outer := a.Label()
	if !fused {
		a.Bind(outer)
	}
	tile := a64TileStart(&a, narrow)
	sb := a.Label()
	if fused {
		a.MOVreg(rA, X3)
		a.MOVreg(rAS, X4)
		a.MOVreg(rAH, X16)
		a.MOVreg(X22, X6) // super-blocks left
		if words {
			a.MOVreg(X23, X12)
			for i := 0; i < a8Words; i++ {
				a.ADDreg(X23, X23, X7)
			}
		}
		for g := 0; g < grp; g++ {
			a.LDRq(vF(g), X14, int32(g*16))
		}
		a.Bind(sb)
	}
	// step12 moves the payload cursor one word plane, and the fused kernel's
	// prefetch cursor with it.
	step12 := func() {
		a.ADDreg(X12, X12, X7)
		if words {
			a.ADDreg(X23, X23, X7)
		}
	}

	// One PRFM per 128-byte line, one tile ahead. A tile reads each word's
	// rows*4 contiguous bytes and then steps a whole plane, and the hardware
	// prefetcher only keeps up when the plane stride is a multiple of 8 KiB; the
	// hint makes the rate independent of the row count and costs the
	// power-of-two shapes nothing. A PRFM never faults, so hinting past the chunk
	// is harmless. See docs/engineering-history/cpu-kernels.md, "The container
	// layout's plane stride".
	pf := rows * 4
	if a64pf != 0 {
		pf = a64pf
	}
	if fused {
		// One tile ahead in the same plane is the NEXT GROUP's words, which
		// the fused kernel reaches only after the whole of k.
		pf = 0
	}
	// hint is the fused kernel's word-ahead prefetch for row group g.
	hint := func(g int) {
		switch {
		case g*16%128 != 0:
		case words:
			a.PRFM(X23, int32(g*16))
		case fused && ahead > 0:
			a.PRFM(X12, int32(g*16+ahead*rows*4))
		}
	}

	// foldSub is the integer fold's epilogue for sub-block sn: the sub-block
	// scale rides one MLA into vS, the bias term one FMLA into vB, and the float
	// fold into vF happens once, at the last sub-block. It is
	// EmitA64PackedMatMulStationary's integer form operation for operation --
	// the same int32 sums, the same f32 bias chain, the same final two FMLAs
	// from the same d, 8*dmin (or 4*d) and d_act -- so a prefill through that
	// GEMM and a decode through this kernel give the same bits.
	foldSub := func(sn int, pairOff int32, sh byte) {
		last := sn == perSuper-1
		if biasArray {
			a.LDRs(vBias, rAS, pairOff+4) // -sum/8
		} else {
			a.LDRs(vBias, rAH, int32(4*sn)) // the 16-wide half sum
		}
		if last {
			a.LDRs(vDA, rAS, pairOff) // d_act, one per super-block
		}
		for g := 0; g < grp; g++ {
			a.LDRq(vSC, X17, int32(g*16))
			if stream {
				scField(a64SC(&a), t1, vSC, t2, sn, false, nextSC(g))
			} else {
				a64SCByte(&a, t1, vSC, sh)
			}
			if scOff != 0 {
				a.SUB4s(t1, t1, vOff)
			}
			if sn == 0 {
				a.MUL4s(vS(g), VReg(g), t1)
			} else {
				a.MLA4s(vS(g), VReg(g), t1)
			}
			if stream {
				scField(a64SC(&a), t2, vSC, vSC, sn, true, nextSC(g))
				a.SCVTF4s(t2, t2) // m
			} else if biasArray {
				a64SCMinByte(&a, t2, vSC, sh)
				a.SCVTF4s(t2, t2) // m
			} else {
				a.SCVTF4s(t2, t1) // the signed scale
			}
			if sn == 0 {
				a.FMULelem(vB(g), t2, vBias, 0)
			} else {
				a.FMLAelem(vB(g), t2, vBias, 0)
			}
			if !last {
				continue
			}
			a.LDRq(t0, X13, int32(g*16))
			if biasArray {
				a.UZP2h8(t2, t0, t0)
				a.FCVTL(t2, t2)
				a.FADD4s(t2, t2, t2)
				a.FADD4s(t2, t2, t2)
				a.FADD4s(t2, t2, t2) // 8*dmin, exactly
			}
			if !f.d32 { // TQ2_F32's word is the scale already
				a.UZP1h8(t0, t0, t0)
				a.FCVTL(t0, t0)
			}
			if !biasArray {
				a.FADD4s(t2, t0, t0)
				a.FADD4s(t2, t2, t2) // 4*d, exactly
			}
			a.SCVTF4s(vS(g), vS(g))
			a.FMULelem(t1, t0, vDA, 0)
			a.FMLA4s(vF(g), vS(g), t1)
			a.FMULelem(t1, t2, vDA, 0)
			a.FMLA4s(vF(g), vB(g), t1)
		}
	}

	for sn := 0; sn < perSuper; sn++ {
		pairOff := f.pairOff(sn)
		sh := f.scShift(sn)
		for g := 0; g < grp; g++ {
			a.MOVIzero(VReg(g))
		}
		for i := 0; i < nact; i++ {
			a.LDRq(vAct(i), rA, int32(sub*sn+16*i))
		}
		if combine {
			// X11 is the secondary plane's cursor, and it is the only GPR this
			// costs. X12 is at this super-block's primary word 0 and the
			// secondary words follow the pw primary ones, so the single
			// secondary plane is pw strides on.
			a.MOVreg(X11, X12)
			for i := 0; i < pw; i++ {
				a.ADDreg(X11, X11, X7)
			}
		}
		for w := 0; w < pw; w++ {
			if chain && bits != 2 {
				bcast(bA, w)
				if bits == 4 {
					bcast(bB, pw+w)
				}
			}
			// dot is the payload loop's SDOTelem of element e into row group
			// g, or the chained link through e's broadcast b.
			dot := func(g int, src VReg, e int, b VReg, lastLink bool) {
				if chain {
					a.DotChain(VReg(g), src, b, lastLink || !pair)
					return
				}
				a.SDOTelem(VReg(g), src, vAct(e/4), uint8(e%4))
			}
			if bits == 2 {
				// TQ2's four 2-bit fields a byte, two at a time: field fi of
				// word w multiplies activation chunk fi*pw+w (packSub), and a
				// pair shares a chain where the dots are widened.
				for f0 := 0; f0 < f.fields; f0 += 2 {
					if chain {
						bcast(bA, f0*pw+w)
						bcast(bB, (f0+1)*pw+w)
					}
					for g := 0; g < grp; g++ {
						a.LDRq(vPay, X12, int32(g*16))
						if f0 == 0 {
							if pf > 0 && g%8 == 0 {
								a.PRFM(X12, int32(g*16+pf))
							}
							hint(g)
						}
						for i := 0; i < 2; i++ {
							fi := f0 + i
							if fi == 0 {
								a.AND16b(t0, vPay, vMask)
							} else {
								a.USHR16b(t0, vPay, byte(2*fi))
								a.AND16b(t0, t0, vMask)
							}
							dot(g, t0, fi*pw+w, []VReg{bA, bB}[i], i == 1)
						}
					}
				}
				step12()
				continue
			}
			for g := 0; g < grp; g++ {
				a.LDRq(vPay, X12, int32(g*16))
				if pf > 0 && g%8 == 0 {
					a.PRFM(X12, int32(g*16+pf))
				}
				hint(g)
				switch {
				case combine:
					a.LDRq(vSec, X11, int32(g*16))
					if pf > 0 && g%8 == 0 {
						a.PRFM(X11, int32(g*16+pf))
					}
					// X12's hint never reaches this plane -- X12 steps over it
					// unread -- so the secondary gets its own, once a sub-block.
					if w == 0 && fused && ahead > 0 && !words && g*16%128 == 0 {
						a.PRFM(X11, int32(g*16+ahead*rows*4))
					}
					for half := 0; half < 2; half++ {
						e := w + half*pw
						if half == 0 {
							a.AND16b(t0, vPay, vMask)
						} else {
							// No mask after the shift: USHR16b is a per-byte
							// shift, so bits 7:4 are already zero. (The two-pass
							// form keeps its dead AND, transliterated from amd64's
							// 16-bit VPSRLW, so its emitted bytes do not move.)
							a.USHR16b(t0, vPay, 4)
						}
						// The secondary field for chunk e sits at bit hi*e of
						// its byte and has to land at bit `bits`; BIT then
						// overwrites exactly vHi's bits of the primary, which
						// is the AND and the ORR in one vector instruction.
						switch s := hi * (e % lanesOfA64(hi)); {
						case s < bits:
							a.SHL16b(vSecSh, vSec, byte(bits-s))
							a.BIT16b(t0, vSecSh, vHi)
						case s == bits:
							// Already aligned: no shift, and no MOVvec either.
							a.BIT16b(t0, vSec, vHi)
						default:
							a.USHR16b(vSecSh, vSec, byte(s-bits))
							a.BIT16b(t0, vSecSh, vHi)
						}
						dot(g, t0, e, []VReg{bA, bB}[half], half == 1)
					}
				case bits == 4:
					// The quants are 0..15, which is inside int8's positive
					// range, so SDOT reads them directly -- no USDOT and no
					// i8mm feature to probe.
					a.AND16b(t0, vPay, vMask)
					if codes {
						a.TBL(t0, vLut, t0)
					}
					dot(g, t0, w, bA, false)
					a.USHR16b(t0, vPay, 4)
					if !chain {
						// Dead (USHR16b is a per-byte shift), kept so the
						// SDOT kernel's bytes do not move.
						a.AND16b(t0, t0, vMask)
					}
					if codes {
						a.TBL(t0, vLut, t0)
					}
					dot(g, t0, pw+w, bB, true)
				default:
					dot(g, vPay, w, bA, true)
				}
			}
			step12()
		}
		// --- the secondary plane ---
		if combine {
			// It was read in place through X11, but X12 still has to step over
			// it: the outer base advances by perSuper*(pw+hw) words and the
			// tile loop re-derives X12 from a base that assumes every word of
			// every super-block was walked.
			for i := 0; i < hw; i++ {
				step12()
			}
		} else if hw > 0 {
			lanes := 8 / hi
			for hwi := 0; hwi < hw; hwi++ {
				for g := 0; g < grp; g++ {
					a.LDRq(vPay, X12, int32(g*16))
					if pf > 0 && g*16%128 == 0 {
						a.PRFM(X12, int32(g*16+pf))
					}
					hint(g)
					for half := 0; half < 2; half++ {
						for w := 0; w < pw; w++ {
							gi := w + half*pw
							if gi/lanes != hwi {
								continue
							}
							switch s := hi * (gi % lanes); {
							case s < bits:
								a.SHL16b(t0, vPay, byte(bits-s))
							case s == bits:
								a.MOVvec(t0, vPay)
							default:
								a.USHR16b(t0, vPay, byte(s-bits))
							}
							a.AND16b(t0, t0, vHi)
							e := w
							if half == 1 {
								e = pw + w
							}
							a.SDOTelem(VReg(g), t0, vAct(e/4), uint8(e%4))
						}
					}
				}
				step12()
			}
		}
		// --- epilogue ---
		//
		// Four rows' d words are sixteen contiguous bytes, `d | dmin<<16` each, so
		// one LDRq holds all eight halves and UZP1/UZP2 split d from dmin, instead of
		// gathering lane by lane. The loop invariants are hoisted above. Every
		// product is computed in the same order as the gathering form, so the output
		// is bit-identical.
		if intFold {
			foldSub(sn, pairOff, sh)
			if stepSC(sn) {
				a.ADDreg(X17, X17, X7)
			}
			continue
		}
		a.LDRs(vDA, rAS, pairOff)
		a.DUPs4(vDA, vDA)
		switch {
		case corrK == 0:
		case biasArray:
			a.LDRs(vBias, rAS, pairOff+4)
			a.DUPs4(vBias, vBias)
			a.FMUL4s(vBias, vBias, vDA) // bscale carries d_act
			a.FMUL4s(vBias, vBias, vK)  // ... and the folded 8
		case sub == Q8Block/2:
			a.LDRs(vBias, rAH, int32(4*sn))
			a.DUPs4(vBias, vBias)
			a.FMUL4s(vBias, vBias, vK)
		default:
			a.LDRs(vBias, rAS, pairOff+4)
			a.DUPs4(vBias, vBias)
			a.FMUL4s(vBias, vBias, vK)
		}
		for g := 0; g < grp; g++ {
			// Four rows' row scales into t0 as four floats (and, for a format
			// with a minimum array, the four minima into t2).
			if e8 {
				// MXFP4's super-scale is a biased E8M0 exponent, so the
				// byte shifted left 23 is the f32: widen u8 to u32, then
				// SHL. Exact, since the value is a power of two. See
				// kernels.DSlots.
				a.LDRs(t0, X13, int32(g*4))
				a.UXTL8h(t0, t0)
				a.UXTL4s(t0, t0)
				a.SHL4s(t0, t0, 23)
			} else if narrow {
				a.LDRd(t0, X13, int32(g*4*2))
				a.FCVTL(t0, t0)
			} else if f.d32 {
				a.LDRq(t0, X13, int32(g*16)) // TQ2_F32: four rows' f32 scales
			} else {
				a.LDRq(t0, X13, int32(g*16))
				if corrK != 0 && biasArray {
					a.UZP2h8(t2, t0, t0)
					a.FCVTL(t2, t2)
				}
				a.UZP1h8(t0, t0, t0)
				a.FCVTL(t0, t0)
			}
			if stream {
				// t3 is free until the accumulator is read below.
				a.LDRq(vSC, X17, int32(g*16))
				scField(a64SC(&a), t1, vSC, t3, sn, false, nextSC(g))
				if scOff != 0 {
					a.SUB4s(t1, t1, vOff)
				}
				a.SCVTF4s(t1, t1)
				a.FMUL4s(t0, t0, t1) // d_row * sub-block scale
			} else if perSuper > 1 {
				a.LDRq(vSC, X17, int32(g*16)) // the packed scale word
				a64SCByte(&a, t1, vSC, sh)
				if scOff != 0 {
					// Per lane, not per byte: SUB16b does not borrow across
					// bytes, so a scale below the offset would wrap to 224+
					// instead of going negative.
					a.SUB4s(t1, t1, vOff)
				}
				a.SCVTF4s(t1, t1)
				a.FMUL4s(t0, t0, t1) // d_row * sub-block scale
			}
			a.SCVTF4s(VReg(g), VReg(g))
			a.FMUL4s(t1, t0, vDA) // d_row * sub-scale * d_act
			acc := t3
			if fused {
				acc = vF(g) // the running sum lives in a register
			} else {
				a.LDRq(acc, X14, int32(g*16))
			}
			a.FMLA4s(acc, VReg(g), t1)
			switch {
			case corrK == 0:
				// signed payload: SDOT needs no bias undone.
			case biasArray:
				// dmin (t2, the HIGH halves of the d words) * the minimum.
				if stream {
					// vSC is dead once read: the straddle's scratch.
					scField(a64SC(&a), t1, vSC, vSC, sn, true, nextSC(g))
					a.SCVTF4s(t1, t1)
					a.FMUL4s(t2, t2, t1) // dmin_row * min
				} else if !minD {
					a64SCMinByte(&a, t1, vSC, sh)
					a.SCVTF4s(t1, t1)
					a.FMUL4s(t2, t2, t1) // dmin_row * min
				}
				a.FMUL4s(t2, t2, vBias) // bscale * d_act * 8
				a.FADD4s(acc, acc, t2)  // amd64 negates the 8 and subtracts
			default:
				a.FMUL4s(t2, vBias, t1)
				a.FADD4s(acc, acc, t2)
			}
			if !fused {
				a.STRq(acc, X14, int32(g*16))
			}
		}
		// The scale word covers perWordOf sub-blocks; step it when it is spent.
		if stepSC(sn) {
			a.ADDreg(X17, X17, X7)
		}
	}

	if fused {
		// Every cursor but d's already walked its super-block; d is indexed
		// [super-block][row] and steps a whole plane.
		if narrow {
			a.ADDreg(X13, X13, X25)
		} else {
			a.ADDreg(X13, X13, X7)
		}
		a.ADDimm(rA, rA, int32(perSuper*sub))
		a.ADDimm(rAS, rAS, int32(perSuper*sub/Q8Block*8))
		if sub == Q8Block/2 {
			a.ADDimm(rAH, rAH, int32(perSuper*4))
		}
		a.SUBimm(X22, X22, 1)
		a.CBNZ(X22, sb)
		for g := 0; g < grp; g++ {
			a.STRq(vF(g), X14, int32(g*16))
		}
	}

	a.ADDimm(X10, X10, int32(rows*4))
	if narrow {
		// DRowBytes a row: two on a narrow f16 scale, ONE on MXFP4's E8M0 byte.
		a.ADDimm(X26, X26, int32(rows*DRowBytes(t)))
	}
	a.SUBimm(X9, X9, 1)
	a.CBNZ(X9, tile)
	if fused {
		a.RET()
		return a.Bytes(), nil
	}

	// The bases move once, after every tile has read them, and the row scale
	// plane is one of them: d is indexed [super-block][row] and advances one
	// plane per outer iteration exactly as the payload does. Dropping this
	// advance makes every super-block read super-block zero's scales.
	if narrow {
		a.ADDreg(X8, X8, X25) // half the payload's stride; see DSuperBytes
	} else {
		a.ADDreg(X8, X8, X7)
	}
	for i := 0; i < perSuper*(pw+hw); i++ {
		a.ADDreg(X2, X2, X7)
	}
	scWords := 1
	switch {
	case stream:
		scWords = kernels.ScStreamWords
	case perSuper > 1:
		scWords = perSuper / perWordOf(biasArray)
	}
	for i := 0; i < scWords; i++ {
		a.ADDreg(X15, X15, X7)
	}
	a.ADDimm(X3, X3, int32(perSuper*sub))
	a.ADDimm(X4, X4, int32(perSuper*sub/Q8Block*8))
	if sub == Q8Block/2 {
		a.ADDimm(X16, X16, int32(perSuper*4))
	}
	a.SUBimm(X6, X6, 1)
	a.CBNZ(X6, outer)
	a.RET()
	return a.Bytes(), nil
}

// lanesOfA64 is how many element chunks share one byte of a secondary plane
// word: eight bits divided by the plane's width. It is packSub's `lanes` and
// hiPlane's, named once so the emitters cannot drift from the packer.
func lanesOfA64(hi int) int { return 8 / hi }

// a64PackedArgs loads the pointers and strides every arm64 packed matrix
// kernel reads. AAPCS64 hands *Args in x0, and arm64 saves nothing, so every
// register below is the kernel's (x18, x27-x30 and sp excepted).
func a64PackedArgs(a *A64) {
	a.LDRx(X1, X0, 0)    // Out
	a.LDRx(X2, X0, 8)    // QS
	a.LDRx(X3, X0, 16)   // A
	a.LDRx(X4, X0, 24)   // AScale
	a.LDRx(X6, X0, 40)   // K, in super-blocks
	a.LDRx(X7, X0, 48)   // RowStr, bytes between payload word planes
	a.LDRx(X8, X0, 144)  // PD
	a.LDRx(X15, X0, 152) // PSC
}

// a64PackedConsts builds the constants an arm64 packed kernel holds for the
// whole walk: the nibble mask, the code table, and the secondary plane's mask
// PRE-SHIFTED by the primary width, so the two planes reach one accumulator
// (3<<4 is 48, and SDOT takes it).
func a64PackedConsts(a *A64, f *packedFormat, vMask, vHi, vLut VReg) {
	if f.bits < 8 {
		a.MOVI16b(vMask, byte(1<<f.bits-1)) // 0x0F a nibble, 0x03 a 2-bit field
	}
	if f.codes {
		a.LDRx(X5, X0, 56) // Scr
		a.LDRq(vLut, X5, codesSlot)
	}
	if f.hi != 0 {
		a.MOVI16b(vHi, byte(((1<<f.hi)-1)<<f.bits))
	}
}

// a64TileStart opens the row-tile loop of the arm64 packed matvec and its
// token tile, and returns its label. X10 is the tile's byte offset and X26 a
// narrow format's d plane offset, a half or a quarter of it. The cursors walk
// across sub-blocks and reset per tile, the opposite nesting to amd64's: the
// tile loop is outside, so X12 (payload) and X17 (scale words) start from the
// super-block base plus this tile's row offset and walk forward, with X13 the
// d words and X14 the outputs; the bases themselves move once per
// super-block, after the tile loop.
func a64TileStart(a *A64, narrow bool) int {
	a.LDRx(X9, X0, 32) // Rows, in tiles
	a.MOVimm(X10, 0)
	if narrow {
		a.MOVimm(X26, 0)
	}
	tile := a.Label()
	a.Bind(tile)
	a.ADDreg(X12, X2, X10)
	a.ADDreg(X17, X15, X10)
	dOff := X10
	if narrow {
		dOff = X26
	}
	a.ADDreg(X13, X8, dOff)
	a.ADDreg(X14, X1, X10)
	return tile
}

// a64SCByte extracts the sub-block scale byte at bit sh of each packed SC
// word lane into dst as an int32; amd64's scByte.
func a64SCByte(a *A64, dst, word VReg, sh byte) {
	a.SHL4s(dst, word, 24-sh)
	a.USHR4s(dst, dst, 24)
}

// a64SCMinByte extracts the minimum byte, eight bits above the scale's.
func a64SCMinByte(a *A64, dst, word VReg, sh byte) {
	a.SHL4s(dst, word, 16-sh)
	a.USHR4s(dst, dst, 24)
}

// a64CorrConst builds the correction's constant factor as f32 in every lane
// of vK (packedFormat.corrK).
func a64CorrConst(a *A64, vK VReg, k int) {
	a.MOVI4s(vK, byte(k))
	a.SCVTF4s(vK, vK)
}

// MaxTiledTokensA64 is how many tokens the arm64 token-tiled packed matmul
// covers for this format, or 0 when it has none.
//
// With 32 NEON registers the tile stays four wide where amd64 narrows. The
// budget is grp*tok accumulators, one payload, the activation words per token,
// the nibble mask, a secondary-plane mask, the correction constant and four
// epilogue temporaries: at worst 16 + 1 + 8 + 1 + 1 + 1 = 28.
func MaxTiledTokensA64(t quant.Type) int {
	q, ok := kernels.QuantOf(t)
	if !ok || !SupportedA64Packed(t) {
		return 0
	}
	sub, _, _, _ := kernels.Layout(q)
	hi := kernels.HiPlane(q)
	grp := packedFusedGroupK(q) / 4
	nact := ((sub / 4) + 3) / 4
	for tok := 4; tok >= 2; tok-- {
		used := grp*tok + 1 + nact*tok + 1 + 1
		if hi != 0 {
			used++
		}
		if kernels.Codes(q) != nil {
			used++ // the code table
		}
		if used+4 <= 32 {
			return tok
		}
	}
	return 0
}

// EmitA64PackedMatMulTiled is EmitPackedMatMulTiled's arm64 twin: tok
// activations against one pass over the weight, in the container layout.
// Without it prefill and the vision tower dispatch once per token, since
// nn.MatMulPacked requires a tiled kernel.
//
// k and nrows are baked, the granularity nn.AddShape already emits at. Args
// match the amd64 tiled kernel: Out is [tok][rows], A is [tok][k], AScale is
// [tok][2*nb] and AHalf is [tok][k/16].
func EmitA64PackedMatMulTiled(t quant.Type, k, nrows, tok int) ([]byte, error) {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return nil, fmt.Errorf("jit: EmitA64PackedMatMulTiled: %s has no device layout", t)
	}
	maxTok := MaxTiledTokensA64(t)
	if maxTok < 2 {
		return nil, fmt.Errorf("jit: EmitA64PackedMatMulTiled: %s has no tiled kernel", t)
	}
	// tok=1 is allowed: it is the remainder kernel. A batch divides nothing, and
	// arm64 has no fused per-token packed kernel to finish with as amd64 does.
	if tok < 1 || tok > maxTok {
		return nil, fmt.Errorf("jit: EmitA64PackedMatMulTiled: %s tok=%d out of 1..%d", t, tok, maxTok)
	}
	f, err := packedFormatK(q, t)
	if err != nil {
		return nil, err
	}
	sub, bits, biasArray := f.sub, f.bits, f.biasArray
	perSuper, scOff := f.perSuper, f.scOff
	minD, hi, codes := f.minD, f.hi, f.codes
	pw, hw := f.pw, f.hw
	grp := packedFusedGroupK(q) / 4
	group := packedFusedGroupK(q)
	if k <= 0 || k%(sub*perSuper) != 0 || nrows <= 0 || nrows%group != 0 {
		return nil, fmt.Errorf("jit: EmitA64PackedMatMulTiled: k=%d nrows=%d", k, nrows)
	}
	nb := k / Q8Block
	nact := ((sub / 4) + 3) / 4
	aTok := int32(k)
	asTok := int32(8 * nb)
	oTok := int32(4 * nrows)
	ahTok := int32(4 * (k / 16))

	next := VReg(0)
	take := func() VReg { r := next; next++; return r }
	acc := make([][]VReg, grp)
	for g := range acc {
		acc[g] = make([]VReg, tok)
		for i := range acc[g] {
			acc[g][i] = take()
		}
	}
	vPay := take()
	act := make([][]VReg, tok)
	for i := range act {
		act[i] = make([]VReg, nact)
		for j := range act[i] {
			act[i][j] = take()
		}
	}
	vMask, vHi, vK := take(), VReg(0), VReg(0)
	if hi != 0 {
		vHi = take()
	}
	vK = take()
	t0, t1, t2, t3 := take(), take(), take(), take()
	vLut := VReg(0)
	if codes {
		vLut = take()
	}
	if int(next)-1 > 31 {
		return nil, fmt.Errorf("jit: EmitA64PackedMatMulTiled: %s tok=%d needs v%d", t, tok, next-1)
	}
	// The payload loop's scratch is t0 alone; t1 and t2 are epilogue-local. See
	// the matvec -- merging the planes costs no register here either, so
	// MaxTiledTokensA64's budget is unchanged and the tile stays four wide.
	vSec, vSecSh := t1, t2

	var a A64
	// A widened SDOT (sdotemu.go) needs two scratch vectors and the tile has
	// spent the rest: t3, which the epilogue writes before it reads, and vK,
	// the correction constant, which the payload loop never reads and the
	// epilogue rebuilds when it was clobbered.
	a.DotScratch(t3, vK)
	// Container v27's SC stream (scstream.go): X25 -- a narrow format's DStr,
	// and a stream format is never narrow -- addresses a straddling field's
	// next word, one plane stride on.
	stream := f.stream
	nextSC := func(g int) func(VReg) {
		return func(r VReg) {
			a.ADDreg(X25, X17, X7)
			a.LDRq(r, X25, int32(g*16))
		}
	}
	a64PackedArgs(&a)
	a.LDRx(X16, X0, 64) // AHalf
	// The narrow d plane's own stride; see EmitA64PackedMatVec.
	narrow, e8 := f.narrow, f.e8
	if narrow {
		a.LDRx(X25, X0, 160) // DStr
	}
	// The same plane merge as the matvec, and it pays more here: `dot` issues tok
	// SDOTs per element chunk while the unpack is issued once, so halving the
	// chunks removes tok-1 instructions per chunk.
	//
	// It also needs !narrow: this kernel's only spare GPRs are X25 and X26, which
	// a narrow format spends on DStr and the d plane's offset. No format is both
	// narrow and two-plane, so the guard only keeps an unreachable combination on
	// the two-pass path.
	combine := hi != 0 && hw == 1 && bits == 4 && !narrow
	// The token strides go in registers, not immediates: a scaled `ldr q`
	// immediate tops out at 65520 bytes, and gemma-2b's 16384-row FFN has an
	// output token stride of 65536. The emitter panics rather than truncating.
	a.MOVimm(X19, int64(aTok))
	a.MOVimm(X20, int64(asTok))
	a.MOVimm(X21, int64(oTok))
	a.MOVimm(X22, int64(ahTok))
	a64PackedConsts(&a, &f, vMask, vHi, vLut)
	corrK := f.corrK() // SDOT is signed by signed: nothing to undo
	if corrK != 0 {
		a64CorrConst(&a, vK, corrK)
	}

	outer := a.Label()
	a.Bind(outer)
	tile := a64TileStart(&a, narrow)

	// No PRFM here, unlike EmitA64PackedMatVec: a tile does tok times the
	// arithmetic per byte, so the hint measured slightly slower in prefill. See
	// docs/engineering-history/cpu-kernels.md, "The container layout's plane
	// stride".
	for sn := 0; sn < perSuper; sn++ {
		pairOff := f.pairOff(sn)
		sh := f.scShift(sn)
		for g := 0; g < grp; g++ {
			for i := 0; i < tok; i++ {
				a.MOVIzero(acc[g][i])
			}
		}
		a.MOVreg(X5, X3)
		for i := 0; i < tok; i++ {
			for j := 0; j < nact; j++ {
				a.LDRq(act[i][j], X5, int32(sub*sn+16*j))
			}
			a.ADDreg(X5, X5, X19)
		}
		dot := func(g int, src VReg, e int) {
			for i := 0; i < tok; i++ {
				a.SDOTelem(acc[g][i], src, act[i][e/4], uint8(e%4))
			}
		}
		if combine {
			// X26 is the secondary plane's cursor; it is free exactly because
			// this format is not narrow.
			a.MOVreg(X26, X12)
			for i := 0; i < pw; i++ {
				a.ADDreg(X26, X26, X7)
			}
		}
		for w := 0; w < pw; w++ {
			for g := 0; g < grp; g++ {
				a.LDRq(vPay, X12, int32(g*16))
				switch {
				case combine:
					a.LDRq(vSec, X26, int32(g*16))
					for half := 0; half < 2; half++ {
						e := w + half*pw
						if half == 0 {
							a.AND16b(t0, vPay, vMask)
						} else {
							// USHR16b is per-BYTE: the top nibble is already
							// zero and the two-pass form's AND here is dead.
							a.USHR16b(t0, vPay, 4)
						}
						switch s := hi * (e % lanesOfA64(hi)); {
						case s < bits:
							a.SHL16b(vSecSh, vSec, byte(bits-s))
							a.BIT16b(t0, vSecSh, vHi)
						case s == bits:
							a.BIT16b(t0, vSec, vHi)
						default:
							a.USHR16b(vSecSh, vSec, byte(s-bits))
							a.BIT16b(t0, vSecSh, vHi)
						}
						dot(g, t0, e)
					}
				case bits == 4:
					a.AND16b(t0, vPay, vMask)
					if codes {
						a.TBL(t0, vLut, t0)
					}
					dot(g, t0, w)
					a.USHR16b(t0, vPay, 4)
					a.AND16b(t0, t0, vMask)
					if codes {
						a.TBL(t0, vLut, t0)
					}
					dot(g, t0, pw+w)
				case bits == 2:
					// TQ2: field fi of word w against activation chunk fi*pw+w.
					for fi := 0; fi < f.fields; fi++ {
						if fi == 0 {
							a.AND16b(t0, vPay, vMask)
						} else {
							a.USHR16b(t0, vPay, byte(2*fi))
							a.AND16b(t0, t0, vMask)
						}
						dot(g, t0, fi*pw+w)
					}
				default:
					dot(g, vPay, w)
				}
			}
			a.ADDreg(X12, X12, X7)
		}
		if combine {
			// Read in place through X26; X12 still steps over the plane so the
			// super-block bases stay pw+hw words apart.
			for i := 0; i < hw; i++ {
				a.ADDreg(X12, X12, X7)
			}
		} else if hw > 0 {
			lanes := 8 / hi
			for hwi := 0; hwi < hw; hwi++ {
				for g := 0; g < grp; g++ {
					a.LDRq(vPay, X12, int32(g*16))
					for half := 0; half < 2; half++ {
						for w := 0; w < pw; w++ {
							gi := w + half*pw
							if gi/lanes != hwi {
								continue
							}
							switch s := hi * (gi % lanes); {
							case s < bits:
								a.SHL16b(t0, vPay, byte(bits-s))
							case s == bits:
								a.MOVvec(t0, vPay)
							default:
								a.USHR16b(t0, vPay, byte(s-bits))
							}
							a.AND16b(t0, t0, vHi)
							e := w
							if half == 1 {
								e = pw + w
							}
							dot(g, t0, e)
						}
					}
				}
				a.ADDreg(X12, X12, X7)
			}
		}
		if a.dotEmu && corrK != 0 {
			a64CorrConst(&a, vK, corrK)
		}
		// --- epilogue: d_row and the sub-block scale are shared by every
		// token, so they are computed once per row group and reused.
		// Every token's d_act and bias are gathered once per sub-block, lane i
		// token i, into activation registers that are dead until the next
		// sub-block. The products keep their order, so the tile stays
		// bit-identical to the row loop (TestMatMulPackedMatchesTheRowLoopExactly).
		vDA4, vB4, gathered := VReg(0), VReg(0), false
		if flat := actFlat(act); len(flat) >= 2 && tok <= 4 {
			vDA4, vB4, gathered = flat[0], flat[1], true
			gather := func(dst VReg, base, stride XReg, off int32) {
				a.MOVreg(X23, base)
				for i := 0; i < tok; i++ {
					if i == 0 {
						a.LDRs(dst, X23, off)
					} else {
						a.LDRs(t1, X23, off)
						a.INSs(dst, uint8(i), t1, 0)
					}
					if i+1 < tok {
						a.ADDreg(X23, X23, stride)
					}
				}
			}
			gather(vDA4, X4, X20, pairOff)
			switch {
			case corrK == 0:
			case biasArray:
				gather(vB4, X4, X20, pairOff+4)
				a.FMUL4s(vB4, vB4, vDA4) // bscale carries d_act
				a.FMUL4s(vB4, vB4, vK)
			case sub == Q8Block/2:
				gather(vB4, X16, X22, int32(4*sn))
				a.FMUL4s(vB4, vB4, vK)
			default:
				gather(vB4, X4, X20, pairOff+4)
				a.FMUL4s(vB4, vB4, vK)
			}
		}
		for g := 0; g < grp; g++ {
			// Four rows' f16 row scales into t0. On a narrow plane they are
			// four ADJACENT halves: one 8-byte load and one widen.
			if e8 {
				// E8M0 scale: see EmitA64PackedMatVec.
				a.LDRs(t0, X13, int32(g*4))
				a.UXTL8h(t0, t0)
				a.UXTL4s(t0, t0)
				a.SHL4s(t0, t0, 23)
			} else if narrow {
				a.LDRd(t0, X13, int32(g*4*2))
				a.FCVTL(t0, t0)
			} else if f.d32 {
				a.LDRq(t0, X13, int32(g*16)) // TQ2_F32: four rows' f32 scales
			} else {
				// Four rows' `d | dmin<<16` words in one load, split by UZP
				// as in the matvec. dmin lands in the payload register,
				// which is dead in the epilogue.
				a.LDRq(t1, X13, int32(g*16))
				if biasArray {
					a.UZP2h8(vPay, t1, t1)
					a.FCVTL(vPay, vPay)
				}
				a.UZP1h8(t0, t1, t1)
				a.FCVTL(t0, t0)
			}
			if stream {
				// t3 is free until the accumulator is read below.
				a.LDRq(t2, X17, int32(g*16))
				scField(a64SC(&a), t1, t2, t3, sn, false, nextSC(g))
				if scOff != 0 {
					a.MOVI4s(t3, byte(-scOff))
					a.SUB4s(t1, t1, t3)
				}
				a.SCVTF4s(t1, t1)
				a.FMUL4s(t0, t0, t1)
			} else if perSuper > 1 {
				a.LDRq(t2, X17, int32(g*16))
				a64SCByte(&a, t1, t2, sh)
				if scOff != 0 {
					a.MOVI4s(t3, byte(-scOff))
					a.SUB4s(t1, t1, t3)
				}
				a.SCVTF4s(t1, t1)
				a.FMUL4s(t0, t0, t1)
			}
			var dmin VReg = 0
			if biasArray {
				// dmin*min is shared by every token too.
				dmin = vPay // gathered with d above
				if stream {
					a.LDRq(t2, X17, int32(g*16))
					scField(a64SC(&a), t2, t2, t1, sn, true, nextSC(g))
					a.SCVTF4s(t2, t2)
					a.FMUL4s(dmin, dmin, t2)
				} else if !minD {
					a.LDRq(t2, X17, int32(g*16))
					a64SCMinByte(&a, t2, t2, sh)
					a.SCVTF4s(t2, t2)
					a.FMUL4s(dmin, dmin, t2)
				}
			}
			a.ADDimm(X11, X14, int32(g*16))
			if gathered {
				for i := 0; i < tok; i++ {
					a.SCVTF4s(acc[g][i], acc[g][i])
					a.FMULelem(t2, t0, vDA4, uint8(i)) // d_row * sub-scale * d_act
					a.LDRq(t3, X11, 0)
					a.FMLA4s(t3, acc[g][i], t2)
					switch {
					case corrK == 0:
					case biasArray:
						a.FMULelem(t2, dmin, vB4, uint8(i))
						a.FADD4s(t3, t3, t2)
					default:
						a.FMULelem(t1, t2, vB4, uint8(i))
						a.FADD4s(t3, t3, t1)
					}
					a.STRq(t3, X11, 0)
					if i+1 < tok {
						a.ADDreg(X11, X11, X21)
					}
				}
				continue
			}
			a.MOVreg(X23, X4)
			a.MOVreg(X24, X16)
			for i := 0; i < tok; i++ {
				a.SCVTF4s(acc[g][i], acc[g][i])
				a.LDRs(t1, X23, pairOff)
				a.DUPs4(t1, t1)      // d_act
				a.FMUL4s(t2, t0, t1) // d_row * sub-scale * d_act
				a.LDRq(t3, X11, 0)
				a.FMLA4s(t3, acc[g][i], t2)
				switch {
				case corrK == 0:
				case biasArray:
					a.LDRs(t2, X23, pairOff+4)
					a.DUPs4(t2, t2)
					a.FMUL4s(t2, t2, t1) // bscale carries d_act
					a.FMUL4s(t2, t2, vK)
					a.FMUL4s(t2, t2, dmin)
					a.FADD4s(t3, t3, t2)
				case sub == Q8Block/2:
					a.LDRs(t1, X24, int32(4*sn))
					a.DUPs4(t1, t1)
					a.FMUL4s(t1, t1, vK)
					a.FMUL4s(t1, t1, t2)
					a.FADD4s(t3, t3, t1)
				default:
					a.LDRs(t1, X23, pairOff+4)
					a.DUPs4(t1, t1)
					a.FMUL4s(t1, t1, vK)
					a.FMUL4s(t1, t1, t2)
					a.FADD4s(t3, t3, t1)
				}
				a.STRq(t3, X11, 0)
				a.ADDreg(X11, X11, X21)
				a.ADDreg(X23, X23, X20)
				a.ADDreg(X24, X24, X22)
			}
		}
		if f.scWordDone(sn) {
			a.ADDreg(X17, X17, X7)
		}
	}

	a.ADDimm(X10, X10, int32(group*4))
	if narrow {
		// DRowBytes a row; see EmitA64PackedMatVec's twin.
		a.ADDimm(X26, X26, int32(group*DRowBytes(t)))
	}
	a.SUBimm(X9, X9, 1)
	a.CBNZ(X9, tile)

	// The bases move once, after every tile has read them; see the matvec.
	if narrow {
		a.ADDreg(X8, X8, X25) // half the payload's stride; see DSuperBytes
	} else {
		a.ADDreg(X8, X8, X7)
	}
	for i := 0; i < perSuper*(pw+hw); i++ {
		a.ADDreg(X2, X2, X7)
	}
	scWords := 1
	switch {
	case stream:
		scWords = kernels.ScStreamWords
	case perSuper > 1:
		scWords = perSuper / perWordOf(biasArray)
	}
	for i := 0; i < scWords; i++ {
		a.ADDreg(X15, X15, X7)
	}
	a.ADDimm(X3, X3, int32(perSuper*sub))
	a.ADDimm(X4, X4, int32(perSuper*sub/Q8Block*8))
	if sub == Q8Block/2 {
		a.ADDimm(X16, X16, int32(perSuper*4))
	}
	a.SUBimm(X6, X6, 1)
	a.CBNZ(X6, outer)
	a.RET()
	return a.Bytes(), nil
}

// EmitA64PackedMatVecFusedWin is EmitA64PackedMatVecFused with the integer
// super-block fold (StationaryIntAccA64) where the activation window allows it:
// the sub-block scale is one MLA into an int32 sum and the float epilogue runs
// once per super-block, where the float form ran it per sub-block -- about 19
// instructions a row group a sub-block against ~7. Where the window does not
// allow it, it is the float kernel.
func EmitA64PackedMatVecFusedWin(t quant.Type, rows, ahead, win int) ([]byte, error) {
	return emitA64Packed(t, rows, true, ahead, StationaryIntAccA64(t, win), 0)
}

// actFlat lists a tile's activation registers, token-major.
func actFlat(act [][]VReg) []VReg {
	var out []VReg
	for _, r := range act {
		out = append(out, r...)
	}
	return out
}
