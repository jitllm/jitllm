//go:build arm64

package cpu

// EmitMoETopK is the NEON twin of the AVX2 kernel in topk.go, which carries the
// contract, the constant block's layout and the reasoning. The two differ only
// in width and in how a mask is applied.
//
//	x0  the Args block
//	x1  the element cursor      x2  Scr        x3  p
//	x4  Scratch                 x5  sel        x6  ord
//	x7  wt                      x9  ow         x10 sum
//	x11, x12 loop counters
//
//	v0  the running maximum     v1  its index  v2  the element index
//	v3  the index step          v11 prevVal    v12 prevIdx
//	v13 INT_MAX                 v14 NaN        v4..v10, v15 temporaries
//
// A masked select is one instruction here: BIT (Bitwise Insert if True) is
// vd = (vn & vm) | (vd & ~vm), where the AVX2 kernel spends an AND, an ANDN
// and an OR. The tail's NaN filler is an INS into a register full of NaN.
//
// The two folds are single instructions: SMINV for the lowest tying index,
// ADDV for the one surviving weight lane. ADDV is the integer reduce on
// purpose: with at most one nonzero lane it is a select, and the survivor's
// bits arrive unchanged, including a -0.0 a float add would cancel.
func EmitMoETopK(n, k int, norm bool) ([]byte, error) {
	return EmitMoERoute(n, k, MoEGate{Norm: norm})
}

// EmitMoERoute is the NEON twin of the DeepSeek-V3 router in topk.go, which
// carries the contract and the reasoning. V16 is the group accumulator (the
// SSE kernel keeps it in memory), and X13/X14/X15 are the cursors the extra
// phases need, so nothing the selection loaded has to be reloaded.
func EmitMoERoute(n, k int, g MoEGate) ([]byte, error) {
	if err := moeRouteCheck(n, k, g); err != nil {
		return nil, err
	}
	return must("moe_topk", emitMoETopKA64(n, k, g))
}

func emitMoETopKA64(n, k int, g MoEGate) []byte {
	const lanes = 4
	nv, nt := n/lanes, n%lanes
	pl := moePlanes(n, k, g)
	pad := pl.pad
	wtOff := int32(4 * pl.wt)
	biOff := int32(4 * pl.bi)
	gsOff := int32(4 * pl.gs)
	keepOff := int32(4 * pl.keep)
	vecs := pad / lanes

	var a A64
	a.LDRx(X2, X0, 56)  // Scr
	a.LDRx(X3, X0, 112) // Q32      -> p
	a.LDRx(X4, X0, 72)  // Scratch
	a.LDRx(X5, X0, 96)  // ASum     -> sel
	a.LDRx(X6, X0, 104) // AHalfSum -> ord
	a.LDRx(X7, X0, 0)   // Out      -> wt
	a.LDRx(X9, X0, 120) // Out2     -> ow
	a.LDRx(X10, X0, 24) // AScale   -> sum

	// addOff is `rd = rn + off` for an offset ADD's twelve-bit immediate cannot
	// reach. X16 is the architecture's intra-procedure scratch and this kernel
	// is a leaf that calls nothing, so nothing of the caller's lives there.
	addOff := func(rd, rn XReg, off int32) {
		if off <= 4095 {
			a.ADDimm(rd, rn, off)
			return
		}
		a.MOVimm(X16, int64(off))
		a.ADDreg(rd, rn, X16)
	}

	a.LDRs(V13, X2, moeMaxIOff)
	a.DUPs4(V13, V13)
	for v := 0; v < vecs; v++ {
		a.STRq(V13, X4, int32(16*v))
	}
	a.MOVIzero(V4)
	for v := 0; v < vecs; v++ {
		a.STRq(V4, X4, wtOff+int32(16*v))
	}

	// One pass of the lexicographic walk, leaving v11 = the winner's value and
	// v12 = its id.
	lexPass := func(base XReg, cnt, i int) {
		cnv, cnt4 := cnt/lanes, cnt%lanes
		a.LDRs(V0, X2, moeNegInfOff)
		a.DUPs4(V0, V0)
		a.MOVvec(V1, V13)         // the index, INT_MAX = this lane is empty
		a.LDRq(V2, X2, moeIdxOff) // 0..3
		a.MOVreg(X1, base)

		body := func() {
			if i == 0 {
				a.FCMEQ4s(V5, V4, V4) // ordered: every element but a NaN
			} else {
				a.FCMGT4s(V5, V11, V4) // prevVal > p, i.e. p < prevVal
				a.FCMEQ4s(V6, V4, V11)
				a.CMGT4s(V7, V2, V12) // and a higher id than the tie it lost
				a.AND16b(V6, V6, V7)
				a.ORR16b(V5, V5, V6)
			}
			// FCMGT is false for a NaN on either side, which is `v > best`
			// exactly; FMAX would propagate a NaN into the running maximum.
			a.FCMGT4s(V8, V4, V0)
			a.CMEQ4s(V9, V1, V13) // unless this lane is still empty
			a.ORR16b(V8, V8, V9)
			a.AND16b(V5, V5, V8)
			a.BIT16b(V0, V4, V5)
			a.BIT16b(V1, V2, V5)
			a.ADD4s(V2, V2, V3)
		}

		if cnv > 0 {
			a.LDRs(V3, X2, moeStep4Off)
			a.DUPs4(V3, V3)
			countLoopA64(&a, X11, cnv, func() {
				a.LDRq(V4, X1, 0)
				body()
				a.ADDimm(X1, X1, 16)
			})
		}
		if cnt4 > 0 {
			a.LDRs(V14, X2, moeNaNOff)
			a.DUPs4(V14, V14)
			a.LDRs(V3, X2, moeOneIOff)
			a.DUPs4(V3, V3)
			countLoopA64(&a, X12, cnt4, func() {
				a.LDRs(V10, X1, 0)
				a.MOVvec(V4, V14)     // NaN in every lane
				a.INSs(V4, 0, V10, 0) // and the element in lane 0
				body()
				a.ADDimm(X1, X1, 4)
			})
		}

		// The fold. The per-lane maxima survive in v15; no lane of v0 can hold a
		// NaN -- FCMGT never selected one -- so FMAXV's NaN rule is unreachable.
		a.MOVvec(V15, V0)
		a.FMAXV(V0, V0)
		a.DUPs4(V0, V0)
		a.FCMEQ4s(V5, V15, V0)
		a.MOVvec(V6, V13)
		a.BIT16b(V6, V1, V5) // the index where this lane ties, INT_MAX where not
		a.SMINV(V6, V6)
		a.DUPs4(V12, V6) // the winner's id, and pass i+1's prevIdx
		a.CMEQ4s(V5, V1, V12)
		a.MOVIzero(V6)
		a.BIT16b(V6, V15, V5)
		a.ADDV(V6, V6)
		a.DUPs4(V11, V6) // its probability, and pass i+1's prevVal
	}

	// ---- PHASE A: the plane the selection walks (topk.go). ----
	plane := X3
	if g.needsBiasedPlane() {
		plane = X13
		a.MOVreg(X1, X3)
		addOff(X13, X4, biOff)
		if g.Bias {
			a.LDRx(X14, X0, 128) // Q2 -> the selection bias
		}
		if nv > 0 {
			countLoopA64(&a, X11, nv, func() {
				a.LDRq(V0, X1, 0)
				if g.Bias {
					a.LDRq(V1, X14, 0)
					a.FADD4s(V0, V0, V1)
					a.ADDimm(X14, X14, 16)
				}
				a.STRq(V0, X13, 0)
				a.ADDimm(X1, X1, 16)
				a.ADDimm(X13, X13, 16)
			})
		}
		if nt > 0 {
			countLoopA64(&a, X12, nt, func() {
				a.LDRs(V0, X1, 0)
				if g.Bias {
					a.LDRs(V1, X14, 0)
					a.FADD4s(V0, V0, V1)
					a.ADDimm(X14, X14, 4)
				}
				a.STRs(V0, X13, 0)
				a.ADDimm(X1, X1, 4)
				a.ADDimm(X13, X13, 4)
			})
		}
	}

	// ---- PHASE B: the expert groups (topk.go). ----
	if g.Grouped() {
		sz := n / g.NGroup
		take := 2
		if sz < take {
			take = sz
		}
		gvecs := pl.gPad / lanes

		a.MOVIzero(V4)
		for v := 0; v < gvecs; v++ {
			a.STRq(V4, X4, gsOff+int32(16*v))
		}

		// B1: gs[g] = the sum of the group's top two biased scores.
		addOff(X13, X4, biOff)
		addOff(X14, X4, gsOff)
		countLoopA64(&a, X15, g.NGroup, func() {
			a.MOVIzero(V16) // ggml_sum_rows accumulates from zero
			for t := 0; t < take; t++ {
				lexPass(X13, sz, t)
				a.FADD4s(V16, V16, V11)
			}
			a.STRs(V16, X14, 0)
			addOff(X13, X13, int32(4*sz))
			a.ADDimm(X14, X14, 4)
		})

		// B2: the best NGroupUsed groups; only the threshold is wanted.
		addOff(X13, X4, gsOff)
		lexPass(X13, g.NGroup, 0)
		if g.NGroupUsed > 1 {
			countLoopA64(&a, X15, g.NGroupUsed-1, func() { lexPass(X13, g.NGroup, 1) })
		}

		// B3: kept(g) = gs > thrVal || (gs == thrVal && g <= thrIdx).
		addOff(X13, X4, gsOff)
		addOff(X14, X4, keepOff)
		a.LDRq(V2, X2, moeIdxOff)
		a.LDRs(V3, X2, moeStep4Off)
		a.DUPs4(V3, V3)
		countLoopA64(&a, X11, gvecs, func() {
			a.LDRq(V4, X13, 0)
			a.FCMGT4s(V5, V4, V11) // gs > thrVal
			a.FCMEQ4s(V6, V4, V11) // gs == thrVal
			a.CMGT4s(V7, V2, V12)  // g > thrIdx
			a.BIC16b(V8, V6, V7)   // ... and NOT that: g <= thrIdx
			a.ORR16b(V5, V5, V8)
			a.STRq(V5, X14, 0)
			a.ADD4s(V2, V2, V3)
			a.ADDimm(X13, X13, 16)
			a.ADDimm(X14, X14, 16)
		})

		// B4: every expert of a group that was not kept goes to -Inf.
		a.LDRs(V17, X2, moeNegInfOff)
		a.DUPs4(V17, V17)
		addOff(X13, X4, biOff)
		addOff(X14, X4, keepOff)
		svec, stail := sz/lanes, sz%lanes
		countLoopA64(&a, X15, g.NGroup, func() {
			a.LDRs(V5, X14, 0)
			a.DUPs4(V5, V5)
			if svec > 0 {
				countLoopA64(&a, X11, svec, func() {
					a.LDRq(V4, X13, 0)
					a.MOVvec(V6, V17)    // -Inf everywhere
					a.BIT16b(V6, V4, V5) // ... and the score where kept
					a.STRq(V6, X13, 0)
					a.ADDimm(X13, X13, 16)
				})
			}
			if stail > 0 {
				countLoopA64(&a, X12, stail, func() {
					a.LDRs(V4, X13, 0)
					a.MOVvec(V6, V17)
					a.BIT16b(V6, V4, V5)
					a.STRs(V6, X13, 0)
					a.ADDimm(X13, X13, 4)
				})
			}
			a.ADDimm(X14, X14, 4)
		})
	}

	if g.needsBiasedPlane() {
		addOff(X13, X4, biOff)
	}

	// ---- PHASE C: the selection. ----
	for i := 0; i < k; i++ {
		lexPass(plane, n, i)
		a.STRs(V12, X5, int32(4*i))
		a.STRs(V12, X4, int32(4*i))
		a.STRs(V11, X4, wtOff+int32(4*i))
	}

	// ---- PHASE D: the unbiased weight of each winner (topk.go). ----
	if g.Bias {
		for i := 0; i < k; i++ {
			moeGatherIDA64(&a, X3, int32(4*i), nv, nt)
			a.STRs(V10, X4, wtOff+int32(4*i))
		}
	}

	// The divisor, in selection order, one addition at a time.
	if g.Norm {
		a.MOVIzero(V0)
		for i := 0; i < k; i++ {
			a.LDRs(V4, X4, wtOff+int32(4*i))
			a.FADD4s(V0, V0, V4)
		}
		// The divisor's floor; see topk.go and moeSumFloor.
		a.LDRs(V4, X2, moeSumFloor)
		a.FADD4s(V0, V0, V4)
		a.DUPs4(V0, V0)
	} else {
		a.LDRs(V0, X2, moeOneFOff)
		a.DUPs4(V0, V0)
	}
	a.STRs(V0, X10, 0)
	if g.Scale {
		a.LDRs(V1, X2, moeScaleOff)
		a.DUPs4(V1, V1)
	}
	if g.ExpScale {
		a.LDRx(X14, X0, 136) // AScale2 -> the per-expert scale; see topk.go
	}
	for i := 0; i < k; i++ {
		if g.ExpScale {
			moeGatherIDA64(&a, X14, int32(4*i), nv, nt)
		}
		a.LDRs(V4, X4, wtOff+int32(4*i))
		a.FDIV4s(V4, V4, V0)
		if g.Scale {
			a.FMUL4s(V4, V4, V1) // after the division; see topk.go
		}
		if g.ExpScale {
			a.FMUL4s(V4, V4, V10) // last; see topk.go
		}
		a.STRs(V4, X7, int32(4*i))
		a.STRs(V4, X4, wtOff+int32(4*i))
	}

	// Ascending expert id, by successive minimum over the pad.
	a.LDRs(V12, X2, moeMinusOne)
	a.DUPs4(V12, V12)
	for r := 0; r < k; r++ {
		for v := 0; v < vecs; v++ {
			a.LDRq(V4, X4, int32(16*v))
			a.CMGT4s(V5, V4, V12)
			a.MOVvec(V6, V13)
			a.BIT16b(V6, V4, V5)
			if v == 0 {
				a.MOVvec(V9, V6)
			} else {
				a.SMIN4s(V9, V9, V6)
			}
		}
		a.SMINV(V9, V9)
		a.DUPs4(V12, V9)
		a.STRs(V12, X6, int32(4*r))
		a.MOVIzero(V10)
		for v := 0; v < vecs; v++ {
			a.LDRq(V4, X4, int32(16*v))
			a.CMEQ4s(V5, V4, V12)
			a.LDRq(V6, X4, wtOff+int32(16*v))
			a.AND16b(V6, V6, V5)
			a.ORR16b(V10, V10, V6)
		}
		a.ADDV(V10, V10)
		a.STRs(V10, X9, int32(4*r))
	}

	a.RET()
	return a.Bytes()
}

// countLoopA64 emits body exactly cnt times: inline for one or two iterations,
// a counted loop beyond that. cnt is known at emit time, so no zero test is
// needed.
func countLoopA64(a *A64, r XReg, cnt int, body func()) {
	if cnt <= 0 {
		return
	}
	if cnt <= 2 {
		for i := 0; i < cnt; i++ {
			body()
		}
		return
	}
	a.MOVimm(r, int64(cnt))
	lp := a.Label()
	a.Bind(lp)
	body()
	a.SUBimm(r, r, 1)
	a.CBNZ(r, lp)
}

// moeGatherIDA64 is topk.go's moeGatherID on NEON: V10's lane 0 is src[id]
// for the id at X4+idOff. Clobbers V2-V7, V12, X1, X11 and X12.
func moeGatherIDA64(a *A64, src XReg, idOff int32, nv, nt int) {
	a.LDRs(V12, X4, idOff) // the id, as 32 bits
	a.DUPs4(V12, V12)
	a.MOVIzero(V10)
	a.LDRq(V2, X2, moeIdxOff)
	a.MOVreg(X1, src)
	if nv > 0 {
		a.LDRs(V3, X2, moeStep4Off)
		a.DUPs4(V3, V3)
		countLoopA64(a, X11, nv, func() {
			a.LDRq(V4, X1, 0)
			a.CMEQ4s(V5, V2, V12)
			a.AND16b(V6, V4, V5)
			a.ORR16b(V10, V10, V6)
			a.ADD4s(V2, V2, V3)
			a.ADDimm(X1, X1, 16)
		})
	}
	if nt > 0 {
		a.LDRs(V3, X2, moeOneIOff)
		a.DUPs4(V3, V3)
		countLoopA64(a, X12, nt, func() {
			a.LDRs(V4, X1, 0)
			a.MOVvec(V7, V13)    // INT_MAX, which matches no id
			a.INSs(V7, 0, V2, 0) // ... except in lane 0
			a.CMEQ4s(V5, V7, V12)
			a.AND16b(V6, V4, V5)
			a.ORR16b(V10, V10, V6)
			a.ADD4s(V2, V2, V3)
			a.ADDimm(X1, X1, 4)
		})
	}
	// At most one lane matched, so the integer reduce is a select and the
	// survivor's bits arrive unchanged, -0.0 included.
	a.ADDV(V10, V10)
}
