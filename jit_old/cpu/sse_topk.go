package cpu

// The SSE tier's MoE router top-k: EmitMoETopK (topk.go) on legacy SSE, with
// the AVX2 twin's ABI and constant block unchanged.
//
// It walks four elements an iteration where AVX2 walks eight, and unlike the
// norms that is allowed: nothing here is arithmetic except the renormalising
// sum, which is a k-term chain in selection order on every tier. Every other
// output is an input selected by comparison, and ties are broken by index,
// so the grouping does not matter and the gate is equality
// (moetopk_test.go).
//
// PMINSD and BLENDPS (SSE4.1) decide the tier: ids are 32-bit with a
// 0x7FFFFFFF sentinel, and the blend lets the ragged tail reuse the vector
// body (see topk.go).
//
// Registers: RSI the element cursor, RBX Scr, R12 p, R13 the scratch, R8 sel,
// R9 ord, R10 wt, R11 ow, R15 sum; XMM0 the running maximum, XMM1 its index,
// XMM2 the element index, XMM3 the index step, XMM11 prevVal, XMM12 prevIdx,
// XMM13 INT_MAX, XMM14 NaN, XMM4..XMM10 and XMM15 temporaries.
func emitMoETopKSSEBody(n, k int, g MoEGate) []byte {
	const lanes = 4
	nv, nt := n/lanes, n%lanes
	pl := moePlanes(n, k, g)
	pad := pl.pad
	wtOff := int32(4 * pl.wt)
	biOff := int32(4 * pl.bi)
	gsOff := int32(4 * pl.gs)
	keepOff := int32(4 * pl.keep)
	vecs := pad / lanes

	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RBX, At(RDI, 56))  // Scr
	a.MOVLoad(R12, At(RDI, 112)) // Q32      -> p
	a.MOVLoad(R13, At(RDI, 72))  // Scratch
	a.MOVLoad(R8, At(RDI, 96))   // ASum     -> sel
	a.MOVLoad(R9, At(RDI, 104))  // AHalfSum -> ord
	a.MOVLoad(R10, At(RDI, 0))   // Out      -> wt
	a.MOVLoad(R11, At(RDI, 120)) // Out2     -> ow
	a.MOVLoad(R15, At(RDI, 24))  // AScale   -> sum

	bcastD(&a, XMM13, At(RBX, moeMaxIOff))
	for v := 0; v < vecs; v++ {
		a.MOVDQUStore(At(R13, int32(16*v)), XMM13)
	}
	a.PXOR(XMM4, XMM4, XMM4)
	for v := 0; v < vecs; v++ {
		a.MOVDQUStore(At(R13, wtOff+int32(16*v)), XMM4)
	}

	// One pass of the lexicographic walk, leaving XMM11 = the winner's value
	// and XMM12 = its id. topk.go's lexPass is the same function eight lanes
	// wide and carries the reasoning.
	lexPass := func(base Reg, cnt, i int) {
		cnv, cnt4 := cnt/lanes, cnt%lanes
		bcastSS(&a, XMM0, At(RBX, moeNegInfOff))
		a.MOVAPS(XMM1, XMM13)                  // the index, INT_MAX = still empty
		a.MOVUPSLoad(XMM2, At(RBX, moeIdxOff)) // 0..3
		a.MOVQ(RSI, base)

		body := func() {
			if i == 0 {
				a.CMPPS(XMM5, XMM4, XMM4, 0) // ordered: every element but a NaN
			} else {
				a.CMPPS(XMM5, XMM4, XMM11, 1) // p < prevVal
				a.CMPPS(XMM6, XMM4, XMM11, 0) // p == prevVal
				a.PCMPGTD(XMM7, XMM2, XMM12)  // and a higher id than the tie it lost
				a.PAND(XMM6, XMM6, XMM7)
				a.POR(XMM5, XMM5, XMM6)
			}
			// Predicate 1 spells `p > max`: legacy CMPPS has no ordered
			// greater-than, so the operands swap. (max < p) ordered is false
			// for a NaN on either side.
			a.CMPPS(XMM8, XMM0, XMM4, 1)
			a.PCMPEQD(XMM9, XMM1, XMM13) // unless this lane is still empty
			a.POR(XMM8, XMM8, XMM9)
			a.PAND(XMM5, XMM5, XMM8)
			a.PAND(XMM6, XMM4, XMM5)
			a.PANDN(XMM7, XMM5, XMM0)
			a.POR(XMM0, XMM6, XMM7)
			a.PAND(XMM6, XMM2, XMM5)
			a.PANDN(XMM7, XMM5, XMM1)
			a.POR(XMM1, XMM6, XMM7)
			a.PADDD(XMM2, XMM2, XMM3)
		}

		if cnv > 0 {
			bcastD(&a, XMM3, At(RBX, moeStep4Off))
			countLoopSSE(&a, RAX, cnv, func() {
				a.MOVUPSLoad(XMM4, At(RSI, 0))
				body()
				a.ADDimm(RSI, 16)
			})
		}
		if cnt4 > 0 {
			bcastSS(&a, XMM14, At(RBX, moeNaNOff))
			bcastD(&a, XMM3, At(RBX, moeOneIOff))
			countLoopSSE(&a, RCX, cnt4, func() {
				a.MOVSSLoad(XMM10, At(RSI, 0))
				// Lane 0 is the element; the three a scalar load zeroed become
				// NaN, which no branch of the predicate admits.
				a.BLENDPS(XMM4, XMM14, XMM10, 0x01)
				body()
				a.ADDimm(RSI, 4)
			})
		}

		// The fold. The per-lane maxima survive in XMM15.
		a.MOVAPS(XMM15, XMM0)
		hmax4SSE(&a, XMM0, XMM6)
		a.CMPPS(XMM5, XMM15, XMM0, 0)
		a.PAND(XMM6, XMM1, XMM5)
		a.PANDN(XMM7, XMM5, XMM13)
		a.POR(XMM6, XMM6, XMM7)
		hmin4SSE(&a, XMM6, XMM4)
		a.MOVAPS(XMM12, XMM6) // the winner's id, and pass i+1's prevIdx
		// Its probability is selected by that id rather than taken from the max
		// fold, which is exact for -0.0 where MAXPS is not (topk.go).
		a.PCMPEQD(XMM5, XMM1, XMM12)
		a.PAND(XMM6, XMM15, XMM5)
		hor4SSE(&a, XMM6, XMM4)
		a.MOVAPS(XMM11, XMM6) // and pass i+1's prevVal
	}

	// ---- PHASE A: the plane the selection walks (topk.go). ----
	plane := R12
	if g.needsBiasedPlane() {
		plane = RDX
		a.MOVQ(RSI, R12)
		a.MOVQ(RDX, R13)
		a.ADDimm(RDX, biOff)
		if g.Bias {
			a.MOVLoad(R8, At(RDI, 128)) // Q2 -> the selection bias
		}
		if nv > 0 {
			countLoopSSE(&a, RAX, nv, func() {
				a.MOVUPSLoad(XMM0, At(RSI, 0))
				if g.Bias {
					a.MOVUPSLoad(XMM1, At(R8, 0))
					a.ADDPS(XMM0, XMM0, XMM1)
					a.ADDimm(R8, 16)
				}
				a.MOVUPSStore(At(RDX, 0), XMM0)
				a.ADDimm(RSI, 16)
				a.ADDimm(RDX, 16)
			})
		}
		if nt > 0 {
			countLoopSSE(&a, RCX, nt, func() {
				a.MOVSSLoad(XMM0, At(RSI, 0))
				if g.Bias {
					a.MOVSSLoad(XMM1, At(R8, 0))
					a.ADDPS(XMM0, XMM0, XMM1)
					a.ADDimm(R8, 4)
				}
				a.MOVSSStore(At(RDX, 0), XMM0)
				a.ADDimm(RSI, 4)
				a.ADDimm(RDX, 4)
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

		a.PXOR(XMM4, XMM4, XMM4)
		for v := 0; v < gvecs; v++ {
			a.MOVDQUStore(At(R13, gsOff+int32(16*v)), XMM4)
		}

		// B1: gs[g] = the sum of the group's top two biased scores.
		//
		// The accumulator lives in gs[g] itself, where the AVX2 kernel
		// keeps it in Y10: this lexPass uses all sixteen XMM. It is the
		// same `0 + m1 + m2` in the same order, so the tiers agree bit for
		// bit.
		a.MOVQ(RDX, R13)
		a.ADDimm(RDX, biOff)
		a.MOVQ(R8, R13)
		a.ADDimm(R8, gsOff)
		countLoopSSE(&a, R9, g.NGroup, func() {
			for t := 0; t < take; t++ {
				lexPass(RDX, sz, t)
				a.MOVSSLoad(XMM0, At(R8, 0))
				a.ADDPS(XMM0, XMM0, XMM11)
				a.MOVSSStore(At(R8, 0), XMM0)
			}
			a.ADDimm(RDX, int32(4*sz))
			a.ADDimm(R8, 4)
		})

		// B2: the best NGroupUsed groups; only the threshold is wanted.
		a.MOVQ(RDX, R13)
		a.ADDimm(RDX, gsOff)
		lexPass(RDX, g.NGroup, 0)
		if g.NGroupUsed > 1 {
			countLoopSSE(&a, R9, g.NGroupUsed-1, func() { lexPass(RDX, g.NGroup, 1) })
		}

		// B3: kept(g) = gs > thrVal || (gs == thrVal && g <= thrIdx).
		a.MOVQ(RSI, R13)
		a.ADDimm(RSI, gsOff)
		a.MOVQ(RDX, R13)
		a.ADDimm(RDX, keepOff)
		a.MOVUPSLoad(XMM2, At(RBX, moeIdxOff))
		bcastD(&a, XMM3, At(RBX, moeStep4Off))
		countLoopSSE(&a, RAX, gvecs, func() {
			a.MOVUPSLoad(XMM4, At(RSI, 0))
			a.CMPPS(XMM5, XMM11, XMM4, 1) // thrVal < gs, i.e. gs > thrVal
			a.CMPPS(XMM6, XMM4, XMM11, 0) // gs == thrVal
			a.PCMPGTD(XMM7, XMM2, XMM12)  // g > thrIdx
			a.PANDN(XMM8, XMM7, XMM6)     // ... and NOT that: g <= thrIdx
			a.POR(XMM5, XMM5, XMM8)
			a.MOVDQUStore(At(RDX, 0), XMM5)
			a.PADDD(XMM2, XMM2, XMM3)
			a.ADDimm(RSI, 16)
			a.ADDimm(RDX, 16)
		})

		// B4: every expert of a group that was not kept goes to -Inf.
		bcastSS(&a, XMM14, At(RBX, moeNegInfOff))
		a.MOVQ(RDX, R13)
		a.ADDimm(RDX, biOff)
		a.MOVQ(R8, R13)
		a.ADDimm(R8, keepOff)
		svec, stail := sz/lanes, sz%lanes
		mask := func(ld, st func()) {
			ld()
			a.PAND(XMM6, XMM4, XMM5)
			a.PANDN(XMM7, XMM5, XMM14)
			a.POR(XMM4, XMM6, XMM7)
			st()
		}
		countLoopSSE(&a, R9, g.NGroup, func() {
			bcastSS(&a, XMM5, At(R8, 0))
			if svec > 0 {
				countLoopSSE(&a, RAX, svec, func() {
					mask(func() { a.MOVUPSLoad(XMM4, At(RDX, 0)) },
						func() { a.MOVUPSStore(At(RDX, 0), XMM4) })
					a.ADDimm(RDX, 16)
				})
			}
			if stail > 0 {
				countLoopSSE(&a, RCX, stail, func() {
					mask(func() { a.MOVSSLoad(XMM4, At(RDX, 0)) },
						func() { a.MOVSSStore(At(RDX, 0), XMM4) })
					a.ADDimm(RDX, 4)
				})
			}
			a.ADDimm(R8, 4)
		})
	}

	if g.needsBiasedPlane() {
		a.MOVLoad(R8, At(RDI, 96))  // ASum     -> sel
		a.MOVLoad(R9, At(RDI, 104)) // AHalfSum -> ord
		a.MOVQ(RDX, R13)
		a.ADDimm(RDX, biOff)
	}

	// ---- PHASE C: the selection. ----
	for i := 0; i < k; i++ {
		lexPass(plane, n, i)
		a.MOVSSStore(At(R8, int32(4*i)), XMM12)
		a.MOVSSStore(At(R13, int32(4*i)), XMM12)
		a.MOVSSStore(At(R13, wtOff+int32(4*i)), XMM11)
	}

	// ---- PHASE D: the unbiased weight of each winner (topk.go). ----
	if g.Bias {
		for i := 0; i < k; i++ {
			moeGatherIDSSE(&a, R12, int32(4*i), nv, nt)
			a.MOVSSStore(At(R13, wtOff+int32(4*i)), XMM10)
		}
	}

	// The divisor, in selection order, one addition at a time.
	if g.Norm {
		a.PXOR(XMM0, XMM0, XMM0)
		for i := 0; i < k; i++ {
			a.MOVSSLoad(XMM4, At(R13, wtOff+int32(4*i)))
			a.ADDPS(XMM0, XMM0, XMM4)
		}
		// The divisor's floor; see topk.go and moeSumFloor.
		a.MOVSSLoad(XMM4, At(RBX, moeSumFloor))
		a.ADDPS(XMM0, XMM0, XMM4)
		a.SHUFPS(XMM0, XMM0, XMM0, 0)
	} else {
		bcastSS(&a, XMM0, At(RBX, moeOneFOff))
	}
	a.MOVSSStore(At(R15, 0), XMM0)
	if g.Scale {
		bcastSS(&a, XMM1, At(RBX, moeScaleOff))
	}
	if g.ExpScale {
		a.MOVLoad(RDX, At(RDI, 136)) // AScale2 -> the per-expert scale; see topk.go
	}
	for i := 0; i < k; i++ {
		if g.ExpScale {
			moeGatherIDSSE(&a, RDX, int32(4*i), nv, nt)
		}
		a.MOVSSLoad(XMM4, At(R13, wtOff+int32(4*i)))
		a.DIVPS(XMM4, XMM4, XMM0)
		if g.Scale {
			a.MULPS(XMM4, XMM4, XMM1) // after the division; see topk.go
		}
		if g.ExpScale {
			a.MULPS(XMM4, XMM4, XMM10) // last; see topk.go
		}
		a.MOVSSStore(At(R10, int32(4*i)), XMM4)
		a.MOVSSStore(At(R13, wtOff+int32(4*i)), XMM4)
	}

	// Ascending expert id, by successive minimum over the pad.
	bcastD(&a, XMM12, At(RBX, moeMinusOne))
	for r := 0; r < k; r++ {
		for v := 0; v < vecs; v++ {
			a.MOVDQULoad(XMM4, At(R13, int32(16*v)))
			a.PCMPGTD(XMM5, XMM4, XMM12)
			a.PAND(XMM6, XMM4, XMM5)
			a.PANDN(XMM7, XMM5, XMM13)
			a.POR(XMM6, XMM6, XMM7)
			if v == 0 {
				a.MOVAPS(XMM9, XMM6)
			} else {
				a.PMINSD(XMM9, XMM9, XMM6)
			}
		}
		hmin4SSE(&a, XMM9, XMM4)
		a.MOVAPS(XMM12, XMM9)
		a.MOVSSStore(At(R9, int32(4*r)), XMM12)
		a.PXOR(XMM10, XMM10, XMM10)
		for v := 0; v < vecs; v++ {
			a.MOVDQULoad(XMM4, At(R13, int32(16*v)))
			a.PCMPEQD(XMM5, XMM4, XMM12)
			a.MOVDQULoad(XMM6, At(R13, wtOff+int32(16*v)))
			a.PAND(XMM6, XMM6, XMM5)
			a.POR(XMM10, XMM10, XMM6)
		}
		hor4SSE(&a, XMM10, XMM4)
		a.MOVSSStore(At(R11, int32(4*r)), XMM10)
	}

	// No VZEROUPPER: it is VEX-encoded and faults on a host with no AVX.
	a.RET()
	return a.Bytes()
}

// hmin4SSE leaves the signed minimum of x's four lanes in every lane of x,
// clobbering t. It must be the integer PMINSD: 0x7FFFFFFF read as a float
// is a NaN, and a float minimum would follow the NaN rule instead.
func hmin4SSE(a *Buf, x, t Reg) {
	a.SHUFPS(t, x, x, 0x4E)
	a.PMINSD(x, x, t)
	a.SHUFPS(t, x, x, 0xB1)
	a.PMINSD(x, x, t)
	a.SHUFPS(x, x, x, 0)
}

// hor4SSE leaves the bitwise OR of x's four lanes in every lane of x,
// clobbering t. Callers mask x so at most one lane is nonzero, making this a
// select; OR rather than add keeps a -0.0's bits.
func hor4SSE(a *Buf, x, t Reg) {
	a.SHUFPS(t, x, x, 0x4E)
	a.POR(x, x, t)
	a.SHUFPS(t, x, x, 0xB1)
	a.POR(x, x, t)
	a.SHUFPS(x, x, x, 0)
}

// countLoopSSE is countLoop (topk.go) for this file, which has no build tag and
// therefore compiles on architectures where that one does not exist.
func countLoopSSE(a *Buf, r Reg, cnt int, body func()) {
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
	a.DEC(r)
	a.JNZ(lp)
}

// EmitMoETopKSSE is the SSE tier's entry in the Emitters table; the body is
// above. A shape it cannot serve is an ordinary error, not ErrNoSSEKernel:
// the kernel exists.
func EmitMoETopKSSE(n, k int, norm bool) ([]byte, error) {
	return EmitMoERouteSSE(n, k, MoEGate{Norm: norm})
}

// EmitMoERouteSSE is the SSE tier's EmitMoERoute: the DeepSeek-V3 router's
// bias, groups and scaling factor, four lanes at a time. topk.go carries the
// contract and the phase-by-phase reasoning.
func EmitMoERouteSSE(n, k int, g MoEGate) ([]byte, error) {
	if err := moeRouteCheck(n, k, g); err != nil {
		return nil, err
	}
	return must("moe_topk_sse", emitMoETopKSSEBody(n, k, g))
}

// moeGatherIDSSE is topk.go's moeGatherID at four lanes: XMM10's lane 0 is
// src[id] for the id at R13+idOff.
func moeGatherIDSSE(a *Buf, src Reg, idOff int32, nv, nt int) {
	bcastSS(a, XMM12, At(R13, idOff)) // the id, as 32 bits
	a.PXOR(XMM10, XMM10, XMM10)
	a.MOVUPSLoad(XMM2, At(RBX, moeIdxOff))
	a.MOVQ(RSI, src)
	if nv > 0 {
		bcastD(a, XMM3, At(RBX, moeStep4Off))
		countLoopSSE(a, RAX, nv, func() {
			a.MOVUPSLoad(XMM4, At(RSI, 0))
			a.PCMPEQD(XMM5, XMM2, XMM12)
			a.PAND(XMM6, XMM4, XMM5)
			a.POR(XMM10, XMM10, XMM6)
			a.PADDD(XMM2, XMM2, XMM3)
			a.ADDimm(RSI, 16)
		})
	}
	if nt > 0 {
		bcastD(a, XMM3, At(RBX, moeOneIOff))
		countLoopSSE(a, RCX, nt, func() {
			a.MOVSSLoad(XMM4, At(RSI, 0))
			a.BLENDPS(XMM7, XMM13, XMM2, 0x01)
			a.PCMPEQD(XMM5, XMM7, XMM12)
			a.PAND(XMM6, XMM4, XMM5)
			a.POR(XMM10, XMM10, XMM6)
			a.PADDD(XMM2, XMM2, XMM3)
			a.ADDimm(RSI, 4)
		})
	}
	hor4SSE(a, XMM10, XMM4)
}
