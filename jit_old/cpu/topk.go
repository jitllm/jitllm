//go:build amd64

package cpu

// EmitMoETopK generates the mixture-of-experts router's selection,
// renormalisation and ordering as one kernel. This is the AVX2 tier;
// sse_topk.go and topk_a64.go are the other two, and topk_const.go holds the
// constant block and the scratch layout all three read.
//
//	Q32      p, the softmax over ALL experts   (f32, n)
//	Scr      MoETopKConsts()
//	Scratch  MoETopKScratch(k) float32s, per caller
//	ASum     sel, the ids in SELECTION order   (int32, k, written)
//	AHalfSum ord, the same ids ASCENDING       (int32, k, written)
//	Out      wt,  sel's probabilities / sum    (f32, k, written)
//	Out2     ow,  ord's probabilities / sum    (f32, k, written)
//	AScale   sum, the divisor                  (f32, 1, written)
//
// n, k and whether the architecture renormalises are baked. Ties go to the
// lowest index, as ggml_argsort's descending order does. `sum` accumulates in
// selection order and `ord` is ascending, because Forward, Prefill and
// ForwardBatch are gated bit-identical on a mixture and moeBatch visits the
// bank expert-major. Selection is a lexicographic walk over the keys
// (p[e], -e), eligible(e) = p[e] < prevVal || (p[e] == prevVal && e > prevIdx),
// and the ragged tail runs the vector body over NaN pads, which no pass takes.
//
// Why a walk, the NaN and -Inf cases, and the tail: docs/engineering-history/
// cpu-kernels.md, "jit_old/cpu/topk.go: EmitMoETopK".
func EmitMoETopK(n, k int, norm bool) ([]byte, error) {
	return EmitMoERoute(n, k, MoEGate{Norm: norm})
}

// EmitMoERoute is that kernel generalised to the DeepSeek-V3 router, which R1,
// Kimi-K2 and GLM-4-MoE share. MoEGate (topk_const.go) is the configuration.
//
//	Q32      the post-gating scores of ALL experts  (f32, n)   sigmoid or softmax
//	Q2       e_score_correction_bias                (f32, n)   read iff g.Bias
//	Scr      MoETopKConstsFor(routed_scaling_factor)
//	Scratch  MoERouteScratch(n, k, g) float32s, per caller
//	...      ASum/AHalfSum/Out/Out2/AScale exactly as above
//
// The bias moves the selection and never the weight: the walk maximises over
// `scores + bias`, so phase D goes back to the unbiased plane and gathers each
// winner's score by id.
//
// "The best two of a group", "the best topk_group of the groups" and "the
// best k of the experts" are the same lexicographic walk at three shapes, so
// lexPass is emitted for all of them.
//
// The kept groups are found from a threshold rather than a set, which would
// need a data-dependent store address. After n_group_used passes the last
// winner is the threshold, and
//
//	kept(g) = gs[g] > thrVal || (gs[g] == thrVal && g <= thrIdx)
//
// is the exact complement of the eligibility predicate.
func EmitMoERoute(n, k int, g MoEGate) ([]byte, error) {
	if err := moeRouteCheck(n, k, g); err != nil {
		return nil, err
	}
	const lanes = 8
	nv, nt := n/lanes, n%lanes
	pl := moePlanes(n, k, g)
	pad := pl.pad
	wtOff := int32(4 * pl.wt) // the weight pad's byte offset inside the scratch
	biOff := int32(4 * pl.bi)
	gsOff := int32(4 * pl.gs)
	keepOff := int32(4 * pl.keep)
	vecs := pad / lanes

	var a Buf
	a.MOVLoad(RBX, At(RDI, 56))  // Scr      -> the constants
	a.MOVLoad(R12, At(RDI, 112)) // Q32      -> p
	a.MOVLoad(R13, At(RDI, 72))  // Scratch
	a.MOVLoad(R8, At(RDI, 96))   // ASum     -> sel
	a.MOVLoad(R9, At(RDI, 104))  // AHalfSum -> ord
	a.MOVLoad(R10, At(RDI, 0))   // Out      -> wt
	a.MOVLoad(R11, At(RDI, 120)) // Out2     -> ow
	a.MOVLoad(R15, At(RDI, 24))  // AScale   -> sum

	// Y13 holds INT_MAX for the whole kernel: it is the empty-lane sentinel,
	// the non-tying lane's contribution to the index fold, and the filler the
	// scratch's pad lanes carry.
	a.VPBROADCASTD(Y13, At(RBX, moeMaxIOff))
	for v := 0; v < vecs; v++ {
		a.VMOVDQUStore(At(R13, int32(32*v)), Y13)
	}
	// The weight pad is zeroed so the output is a function of the inputs alone;
	// nothing reads those lanes, but the scratch is reused across calls.
	a.VPXOR(Y4, Y4, Y4)
	for v := 0; v < vecs; v++ {
		a.VMOVDQUStore(At(R13, wtOff+int32(32*v)), Y4)
	}

	// One pass of the lexicographic walk over cnt elements from base, for pass
	// index i. It leaves Y11 = the winner's value and Y12 = its id, both
	// broadcast, which are also pass i+1's prevVal and prevIdx.
	lexPass := func(base Reg, cnt, i int) {
		cnv, cnt8 := cnt/lanes, cnt%lanes
		a.VBROADCASTSS(Y0, At(RBX, moeNegInfOff)) // the running maximum
		a.VMOVAPSReg(Y1, Y13)                     // its index, INT_MAX = empty
		a.VMOVDQULoad(Y2, At(RBX, moeIdxOff))     // the element index, 0..7
		a.MOVQ(RSI, base)

		body := func() {
			if i == 0 {
				// Ordered equality with itself: every element but a NaN.
				a.VCMPPS(Y5, Y4, Y4, 0)
			} else {
				a.VCMPPS(Y5, Y4, Y11, 1) // _CMP_LT_OS: p < prevVal
				a.VCMPPS(Y6, Y4, Y11, 0) // _CMP_EQ_OQ: p == prevVal
				a.VPCMPGTD(Y7, Y2, Y12)  // and a HIGHER id than the tie it lost
				a.VPAND(Y6, Y6, Y7)
				a.VPOR(Y5, Y5, Y6)
			}
			a.VCMPPS(Y8, Y4, Y0, 14) // _CMP_GT_OS: strictly, so a tie keeps the
			a.VPCMPEQD(Y9, Y1, Y13)  // earlier index -- unless the lane is empty
			a.VPOR(Y8, Y8, Y9)
			a.VPAND(Y5, Y5, Y8)
			a.VPAND(Y6, Y4, Y5)  // there is no blend in this assembler, so the
			a.VPANDN(Y7, Y5, Y0) // mask's and/andnot/or is the blend
			a.VPOR(Y0, Y6, Y7)
			a.VPAND(Y6, Y2, Y5)
			a.VPANDN(Y7, Y5, Y1)
			a.VPOR(Y1, Y6, Y7)
			a.VPADDD(Y2, Y2, Y3)
		}

		if cnv > 0 {
			a.VPBROADCASTD(Y3, At(RBX, moeStep8Off))
			countLoop(&a, RAX, cnv, func() {
				a.VMOVDQULoad(Y4, At(RSI, 0))
				body()
				a.ADDimm(RSI, 32)
			})
		}
		if cnt8 > 0 {
			a.VBROADCASTSS(Y14, At(RBX, moeNaNOff))
			a.VPBROADCASTD(Y3, At(RBX, moeOneIOff))
			countLoop(&a, RCX, cnt8, func() {
				a.VMOVSSLoad(Y4, At(RSI, 0))
				// Lane 0 is the element; the seven a scalar load zeroed become
				// NaN, which no branch of the predicate admits. Y2's lanes 1..7
				// are meaningless from here and nothing reads them.
				a.VPBLENDD(Y4, Y14, Y4, 0x01)
				body()
				a.ADDimm(RSI, 4)
			})
		}

		// The fold. The PER-LANE maxima must survive it, because the winner's
		// index is found by asking which lanes EQUAL the overall maximum.
		a.VMOVAPSReg(Y15, Y0)
		a.VEXTRACTF128(Y4, Y0, 1)
		a.VMAXPS(Y0, Y0, Y4)
		a.VSHUFPS(Y4, Y0, Y0, 0x4E)
		a.VMAXPS(Y0, Y0, Y4)
		a.VSHUFPS(Y4, Y0, Y0, 0xB1)
		a.VMAXPS(Y0, Y0, Y4)
		a.VBROADCASTSSReg(Y0, Y0)
		// Lanes holding it keep their index; the rest contribute INT_MAX, so the
		// signed minimum over the lanes is the lowest index that ties. A lane
		// that never improved already carries INT_MAX and cannot win.
		a.VCMPPS(Y5, Y15, Y0, 0)
		a.VPAND(Y6, Y1, Y5)
		a.VPANDN(Y7, Y5, Y13)
		a.VPOR(Y6, Y6, Y7)
		a.VEXTRACTI128(Y4, Y6, 1)
		a.VPMINSD(Y6, Y6, Y4)
		a.VSHUFPS(Y4, Y6, Y6, 0x4E)
		a.VPMINSD(Y6, Y6, Y4)
		a.VSHUFPS(Y4, Y6, Y6, 0xB1)
		a.VPMINSD(Y6, Y6, Y4)
		a.VBROADCASTSSReg(Y12, Y6) // the winner's id, and pass i+1's prevIdx

		// The winner's value is selected by its index, not taken from the max
		// fold, because of -0.0: VMAXPS returns its second operand when both are
		// zeros, so the fold can differ from p[winner] in its bits. Per-lane best
		// indices are unique, so exactly one lane matches and the OR fold is its
		// bits.
		a.VPCMPEQD(Y5, Y1, Y12)
		a.VPAND(Y6, Y15, Y5)
		a.VEXTRACTI128(Y4, Y6, 1)
		a.VPOR(Y6, Y6, Y4)
		a.VSHUFPS(Y4, Y6, Y6, 0x4E)
		a.VPOR(Y6, Y6, Y4)
		a.VSHUFPS(Y4, Y6, Y6, 0xB1)
		a.VPOR(Y6, Y6, Y4)
		a.VBROADCASTSSReg(Y11, Y6) // and pass i+1's prevVal
	}

	// ---- PHASE A: the plane the selection walks. ----
	//
	// The caller's scores, or a scratch copy of `scores + bias` when a bias is
	// present or a group has to be masked out.
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
			countLoop(&a, RAX, nv, func() {
				a.VMOVDQULoad(Y0, At(RSI, 0))
				if g.Bias {
					a.VMOVDQULoad(Y1, At(R8, 0))
					a.VADDPS(Y0, Y0, Y1)
					a.ADDimm(R8, 32)
				}
				a.VMOVDQUStore(At(RDX, 0), Y0)
				a.ADDimm(RSI, 32)
				a.ADDimm(RDX, 32)
			})
		}
		if nt > 0 {
			countLoop(&a, RCX, nt, func() {
				a.VMOVSSLoad(Y0, At(RSI, 0))
				if g.Bias {
					a.VMOVSSLoad(Y1, At(R8, 0))
					a.VADDPS(Y0, Y0, Y1)
					a.ADDimm(R8, 4)
				}
				a.VMOVSSStore(At(RDX, 0), Y0)
				a.ADDimm(RSI, 4)
				a.ADDimm(RDX, 4)
			})
		}
	}

	// ---- PHASE B: the expert groups. ----
	if g.Grouped() {
		sz := n / g.NGroup
		take := 2 // llama.cpp's ggml_argsort_top_k(selection_groups, 2)
		if sz < take {
			// A group of one has no second element. Neither reference
			// defines this and no shipped config reaches it; take what
			// there is.
			take = sz
		}
		gvecs := pl.gPad / lanes

		// B0: the group-score plane is zeroed whole, including the lanes past
		// NGroup. B3 reads it by vectors and a scratch buffer is reused across
		// calls, so those lanes have to be a function of this call.
		a.VPXOR(Y4, Y4, Y4)
		for v := 0; v < gvecs; v++ {
			a.VMOVDQUStore(At(R13, gsOff+int32(32*v)), Y4)
		}

		// B1: gs[g] = the sum of the group's top two biased scores, one group
		// per loop iteration (inline copies would be ~880 instructions on V3).
		a.MOVQ(RDX, R13)
		a.ADDimm(RDX, biOff)
		a.MOVQ(R8, R13)
		a.ADDimm(R8, gsOff)
		countLoop(&a, R9, g.NGroup, func() {
			a.VPXOR(Y10, Y10, Y10) // ggml_sum_rows accumulates from zero
			for t := 0; t < take; t++ {
				lexPass(RDX, sz, t)
				a.VADDPS(Y10, Y10, Y11)
			}
			a.VMOVSSStore(At(R8, 0), Y10)
			a.ADDimm(RDX, int32(4*sz))
			a.ADDimm(R8, 4)
		})

		// B2: the best NGroupUsed groups. Only the THRESHOLD is wanted, so the
		// passes after the first are identical code and run as a loop.
		a.MOVQ(RDX, R13)
		a.ADDimm(RDX, gsOff)
		lexPass(RDX, g.NGroup, 0)
		if g.NGroupUsed > 1 {
			countLoop(&a, R9, g.NGroupUsed-1, func() { lexPass(RDX, g.NGroup, 1) })
		}

		// B3: kept(g), as an all-ones / all-zeros mask per group.
		a.MOVQ(RSI, R13)
		a.ADDimm(RSI, gsOff)
		a.MOVQ(RDX, R13)
		a.ADDimm(RDX, keepOff)
		a.VMOVDQULoad(Y2, At(RBX, moeIdxOff))
		a.VPBROADCASTD(Y3, At(RBX, moeStep8Off))
		countLoop(&a, RAX, gvecs, func() {
			a.VMOVDQULoad(Y4, At(RSI, 0))
			a.VCMPPS(Y5, Y4, Y11, 14) // gs > thrVal
			a.VCMPPS(Y6, Y4, Y11, 0)  // gs == thrVal
			a.VPCMPGTD(Y7, Y2, Y12)   // g > thrIdx
			a.VPANDN(Y8, Y7, Y6)      // ... and NOT that: g <= thrIdx
			a.VPOR(Y5, Y5, Y8)
			a.VMOVDQUStore(At(RDX, 0), Y5)
			a.VPADDD(Y2, Y2, Y3)
			a.ADDimm(RSI, 32)
			a.ADDimm(RDX, 32)
		})

		// B4: every expert of a group that was not kept goes to -Inf, which is
		// exactly ggml_fill(-INFINITY) + ggml_set_rows of the kept ones.
		a.VBROADCASTSS(Y14, At(RBX, moeNegInfOff))
		a.MOVQ(RDX, R13)
		a.ADDimm(RDX, biOff)
		a.MOVQ(R8, R13)
		a.ADDimm(R8, keepOff)
		svec, stail := sz/lanes, sz%lanes
		mask := func(ld, st func()) {
			ld()
			a.VPAND(Y6, Y4, Y5)
			a.VPANDN(Y7, Y5, Y14)
			a.VPOR(Y4, Y6, Y7)
			st()
		}
		countLoop(&a, R9, g.NGroup, func() {
			a.VBROADCASTSS(Y5, At(R8, 0))
			if svec > 0 {
				countLoop(&a, RAX, svec, func() {
					mask(func() { a.VMOVDQULoad(Y4, At(RDX, 0)) },
						func() { a.VMOVDQUStore(At(RDX, 0), Y4) })
					a.ADDimm(RDX, 32)
				})
			}
			if stail > 0 {
				countLoop(&a, RCX, stail, func() {
					mask(func() { a.VMOVSSLoad(Y4, At(RDX, 0)) },
						func() { a.VMOVSSStore(At(RDX, 0), Y4) })
					a.ADDimm(RDX, 4)
				})
			}
			a.ADDimm(R8, 4)
		})
	}

	// Phases A and B borrow R8 and R9; the selection needs them back. Emitted
	// only when a phase ran, so the plain router's bytes do not move.
	if g.needsBiasedPlane() {
		a.MOVLoad(R8, At(RDI, 96))  // ASum     -> sel
		a.MOVLoad(R9, At(RDI, 104)) // AHalfSum -> ord
		a.MOVQ(RDX, R13)
		a.ADDimm(RDX, biOff)
	}

	// ---- PHASE C: the selection. ----
	for i := 0; i < k; i++ {
		lexPass(plane, n, i)
		a.VMOVSSStore(At(R8, int32(4*i)), Y12)        // sel[i]
		a.VMOVSSStore(At(R13, int32(4*i)), Y12)       // the id pad
		a.VMOVSSStore(At(R13, wtOff+int32(4*i)), Y11) // the weight pad, raw
	}

	// ---- PHASE D: the unbiased weight of each winner. ----
	//
	// Re-read from the scores by id: `(s+b)-b` is not s in f32, and llama.cpp
	// gathers from the unbiased probs.
	if g.Bias {
		for i := 0; i < k; i++ {
			moeGatherID(&a, R12, int32(4*i), nv, nt)
			a.VMOVSSStore(At(R13, wtOff+int32(4*i)), Y10)
		}
	}

	// The divisor, in selection order, one addition at a time (see above).
	if g.Norm {
		a.VPXOR(Y0, Y0, Y0)
		for i := 0; i < k; i++ {
			a.VMOVSSLoad(Y4, At(R13, wtOff+int32(4*i)))
			a.VADDPS(Y0, Y0, Y4)
		}
		// The divisor's floor, DeepSeek's `sum + 1e-20` rather than llama.cpp's
		// clamp: an all-underflowed selection would otherwise divide 0 by 0. See
		// oracle.MoEDivergences and the comment at moeSumFloor.
		a.VMOVSSLoad(Y4, At(RBX, moeSumFloor))
		a.VADDPS(Y0, Y0, Y4)
		a.VBROADCASTSSReg(Y0, Y0)
	} else {
		a.VBROADCASTSS(Y0, At(RBX, moeOneFOff))
	}
	a.VMOVSSStore(At(R15, 0), Y0)
	// AScale is the divisor, not the scaling factor; the trace and the batched
	// path's sum are checked against it.
	if g.Scale {
		a.VBROADCASTSS(Y1, At(RBX, moeScaleOff))
	}
	if g.ExpScale {
		// RDX held the biased plane, which phase C was the last to read.
		a.MOVLoad(RDX, At(RDI, 136)) // AScale2 -> the per-expert scale
	}
	for i := 0; i < k; i++ {
		if g.ExpScale {
			// The selected expert's own scale, gathered by id as phase D
			// gathers its score; Y10 holds it in lane 0.
			moeGatherID(&a, RDX, int32(4*i), nv, nt)
		}
		a.VMOVSSLoad(Y4, At(R13, wtOff+int32(4*i)))
		a.VDIVPS(Y4, Y4, Y0)
		if g.Scale {
			// After the division, as ggml_div then ggml_scale and HF's
			// `/ denominator` then `* routed_scaling_factor`; the other order
			// is a different float32.
			a.VMULPS(Y4, Y4, Y1)
		}
		if g.ExpScale {
			// Last, as Gemma 4's router multiplies the renormalised weights by
			// per_expert_scale.
			a.VMULPS(Y4, Y4, Y10)
		}
		a.VMOVSSStore(At(R10, int32(4*i)), Y4)       // wt[i]
		a.VMOVSSStore(At(R13, wtOff+int32(4*i)), Y4) // and back into the pad
	}

	// Ascending expert id by successive minimum, which needs no data-dependent
	// store address: pass r takes the smallest id strictly above pass r-1's.
	// The pad's INT_MAX lanes are above every real id.
	a.VPBROADCASTD(Y12, At(RBX, moeMinusOne))
	for r := 0; r < k; r++ {
		for v := 0; v < vecs; v++ {
			a.VMOVDQULoad(Y4, At(R13, int32(32*v)))
			a.VPCMPGTD(Y5, Y4, Y12)
			a.VPAND(Y6, Y4, Y5)
			a.VPANDN(Y7, Y5, Y13)
			a.VPOR(Y6, Y6, Y7)
			if v == 0 {
				a.VMOVAPSReg(Y9, Y6)
			} else {
				a.VPMINSD(Y9, Y9, Y6)
			}
		}
		a.VEXTRACTI128(Y4, Y9, 1)
		a.VPMINSD(Y9, Y9, Y4)
		a.VSHUFPS(Y4, Y9, Y9, 0x4E)
		a.VPMINSD(Y9, Y9, Y4)
		a.VSHUFPS(Y4, Y9, Y9, 0xB1)
		a.VPMINSD(Y9, Y9, Y4)
		a.VBROADCASTSSReg(Y12, Y9)
		a.VMOVSSStore(At(R9, int32(4*r)), Y12) // ord[r]
		// Its weight is the pad lane whose id matches. The ids are unique, so
		// one lane survives the mask and the OR fold is that lane exactly.
		a.VPXOR(Y10, Y10, Y10)
		for v := 0; v < vecs; v++ {
			a.VMOVDQULoad(Y4, At(R13, int32(32*v)))
			a.VPCMPEQD(Y5, Y4, Y12)
			a.VMOVDQULoad(Y6, At(R13, wtOff+int32(32*v)))
			a.VPAND(Y6, Y6, Y5)
			a.VPOR(Y10, Y10, Y6)
		}
		a.VEXTRACTI128(Y4, Y10, 1)
		a.VPOR(Y10, Y10, Y4)
		a.VSHUFPS(Y4, Y10, Y10, 0x4E)
		a.VPOR(Y10, Y10, Y4)
		a.VSHUFPS(Y4, Y10, Y10, 0xB1)
		a.VPOR(Y10, Y10, Y4)
		a.VMOVSSStore(At(R11, int32(4*r)), Y10) // ow[r]
	}

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// moeGatherID leaves in Y10's lane 0 the float at src[id], where id is the
// int32 at R13+idOff (the id pad), by a compare-and-OR walk over all n lanes
// -- no data-dependent address. Exactly one lane matches, so the OR fold is
// that lane's bits, and an add would cancel a -0.0 against the zeros.
// Clobbers Y2-Y7, Y12, RSI, RAX and RCX; reads Y13 (INT_MAX) and RBX.
func moeGatherID(a *Buf, src Reg, idOff int32, nv, nt int) {
	a.VBROADCASTSS(Y12, At(R13, idOff)) // the id, as 32 bits
	a.VPXOR(Y10, Y10, Y10)
	a.VMOVDQULoad(Y2, At(RBX, moeIdxOff))
	a.MOVQ(RSI, src)
	if nv > 0 {
		a.VPBROADCASTD(Y3, At(RBX, moeStep8Off))
		countLoop(a, RAX, nv, func() {
			a.VMOVDQULoad(Y4, At(RSI, 0))
			a.VPCMPEQD(Y5, Y2, Y12)
			a.VPAND(Y6, Y4, Y5)
			a.VPOR(Y10, Y10, Y6)
			a.VPADDD(Y2, Y2, Y3)
			a.ADDimm(RSI, 32)
		})
	}
	if nt > 0 {
		// The tail guards the index here, where the selection guards the
		// value: lanes 1..7 hold real expert ids after a scalar load, so they
		// are set to INT_MAX, which matches no id.
		a.VPBROADCASTD(Y3, At(RBX, moeOneIOff))
		countLoop(a, RCX, nt, func() {
			a.VMOVSSLoad(Y4, At(RSI, 0))
			a.VPBLENDD(Y7, Y13, Y2, 0x01)
			a.VPCMPEQD(Y5, Y7, Y12)
			a.VPAND(Y6, Y4, Y5)
			a.VPOR(Y10, Y10, Y6)
			a.VPADDD(Y2, Y2, Y3)
			a.ADDimm(RSI, 4)
		})
	}
	a.VEXTRACTI128(Y4, Y10, 1)
	a.VPOR(Y10, Y10, Y4)
	a.VSHUFPS(Y4, Y10, Y10, 0x4E)
	a.VPOR(Y10, Y10, Y4)
	a.VSHUFPS(Y4, Y10, Y10, 0xB1)
	a.VPOR(Y10, Y10, Y4)
}

// countLoop emits body exactly cnt times: inline for one or two iterations, a
// counted loop beyond that. cnt is known at emit time, so no zero test is
// needed.
func countLoop(a *Buf, r Reg, cnt int, body func()) {
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
