//go:build arm64

package cpu

import "fmt"

// The arm64 attention kernels, the NEON twins of jit_old/cpu/attn.go. Both are
// pure float32; a NEON vector is four lanes against AVX2's eight, so every
// loop runs twice the iterations over the same head. hd and kvStride are baked.

// EmitAttnScores generates scores[t] = dot(q, K[t]) for t in [0, Rows).
//
// It reads Out (scores), W (the K cache), Q32 (the query) and Rows. The last
// hd%4 dimensions (hd%8 on amd64) are baked scalar code, so every head
// dimension runs generated on both architectures.
func EmitAttnScores(hd, kvStride int, kv KVFmt) ([]byte, error) {
	ka, err := newKVAddr(kv, hd, kvStride)
	if err != nil {
		return nil, err
	}
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnScores: hd=%d must be positive", hd)
	}
	nv := hd / 4
	var a A64
	a.LDRx(X1, X0, 0)   // Out: scores
	a.LDRx(X2, X0, 8)   // W:   K cache row 0
	a.LDRx(X3, X0, 32)  // Rows: positions
	a.LDRx(X4, X0, 112) // Q32: query

	row := a.Label()
	a.Bind(row)
	// Four accumulator chains: one chain is latency-bound on the FMLA.
	for i := 0; i < 4; i++ {
		a.MOVIzero(VReg(i))
	}
	// The query is loaded per vector rather than held: it is L1-hot, and
	// holding it would cap hd at the register file (hd=256 is sixty-four NEON
	// vectors).
	for i := 0; i < nv; i++ {
		a.kvLoad(V4, X2, ka, i*4, V6)
		a.LDRq(V5, X4, int32(i)*16)
		a.FMLA4s(VReg(i%4), V4, V5)
	}
	// The last hd%4 dimensions into chain 0, one scalar lane each.
	for d := int32(nv * 4); d < int32(hd); d++ {
		a.kvLoad1(V4, X2, ka, int(d), V6)
		a.LDRs(V5, X4, 4*d)
		a.FMLA4s(V0, V4, V5)
	}
	a.FADD4s(V0, V0, V1)
	a.FADD4s(V2, V2, V3)
	a.FADD4s(V0, V0, V2)
	// Two pairwise adds reduce four lanes to one; arm64 has no HADDPS.
	a.FADDP4s(V0, V0, V0)
	a.FADDP4s(V0, V0, V0)
	a.STRs(V0, X1, 0)
	a.ADDimm(X1, X1, 4)
	a.addBig(X2, ka.stride(kvStride), X9)
	a.SUBimm(X3, X3, 1)
	a.CBNZ(X3, row)

	a.RET()
	return a.Bytes(), nil
}

// EmitAttnAcc generates out[i] = sum over t of att[t] * V[t][i].
//
// It reads Out, W (the V cache), AScale (the softmaxed weights) and Rows.
//
// The output is held in registers across the whole position loop, in column
// blocks; a head wider than one block re-reads V once per block. The block is
// 16 vectors here against amd64's 8, the same 64 floats. Wider is unmeasured.
func EmitAttnAcc(hd, kvStride int, kv KVFmt) ([]byte, error) {
	return emitAttnAcc(hd, kvStride, kv, false)
}

// EmitAttnAccInto is EmitAttnAcc that ADDS INTO Out. See the amd64 twin in
// attn.go for why the bits come out identical to one contiguous call.
func EmitAttnAccInto(hd, kvStride int, kv KVFmt) ([]byte, error) {
	return emitAttnAcc(hd, kvStride, kv, true)
}

func emitAttnAcc(hd, kvStride int, kv KVFmt, acc bool) ([]byte, error) {
	ka, err := newKVAddr(kv, hd, kvStride)
	if err != nil {
		return nil, err
	}
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnAcc: hd=%d must be positive", hd)
	}
	const perBlock = 16 // accumulator vectors held at once; 64 floats
	var a A64
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 8)  // W: V cache
	a.LDRx(X3, X0, 24) // AScale: attention weights
	a.LDRx(X4, X0, 32) // Rows

	for base := 0; base < hd/4; base += perBlock {
		nv := min(perBlock, hd/4-base)
		for v := 0; v < nv; v++ {
			if acc {
				a.LDRq(VReg(v), X1, int32(base+v)*16)
			} else {
				a.MOVIzero(VReg(v))
			}
		}
		a.MOVreg(X5, X2) // V cursor for this block
		a.MOVreg(X6, X3) // weight cursor
		a.MOVreg(X7, X4) // position counter
		pos := a.Label()
		a.Bind(pos)
		// att[t] into lane 0, then FMLA by element -- no broadcast needed, which
		// is where this differs from amd64's VBROADCASTSS.
		a.LDRs(V16, X6, 0)
		for v := 0; v < nv; v++ {
			a.kvLoad(V17, X5, ka, (base+v)*4, V18)
			a.FMLAelem(VReg(v), V17, V16, 0)
		}
		a.addBig(X5, ka.stride(kvStride), X9)
		a.ADDimm(X6, X6, 4)
		a.SUBimm(X7, X7, 1)
		a.CBNZ(X7, pos)
		for v := 0; v < nv; v++ {
			a.STRq(VReg(v), X1, int32(base+v)*16)
		}
	}
	// The last hd%4 dimensions: one lane-0 accumulator each, positions in the
	// same order, so a paged window keeps one contiguous call's bits.
	if tail := hd % 4; tail > 0 {
		d0 := int32(hd - tail)
		for j := 0; j < tail; j++ {
			if acc {
				a.LDRs(VReg(j), X1, 4*(d0+int32(j)))
			} else {
				a.MOVIzero(VReg(j))
			}
		}
		a.MOVreg(X5, X2)
		a.MOVreg(X6, X3)
		a.MOVreg(X7, X4)
		pos := a.Label()
		a.Bind(pos)
		a.LDRs(V16, X6, 0)
		for j := 0; j < tail; j++ {
			a.kvLoad1(V17, X5, ka, int(d0)+j, V18)
			a.FMLAelem(VReg(j), V17, V16, 0)
		}
		a.addBig(X5, ka.stride(kvStride), X9)
		a.ADDimm(X6, X6, 4)
		a.SUBimm(X7, X7, 1)
		a.CBNZ(X7, pos)
		for j := 0; j < tail; j++ {
			a.STRs(VReg(j), X1, 4*(d0+int32(j)))
		}
	}
	a.RET()
	return a.Bytes(), nil
}

// EmitAttnScores2 is EmitAttnScores for two query heads at once: it loads each
// K vector once and dots it against both queries (see the amd64 twin). With 32
// vector registers the amd64 budget does not bind; wider is unmeasured.
func EmitAttnScores2(hd, kvStride int, kv KVFmt) ([]byte, error) {
	ka, err := newKVAddr(kv, hd, kvStride)
	if err != nil {
		return nil, err
	}
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnScores2: hd=%d must be positive", hd)
	}
	nv := hd / 4
	var a A64
	a.LDRx(X1, X0, 0)   // Out:  scores, head 0
	a.LDRx(X2, X0, 8)   // W:    K cache row 0
	a.LDRx(X3, X0, 32)  // Rows: positions
	a.LDRx(X4, X0, 112) // Q32:  query 0
	a.LDRx(X5, X0, 120) // Out2: scores, head 1
	a.LDRx(X6, X0, 128) // Q2:   query 1

	row := a.Label()
	a.Bind(row)
	// V0-V3 accumulate head 0 and V8-V11 head 1, four chains each for the FMLA
	// latency reason EmitAttnScores records.
	for i := 0; i < 4; i++ {
		a.MOVIzero(VReg(i))
		a.MOVIzero(VReg(8 + i))
	}
	for i := 0; i < nv; i++ {
		// One K load, two FMLAs.
		a.kvLoad(V4, X2, ka, i*4, V7)
		a.LDRq(V5, X4, int32(i)*16)
		a.FMLA4s(VReg(i%4), V4, V5)
		a.LDRq(V6, X6, int32(i)*16)
		a.FMLA4s(VReg(8+i%4), V4, V6)
	}
	// The single-head tail, per head, so the pair stays bit-identical.
	for d := int32(nv * 4); d < int32(hd); d++ {
		a.kvLoad1(V4, X2, ka, int(d), V7)
		a.LDRs(V5, X4, 4*d)
		a.FMLA4s(V0, V4, V5)
		a.LDRs(V6, X6, 4*d)
		a.FMLA4s(V8, V4, V6)
	}
	// The single-head reduction, twice, in the same order: the paired kernel
	// must be BIT-IDENTICAL to two single-head calls.
	a.FADD4s(V0, V0, V1)
	a.FADD4s(V2, V2, V3)
	a.FADD4s(V0, V0, V2)
	a.FADDP4s(V0, V0, V0)
	a.FADDP4s(V0, V0, V0)
	a.STRs(V0, X1, 0)

	a.FADD4s(V8, V8, V9)
	a.FADD4s(V10, V10, V11)
	a.FADD4s(V8, V8, V10)
	a.FADDP4s(V8, V8, V8)
	a.FADDP4s(V8, V8, V8)
	a.STRs(V8, X5, 0)

	a.ADDimm(X1, X1, 4)
	a.ADDimm(X5, X5, 4)
	a.addBig(X2, ka.stride(kvStride), X9)
	a.SUBimm(X3, X3, 1)
	a.CBNZ(X3, row)

	a.RET()
	return a.Bytes(), nil
}

// EmitAttnAcc2 is EmitAttnAcc for two heads sharing one kv head: one V load,
// two weighted accumulations. Thirty-two registers hold both heads' sixteen
// vectors plus V and two weights, so unlike amd64 this keeps the single-head
// block width and needs no second walk.
func EmitAttnAcc2(hd, kvStride int, kv KVFmt) ([]byte, error) {
	return emitAttnAcc2(hd, kvStride, kv, false)
}

// EmitAttnAcc2Into is EmitAttnAcc2 that ADDS INTO both outputs. See the amd64
// twin in attn.go.
func EmitAttnAcc2Into(hd, kvStride int, kv KVFmt) ([]byte, error) {
	return emitAttnAcc2(hd, kvStride, kv, true)
}

func emitAttnAcc2(hd, kvStride int, kv KVFmt, acc bool) ([]byte, error) {
	ka, err := newKVAddr(kv, hd, kvStride)
	if err != nil {
		return nil, err
	}
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnAcc2: hd=%d must be positive", hd)
	}
	const perBlock = 8 // per head; two heads is 16 of 32 vectors
	var a A64
	a.LDRx(X1, X0, 0)    // Out
	a.LDRx(X2, X0, 8)    // W: V cache
	a.LDRx(X3, X0, 24)   // AScale
	a.LDRx(X4, X0, 32)   // Rows
	a.LDRx(X10, X0, 120) // Out2
	a.LDRx(X11, X0, 136) // AScale2

	for base := 0; base < hd/4; base += perBlock {
		nv := min(perBlock, hd/4-base)
		for v := 0; v < nv; v++ {
			if acc {
				a.LDRq(VReg(v), X1, int32(base+v)*16)
				a.LDRq(VReg(16+v), X10, int32(base+v)*16)
			} else {
				a.MOVIzero(VReg(v))      // head 0: V0..
				a.MOVIzero(VReg(16 + v)) // head 1: V16..
			}
		}
		a.MOVreg(X5, X2)   // V cursor
		a.MOVreg(X6, X3)   // weight cursor, head 0
		a.MOVreg(X12, X11) // weight cursor, head 1
		a.MOVreg(X7, X4)
		pos := a.Label()
		a.Bind(pos)
		a.LDRs(V14, X6, 0)
		a.LDRs(V15, X12, 0)
		for v := 0; v < nv; v++ {
			// One V load, two FMLAs.
			a.kvLoad(V13, X5, ka, (base+v)*4, V31)
			a.FMLAelem(VReg(v), V13, V14, 0)
			a.FMLAelem(VReg(16+v), V13, V15, 0)
		}
		a.addBig(X5, ka.stride(kvStride), X9)
		a.ADDimm(X6, X6, 4)
		a.ADDimm(X12, X12, 4)
		a.SUBimm(X7, X7, 1)
		a.CBNZ(X7, pos)
		for v := 0; v < nv; v++ {
			a.STRq(VReg(v), X1, int32(base+v)*16)
			a.STRq(VReg(16+v), X10, int32(base+v)*16)
		}
	}
	if tail := hd % 4; tail > 0 {
		d0 := int32(hd - tail)
		for j := 0; j < tail; j++ {
			d := 4 * (d0 + int32(j))
			if acc {
				a.LDRs(VReg(j), X1, d)
				a.LDRs(VReg(16+j), X10, d)
			} else {
				a.MOVIzero(VReg(j))
				a.MOVIzero(VReg(16 + j))
			}
		}
		a.MOVreg(X5, X2)
		a.MOVreg(X6, X3)
		a.MOVreg(X12, X11)
		a.MOVreg(X7, X4)
		pos := a.Label()
		a.Bind(pos)
		a.LDRs(V14, X6, 0)
		a.LDRs(V15, X12, 0)
		for j := 0; j < tail; j++ {
			a.kvLoad1(V13, X5, ka, int(d0)+j, V31)
			a.FMLAelem(VReg(j), V13, V14, 0)
			a.FMLAelem(VReg(16+j), V13, V15, 0)
		}
		a.addBig(X5, ka.stride(kvStride), X9)
		a.ADDimm(X6, X6, 4)
		a.ADDimm(X12, X12, 4)
		a.SUBimm(X7, X7, 1)
		a.CBNZ(X7, pos)
		for j := 0; j < tail; j++ {
			d := 4 * (d0 + int32(j))
			a.STRs(VReg(j), X1, d)
			a.STRs(VReg(16+j), X10, d)
		}
	}
	a.RET()
	return a.Bytes(), nil
}

// kvLoad brings four cache elements from element e into a 4S vector as
// float32. f16 pays one extra instruction per four elements (LDRd + FCVTL
// against LDRq). q8 loads four int8 (LDR s), sign-extends them twice, converts
// and multiplies by the block's d, loaded into lane 0 of tmp: e is a multiple
// of four, so the four share one 32-element block.
func (a *A64) kvLoad(dst VReg, base XReg, k kvAddr, e int, tmp VReg) {
	switch k.f {
	case KVF16:
		a.LDRd(dst, base, k.off(e))
		a.FCVTL(dst, dst)
	case KVQ8:
		a.LDRs(dst, base, k.off(e))
		a.kvQ8Widen(dst, base, k, e, tmp)
	default:
		a.LDRq(dst, base, k.off(e))
	}
}

// kvQ8Widen turns the int8 in dst's low lanes into float32 scaled by the d
// covering element e.
func (a *A64) kvQ8Widen(dst VReg, base XReg, k kvAddr, e int, tmp VReg) {
	a.SXTL8h(dst, dst)
	a.SXTL4s(dst, dst)
	a.SCVTF4s(dst, dst)
	a.LDRs(tmp, base, k.scale(e))
	a.FMULelem(dst, dst, tmp, 0)
}

// kvLoad1 brings element e into lane 0, the rest zeroed: LDR h and a widen for
// f16, LDR s for f32, LDR b and the q8 widening for q8 (the zeroed lanes
// convert to +0).
func (a *A64) kvLoad1(dst VReg, base XReg, k kvAddr, e int, tmp VReg) {
	switch k.f {
	case KVF16:
		a.LDRh(dst, base, k.off(e))
		a.FCVTL(dst, dst)
	case KVQ8:
		a.LDRb(dst, base, k.off(e))
		a.kvQ8Widen(dst, base, k, e, tmp)
	default:
		a.LDRs(dst, base, k.off(e))
	}
}

// EmitAttnScoresTiled is the NEON twin of EmitAttnScoresTiled: qt queries share
// one pass over K (see the amd64 comment). NEON has no memory-operand FMA, so
// the query is loaded per vector either way; what tiling removes is the K
// load. The tower runs the same qt on both architectures.
func EmitAttnScoresTiled(hd, kvStride, qStride, scoreStride, qt int, kv KVFmt) ([]byte, error) {
	ka, err := newKVAddr(kv, hd, kvStride)
	if err != nil {
		return nil, err
	}
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnScoresTiled: hd=%d must be positive", hd)
	}
	if qt < 1 || qt > 30 {
		return nil, fmt.Errorf("jit: EmitAttnScoresTiled: qt=%d needs qt+2 of 32 vector registers", qt)
	}
	// LDRq's unsigned offset is scaled by 16, so a query row has to start on a
	// vector boundary for the baked displacement to be encodable at all.
	if qStride%4 != 0 {
		return nil, fmt.Errorf("jit: EmitAttnScoresTiled: qStride=%d must be a multiple of 4", qStride)
	}
	nv := hd / 4
	vk, vq := VReg(qt), VReg(qt+1)

	var a A64
	a.LDRx(X1, X0, 0)   // Out:  scores, row j at j*scoreStride
	a.LDRx(X2, X0, 8)   // W:    K cache row 0
	a.LDRx(X3, X0, 32)  // Rows: positions
	a.LDRx(X4, X0, 112) // Q32:  query 0, row j at j*qStride

	row := a.Label()
	a.Bind(row)
	for j := 0; j < qt; j++ {
		a.MOVIzero(VReg(j))
	}
	for i := 0; i < nv; i++ {
		a.kvLoad(vk, X2, ka, i*4, vq)
		for j := 0; j < qt; j++ {
			a.LDRq(vq, X4, int32(j*qStride+i*4)*4)
			a.FMLA4s(VReg(j), vk, vq)
		}
	}
	// The last hd%4 dimensions. LDR s's scaled offset tops out at 16380, which
	// j*qStride*4 passes on any real tower, so the query row walks a cursor.
	for d := nv * 4; d < hd; d++ {
		a.kvLoad1(vk, X2, ka, d, vq)
		a.MOVreg(X6, X4)
		a.addBig(X6, int32(d)*4, X9)
		for j := 0; j < qt; j++ {
			a.LDRs(vq, X6, 0)
			a.FMLA4s(VReg(j), vk, vq)
			if j != qt-1 {
				a.addBig(X6, int32(qStride)*4, X9)
			}
		}
	}
	// scoreStride*4 overruns STR's scaled 12-bit immediate at any real patch
	// count (7*1024*4 = 28672 against a 16380 ceiling), so the row cursor walks
	// instead of being an immediate.
	a.MOVreg(X5, X1)
	for j := 0; j < qt; j++ {
		a.FADDP4s(VReg(j), VReg(j), VReg(j))
		a.FADDP4s(VReg(j), VReg(j), VReg(j))
		a.STRs(VReg(j), X5, 0)
		if j != qt-1 {
			a.addBig(X5, int32(scoreStride)*4, X9)
		}
	}
	a.ADDimm(X1, X1, 4)
	a.addBig(X2, ka.stride(kvStride), X9)
	a.SUBimm(X3, X3, 1)
	a.CBNZ(X3, row)

	a.RET()
	return a.Bytes(), nil
}

// EmitKVWiden is the NEON twin of the amd64 EmitKVWiden: Rows q8_0 head rows
// widened to float32 through the attention kernels' own load.
func EmitKVWiden(hd int) ([]byte, error) {
	ka, err := newKVAddr(KVQ8, hd, hd)
	if err != nil {
		return nil, err
	}
	var a A64
	a.LDRx(X1, X0, 0)  // Out: f32 rows
	a.LDRx(X2, X0, 8)  // W:   q8 rows
	a.LDRx(X3, X0, 32) // Rows
	done, row := a.Label(), a.Label()
	a.CBZ(X3, done)
	a.Bind(row)
	for i := 0; i < hd/4; i++ {
		a.kvLoad(V0, X2, ka, i*4, V1)
		a.STRq(V0, X1, int32(i*16))
	}
	for d := hd / 4 * 4; d < hd; d++ {
		a.kvLoad1(V0, X2, ka, d, V1)
		a.STRs(V0, X1, int32(4*d))
	}
	a.addBig(X1, int32(4*hd), X9)
	a.addBig(X2, ka.stride(hd), X9)
	a.SUBimm(X3, X3, 1)
	a.CBNZ(X3, row)
	a.Bind(done)
	a.RET()
	return a.Bytes(), nil
}
