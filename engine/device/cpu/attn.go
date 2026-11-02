//go:build amd64

package cpu

import "fmt"

// The attention kernels.
//
// Attention is the one op whose cost grows with the conversation: a small
// share of decode at a short context, the majority at several hundred
// positions. Both kernels are pure float32 (eight AVX2 lanes). The head
// dimension and the KV stride are baked; they come from the model header.

// EmitAttnScores generates scores[t] = dot(q, K[t]) for t in [0, Rows).
//
// It reads Out (scores), W (the K cache), Q32 (the query) and Rows. The query
// is a memory operand of the FMA rather than held in registers: it is reused
// for every position, so it is L1-hot, and holding it would cap hd at the
// register file (hd=256 is thirty-two vectors).
func EmitAttnScores(hd, kvStride int, kv KVFmt) ([]byte, error) {
	ka, err := newKVAddr(kv, hd, kvStride)
	if err != nil {
		return nil, err
	}
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnScores: hd=%d must be positive", hd)
	}
	nv := hd / 8
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))   // Out: scores
	a.MOVLoad(RDX, At(RDI, 8))   // W:   K cache row 0
	a.MOVLoad(RAX, At(RDI, 32))  // Rows: positions
	a.MOVLoad(RSI, At(RDI, 112)) // Q32: query

	row := a.Label()
	a.Bind(row)
	// Four accumulator chains: one chain is latency-bound on the FMA. Four
	// covers a 4-cycle latency at one FMA per cycle.
	for i := 0; i < 4; i++ {
		a.VPXOR(Reg(i), Reg(i), Reg(i))
	}
	for i := 0; i < nv; i++ {
		a.kvLoad(Y4, RDX, ka, i*8, Y5)
		a.VFMADD231PSMem(Reg(i%4), Y4, At(RSI, int32(i)*32))
	}
	// hd is baked, so the last hd%8 dimensions are unrolled scalar FMAs into
	// chain 0: every other lane of both operands is zero and adds nothing.
	for d := int32(nv * 8); d < int32(hd); d++ {
		a.kvLoad1(Y4, RDX, ka, int(d), Y5)
		a.VMOVSSLoad(Y5, At(RSI, 4*d))
		a.VFMADD231PS(Y0, Y4, Y5)
	}
	a.VADDPS(Y0, Y0, Y1)
	a.VADDPS(Y2, Y2, Y3)
	a.VADDPS(Y0, Y0, Y2)
	a.VEXTRACTF128(Y1, Y0, 1)
	a.VADDPSx(Y0, Y0, Y1)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VMOVSSStore(At(RCX, 0), Y0)
	a.ADDimm(RCX, 4)
	a.ADDimm(RDX, ka.stride(kvStride))
	a.DEC(RAX)
	a.JNZ(row)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// EmitAttnAcc generates out[i] = sum over t of att[t] * V[t][i].
//
// It reads Out, W (the V cache), AScale (the softmaxed weights) and Rows.
//
// The output is held in registers across the whole position loop, in column
// blocks of eight vectors, rather than a load/fma/store round trip per
// position. A head wider than 64 floats takes several blocks, re-reading V
// (usually L2-resident) once per block.
func EmitAttnAcc(hd, kvStride int, kv KVFmt) ([]byte, error) {
	return emitAttnAcc(hd, kvStride, kv, false)
}

// EmitAttnAccInto is EmitAttnAcc that adds into Out instead of overwriting it,
// so a window split across several calls sums to what one call would have.
//
// It exists for the paged KV cache, and the bits are the point: the loop nests
// d-block outer and position inner, so for a fixed output dimension additions
// happen in increasing position order. Seeding the accumulators from Out makes
// page 0 then page 1 produce exactly the FMAs of one contiguous call, so paged
// and unpaged agree bit for bit rather than within an NMSE bound.
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
	const perBlock = 8 // accumulator vectors held at once
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W: V cache
	a.MOVLoad(R8, At(RDI, 24))  // AScale: attention weights
	a.MOVLoad(RAX, At(RDI, 32)) // Rows

	for base := 0; base < hd/8; base += perBlock {
		nv := min(perBlock, hd/8-base)
		for v := 0; v < nv; v++ {
			if acc {
				a.VMOVDQULoad(Reg(v), At(RCX, int32(base+v)*32))
			} else {
				a.VPXOR(Reg(v), Reg(v), Reg(v))
			}
		}
		a.MOVQ(RSI, RDX) // V cursor for this block
		a.MOVQ(R9, R8)   // weight cursor
		a.MOVQ(R11, RAX)
		pos := a.Label()
		a.Bind(pos)
		a.VBROADCASTSS(Y15, At(R9, 0))
		for v := 0; v < nv; v++ {
			a.kvLoad(Y14, RSI, ka, (base+v)*8, Y13)
			a.VFMADD231PS(Reg(v), Y14, Y15)
		}
		a.ADDimm(RSI, ka.stride(kvStride))
		a.ADDimm(R9, 4)
		a.DEC(R11)
		a.JNZ(pos)
		for v := 0; v < nv; v++ {
			a.VMOVDQUStore(At(RCX, int32(base+v)*32), Reg(v))
		}
	}
	// The last hd%8 dimensions: one accumulator per dimension, in lane 0,
	// walking the positions in the same order -- so a paged window still sums
	// to one contiguous call's bits.
	if tail := hd % 8; tail > 0 {
		d0 := int32(hd - tail)
		for j := 0; j < tail; j++ {
			if acc {
				a.VMOVSSLoad(Reg(j), At(RCX, 4*(d0+int32(j))))
			} else {
				a.VPXOR(Reg(j), Reg(j), Reg(j))
			}
		}
		a.MOVQ(RSI, RDX)
		a.MOVQ(R9, R8)
		a.MOVQ(R11, RAX)
		pos := a.Label()
		a.Bind(pos)
		a.VBROADCASTSS(Y15, At(R9, 0))
		for j := 0; j < tail; j++ {
			a.kvLoad1(Y14, RSI, ka, int(d0)+j, Y13)
			a.VFMADD231PS(Reg(j), Y14, Y15)
		}
		a.ADDimm(RSI, ka.stride(kvStride))
		a.ADDimm(R9, 4)
		a.DEC(R11)
		a.JNZ(pos)
		for j := 0; j < tail; j++ {
			a.VMOVSSStore(At(RCX, 4*(d0+int32(j))), Reg(j))
		}
	}
	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// EmitAttnScores2 is EmitAttnScores for two query heads at once: it loads each
// K vector once and dots it against both queries.
//
// With GQA every kv head is shared by several query heads, and the single-head
// kernel walks the same K rows once per query head. Co-scheduling those heads
// onto one worker does not help (it still issues separate traversals); only a
// kernel can share the load.
//
// Two, not four, because of the register budget: each head keeps four
// accumulator chains, so two heads use eight of sixteen YMM registers plus
// one for K, and four heads would use all sixteen before loading anything.
//
// It reads Out and Out2 (the two score rows), W (the K cache), Rows, and Q32 and
// Q2 (the two queries). The queries stay memory operands of the FMA, as in the
// single-head kernel.
func EmitAttnScores2(hd, kvStride int, kv KVFmt) ([]byte, error) {
	ka, err := newKVAddr(kv, hd, kvStride)
	if err != nil {
		return nil, err
	}
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnScores2: hd=%d must be positive", hd)
	}
	nv := hd / 8
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))   // Out:  scores, head 0
	a.MOVLoad(RDX, At(RDI, 8))   // W:    K cache row 0
	a.MOVLoad(RAX, At(RDI, 32))  // Rows: positions
	a.MOVLoad(RSI, At(RDI, 112)) // Q32:  query 0
	a.MOVLoad(R8, At(RDI, 120))  // Out2: scores, head 1
	a.MOVLoad(R9, At(RDI, 128))  // Q2:   query 1

	row := a.Label()
	a.Bind(row)
	// Y0-Y3 accumulate head 0, Y5-Y8 head 1, Y4 carries the shared K vector.
	for i := 0; i < 4; i++ {
		a.VPXOR(Reg(i), Reg(i), Reg(i))
		a.VPXOR(Reg(5+i), Reg(5+i), Reg(5+i))
	}
	for i := 0; i < nv; i++ {
		// One K load, two FMAs.
		a.kvLoad(Y4, RDX, ka, i*8, Y9)
		a.VFMADD231PSMem(Reg(i%4), Y4, At(RSI, int32(i)*32))
		a.VFMADD231PSMem(Reg(5+i%4), Y4, At(R9, int32(i)*32))
	}
	// The tail exactly as the single-head kernel does it, per head, so the
	// pair stays bit-identical to two single calls.
	for d := int32(nv * 8); d < int32(hd); d++ {
		a.kvLoad1(Y4, RDX, ka, int(d), Y9)
		a.VMOVSSLoad(Y9, At(RSI, 4*d))
		a.VFMADD231PS(Y0, Y4, Y9)
		a.VMOVSSLoad(Y9, At(R9, 4*d))
		a.VFMADD231PS(Y5, Y4, Y9)
	}
	// The reduction tree is the single-head one, twice, in the same order --
	// the paired kernel must be BIT-IDENTICAL to two single-head calls.
	a.VADDPS(Y0, Y0, Y1)
	a.VADDPS(Y2, Y2, Y3)
	a.VADDPS(Y0, Y0, Y2)
	a.VEXTRACTF128(Y1, Y0, 1)
	a.VADDPSx(Y0, Y0, Y1)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VMOVSSStore(At(RCX, 0), Y0)

	a.VADDPS(Y5, Y5, Y6)
	a.VADDPS(Y7, Y7, Y8)
	a.VADDPS(Y5, Y5, Y7)
	a.VEXTRACTF128(Y6, Y5, 1)
	a.VADDPSx(Y5, Y5, Y6)
	a.VHADDPSx(Y5, Y5, Y5)
	a.VHADDPSx(Y5, Y5, Y5)
	a.VMOVSSStore(At(R8, 0), Y5)

	a.ADDimm(RCX, 4)
	a.ADDimm(R8, 4)
	a.ADDimm(RDX, ka.stride(kvStride))
	a.DEC(RAX)
	a.JNZ(row)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// EmitAttnAcc2 is EmitAttnAcc for two heads sharing one kv head: it loads each
// V vector once and accumulates it into both outputs under their own weights.
//
// The dimension tile is forced by the register file: two heads' whole outputs
// at hd=64 would be sixteen vectors, so each pass covers half the dimensions
// (four accumulators per head, two broadcasts, one V: eleven of sixteen). The
// cost is two position walks; each touches a different half of every V vector,
// so V bytes are not re-read, but the weight reads and loop overhead repeat.
//
// It reads Out and Out2, W (the V cache), AScale and AScale2 (the two softmaxed
// weight rows) and Rows.
func EmitAttnAcc2(hd, kvStride int, kv KVFmt) ([]byte, error) {
	return emitAttnAcc2(hd, kvStride, kv, false)
}

// EmitAttnAcc2Into is EmitAttnAcc2 that adds into both outputs, so a paired
// window split across KV pages sums to what one call would have given, and a
// paged cache keeps the paired kernel at long context.
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
	const perBlock = 4 // accumulator vectors per head, per pass
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))   // Out
	a.MOVLoad(RDX, At(RDI, 8))   // W: V cache
	a.MOVLoad(R8, At(RDI, 24))   // AScale
	a.MOVLoad(RAX, At(RDI, 32))  // Rows
	a.MOVLoad(R10, At(RDI, 120)) // Out2
	a.MOVLoad(R12, At(RDI, 136)) // AScale2

	for base := 0; base < hd/8; base += perBlock {
		nv := min(perBlock, hd/8-base)
		for v := 0; v < nv; v++ {
			if acc {
				a.VMOVDQULoad(Reg(v), At(RCX, int32(base+v)*32))
				a.VMOVDQULoad(Reg(4+v), At(R10, int32(base+v)*32))
			} else {
				a.VPXOR(Reg(v), Reg(v), Reg(v))       // head 0: Y0-Y3
				a.VPXOR(Reg(4+v), Reg(4+v), Reg(4+v)) // head 1: Y4-Y7
			}
		}
		a.MOVQ(RSI, RDX) // V cursor for this tile
		a.MOVQ(R9, R8)   // weight cursor, head 0
		a.MOVQ(R13, R12) // weight cursor, head 1
		a.MOVQ(R11, RAX)
		pos := a.Label()
		a.Bind(pos)
		a.VBROADCASTSS(Y15, At(R9, 0))
		a.VBROADCASTSS(Y13, At(R13, 0))
		for v := 0; v < nv; v++ {
			// One V load, two FMAs, as in the paired scores kernel.
			a.kvLoad(Y14, RSI, ka, (base+v)*8, Y12)
			a.VFMADD231PS(Reg(v), Y14, Y15)
			a.VFMADD231PS(Reg(4+v), Y14, Y13)
		}
		a.ADDimm(RSI, ka.stride(kvStride))
		a.ADDimm(R9, 4)
		a.ADDimm(R13, 4)
		a.DEC(R11)
		a.JNZ(pos)
		for v := 0; v < nv; v++ {
			a.VMOVDQUStore(At(RCX, int32(base+v)*32), Reg(v))
			a.VMOVDQUStore(At(R10, int32(base+v)*32), Reg(4+v))
		}
	}
	// The last hd%8 dimensions, four at a time for the register budget above:
	// one lane-0 accumulator per (dimension, head).
	for d0 := int32(hd / 8 * 8); d0 < int32(hd); d0 += perBlock {
		nd := min(perBlock, hd-int(d0))
		for j := 0; j < nd; j++ {
			d := 4 * (d0 + int32(j))
			if acc {
				a.VMOVSSLoad(Reg(j), At(RCX, d))
				a.VMOVSSLoad(Reg(4+j), At(R10, d))
			} else {
				a.VPXOR(Reg(j), Reg(j), Reg(j))
				a.VPXOR(Reg(4+j), Reg(4+j), Reg(4+j))
			}
		}
		a.MOVQ(RSI, RDX)
		a.MOVQ(R9, R8)
		a.MOVQ(R13, R12)
		a.MOVQ(R11, RAX)
		pos := a.Label()
		a.Bind(pos)
		a.VBROADCASTSS(Y15, At(R9, 0))
		a.VBROADCASTSS(Y13, At(R13, 0))
		for j := 0; j < nd; j++ {
			a.kvLoad1(Y14, RSI, ka, int(d0)+j, Y12)
			a.VFMADD231PS(Reg(j), Y14, Y15)
			a.VFMADD231PS(Reg(4+j), Y14, Y13)
		}
		a.ADDimm(RSI, ka.stride(kvStride))
		a.ADDimm(R9, 4)
		a.ADDimm(R13, 4)
		a.DEC(R11)
		a.JNZ(pos)
		for j := 0; j < nd; j++ {
			d := 4 * (d0 + int32(j))
			a.VMOVSSStore(At(RCX, d), Reg(j))
			a.VMOVSSStore(At(R10, d), Reg(4+j))
		}
	}
	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// kvLoad brings eight cache elements starting at element e into a YMM as
// float32. VCVTPH2PS widens eight halves straight from memory, so the f16
// kernel issues exactly what the f32 one does and reads half the bytes. A q8
// row sign-extends eight int8 (VPMOVSXBD from memory), converts, and scales by
// the block's d, broadcast into tmp: e is a multiple of eight, so the eight
// never straddle a 32-element block.
func (a *Buf) kvLoad(dst, base Reg, k kvAddr, e int, tmp Reg) {
	switch k.f {
	case KVF16:
		a.VCVTPH2PS(dst, At(base, k.off(e)))
	case KVQ8:
		a.VPMOVSXBD(dst, At(base, k.off(e)))
		a.VCVTDQ2PS(dst, dst)
		a.VBROADCASTSS(tmp, At(base, k.scale(e)))
		a.VMULPS(dst, dst, tmp)
	default:
		a.VMOVDQULoad(dst, At(base, k.off(e)))
	}
}

// kvLoad1 brings element e into lane 0 -- the tail of a head dimension that is
// not a whole vector. VPINSRW reads exactly the two bytes of an f16 element,
// where VCVTPH2PS from memory would read past the end of the row; f32 and f16
// zero lanes 1..7.
//
// A q8 element is kvLoad's eight-byte read starting at e, which leaves lanes
// 1..7 holding the next elements (finite, and never stored or multiplied by
// anything but a zero lane: every caller pairs the tail with a lane-0 operand
// or stores lane 0 alone). The read stays inside the row: past the int8 run
// come the row's scale pairs, at least eight bytes.
func (a *Buf) kvLoad1(dst, base Reg, k kvAddr, e int, tmp Reg) {
	switch k.f {
	case KVF16:
		a.VPXOR(dst, dst, dst)
		a.VPINSRWLoad(dst, dst, At(base, k.off(e)), 0)
		a.VCVTPH2PSReg(dst, dst)
	case KVQ8:
		a.kvLoad(dst, base, k, e, tmp)
	default:
		a.VMOVSSLoad(dst, At(base, k.off(e)))
	}
}

// EmitAttnScoresTiled generates scores[j][t] = dot(q[j], K[t]) for qt queries at
// once, sharing one pass over K.
//
// It is a traffic fix: EmitAttnScores reads the whole K block per query, which
// for a vision tower (1024 patches, bidirectional) is memory bound at 0.25
// MAC/byte. qt queries per pass divide the K traffic by qt.
//
// Only the scores side tiles. Its accumulator is qt scalars (qt+2 registers),
// while AttnAcc's is qt x hd floats and must stay resident: qt*hd <= 128 caps it
// at 2 for hd=64, and splitting hd to widen qt multiplies the V passes by the
// same factor. That is why AttnAcc2 exists and AttnAcc4 does not.
//
// Every stride is baked, like hd and kvStride. qStride and scoreStride are
// elements, not bytes.
func EmitAttnScoresTiled(hd, kvStride, qStride, scoreStride, qt int, kv KVFmt) ([]byte, error) {
	ka, err := newKVAddr(kv, hd, kvStride)
	if err != nil {
		return nil, err
	}
	if hd <= 0 {
		return nil, fmt.Errorf("jit: EmitAttnScoresTiled: hd=%d must be positive", hd)
	}
	if qt < 1 || qt > 14 {
		return nil, fmt.Errorf("jit: EmitAttnScoresTiled: qt=%d needs qt+2 of 16 vector registers", qt)
	}
	nv := hd / 8
	kreg, treg := Reg(qt), Reg(qt+1)

	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))   // Out:  scores, row j at j*scoreStride
	a.MOVLoad(RDX, At(RDI, 8))   // W:    K cache row 0
	a.MOVLoad(RAX, At(RDI, 32))  // Rows: positions
	a.MOVLoad(RSI, At(RDI, 112)) // Q32:  query 0, row j at j*qStride

	row := a.Label()
	a.Bind(row)
	for j := 0; j < qt; j++ {
		a.VPXOR(Reg(j), Reg(j), Reg(j))
	}
	// K is loaded once per (key, vector) and feeds qt FMAs. The queries stay
	// memory operands as in EmitAttnScores. The qt chains are mutually
	// independent, so they cover the FMA latency without extra accumulators.
	for i := 0; i < nv; i++ {
		a.kvLoad(kreg, RDX, ka, i*8, treg)
		for j := 0; j < qt; j++ {
			a.VFMADD231PSMem(Reg(j), kreg, At(RSI, int32(j*qStride+i*8)*4))
		}
	}
	for d := nv * 8; d < hd; d++ {
		a.kvLoad1(kreg, RDX, ka, d, treg)
		for j := 0; j < qt; j++ {
			a.VMOVSSLoad(treg, At(RSI, int32(j*qStride+d)*4))
			a.VFMADD231PS(Reg(j), kreg, treg)
		}
	}
	for j := 0; j < qt; j++ {
		a.VEXTRACTF128(treg, Reg(j), 1)
		a.VADDPSx(Reg(j), Reg(j), treg)
		a.VHADDPSx(Reg(j), Reg(j), Reg(j))
		a.VHADDPSx(Reg(j), Reg(j), Reg(j))
		a.VMOVSSStore(At(RCX, int32(j*scoreStride)*4), Reg(j))
	}
	a.ADDimm(RCX, 4)
	a.ADDimm(RDX, ka.stride(kvStride))
	a.DEC(RAX)
	a.JNZ(row)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// EmitKVWiden generates the q8 cache's dequantization: Rows q8_0 head rows of
// hd elements (cpu.KVRowBytes apart, from W) widened to Rows float32 rows of
// hd (from Out), each element d*q -- the exact value the q8 attention kernels
// read. It is how a q8 history leaves the host's format (a migration to a
// device that holds float32), through the attention kernels' own load.
func EmitKVWiden(hd int) ([]byte, error) {
	ka, err := newKVAddr(KVQ8, hd, hd)
	if err != nil {
		return nil, err
	}
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out: f32 rows
	a.MOVLoad(RDX, At(RDI, 8))  // W:   q8 rows
	a.MOVLoad(RAX, At(RDI, 32)) // Rows
	done, row := a.Label(), a.Label()
	a.TESTQ(RAX, RAX)
	a.JZ(done)
	a.Bind(row)
	for i := 0; i < hd/8; i++ {
		a.kvLoad(Y0, RDX, ka, i*8, Y1)
		a.VMOVDQUStore(At(RCX, int32(i*32)), Y0)
	}
	for d := hd / 8 * 8; d < hd; d++ {
		a.kvLoad1(Y0, RDX, ka, d, Y1)
		a.VMOVSSStore(At(RCX, int32(4*d)), Y0)
	}
	a.ADDimm(RCX, int32(4*hd))
	a.ADDimm(RDX, ka.stride(hd))
	a.DEC(RAX)
	a.JNZ(row)
	a.Bind(done)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}
