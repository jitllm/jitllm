package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// FlashTile is FlashPrefillTile's blocking: SG simdgroups of eight query rows
// each, and TK keys a trip.
type FlashTile struct{ SG, TK int }

// Rows is how many query rows one workgroup takes.
func (t FlashTile) Rows() int { return 8 * t.SG }

// Threads is the workgroup width.
func (t FlashTile) Threads() int { return ir.SubgroupLanes * t.SG }

// FlashTileGroups is how many workgroups FlashPrefillTile launches.
func FlashTileGroups(heads, rows int, t FlashTile) int { return heads * (rows / t.Rows()) }

// FlashPrefillTileGroups is FlashTileGroups for a shape, a paged one's key
// splits included.
func FlashPrefillTileGroups(s FlashPrefill70Shape, t FlashTile) int {
	return FlashTileGroups(s.Heads, s.Rows, t) * flashPrefillSplits(s)
}

// flashTileShared is FlashPrefillTile's workgroup memory in bytes: K staged
// as binary16 rows padded by eight halves, V as float32 rows padded by four,
// the scores (and over them the probabilities) and the output as float32.
func flashTileShared(hd int, t FlashTile) int {
	qb := t.Rows()
	return t.TK*(hd+8)*2 + t.TK*(hd+4)*4 + qb*(t.TK+4)*4 + qb*(hd+4)*4
}

// FlashTileFor picks the blocking for a head width: the most query rows and
// then the most keys whose workgroup memory fits Metal's 32 KiB, or false.
func FlashTileFor(hd int) (FlashTile, bool) {
	for _, t := range []FlashTile{{4, 32}, {4, 16}, {2, 32}, {2, 16}, {2, 8}, {1, 32}, {1, 16}} {
		if flashTileShared(hd, t) <= 32<<10 && (t.TK*hd/2)%t.Threads() == 0 && (t.TK*hd/4)%t.Threads() == 0 {
			return t, true
		}
	}
	return FlashTile{}, false
}

// FlashPrefillTile is FlashPrefill70's attention on a collective matrix unit
// (the ir tile ops, which Metal lowers to simdgroup_matrix) for a prompt
// chunk: scores, a running softmax and the weighted sum of V per key tile in
// one kernel, the scores never written to memory. It follows llama.cpp's
// kernel_flash_attn_ext (ggml-metal/kernels/fa_common.metal): simdgroup 8x8
// products for Q K^T and P V, the softmax between them through threadgroup
// memory. Without it a Metal prompt's gap to llama.cpp grew with its length.
//
// The softmax goes through memory because a simdgroup_matrix has no documented
// lane layout (msl/tile.go), unlike a Volta fragment: S is stored, four lanes
// a row take the maximum, the sum and the probabilities, and the output O
// lives in threadgroup memory, where those lanes rescale it by
// exp(m_old - m_new) before the next P V accumulates into it.
//
// Q and K drop to binary16, as CUDA's AttnScoresMMA's do; P and V stay
// float32, as CUDA's AttnAcc does, and every accumulator is float32. With P and
// V in binary16 too, a q/k-normed model's prefill (synth-apertus) left the
// host by NMSE 1.3e-04 and an argmax, where CUDA's tiles read 1e-07: a
// normed row's softmax is nearly one-hot, and both halves' roundings land on
// the few keys it keeps. P is written over its own scores (each lane reads
// its keys' scores before it writes their probabilities), so float32 P costs
// no memory and every head width keeps its blocking. The per-element
// transform is FlashPrefill70's: the scale, the softcap, then -inf for a key
// at or past the row's causal count or before its window; a V row past the
// workgroup's widest count is staged as zero (0 * NaN is NaN, RULE 13).
//
// Parameters: pQ ([row][head][dim] f32), pK (the transposed f32 cache at
// KStride), pV ([position][kv head][dim] f32), pN (element 0 the uniform
// width, 1+r row r's count), pOut ([row][head][dim]). Launch FlashTileGroups
// groups of t.Threads().
func FlashPrefillTile(s FlashPrefill70Shape, t FlashTile) (*ir.Kernel, error) {
	hd := s.Dim
	qb, tk, nsg := t.Rows(), t.TK, t.SG
	nth := int64(t.Threads())
	switch {
	case s.KVHeads <= 0 || s.Heads%s.KVHeads != 0:
		return nil, fmt.Errorf("kernels: FlashPrefillTile: %d heads over %d kv heads", s.Heads, s.KVHeads)
	case hd <= 0 || hd%8 != 0:
		return nil, fmt.Errorf("kernels: FlashPrefillTile: head width %d is not a multiple of 8", hd)
	case s.Rows <= 0 || s.Rows%qb != 0:
		return nil, fmt.Errorf("kernels: FlashPrefillTile: %d rows, want a multiple of %d", s.Rows, qb)
	case s.Page == 0 && s.KStride <= 0:
		return nil, fmt.Errorf("kernels: FlashPrefillTile: needs the transposed K cache")
	case s.Window < 0 || s.Softcap < 0:
		return nil, fmt.Errorf("kernels: FlashPrefillTile: window %d softcap %v", s.Window, s.Softcap)
	case tk%8 != 0 || nsg < 1 || flashTileShared(hd, t) > 32<<10:
		return nil, fmt.Errorf("kernels: FlashPrefillTile: tile %+v at head width %d", t, hd)
	case int64(tk*hd/2)%nth != 0 || int64(tk*hd/4)%nth != 0:
		return nil, fmt.Errorf("kernels: FlashPrefillTile: %d threads do not tile %d keys x %d dims", nth, tk, hd)
	}
	if err := flashPrefillCheck("FlashPrefillTile", s); err != nil {
		return nil, err
	}
	P, S := s.Page, flashPrefillSplits(s)
	if P > 0 && 32%tk != 0 {
		return nil, fmt.Errorf("kernels: FlashPrefillTile: a paged key tile of %d does not divide 32", tk)
	}
	gqa := s.Heads / s.KVHeads
	kvDim := s.KVHeads * hd
	ks := int64(hd + 8) // halves between staged K rows (and Q rows)
	vs := int64(hd + 4) // floats between staged V rows
	ss := int64(tk + 4) // floats between score rows, and probability rows
	os := int64(hd + 4) // floats between output rows
	kpl := tk / 4       // keys a lane takes in the softmax
	dpl := hd / 4       // output dims a lane rescales and stores

	name := fmt.Sprintf("flashtile_d%d_q%d_k%d", hd, qb, tk)
	if P > 0 {
		name = fmt.Sprintf("flashtilepaged_d%d_q%d_k%d_s%d", hd, qb, tk, s.Splits)
	}
	if s.Map {
		name += "_map"
	}
	b := ir.New(name, [3]int{int(nth), 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pV := b.Param("pV", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	var pTab, pDesc ir.Value
	if P > 0 {
		pTab, pDesc = b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	}
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	// rowOf is a query slot's row: itself, or the run table's (Map).
	rowOf := func(slot ir.Value) ir.Value { return slot }
	if s.Map {
		pMap := b.Param("pMap", ir.U32)
		rowOf = func(slot ir.Value) ir.Value { return b.Load(ir.U32, pMap, slot, 0) }
	}
	f := b.ConstF32
	add := func(v ir.Value, n int64) ir.Value {
		if n == 0 {
			return v
		}
		return b.Add(ir.U32, v, c(n))
	}
	negInf := b.Bitcast(ir.F32, c(0xFF800000))
	zero := f(0)

	shK := b.Shared("ftK", ir.U32, tk*int(ks)/2)
	shV := b.Shared("ftV", ir.U32, tk*int(vs))
	shS := b.Shared("ftS", ir.F32, qb*int(ss)) // the scores, then P over them
	// The output, float32 -- and before the loop the queries' binary16 staging,
	// which fits in it (qb*(hd+8)/2 words against qb*(hd+4)).
	shO := b.Shared("ftO", ir.U32, qb*int(os))

	tid := b.TID()
	wg := b.CTAID()
	var spl ir.Value
	if P > 0 {
		// The split varies fastest, then the head.
		spl = urem(b, wg, S)
		wg = udiv(b, wg, S)
	}
	h := b.Rem(ir.U32, wg, c(int64(s.Heads)))
	qblk := b.Div(ir.U32, wg, c(int64(s.Heads)))
	kvBase := b.Mul(ir.U32, b.Div(ir.U32, h, c(int64(gqa))), c(int64(hd)))
	row0 := b.Mul(ir.U32, qblk, c(int64(qb)))
	var maxcnt, ntiles, lastPos, kStart ir.Value
	var span prefillSpan
	if P > 0 {
		// The block's keys, [keyStart of its first row, keyEnd of its last),
		// in tk-key tiles from a multiple of 32, which lie in one page each;
		// this split's run of them.
		span = tileSpanOf(b, pDesc, rowOf(row0), rowOf(add(row0, int64(qb-1))))
		kStart, ntiles = span.tileRun(b, spl, S, tk)
	} else {
		maxcnt = b.Load(ir.U32, pN, row0, int64(qb)) // the block's last row's count
		ntiles = b.Div(ir.U32, add(maxcnt, int64(tk-1)), c(int64(tk)))
		lastPos = b.Sub(ir.U32, b.Max(ir.U32, maxcnt, c(1)), c(1))
	}

	// Stage the queries as binary16 in the output's memory, load each
	// simdgroup's hd/8 tiles into registers, then clear the output.
	for it := int64(0); it < int64(qb*hd/2)/nth; it++ {
		w := add(tid, nth*it)
		qi := b.Div(ir.U32, w, c(int64(hd/2)))
		p := b.Rem(ir.U32, w, c(int64(hd/2)))
		src := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, rowOf(b.Add(ir.U32, row0, qi)), c(int64(s.Heads))), h),
			c(int64(hd))), b.Shl(ir.U32, p, c(1)))
		e := b.LoadV(ir.F32, pQ, src, 0, 2)
		b.Store(shO, b.Add(ir.U32, b.Mul(ir.U32, qi, c(ks/2)), p), b.PackF16(e[0], e[1]), 0)
	}
	b.Barrier()
	sg := b.SubgroupIndex()
	aT := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF16, Use: ir.TileA}
	bT := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF16, Use: ir.TileB}
	aP := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF32, Use: ir.TileA}
	bV := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF32, Use: ir.TileB}
	cT := ir.TileType{Rows: 8, Cols: 8, Elem: ir.TileF32, Use: ir.TileAcc}
	sgQ := b.Mul(ir.U32, sg, c(8*ks)) // halves: the simdgroup's first query row
	qt := make([]ir.Value, hd/8)
	for dk := range qt {
		qt[dk] = b.TileLoad(aT, shO, add(sgQ, int64(8*dk)), c(ks), false)
	}
	b.Barrier()
	for it := int64(0); it < int64(qb)*os/nth; it++ {
		b.Store(shO, add(tid, nth*it), c(0), 0)
	}

	// Per lane: its row of the simdgroup's eight, its quarter of the row.
	lane := b.And(ir.U32, tid, c(31))
	rl := b.Shr(ir.U32, lane, c(2))
	part := b.And(ir.U32, lane, c(3))
	row := b.Add(ir.U32, b.Mul(ir.U32, sg, c(8)), rl) // within the block
	var cnt, kst ir.Value                             // paged: the row's keyEnd and keyStart
	if P > 0 {
		kst, cnt = pagedDesc(b, pDesc, rowOf(b.Add(ir.U32, row0, row)), PRowStart), pagedDesc(b, pDesc, rowOf(b.Add(ir.U32, row0, row)), PRowEnd)
	} else {
		cnt = b.Load(ir.U32, pN, b.Add(ir.U32, row0, row), 1)
	}
	var winLo ir.Value
	if s.Window > 0 {
		winLo = b.Sub(ir.U32, cnt, b.Min(ir.U32, cnt, c(int64(s.Window))))
	}
	keyL := b.Mul(ir.U32, part, c(int64(kpl)))
	dimL := b.Mul(ir.U32, part, c(int64(dpl)))
	sRow := b.Add(ir.U32, b.Mul(ir.U32, row, c(ss)), keyL)
	oRow := b.Add(ir.U32, b.Mul(ir.U32, row, c(os)), dimL)
	sgS := b.Mul(ir.U32, sg, c(8*ss))
	sgO := b.Mul(ir.U32, sg, c(8*os))
	zeroU := c(0)
	b.Barrier()
	// Paged: everything of a staging item's address but the tile's page and
	// offset, and the row's key span, hoisted out of the tile loop.
	var kOff, vOff []ir.Value
	var kspan, hlo ir.Value
	if P > 0 {
		for it := int64(0); it < int64(tk*hd/2)/nth; it++ {
			w := add(tid, nth*it)
			key, dp := b.Rem(ir.U32, w, c(int64(tk))), b.Div(ir.U32, w, c(int64(tk)))
			kOff = append(kOff, b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, kvBase, b.Shl(ir.U32, dp, c(1))), c(int64(P))), key))
		}
		for it := int64(0); it < int64(tk*hd/4)/nth; it++ {
			w := add(tid, nth*it)
			dq, key := b.Rem(ir.U32, w, c(int64(hd/4))), b.Div(ir.U32, w, c(int64(hd/4)))
			vOff = append(vOff, b.Add(ir.U32, b.Mul(ir.U32, key, c(int64(kvDim))), b.Add(ir.U32, kvBase, b.Shl(ir.U32, dq, c(2)))))
		}
		kspan, hlo = b.Sub(ir.U32, cnt, kst), b.Sub(ir.U32, span.hi, span.lo)
	}

	var kt, m, l, k0, mo, lo, pg, kTile, vTile ir.Value
	if P > 0 {
		// The run's tiles a page at a time: an outer loop over the pages it
		// touches reads each page's id once, and the tiles inside it need no
		// lookup. A table read per tile, a dependent load at the head of
		// every trip, measurably slowed d128 on Metal.
		kEnd := b.Add(ir.U32, kStart, b.Mul(ir.U32, ntiles, c(int64(tk))))
		pg0 := pageDiv(b, kStart, P)
		npg := b.Select(ir.U32, b.Lt(ir.U32, zeroU, ntiles),
			add(b.Sub(ir.U32, pageDiv(b, b.Sub(ir.U32, kEnd, c(1)), P), pg0), 1), zeroU)
		b.LoopN(npg)
		mo, lo, pg = b.Phi(ir.F32, negInf), b.Phi(ir.F32, zero), b.Phi(ir.U32, pg0)
		t0 := b.Max(ir.U32, kStart, b.Mul(ir.U32, pg, c(int64(P))))
		pid := pageOf(b, pTab, span.tab, t0, P)
		t1 := b.Min(ir.U32, kEnd, b.Mul(ir.U32, add(pg, 1), c(int64(P))))
		b.LoopN(udiv(b, b.Sub(ir.U32, t1, t0), tk))
		m, l, k0 = b.Phi(ir.F32, mo), b.Phi(ir.F32, lo), b.Phi(ir.U32, t0)
		off := pageRem(b, k0, P)
		kTile = b.Add(ir.U32, b.Mul(ir.U32, pid, c(int64(P*kvDim))), off)
		vTile = b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pid, c(int64(P))), off), c(int64(kvDim)))
	} else {
		b.LoopN(ntiles)
		kt = b.Phi(ir.U32, zeroU)
		m = b.Phi(ir.F32, negInf)
		l = b.Phi(ir.F32, zero)
		k0 = b.Mul(ir.U32, kt, c(int64(tk)))
	}

	// Stage K and V. The barrier first: the last trip's products read them.
	b.Barrier()
	for it := int64(0); it < int64(tk*hd/2)/nth; it++ {
		w := add(tid, nth*it)
		key := b.Rem(ir.U32, w, c(int64(tk))) // consecutive threads, consecutive keys: coalesced
		dp := b.Div(ir.U32, w, c(int64(tk)))
		var lo, hi ir.Value
		if P > 0 {
			// A page's K is [kvRow][P]: the tile's keys are adjacent.
			e := b.Add(ir.U32, kTile, kOff[it])
			lo, hi = b.Load(ir.F32, pK, e, 0), b.Load(ir.F32, pK, e, int64(P))
		} else {
			pos := b.Min(ir.U32, b.Add(ir.U32, k0, key), c(int64(s.KStride-1)))
			e := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, kvBase, b.Shl(ir.U32, dp, c(1))), c(int64(s.KStride))), pos)
			lo = b.Load(ir.F32, pK, e, 0)
			hi = b.Load(ir.F32, pK, e, int64(s.KStride))
		}
		b.Store(shK, b.Add(ir.U32, b.Mul(ir.U32, key, c(ks/2)), dp), b.PackF16(lo, hi), 0)
	}
	for it := int64(0); it < int64(tk*hd/4)/nth; it++ {
		w := add(tid, nth*it)
		dq := b.Rem(ir.U32, w, c(int64(hd/4))) // consecutive threads, consecutive dims
		key := b.Div(ir.U32, w, c(int64(hd/4)))
		p := b.Add(ir.U32, k0, key)
		dst := func() ir.Value { return b.Add(ir.U32, b.Mul(ir.U32, key, c(vs)), b.Shl(ir.U32, dq, c(2))) }
		if P > 0 && s.F16 {
			// Packed f16 V: two words of binary16 pairs, widened, zero where
			// the position is outside the block's keys.
			ok := b.Lt(ir.U32, b.Sub(ir.U32, p, span.lo), hlo)
			if pagedFault == "vmask" {
				ok = span.in(b, p)
			}
			idx := b.Add(ir.U32, vTile, vOff[it])
			w := b.LoadV(ir.F32, pV, b.Shr(ir.U32, idx, c(1)), 0, 2)
			var f4 []ir.Value
			for _, x := range w {
				u := b.Bitcast(ir.U32, x)
				for _, h := range []ir.Value{b.CvtF16H(u), b.CvtF16H(b.Shr(ir.U32, u, c(16)))} {
					f4 = append(f4, b.Bitcast(ir.U32, b.Select(ir.F32, ok, h, zero)))
				}
			}
			b.StoreV(shV, dst(), 0, f4...)
			continue
		}
		var ok ir.Value
		var v []ir.Value
		if P > 0 {
			ok = b.Lt(ir.U32, b.Sub(ir.U32, p, span.lo), hlo)
			if pagedFault == "vmask" {
				ok = span.in(b, p)
			}
			v = vRow4(b, pV, b.Add(ir.U32, vTile, vOff[it]), 0, false)
		} else {
			ok = b.Lt(ir.U32, p, maxcnt)
			src := b.Add(ir.U32, b.Mul(ir.U32, b.Min(ir.U32, p, lastPos), c(int64(kvDim))),
				b.Add(ir.U32, kvBase, b.Shl(ir.U32, dq, c(2))))
			v = b.LoadV(ir.F32, pV, src, 0, 4)
		}
		for x := range v {
			v[x] = b.Select(ir.F32, ok, v[x], zero)
		}
		b.StoreV(shV, dst(), 0, b.Bitcast(ir.U32, v[0]), b.Bitcast(ir.U32, v[1]),
			b.Bitcast(ir.U32, v[2]), b.Bitcast(ir.U32, v[3]))
	}
	b.Barrier()

	// S = Q K^T for the simdgroup's eight rows, into shared memory.
	for j := 0; j < tk/8; j++ {
		acc := b.TileSplat(cT, zero)
		for dk := 0; dk < hd/8; dk++ {
			kb := b.TileLoad(bT, shK, c(int64(8*j)*ks+int64(8*dk)), c(ks), true)
			acc = b.TileMMA(qt[dk], kb, acc)
		}
		b.TileStore(shS, add(sgS, int64(8*j)), c(ss), acc, false)
	}
	b.Barrier()

	// The transform, the running maximum and sum, P over the scores, O rescaled.
	sc := f(s.Scale)
	val := make([]ir.Value, kpl)
	t0 := negInf
	for i := range val {
		x := b.Mul(ir.F32, b.Load(ir.F32, shS, sRow, int64(i)), sc)
		if s.Softcap > 0 {
			e := b.Exp(b.Mul(ir.F32, f(2/s.Softcap), x))
			x = b.Mul(ir.F32, f(s.Softcap), b.Sub(ir.F32, f(1), b.Div(ir.F32, f(2), b.Add(ir.F32, e, f(1)))))
		}
		key := b.Add(ir.U32, k0, add(keyL, int64(i)))
		if P > 0 {
			x = b.Select(ir.F32, b.Lt(ir.U32, b.Sub(ir.U32, key, kst), kspan), x, negInf)
		} else {
			x = b.Select(ir.F32, b.Lt(ir.U32, key, cnt), x, negInf)
		}
		if s.Window > 0 {
			x = b.Select(ir.F32, b.Lt(ir.U32, key, winLo), negInf, x)
		}
		val[i] = x
		t0 = b.Max(ir.F32, t0, x)
	}
	for _, mk := range []int64{1, 2} {
		t0 = b.Max(ir.F32, t0, b.ShuffleXor(ir.F32, t0, mk))
	}
	mn := b.Max(ir.F32, m, t0)
	// A row with no key yet stays at -inf; a finite stand-in keeps -inf - -inf
	// out of the exponentials.
	mf := b.Max(ir.F32, mn, f(-1e30))
	alpha := b.Exp(b.Sub(ir.F32, m, mf))
	sum := zero
	for i := 0; i < kpl; i += 2 {
		p0 := b.Exp(b.Sub(ir.F32, val[i], mf))
		p1 := b.Exp(b.Sub(ir.F32, val[i+1], mf))
		sum = b.Add(ir.F32, sum, b.Add(ir.F32, p0, p1))
		b.StoreV(shS, sRow, int64(i), p0, p1)
	}
	for _, mk := range []int64{1, 2} {
		sum = b.Add(ir.F32, sum, b.ShuffleXor(ir.F32, sum, mk))
	}
	lNext := b.Add(ir.F32, b.Mul(ir.F32, l, alpha), sum)
	for d := 0; d < dpl; d++ {
		o := b.Bitcast(ir.F32, b.Load(ir.U32, shO, oRow, int64(d)))
		b.Store(shO, oRow, b.Bitcast(ir.U32, b.Mul(ir.F32, o, alpha)), int64(d))
	}
	b.Barrier()

	// O += P V for the simdgroup's rows.
	pt := make([]ir.Value, tk/8)
	for kk := range pt {
		pt[kk] = b.TileLoad(aP, shS, add(sgS, int64(8*kk)), c(ss), false)
	}
	for dt := 0; dt < hd/8; dt++ {
		oOff := add(sgO, int64(8*dt))
		acc := b.TileLoad(cT, shO, oOff, c(os), false)
		for kk := 0; kk < tk/8; kk++ {
			vb := b.TileLoad(bV, shV, c(int64(8*kk)*vs+int64(8*dt)), c(vs), false)
			acc = b.TileMMA(pt[kk], vb, acc)
		}
		b.TileStore(shO, oOff, c(os), acc, false)
	}
	if P > 0 {
		b.SetPhi(k0, add(k0, int64(tk)))
		b.SetPhi(m, mn)
		b.SetPhi(l, lNext)
		b.EndLoop()
		b.SetPhi(mo, m)
		b.SetPhi(lo, l)
		b.SetPhi(pg, add(pg, 1))
		b.EndLoop()
		m, l = mo, lo
	} else {
		b.SetPhi(kt, add(kt, 1))
		b.SetPhi(m, mn)
		b.SetPhi(l, lNext)
		b.EndLoop()
	}

	// From the phis: a block whose widest count is zero runs no trip, and a
	// body value is then a register nothing wrote.
	b.Barrier()
	if P > 0 && s.Splits > 0 {
		// This run's partials for FlashAttentionMerge: the row's unnormalised
		// sums, its maximum (the finite stand-in its exponentials were taken
		// against) and its sum; the four lanes of a row write eight of its 32
		// replicas each.
		padded := (hd + 31) / 32 * 32
		pb := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, rowOf(b.Add(ir.U32, row0, row)),
			c(int64(s.Heads))), h), c(int64(S))), spl), c(int64(padded+64)))
		partialML(b, pOut, pb, b.Max(ir.F32, m, f(-1e30)), l, part, hd, 4)
		for d := 0; d < dpl; d++ {
			b.Store(pOut, b.Add(ir.U32, pb, dimL), b.Bitcast(ir.F32, b.Load(ir.U32, shO, oRow, int64(d))), int64(d))
		}
		return b.Done(), nil
	}
	inv := b.Div(ir.F32, f(1), b.Max(ir.F32, l, f(1e-30)))
	dst := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, rowOf(b.Add(ir.U32, row0, row)), c(int64(s.Heads))), h),
		c(int64(hd))), dimL)
	for d := 0; d < dpl; d++ {
		o := b.Bitcast(ir.F32, b.Load(ir.U32, shO, oRow, int64(d)))
		b.Store(pOut, dst, b.Mul(ir.F32, o, inv), int64(d))
	}
	return b.Done(), nil
}
