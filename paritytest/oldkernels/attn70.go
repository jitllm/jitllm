package oldkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// AttnScoresMMA70 is the batched attention scores on sm_70's f16 tensor cores
// (ir.MMAVolta): AttnScoresMMA's product for a card whose matrix instruction is
// m8n8k4, with the operands the other way round.
//
// The keys are the A operand and the queries the B, because of where each is
// contiguous. An m8n8k4 lane holds four consecutive k-elements of one row of A
// and of one column of B. The K cache is transposed ([kv head][dim][position]),
// so a key's four dims are kStride apart and load one at a time, but the 32
// lanes then read 32 consecutive positions, which coalesces. A query's four
// dims are adjacent, one 16-byte load shared by the warp's quad-pairs. D comes
// out keys by queries.
//
// Operands drop to binary16 and the accumulator stays float32, as in
// AttnScoresMMA; the scale is applied to the float32 result. The mask is -inf
// per query row past its causal count (and outside a window). Keys past the
// cache are clamped to its last slot for the load only; their scores are
// masked by a Select, which replaces rather than multiplies.
//
// Launch: (rows/(8*nt)) * nHeads * ceil(pN[0]/(32*mt)) warps of 32, in groups
// of 128; the grid is clamped to its last warp.
//
// kStride 0 is a row-major K ([position][kvDim]), MLA's latent cache: one row
// per position that is both the key and (its first KVLoraRank floats) the
// value, so it cannot be transposed without a second copy. A lane's four dims
// of one key are then adjacent (one 16-byte load) and each quad-pair lane
// walks its own row through the unrolled k loop, using every cache line over
// consecutive steps. Keys past the chunk's width are clamped to its last row
// for the load only.
func AttnScoresMMA70(nHeads, headDim, kvDim, gqa, maxSeq int, scale float32, rows, kStride, mt, nt, window int) (*ir.Kernel, error) {
	if gqa <= 0 || nHeads%gqa != 0 {
		return nil, fmt.Errorf("kernels: AttnScoresMMA70: gqa=%d does not divide nHeads=%d", gqa, nHeads)
	}
	if kStride < 0 {
		return nil, fmt.Errorf("kernels: AttnScoresMMA70: kStride=%d", kStride)
	}
	rowMajor := kStride == 0
	if mt < 1 || nt < 1 || headDim%4 != 0 || rows%(8*nt) != 0 || rowMajor && kvDim%4 != 0 {
		return nil, fmt.Errorf("kernels: AttnScoresMMA70: headDim=%d kvDim=%d rows=%d mt=%d nt=%d",
			headDim, kvDim, rows, mt, nt)
	}
	name := fmt.Sprintf("attnscoresmma70_m%dn%d", mt, nt)
	if rowMajor {
		name = fmt.Sprintf("attnscoresmma70r_m%dn%d", mt, nt)
	}
	b := ir.New(name, [3]int{128, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }

	capN := b.Load(ir.U32, pN, c(0), 0)
	ktiles := b.Div(ir.U32, b.Add(ir.U32, capN, c(int64(32*mt-1))), c(int64(32*mt)))
	qgroups := rows / (8 * nt)
	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	warp := b.Min(ir.U32, b.Shr(ir.U32, tid, c(5)),
		b.Sub(ir.U32, b.Mul(ir.U32, ktiles, c(int64(nHeads*qgroups))), c(1)))
	v := voltaLane{b, b.And(ir.U32, tid, c(31))}
	// The key tile varies fastest, so neighbouring warps read neighbouring
	// positions of the same head.
	kt := b.Rem(ir.U32, warp, ktiles)
	rest := b.Div(ir.U32, warp, ktiles)
	h := b.Rem(ir.U32, rest, c(int64(nHeads)))
	qg := b.Div(ir.U32, rest, c(int64(nHeads)))

	q8 := b.Shl(ir.U32, v.quad(), c(3))
	keyW := b.Add(ir.U32, b.Mul(ir.U32, kt, c(int64(32*mt))), q8) // + 32i + row
	qryW := b.Mul(ir.U32, qg, c(int64(8*nt)))                     // + 8j + row
	kvh := b.Mul(ir.U32, b.Div(ir.U32, h, c(int64(gqa))), c(int64(headDim)))
	kBase := make([]ir.Value, mt)
	for i := range kBase {
		if rowMajor {
			key := b.Min(ir.U32, b.Add(ir.U32, keyW, b.Add(ir.U32, v.row(), c(int64(32*i)))), b.Sub(ir.U32, capN, c(1)))
			kBase[i] = b.Add(ir.U32, b.Mul(ir.U32, key, c(int64(kvDim))), kvh)
			continue
		}
		key := b.Min(ir.U32, b.Add(ir.U32, keyW, b.Add(ir.U32, v.row(), c(int64(32*i)))), c(int64(kStride-1)))
		kBase[i] = b.Add(ir.U32, b.Mul(ir.U32, kvh, c(int64(kStride))), key)
	}
	qBase := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, qryW, v.row()), c(int64(nHeads))), h),
		c(int64(headDim)))

	// A key tile at or past the group's widest causal count is not computed:
	// on a causal prompt about half the grid's tiles score keys no query in
	// the group can see, and the softmax and AttnAccMMA70 both stop at the
	// group's widest count, so nothing reads them. A LoopN of 0 or 1 is the
	// IR's only conditional.
	gmax := b.Load(ir.U32, pN, qryW, int64(8*nt))
	live := b.Select(ir.U32, b.Lt(ir.U32, b.Mul(ir.U32, kt, c(int64(32*mt))), gmax), c(1), c(0))
	b.LoopN(live)

	zero := b.ConstF32(0)
	acc := make([][][]ir.Value, mt)
	for i := range acc {
		acc[i] = make([][]ir.Value, nt)
		for j := range acc[i] {
			acc[i][j] = make([]ir.Value, 8)
			for x := range acc[i][j] {
				acc[i][j][x] = zero
			}
		}
	}
	// headDim is fully unrolled: every address is a base plus an immediate.
	for s := 0; s < headDim/4; s++ {
		af := make([][]ir.Value, mt)
		for i := range af {
			var e []ir.Value
			if rowMajor {
				e = b.LoadV(ir.F32, pK, kBase[i], int64(4*s), 4)
			} else {
				e = make([]ir.Value, 4)
				for x := range e {
					e[x] = b.Load(ir.F32, pK, kBase[i], int64(4*s+x)*int64(kStride))
				}
			}
			af[i] = []ir.Value{b.PackF16(e[0], e[1]), b.PackF16(e[2], e[3])}
		}
		bf := make([][]ir.Value, nt)
		for j := range bf {
			e := b.LoadV(ir.F32, pQ, qBase, int64(8*j*nHeads*headDim+4*s), 4)
			bf[j] = []ir.Value{b.PackF16(e[0], e[1]), b.PackF16(e[2], e[3])}
		}
		for i := 0; i < mt; i++ {
			for j := 0; j < nt; j++ {
				acc[i][j] = b.MMA(ir.MMAVolta(), af[i], bf[j], acc[i][j])
			}
		}
	}

	negInf := b.Bitcast(ir.F32, c(0xFF800000))
	sc := b.ConstF32(scale)
	dRow := []ir.Value{b.Add(ir.U32, keyW, v.dRow(0)), b.Add(ir.U32, keyW, v.dRow(2))}
	dCol := []ir.Value{b.Add(ir.U32, qryW, v.dCol(0)), b.Add(ir.U32, qryW, v.dCol(1)),
		b.Add(ir.U32, qryW, v.dCol(4)), b.Add(ir.U32, qryW, v.dCol(5))}
	for j := 0; j < nt; j++ {
		// The four queries this lane's accumulators land on in n-tile j, their
		// causal counts and their score rows.
		var cnt, row [4]ir.Value
		for x := range cnt {
			q := b.Add(ir.U32, dCol[x], c(int64(8*j)))
			cnt[x] = b.Load(ir.U32, pN, q, 1)
			row[x] = b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, q, c(int64(nHeads))), h), c(int64(maxSeq)))
		}
		for i := 0; i < mt; i++ {
			for comp := 0; comp < 8; comp++ {
				key := b.Add(ir.U32, dRow[(comp>>1)&1], c(int64(32*i)))
				x := (comp & 1) + 2*(comp>>2)
				val := b.Select(ir.F32, b.Lt(ir.U32, key, cnt[x]), b.Mul(ir.F32, acc[i][j][comp], sc), negInf)
				val = windowed(b, val, key, cnt[x], window, negInf)
				b.Store(pOut, b.Add(ir.U32, row[x], key), val, 0)
			}
		}
	}
	b.EndLoop()
	return b.Done(), nil
}

// AttnScoresMMA70Warps is how many warps AttnScoresMMA70 launches for a score
// width of nCap positions.
func AttnScoresMMA70Warps(nHeads, rows, nCap, mt, nt int) int {
	return (rows / (8 * nt)) * nHeads * ((nCap + 32*mt - 1) / (32 * mt))
}

// AttnAccMMA70 is the batched attention's weighted sum of V on sm_70's f16
// tensor cores: out = P . V per head, over each query group's causal width.
//
// V^T is the A operand and P^T the B, for the same reason as the scores: a
// lane's A row is one dim of V at four consecutive positions (kvDim apart,
// with 32 lanes reading 32 adjacent dims), and its B column is one query's
// probabilities at those positions, one 16-byte load. D is dims by queries.
//
// The warp walks its group's widest count, so the softmax must have
// normalised that far (SoftmaxRows' qt is 8*nt). The last partial step of
// four positions is masked on both operands (P to zero, V to zero with its
// load clamped to the last counted position): a V slot past the count is
// memory nothing wrote, and 0 * NaN is NaN.
//
// maxSeq is the score row stride and must be a multiple of four (the P loads
// are 16 bytes). Launch: (headDim/(32*mt)) * nHeads * (rows/(8*nt)) warps of 32,
// in groups of 128; the grid is clamped to its last warp.
func AttnAccMMA70(nHeads, headDim, kvDim, gqa, maxSeq, rows, mt, nt int) (*ir.Kernel, error) {
	if gqa <= 0 || nHeads%gqa != 0 {
		return nil, fmt.Errorf("kernels: AttnAccMMA70: gqa=%d does not divide nHeads=%d", gqa, nHeads)
	}
	if mt < 1 || nt < 1 || headDim%(32*mt) != 0 || rows%(8*nt) != 0 || maxSeq%4 != 0 {
		return nil, fmt.Errorf("kernels: AttnAccMMA70: headDim=%d rows=%d stride=%d mt=%d nt=%d",
			headDim, rows, maxSeq, mt, nt)
	}
	b := ir.New(fmt.Sprintf("attnaccmma70_m%dn%d", mt, nt), [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pV := b.Param("pV", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }

	dblocks, qgroups := headDim/(32*mt), rows/(8*nt)
	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	warp := b.Min(ir.U32, b.Shr(ir.U32, tid, c(5)), c(int64(dblocks*nHeads*qgroups-1)))
	v := voltaLane{b, b.And(ir.U32, tid, c(31))}
	db := b.Rem(ir.U32, warp, c(int64(dblocks)))
	rest := b.Div(ir.U32, warp, c(int64(dblocks)))
	h := b.Rem(ir.U32, rest, c(int64(nHeads)))
	qg := b.Div(ir.U32, rest, c(int64(nHeads)))

	q8 := b.Shl(ir.U32, v.quad(), c(3))
	dimW := b.Add(ir.U32, b.Mul(ir.U32, db, c(int64(32*mt))), q8)
	qryW := b.Mul(ir.U32, qg, c(int64(8*nt)))
	kvh := b.Mul(ir.U32, b.Div(ir.U32, h, c(int64(gqa))), c(int64(headDim)))
	vBase := b.Add(ir.U32, kvh, b.Add(ir.U32, dimW, v.row())) // + 32i, + pos*kvDim
	pBase := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, qryW, v.row()), c(int64(nHeads))), h),
		c(int64(maxSeq))) // + 8j*nHeads*maxSeq, + pos

	// The group's widest count: its last query's.
	n := b.Load(ir.U32, pN, qryW, int64(8*nt))
	full := b.Shr(ir.U32, n, c(2))

	zero := b.ConstF32(0)
	step := func(pv, pp ir.Value, d [][][]ir.Value, mask func(x int) (ir.Value, ir.Value)) {
		af := make([][]ir.Value, mt)
		for i := range af {
			e := make([]ir.Value, 4)
			for x := range e {
				if mask == nil {
					e[x] = b.Load(ir.F32, pV, pv, int64(x*kvDim+32*i))
					continue
				}
				ok, at := mask(x)
				e[x] = b.Select(ir.F32, ok, b.Load(ir.F32, pV, b.Add(ir.U32, at, vBase), int64(32*i)), zero)
			}
			af[i] = []ir.Value{b.PackF16(e[0], e[1]), b.PackF16(e[2], e[3])}
		}
		bf := make([][]ir.Value, nt)
		for j := range bf {
			e := b.LoadV(ir.F32, pA, pp, int64(8*j*nHeads*maxSeq), 4)
			if mask != nil {
				for x := range e {
					ok, _ := mask(x)
					e[x] = b.Select(ir.F32, ok, e[x], zero)
				}
			}
			bf[j] = []ir.Value{b.PackF16(e[0], e[1]), b.PackF16(e[2], e[3])}
		}
		for i := 0; i < mt; i++ {
			for j := 0; j < nt; j++ {
				d[i][j] = b.MMA(ir.MMAVolta(), af[i], bf[j], d[i][j])
			}
		}
	}

	b.LoopN(full)
	acc := make([][][]ir.Value, mt)
	for i := range acc {
		acc[i] = make([][]ir.Value, nt)
		for j := range acc[i] {
			acc[i][j] = make([]ir.Value, 8)
			for x := range acc[i][j] {
				acc[i][j][x] = b.Phi(ir.F32, zero)
			}
		}
	}
	pv := b.Phi(ir.U32, vBase)
	pp := b.Phi(ir.U32, pBase)
	d := make([][][]ir.Value, mt)
	for i := range d {
		d[i] = make([][]ir.Value, nt)
		copy(d[i], acc[i])
	}
	step(pv, pp, d, nil)
	for i := range acc {
		for j := range acc[i] {
			for x := range acc[i][j] {
				b.SetPhi(acc[i][j][x], d[i][j][x])
			}
		}
	}
	b.SetPhi(pv, b.Add(ir.U32, pv, c(int64(4*kvDim))))
	b.SetPhi(pp, b.Add(ir.U32, pp, c(4)))
	b.EndLoop()

	// The last partial step: positions 4*full .. 4*full+3, each kept only below
	// n. Its V load is clamped to position n-1 (n >= 1 for every query), so it
	// reads a slot something wrote; the Select throws the value away past n.
	// From the phis, not the body's values: the loop runs zero times when
	// the widest count is under four, and a phi then holds its init.
	p0 := b.Shl(ir.U32, full, c(2))
	lastPos := b.Sub(ir.U32, b.Max(ir.U32, n, c(1)), c(1))
	mask := func(x int) (ir.Value, ir.Value) {
		p := b.Add(ir.U32, p0, c(int64(x)))
		return b.Lt(ir.U32, p, n), b.Mul(ir.U32, b.Min(ir.U32, p, lastPos), c(int64(kvDim)))
	}
	for i := range d {
		copy(d[i], acc[i])
	}
	step(0, b.Add(ir.U32, pBase, p0), d, mask)

	dRow := []ir.Value{b.Add(ir.U32, dimW, v.dRow(0)), b.Add(ir.U32, dimW, v.dRow(2))}
	dCol := []ir.Value{b.Add(ir.U32, qryW, v.dCol(0)), b.Add(ir.U32, qryW, v.dCol(1)),
		b.Add(ir.U32, qryW, v.dCol(4)), b.Add(ir.U32, qryW, v.dCol(5))}
	for i := 0; i < mt; i++ {
		for j := 0; j < nt; j++ {
			for comp := 0; comp < 8; comp++ {
				dim := b.Add(ir.U32, dRow[(comp>>1)&1], c(int64(32*i)))
				q := b.Add(ir.U32, dCol[(comp&1)+2*(comp>>2)], c(int64(8*j)))
				o := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, q, c(int64(nHeads))), h), c(int64(headDim))), dim)
				b.Store(pOut, o, d[i][j][comp], 0)
			}
		}
	}
	return b.Done(), nil
}

// AttnAccMMA70Warps is how many warps AttnAccMMA70 launches.
func AttnAccMMA70Warps(nHeads, headDim, rows, mt, nt int) int {
	return (headDim / (32 * mt)) * nHeads * (rows / (8 * nt))
}
