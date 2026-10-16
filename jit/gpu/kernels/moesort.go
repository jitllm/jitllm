package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// The batched mixture's grouping, on the device. A chunk's router picks k
// experts for each of rows tokens; the grouped expert matvecs want those
// (token, slot) pairs sorted by expert, each expert's run padded to a whole
// token group of unit columns, with a column table naming each column's
// expert. These kernels compute it (RULE 8), launched in this order inside
// the same submission:
//
//	ExpertGroupBlockCount bc[b][e] pairs of block b routed to e
//	ExpertGroupCount    off, cnt   e's pairs in earlier blocks; its total
//	ExpertGroupEnds     ends[e]    end of e's padded run: sum over f <= e
//	ExpertGroupColumns  colE[j]    each column's expert; perm[j] = rows (a
//	                                zero row) and colPair[j] = 0 by default
//	ExpertGroupScatter  perm/pos   each pair's column, stable in selection
//	                                order; colPair its inverse, when asked
//	ExpertGroupTiles    tsel[t]    the expert of each block of u columns
//
// The layout is exactly the reference's (internal/oracle's groupPairs, gated
// bit for bit in tier), so every kernel downstream reads what it always read.
// ends[nExp-1] is the used column count, the one word the host reads back to
// size the grouped launches.
//
// Each kernel is one flat loop per thread -- the IR has no nested loops --
// over at most GroupBlock pairs, the block count or nExp experts (see
// ExpertGroupScatter for why two levels).

// selAt loads pair i's expert: pairs are row-major, k to a row, in a selection
// laid out k+1 wide (ExpertRank's slot k is the runner-up).
func selAt(b *ir.Builder, pSel, i ir.Value, k int) ir.Value {
	ck := b.Const(ir.U32, int64(k))
	r, s := b.Div(ir.U32, i, ck), b.Rem(ir.U32, i, ck)
	return b.Load(ir.U32, pSel, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(k+1))), s), 0)
}

// eqU32 is 1 where x == y and 0 elsewhere: "neither less nor greater", since
// the IR has no Eq op (ExpertRank's idiom).
func eqU32(b *ir.Builder, x, y ir.Value) ir.Value {
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	lt := b.Select(ir.U32, b.Lt(ir.U32, x, y), one, zero)
	gt := b.Select(ir.U32, b.Lt(ir.U32, y, x), one, zero)
	return b.Sub(ir.U32, one, b.Add(ir.U32, lt, gt))
}

// padTo rounds n up to a whole multiple of unit.
func padTo(b *ir.Builder, n ir.Value, unit int) ir.Value {
	cu := b.Const(ir.U32, int64(unit))
	return b.Mul(ir.U32, b.Div(ir.U32, b.Add(ir.U32, n, b.Const(ir.U32, int64(unit-1))), cu), cu)
}

func groupCheck(name string, rows, k, nExp int) error {
	if rows < 1 || k < 1 || nExp < 1 {
		return fmt.Errorf("kernels: %s: rows=%d k=%d nExp=%d", name, rows, k, nExp)
	}
	return nil
}

// GroupBlock is how many pairs one block of the two-level scatter covers: a
// pair's rank among its expert's pairs is its block's offset plus a count over
// at most GroupBlock pairs before it, so no thread walks more than this.
const GroupBlock = 64

// GroupBlocks is how many GroupBlock-pair blocks rows*k pairs make.
func GroupBlocks(rows, k int) int { return (rows*k + GroupBlock - 1) / GroupBlock }

// ExpertGroupBlockCount writes bc[b*nExp+e], the pairs of block b routed to e.
// Launch GroupBlocks(rows, k)*nExp threads.
func ExpertGroupBlockCount(rows, k, nExp int) (*ir.Kernel, error) {
	if err := groupCheck("ExpertGroupBlockCount", rows, k, nExp); err != nil {
		return nil, err
	}
	n, nb := rows*k, GroupBlocks(rows, k)
	b := ir.New("expertgroupblockcount", [3]int{128, 1, 1})
	pSel := b.Param("pSel", ir.U32)
	pBC := b.Param("pBC", ir.U32)
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	t := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, int64(nb*nExp-1)))
	ce := b.Const(ir.U32, int64(nExp))
	blk, e := b.Div(ir.U32, t, ce), b.Rem(ir.U32, t, ce)
	base := b.Mul(ir.U32, blk, b.Const(ir.U32, GroupBlock))
	last := b.Const(ir.U32, int64(n-1))
	b.Loop(GroupBlock)
	acc := b.Phi(ir.U32, zero)
	j := b.Phi(ir.U32, zero)
	i := b.Add(ir.U32, base, j)
	in := b.Select(ir.U32, b.Lt(ir.U32, last, i), zero, one) // i < n
	b.SetPhi(acc, b.Add(ir.U32, acc, b.Mul(ir.U32, in, eqU32(b, selAt(b, pSel, b.Min(ir.U32, i, last), k), e))))
	b.SetPhi(j, b.Add(ir.U32, j, one))
	b.EndLoop()
	b.Store(pBC, t, acc, 0)
	return b.Done(), nil
}

// ExpertGroupCount writes cnt[e], the pairs routed to e, and off[b*nExp+e],
// the pairs of e in blocks before b, from the block counts. Launch
// GroupBlocks(rows, k)*nExp threads; the last block's thread of each expert
// also writes cnt[e].
func ExpertGroupCount(rows, k, nExp int) (*ir.Kernel, error) {
	if err := groupCheck("ExpertGroupCount", rows, k, nExp); err != nil {
		return nil, err
	}
	nb := GroupBlocks(rows, k)
	b := ir.New("expertgroupcount", [3]int{128, 1, 1})
	pBC := b.Param("pBC", ir.U32)
	pOff := b.Param("pOff", ir.U32)
	pCnt := b.Param("pCnt", ir.U32)
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	t := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, int64(nb*nExp-1)))
	ce := b.Const(ir.U32, int64(nExp))
	blk, e := b.Div(ir.U32, t, ce), b.Rem(ir.U32, t, ce)
	b.Loop(int64(nb))
	acc := b.Phi(ir.U32, zero)
	f := b.Phi(ir.U32, zero)
	before := b.Select(ir.U32, b.Lt(ir.U32, f, blk), one, zero)
	b.SetPhi(acc, b.Add(ir.U32, acc, b.Mul(ir.U32, before, b.Load(ir.U32, pBC, b.Add(ir.U32, b.Mul(ir.U32, f, ce), e), 0))))
	b.SetPhi(f, b.Add(ir.U32, f, one))
	b.EndLoop()
	b.Store(pOff, t, acc, 0)
	// Every thread writes cnt[e]; only the last block's value is the total, so
	// the others write theirs to e's slot of a trash row past the nExp real
	// ones (cnt is 2*nExp long).
	isLast := b.Select(ir.U32, b.Lt(ir.U32, blk, b.Const(ir.U32, int64(nb-1))), zero, one)
	at := b.Add(ir.U32, e, b.Mul(ir.U32, b.Sub(ir.U32, one, isLast), ce))
	b.Store(pCnt, at, b.Add(ir.U32, acc, b.Load(ir.U32, pBC, t, 0)), 0)
	return b.Done(), nil
}

// ExpertGroupEnds writes ends[e] = sum over f <= e of cnt[f] padded to unit.
// Launch nExp threads.
func ExpertGroupEnds(nExp, unit int) (*ir.Kernel, error) {
	if nExp < 1 || unit < 1 {
		return nil, fmt.Errorf("kernels: ExpertGroupEnds: nExp=%d unit=%d", nExp, unit)
	}
	b := ir.New("expertgroupends", [3]int{128, 1, 1})
	pCnt := b.Param("pCnt", ir.U32)
	pEnds := b.Param("pEnds", ir.U32)
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	e := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, int64(nExp-1)))
	b.Loop(int64(nExp))
	acc := b.Phi(ir.U32, zero)
	f := b.Phi(ir.U32, zero)
	in := b.Select(ir.U32, b.Lt(ir.U32, e, f), zero, one) // f <= e
	b.SetPhi(acc, b.Add(ir.U32, acc, b.Mul(ir.U32, in, padTo(b, b.Load(ir.U32, pCnt, f, 0), unit))))
	b.SetPhi(f, b.Add(ir.U32, f, one))
	b.EndLoop()
	b.Store(pEnds, e, acc, 0)
	return b.Done(), nil
}

// expertOf is the expert whose padded run holds column j: the number of runs
// that END at or before j. At or past the used columns it is nExp, which the
// callers map to expert 0 -- the reference's value for an unused column.
func expertOf(b *ir.Builder, pEnds, j ir.Value, nExp int) ir.Value {
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	b.Loop(int64(nExp))
	acc := b.Phi(ir.U32, zero)
	f := b.Phi(ir.U32, zero)
	done := b.Select(ir.U32, b.Lt(ir.U32, j, b.Load(ir.U32, pEnds, f, 0)), zero, one) // ends[f] <= j
	b.SetPhi(acc, b.Add(ir.U32, acc, done))
	b.SetPhi(f, b.Add(ir.U32, f, one))
	b.EndLoop()
	return b.Select(ir.U32, b.Lt(ir.U32, acc, b.Const(ir.U32, int64(nExp))), acc, zero)
}

// ExpertGroupColumns writes colE[j] for every one of np columns, and the
// defaults the scatter then overwrites for the real ones: perm[j] = rows (a
// padding column gathers a zero row) and, with pair, colPair[j] = 0. Launch np
// threads.
func ExpertGroupColumns(rows, nExp, np int, pair bool) (*ir.Kernel, error) {
	if rows < 1 || nExp < 1 || np < 1 {
		return nil, fmt.Errorf("kernels: ExpertGroupColumns: rows=%d nExp=%d np=%d", rows, nExp, np)
	}
	b := ir.New("expertgroupcolumns", [3]int{128, 1, 1})
	pEnds := b.Param("pEnds", ir.U32)
	pColE := b.Param("pColE", ir.U32)
	pPerm := b.Param("pPerm", ir.U32)
	var pPair ir.Value
	if pair {
		pPair = b.Param("pPair", ir.U32)
	}
	j := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, int64(np-1)))
	b.Store(pColE, j, expertOf(b, pEnds, j, nExp), 0)
	b.Store(pPerm, j, b.Const(ir.U32, int64(rows)), 0)
	if pair {
		b.Store(pPair, j, b.Const(ir.U32, 0), 0)
	}
	return b.Done(), nil
}

// ExpertGroupScatter places each of rows*k pairs: its column is its expert's
// run start, plus that expert's pairs in earlier blocks (off), plus those
// before it in its own block -- so it stays stable in selection order -- and
// perm[col] = its row, pos[pair] = col and, with pair, colPair[col] = pair.
// Launch rows*k threads, after ExpertGroupColumns.
//
// Two levels because both one-level forms are slow: a pair counting every
// pair before it is O((rows*k)^2), and one thread per expert walking the
// pairs is only nExp threads wide and latency-bound. Here no thread walks
// more than GroupBlock pairs, and there is a thread per pair.
func ExpertGroupScatter(rows, k, nExp, unit, np int, pair bool) (*ir.Kernel, error) {
	if err := groupCheck("ExpertGroupScatter", rows, k, nExp); err != nil {
		return nil, err
	}
	b := ir.New("expertgroupscatter", [3]int{128, 1, 1})
	pSel := b.Param("pSel", ir.U32)
	pCnt := b.Param("pCnt", ir.U32)
	pEnds := b.Param("pEnds", ir.U32)
	pOff := b.Param("pOff", ir.U32)
	pPerm := b.Param("pPerm", ir.U32)
	pPos := b.Param("pPos", ir.U32)
	var pPair ir.Value
	if pair {
		pPair = b.Param("pPair", ir.U32)
	}
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	n := rows * k
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, int64(n-1)))
	e := selAt(b, pSel, i, k)
	blk := b.Div(ir.U32, i, b.Const(ir.U32, GroupBlock))
	base := b.Mul(ir.U32, blk, b.Const(ir.U32, GroupBlock))
	b.Loop(GroupBlock)
	acc := b.Phi(ir.U32, zero)
	j := b.Phi(ir.U32, zero)
	jj := b.Add(ir.U32, base, j)
	before := b.Select(ir.U32, b.Lt(ir.U32, jj, i), one, zero)
	b.SetPhi(acc, b.Add(ir.U32, acc, b.Mul(ir.U32, before, eqU32(b, selAt(b, pSel, b.Min(ir.U32, jj, i), k), e))))
	b.SetPhi(j, b.Add(ir.U32, j, one))
	b.EndLoop()
	// A selection of nExp or more names no expert and takes no column: its pair
	// points at the spare column np, which perm and colPair have room for and
	// the down projection's output keeps zero, so the combine adds nothing for
	// it. A wave of a prompt chunk's experts (streamBank.prefillWave) runs the
	// grouped kernels over the experts the card holds now and routes the
	// chunk's other pairs here.
	named := b.Lt(ir.U32, e, b.Const(ir.U32, int64(nExp)))
	ec := b.Min(ir.U32, e, b.Const(ir.U32, int64(nExp-1)))
	start := b.Sub(ir.U32, b.Load(ir.U32, pEnds, ec, 0), padTo(b, b.Load(ir.U32, pCnt, ec, 0), unit))
	off := b.Load(ir.U32, pOff, b.Add(ir.U32, b.Mul(ir.U32, blk, b.Const(ir.U32, int64(nExp))), ec), 0)
	col := b.Select(ir.U32, named, b.Add(ir.U32, b.Add(ir.U32, start, off), acc), b.Const(ir.U32, int64(np)))
	b.Store(pPerm, col, b.Div(ir.U32, i, b.Const(ir.U32, int64(k))), 0)
	b.Store(pPos, i, col, 0)
	if pair {
		b.Store(pPair, col, i, 0)
	}
	return b.Done(), nil
}

// ExpertGroupTiles writes tsel[t], the expert of columns t*u .. t*u+u-1, for
// nt tiles. u must divide the padding unit so a tile never straddles two runs.
// Launch nt threads.
func ExpertGroupTiles(nExp, nt, u int) (*ir.Kernel, error) {
	if nExp < 1 || nt < 1 || u < 1 {
		return nil, fmt.Errorf("kernels: ExpertGroupTiles: nExp=%d nt=%d u=%d", nExp, nt, u)
	}
	b := ir.New("expertgrouptiles", [3]int{128, 1, 1})
	pEnds := b.Param("pEnds", ir.U32)
	pTsel := b.Param("pTsel", ir.U32)
	t := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, int64(nt-1)))
	j := b.Mul(ir.U32, t, b.Const(ir.U32, int64(u)))
	b.Store(pTsel, t, expertOf(b, pEnds, j, nExp), 0)
	return b.Done(), nil
}
