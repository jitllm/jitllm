package oldkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// DeepSeek V3.2's lightning indexer on a device (engine/model/indexer.go has
// the graph). The indexer's key is cached in the MLA row of a paged pool, at
// column koff of a rowW-wide row-major row, and each query row's history is
// its descriptor's [PRowStart, PRowEnd) through the page table. Four basic
// kernels, each a loop of depth one:
//
//	IdxScoresPaged  score[r][t] = sum_h w[r][h] * scale * relu(q[r][h] . k_t)
//	IdxSelect       the IdxTopK highest-scoring positions of each row
//	IdxGather       the selected positions' attention rows into scratch pages
//	IdxDesc         the gathered pages' descriptors
//
// after which the MLA attention runs unchanged over the gathered pages: a
// row's keys are exactly the positions the indexer kept. Ties at the top-k
// boundary go to the lower position, as the host's selection does.

// IdxScoresPaged scores every position of each row's history. One thread per
// (row, position) up to tcap positions a row; a position past the row's end
// scores its last position (a clamp, never a read past the history), which the
// selection does not count. Parameters: pQ (rows x heads x dim, rotated), pW
// (rows x heads), pK (the pool), pTab, pRow, pOut (rows x tcap).
func IdxScoresPaged(rows, heads, dim, tcap, page, rowW, koff int, scale float32) (*ir.Kernel, error) {
	if err := checkPage("IdxScoresPaged", page); err != nil {
		return nil, err
	}
	if rows < 1 || heads < 1 || dim < 1 || tcap < 1 || koff < 0 || koff+dim > rowW {
		return nil, fmt.Errorf("kernels: IdxScoresPaged: rows=%d heads=%d dim=%d tcap=%d row=%d col=%d",
			rows, heads, dim, tcap, rowW, koff)
	}
	b := ir.New("idxscores", [3]int{128, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pW := b.Param("pW", ir.F32)
	pK := b.Param("pK", ir.F32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*tcap-1)))
	one := b.Const(ir.U32, 1)
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(tcap)))
	t := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(tcap)))
	end := b.Max(ir.U32, pagedDesc(b, pRow, r, PRowEnd), one)
	t = b.Min(ir.U32, t, b.Sub(ir.U32, end, one))
	pid := pageOf(b, pTab, pagedDesc(b, pRow, r, PRowTab), t, page)
	kb := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pid, b.Const(ir.U32, int64(page))),
		pageRem(b, t, page)), b.Const(ir.U32, int64(rowW))), b.Const(ir.U32, int64(koff)))
	qb := b.Mul(ir.U32, r, b.Const(ir.U32, int64(heads*dim)))
	wb := b.Mul(ir.U32, r, b.Const(ir.U32, int64(heads)))
	zf, zu := b.ConstF32(0), b.Const(ir.U32, 0)
	// One loop over every (head, dimension): the dot accumulates in acc and,
	// at each head's last dimension, its ReLU joins the weighted total.
	b.Loop(int64(heads * dim))
	i := b.Phi(ir.U32, zu)
	acc := b.Phi(ir.F32, zf)
	tot := b.Phi(ir.F32, zf)
	h := b.Div(ir.U32, i, b.Const(ir.U32, int64(dim)))
	d := b.Rem(ir.U32, i, b.Const(ir.U32, int64(dim)))
	a := b.Fma(b.Load(ir.F32, pQ, b.Add(ir.U32, qb, i), 0), b.Load(ir.F32, pK, b.Add(ir.U32, kb, d), 0), acc)
	mid := b.Lt(ir.U32, d, b.Const(ir.U32, int64(dim-1)))
	w := b.Mul(ir.F32, b.Load(ir.F32, pW, b.Add(ir.U32, wb, h), 0), b.ConstF32(scale))
	b.SetPhi(tot, b.Select(ir.F32, mid, tot, b.Fma(w, b.Max(ir.F32, a, zf), tot)))
	b.SetPhi(acc, b.Select(ir.F32, mid, a, zf))
	b.SetPhi(i, b.Add(ir.U32, i, one))
	b.EndLoop()
	b.Store(pOut, flat, tot, 0)
	return b.Done(), nil
}

// IdxSelect writes each row's top positions: position t of row r lands in
// slot rank(t) of the row's k+1 slots when it is in the history and its rank
// is below k, and in the spare slot k otherwise. rank(t) counts the positions
// that score higher, or equal and earlier, so the k kept are the host's.
// Parameters: pScore (rows x tcap), pRow, pSel (rows x (k+1), u32).
func IdxSelect(rows, tcap, k int) (*ir.Kernel, error) {
	if rows < 1 || tcap < 1 || k < 1 {
		return nil, fmt.Errorf("kernels: IdxSelect: rows=%d tcap=%d k=%d", rows, tcap, k)
	}
	b := ir.New("idxselect", [3]int{128, 1, 1})
	pScore := b.Param("pScore", ir.F32)
	pRow := b.Param("pRow", ir.U32)
	pSel := b.Param("pSel", ir.U32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*tcap-1)))
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(tcap)))
	t := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(tcap)))
	end := b.Min(ir.U32, pagedDesc(b, pRow, r, PRowEnd), b.Const(ir.U32, int64(tcap)))
	sb := b.Mul(ir.U32, r, b.Const(ir.U32, int64(tcap)))
	st := b.Load(ir.F32, pScore, flat, 0)
	b.LoopN(end)
	u := b.Phi(ir.U32, zero)
	rank := b.Phi(ir.U32, zero)
	su := b.Load(ir.F32, pScore, b.Add(ir.U32, sb, u), 0)
	// Higher, or (neither higher nor lower, so equal) and earlier.
	before := b.Select(ir.U32, b.Lt(ir.F32, st, su), one,
		b.Select(ir.U32, b.Lt(ir.F32, su, st), zero, b.Select(ir.U32, b.Lt(ir.U32, u, t), one, zero)))
	b.SetPhi(rank, b.Add(ir.U32, rank, before))
	b.SetPhi(u, b.Add(ir.U32, u, one))
	b.EndLoop()
	kk := b.Const(ir.U32, int64(k))
	slot := b.Select(ir.U32, b.Lt(ir.U32, t, end), b.Select(ir.U32, b.Lt(ir.U32, rank, kk), rank, kk), kk)
	b.Store(pSel, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(k+1))), slot), t, 0)
	return b.Done(), nil
}

// IdxGatherPages is the gathered pages' count a row: k keys of page P.
func IdxGatherPages(k, page int) int { return (k + page - 1) / page }

// IdxGather copies each row's selected positions' first width elements (the
// MLA latent and rotary key) into row r's pages of a scratch pool of the same
// row-major layout at width: kept key j of row r at page r*IdxGatherPages+j/P.
// A slot past the row's kept count copies its last kept key, which no
// descriptor reads. Parameters: pSrc (the pool), pTab, pRow, pSel, pDst.
func IdxGather(rows, k, width, page, rowW int) (*ir.Kernel, error) {
	if err := checkPage("IdxGather", page); err != nil {
		return nil, err
	}
	if rows < 1 || k < 1 || width < 1 || width > rowW {
		return nil, fmt.Errorf("kernels: IdxGather: rows=%d k=%d width=%d row=%d", rows, k, width, rowW)
	}
	ppr := IdxGatherPages(k, page)
	b := ir.New("idxgather", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	pSel := b.Param("pSel", ir.U32)
	pDst := b.Param("pDst", ir.F32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*k*width-1)))
	one := b.Const(ir.U32, 1)
	e := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(width)))
	j := b.Rem(ir.U32, b.Div(ir.U32, flat, b.Const(ir.U32, int64(width))), b.Const(ir.U32, int64(k)))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(k*width)))
	end := b.Max(ir.U32, pagedDesc(b, pRow, r, PRowEnd), one)
	kept := b.Min(ir.U32, end, b.Const(ir.U32, int64(k)))
	jj := b.Min(ir.U32, j, b.Sub(ir.U32, kept, one))
	t := b.Load(ir.U32, pSel, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(k+1))), jj), 0)
	t = b.Min(ir.U32, t, b.Sub(ir.U32, end, one))
	pid := pageOf(b, pTab, pagedDesc(b, pRow, r, PRowTab), t, page)
	src := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pid, b.Const(ir.U32, int64(page))),
		pageRem(b, t, page)), b.Const(ir.U32, int64(rowW))), e)
	dst := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(ppr*page))), j),
		b.Const(ir.U32, int64(width))), e)
	b.Store(pDst, dst, b.Load(ir.F32, pSrc, src, 0), 0)
	return b.Done(), nil
}

// IdxDesc writes the gathered pages' descriptors: row r's table at
// r*IdxGatherPages, its keys 0..min(end, k), written nowhere.
// Parameters: pRow (the history's), pOut.
func IdxDesc(rows, k, page int) (*ir.Kernel, error) {
	if rows < 1 || k < 1 {
		return nil, fmt.Errorf("kernels: IdxDesc: rows=%d k=%d", rows, k)
	}
	b := ir.New("idxdesc", [3]int{128, 1, 1})
	pRow := b.Param("pRow", ir.U32)
	pOut := b.Param("pOut", ir.U32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*PRowWords-1)))
	w := b.Rem(ir.U32, flat, b.Const(ir.U32, PRowWords))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, PRowWords))
	zero := b.Const(ir.U32, 0)
	kept := b.Min(ir.U32, pagedDesc(b, pRow, r, PRowEnd), b.Const(ir.U32, int64(k)))
	tab := b.Mul(ir.U32, r, b.Const(ir.U32, int64(IdxGatherPages(k, page))))
	isTab := b.Lt(ir.U32, w, b.Const(ir.U32, PRowTab+1))
	isEnd := b.Select(ir.U32, b.Lt(ir.U32, w, b.Const(ir.U32, PRowEnd)), zero,
		b.Select(ir.U32, b.Lt(ir.U32, w, b.Const(ir.U32, PRowEnd+1)), b.Const(ir.U32, 1), zero))
	v := b.Select(ir.U32, isTab, tab, b.Mul(ir.U32, isEnd, kept))
	b.Store(pOut, flat, v, 0)
	return b.Done(), nil
}
