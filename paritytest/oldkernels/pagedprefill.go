package oldkernels

import (
	"fmt"
	"math"
	"math/bits"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// The staged prefill over paged KV (docs/design/device-kv-paging.md): a
// prompt chunk's attention as scores, a softmax and an accumulate, in bounded
// key chunks with a merge -- the paged forms of AttnScoresTiled,
// AttnScoresMMA and AttnScoresMMA70, of the softmax, and of AttnAccTiled and
// AttnAccMMA70.
//
//	PagedAttnScores*     each row's scores into a plane of Splits*Chunk
//	                     slots per (row, head), [row][head][split][Chunk]
//	PagedPrefillSoftmax  each (row, head, split)'s maximum and denominator
//	                     into the merge's partials, and its unnormalised
//	                     weights, 0 on every slot the row does not attend
//	PagedAttnAcc*        each split's weighted sum of V into the partials
//	FlashAttentionMerge  the splits combined, sinks and the normalisation
//	                     applied once
//
// A launch is the rows of ONE sequence in position order (a prefill chunk):
// they share the table at row 0's tabOff, keyStart and keyEnd do not decrease
// from row to row, and a padded row is empty at the last real row's keyEnd
// (keyStart == keyEnd == it), so a query tile's keys are [keyStart of its
// first row, keyEnd of its last). The partition is the launch's: slot j of
// every row is key base+j with base the 64-aligned position at or below row
// 0's keyStart, and split sp is slots [sp*Chunk, (sp+1)*Chunk). All three
// kernels derive it from the same descriptor, so they agree without a
// parameter; the caller picks Splits*Chunk >= keyEnd(last row) - base. The
// plane is Rows*Heads*Splits*Chunk floats -- the span the launch attends,
// a window plus the chunk on a windowed layer, never a capacity.
//
// A key tile (up to 64 keys from a multiple of its width relative to base)
// lies in one page, since base is 64-aligned and P a multiple of 64; the
// tiled FMA scores translate each key on its own. Every address is clamped
// into the tile's own attended keys before its page is looked up, so no
// table entry past the row's table is read, and every value read outside
// those keys is selected away -- a page's unwritten tail is NaN (RULE 13).

// udiv is x/n and urem x%n for a constant n, a shift and a mask where n is a
// power of two. The PTX lowering hands div.u32 its divisor in a register and
// ptxas then emits a whole division (I2F, MUFU.RCP and a correction on sm_70)
// even for 4: in a short-context prefill a warp's setup is a real share of
// its work, and these kernels divide by their tiles and aligns throughout.
func udiv(b *ir.Builder, x ir.Value, n int) ir.Value {
	switch {
	case n == 1:
		return x
	case n > 0 && n&(n-1) == 0:
		return b.Shr(ir.U32, x, b.Const(ir.U32, int64(bits.TrailingZeros(uint(n)))))
	}
	return b.Div(ir.U32, x, b.Const(ir.U32, int64(n)))
}

func urem(b *ir.Builder, x ir.Value, n int) ir.Value {
	switch {
	case n == 1:
		return b.Const(ir.U32, 0)
	case n > 0 && n&(n-1) == 0:
		return b.And(ir.U32, x, b.Const(ir.U32, int64(n-1)))
	}
	return b.Rem(ir.U32, x, b.Const(ir.U32, int64(n)))
}

// prefillCheck validates a staged paged prefill shape.
func prefillCheck(what string, s FlashShape) error {
	if s.Page == 0 {
		return fmt.Errorf("kernels: %s: a paged prefill shape needs a Page", what)
	}
	if err := flashPaged(what, s); err != nil {
		return err
	}
	if s.Heads < 1 || s.KVHeads < 1 || s.Heads%s.KVHeads != 0 || s.Dim < 1 || s.Rows < 1 || s.Splits < 0 ||
		s.Chunk < 64 || s.Chunk%64 != 0 || s.Softcap < 0 {
		return fmt.Errorf("kernels: %s: invalid shape %+v", what, s)
	}
	if s.MLA > 0 && (s.KVHeads != 1 || s.MLA > s.Dim || s.F16) {
		return fmt.Errorf("kernels: %s: MLA wants one KV head, a value within the row and f32: %+v", what, s)
	}
	if s.F16 && (s.Dim%2 != 0 || s.KVHeads*s.Dim%2 != 0) {
		return fmt.Errorf("kernels: %s: packed f16 V needs an even Dim: %+v", what, s)
	}
	if s.Map && s.Splits != 0 {
		return fmt.Errorf("kernels: %s: a run table (Map) is the direct form's: no splits", what)
	}
	return nil
}

// PagedPrefillPlane is the score and weight planes' size in floats.
func PagedPrefillPlane(s FlashShape) int { return s.Rows * s.Heads * splitsOf(s) * s.Chunk }

// prefillSpan is what a query tile of rows [first, last] attends: the table,
// the launch's partition base, and the tile's keys [lo, hi).
type prefillSpan struct{ tab, base, lo, hi, live ir.Value }

func prefillSpanOf(b *ir.Builder, pRow, first, last ir.Value) prefillSpan {
	p := tileSpanOf(b, pRow, first, last)
	p.base = prefillBase(b, pRow)
	return p
}

// tileSpanOf is prefillSpanOf without the partition base, for a kernel that
// needs no partition shared with another.
func tileSpanOf(b *ir.Builder, pRow, first, last ir.Value) prefillSpan {
	lo, hi := pagedDesc(b, pRow, first, PRowStart), pagedDesc(b, pRow, last, PRowEnd)
	return prefillSpan{tab: pagedDesc(b, pRow, first, PRowTab), lo: lo, hi: hi, live: b.Lt(ir.U32, lo, hi)}
}

// prefillBase is the launch's partition base: the 64-aligned position at or
// below row 0's keyStart.
func prefillBase(b *ir.Builder, pRow ir.Value) ir.Value {
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	ks0 := pagedDesc(b, pRow, u(0), PRowStart)
	if pagedFault == "unaligned" {
		return ks0
	}
	return b.Mul(ir.U32, udiv(b, ks0, 64), u(64))
}

// rowMap is a staged prefill kernel's run table (FlashShape.Map): with it,
// query slot i is row pMap[i] of q, the output and the descriptors, and its
// partition base is pBase[i] -- each run's own, since the runs are different
// sequences at different positions -- while the score and weight planes stay
// in slots. pMap and pBase are the kernel's last parameters. Without it a
// slot is its row and the base is the launch's (prefillBase).
type rowMap struct {
	on          bool
	pMap, pBase ir.Value
}

func newRowMap(b *ir.Builder, s FlashShape) rowMap {
	if !s.Map {
		return rowMap{}
	}
	return rowMap{on: true, pMap: b.Param("pMap", ir.U32), pBase: b.Param("pBase", ir.U32)}
}

// row is slot's row.
func (m rowMap) row(b *ir.Builder, slot ir.Value) ir.Value {
	if !m.on {
		return slot
	}
	return b.Load(ir.U32, m.pMap, slot, 0)
}

// base is slot's partition base.
func (m rowMap) base(b *ir.Builder, pRow, slot ir.Value) ir.Value {
	if !m.on {
		return prefillBase(b, pRow)
	}
	return b.Load(ir.U32, m.pBase, slot, 0)
}

// span is prefillSpanOf over the slots first..last: their rows' keys and
// table, and first's base.
func (m rowMap) span(b *ir.Builder, pRow, first, last ir.Value) prefillSpan {
	p := tileSpanOf(b, pRow, m.row(b, first), m.row(b, last))
	p.base = m.base(b, pRow, first)
	return p
}

// tileRun is a flash prefill workgroup's run of key tiles of width tk: the
// tiles of its keys [lo, hi) from the 32-aligned position at or below lo,
// divided into splits runs of whole tiles; run spl starts at key start and
// has n tiles (none for an empty span or a run past its end).
func (p prefillSpan) tileRun(b *ir.Builder, spl ir.Value, splits, tk int) (start, n ir.Value) {
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	a := b.Mul(ir.U32, udiv(b, p.lo, 32), u(32))
	if pagedFault == "unaligned" {
		a = p.lo
	}
	total := b.Select(ir.U32, p.live, udiv(b, b.Add(ir.U32, b.Sub(ir.U32, p.hi, a), u(tk-1)), tk), u(0))
	if splits <= 1 {
		return a, total
	}
	per := udiv(b, b.Add(ir.U32, total, u(splits-1)), splits)
	t0 := b.Min(ir.U32, b.Mul(ir.U32, spl, per), total)
	return b.Add(ir.U32, a, b.Mul(ir.U32, t0, u(tk))), b.Sub(ir.U32, b.Min(ir.U32, b.Add(ir.U32, t0, per), total), t0)
}

// pagedV4 is V elements col..col+3 of position t's row, t in page pid.
func pagedV4(b *ir.Builder, pV, pid, t, col ir.Value, page, kvRow int, f16 bool) []ir.Value {
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	return vRow4(b, pV, b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pid, u(page)), pageRem(b, t, page)), u(kvRow)), col), 0, f16)
}

// vRow4 is V elements idx..idx+3 (element units, whatever the packing) plus
// imm: one 16-byte load, or for packed f16 one 8-byte load unpacked.
func vRow4(b *ir.Builder, pV, idx ir.Value, imm int, f16 bool) []ir.Value {
	if !f16 {
		return b.LoadV(ir.F32, pV, idx, int64(imm), 4)
	}
	w := b.LoadV(ir.F32, pV, b.Shr(ir.U32, idx, b.Const(ir.U32, 1)), int64(imm/2), 2)
	w0, w1 := b.Bitcast(ir.U32, w[0]), b.Bitcast(ir.U32, w[1])
	u16 := b.Const(ir.U32, 16)
	return []ir.Value{b.CvtF16H(w0), b.CvtF16H(b.Shr(ir.U32, w0, u16)), b.CvtF16H(w1), b.CvtF16H(b.Shr(ir.U32, w1, u16))}
}

// partialML stores a flash prefill row's maximum and sum into the merge's
// partials at pb (FlashAttentionMerge's layout at value width vd): the 32
// replicas, of which each of `share` lanes holding the row writes 32/share
// starting at rep*(32/share).
func partialML(b *ir.Builder, pOut, pb, m, l, rep ir.Value, vd, share int) {
	padded := (vd + 31) / 32 * 32
	n := 32 / share
	at := b.Add(ir.U32, pb, b.Mul(ir.U32, rep, b.Const(ir.U32, int64(n))))
	for r := 0; r < n; r++ {
		b.Store(pOut, at, m, int64(padded+r))
		b.Store(pOut, at, l, int64(padded+32+r))
	}
}

// clamp is t moved into the tile's keys [lo, hi-1] -- position 0 for an empty
// tile, whose table's first entry exists -- so its page is one the table
// holds. align > 1 clamps between the aligned tiles holding lo and hi-1, so
// an aligned t stays aligned.
func (p prefillSpan) clamp(b *ir.Builder, t ir.Value, align int) ir.Value {
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	lo, last := p.lo, b.Sub(ir.U32, b.Max(ir.U32, p.hi, u(1)), u(1))
	if align > 1 {
		lo = b.Mul(ir.U32, udiv(b, lo, align), u(align))
		last = b.Mul(ir.U32, udiv(b, last, align), u(align))
	}
	return b.Select(ir.U32, p.live, b.Min(ir.U32, b.Max(ir.U32, t, lo), last), u(0))
}

// in is whether key t is one of the tile's keys: its V is written there.
func (p prefillSpan) in(b *ir.Builder, t ir.Value) ir.Value {
	if pagedFault == "vmask" {
		return b.Lt(ir.U32, b.Const(ir.U32, 0), b.Const(ir.U32, 1))
	}
	return b.Lt(ir.U32, b.Sub(ir.U32, t, p.lo), b.Sub(ir.U32, p.hi, p.lo))
}

// rowAttends is whether row `row` (descriptor words ks, ke) attends key t.
func rowAttends(b *ir.Builder, t, ks, ke ir.Value) ir.Value {
	return b.Lt(ir.U32, b.Sub(ir.U32, t, ks), b.Sub(ir.U32, ke, ks))
}

// softcapped is FlashAttention's per-score transform: the scale, then the
// attention softcap when there is one.
func softcapped(b *ir.Builder, dot ir.Value, s FlashShape) ir.Value {
	x := b.Mul(ir.F32, dot, b.ConstF32(s.Scale))
	if s.Softcap > 0 {
		e := b.Exp(b.Mul(ir.F32, b.ConstF32(2/s.Softcap), x))
		x = b.Mul(ir.F32, b.ConstF32(s.Softcap), b.Sub(ir.F32, b.ConstF32(1), b.Div(ir.F32, b.ConstF32(2), b.Add(ir.F32, e, b.ConstF32(1)))))
	}
	return x
}

// PagedAttnScoresTiledThreads is how many threads PagedAttnScoresTiled takes
// (groups of 128).
func PagedAttnScoresTiledThreads(s FlashShape, qt, kt int) int {
	return s.Rows / qt * s.Heads * splitsOf(s) * s.Chunk / kt
}

// PagedAttnScoresTiled is AttnScoresTiledW over paged KV: a thread carries qt
// query rows and kt slots, strided (slot j, j+G, ... with G the slot count
// over kt) so neighbouring threads read neighbouring keys, and qt+kt loads
// serve qt*kt products. Each key's page is looked up on its own. Row-major K
// when s.MLA > 0 (the MLA region), else a page's transposed K. The score of a
// slot the row does not attend is -inf. Parameters: pQ, pK, pN (not read),
// pOut (the plane), pTab, pRow.
func PagedAttnScoresTiled(s FlashShape, qt, kt int) (*ir.Kernel, error) {
	if err := prefillCheck("PagedAttnScoresTiled", s); err != nil {
		return nil, err
	}
	P, W := s.Page, splitsOf(s)*s.Chunk
	if qt < 1 || kt < 1 || s.Rows%qt != 0 || W%kt != 0 {
		return nil, fmt.Errorf("kernels: PagedAttnScoresTiled: qt=%d kt=%d does not tile rows=%d and %d slots", qt, kt, s.Rows, W)
	}
	G := W / kt
	kvRow, gqa := s.KVHeads*s.Dim, s.Heads/s.KVHeads
	name := "pagedscorestiled"
	if s.Map {
		name += "_map"
	}
	b := ir.New(name, [3]int{128, 1, 1})
	pQ, pK := b.Param("pQ", ir.F32), b.Param("pK", ir.F32)
	b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	pTab, pRow := b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	rm := newRowMap(b, s)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	add := func(a, c ir.Value) ir.Value { return b.Add(ir.U32, a, c) }
	mul := func(a, c ir.Value) ir.Value { return b.Mul(ir.U32, a, c) }
	flat := b.Min(ir.U32, add(mul(b.CTAID(), b.NTID()), b.TID()), u(s.Rows/qt*s.Heads*G-1))
	j := urem(b, flat, G)
	rest := udiv(b, flat, G)
	h := urem(b, rest, s.Heads)
	qrow := mul(udiv(b, rest, s.Heads), u(qt))
	sp := rm.span(b, pRow, qrow, add(qrow, u(qt-1)))
	kvh := mul(udiv(b, h, gqa), u(s.Dim))
	slots := make([]ir.Value, kt)
	kbase := make([]ir.Value, kt)
	step := int64(P)
	if s.MLA > 0 {
		step = 1
	}
	for x := range slots {
		slots[x] = add(j, u(x*G))
		t := sp.clamp(b, add(sp.base, slots[x]), 1)
		pid := pageOf(b, pTab, sp.tab, t, P)
		if s.MLA > 0 {
			kbase[x] = add(mul(add(mul(pid, u(P)), pageRem(b, t, P)), u(kvRow)), kvh)
		} else {
			kbase[x] = add(add(mul(pid, u(P*kvRow)), mul(kvh, u(P))), pageRem(b, t, P))
		}
	}
	qbase := make([]ir.Value, qt)
	for r := range qbase {
		qbase[r] = mul(add(mul(rm.row(b, add(qrow, u(r))), u(s.Heads)), h), u(s.Dim))
	}
	accs := make([]ir.Value, qt*kt)
	for i := range accs {
		accs[i] = b.ConstF32(0)
	}
	for i := 0; i < s.Dim; i++ {
		qv := make([]ir.Value, qt)
		for r := range qv {
			qv[r] = b.Load(ir.F32, pQ, qbase[r], int64(i))
		}
		for x := range kbase {
			kv := b.Load(ir.F32, pK, kbase[x], int64(i)*step)
			for r := range qv {
				accs[r*kt+x] = b.Fma(qv[r], kv, accs[r*kt+x])
			}
		}
	}
	negInf := b.Bitcast(ir.F32, u(0xFF800000))
	for r := 0; r < qt; r++ {
		row := add(qrow, u(r))
		dr := rm.row(b, row)
		ks, ke := pagedDesc(b, pRow, dr, PRowStart), pagedDesc(b, pRow, dr, PRowEnd)
		ob := mul(add(mul(row, u(s.Heads)), h), u(W))
		for x := range slots {
			ok := rowAttends(b, add(sp.base, slots[x]), ks, ke)
			b.Store(pOut, add(ob, slots[x]), b.Select(ir.F32, ok, softcapped(b, accs[r*kt+x], s), negInf), 0)
		}
	}
	return b.Done(), nil
}

// splitsOf is how many splits a staged prefill shape's partition has: its
// Splits, or one for Splits 0 (the direct form, below).
func splitsOf(s FlashShape) int { return max(1, s.Splits) }

// PagedPrefillSoftmaxGroups is PagedPrefillSoftmax's launch: a warp a group
// at lanes 32, 64 threads an item each at lanes 1 (PagedPrefillSoftmaxWidth).
func PagedPrefillSoftmaxGroups(s FlashShape, lanes int) int {
	items := s.Rows * s.Heads * splitsOf(s)
	if lanes == 32 {
		return items
	}
	return (items + 63) / 64
}

// PagedPrefillSoftmaxWidth is PagedPrefillSoftmax's workgroup width.
func PagedPrefillSoftmaxWidth(lanes int) int {
	if lanes == 32 {
		return 32
	}
	return 64
}

// PagedPrefillSoftmax reduces each (row, head, split) over the slots of the
// split the row attends: its maximum m and denominator l go to the merge's
// partials (FlashAttentionMerge's layout at the value width), and
// exp(score - m) to pOut. The sink is the merge's.
//
// Weights are written only where the accumulate reads them: over the part of
// the chunk the row's query tile of qt rows attends (the accumulate's tile:
// PagedAttnAccTiled's qt, 8*nt for PagedAttnAccMMA70), widened to whole
// blocks of 8. Writing the whole chunk cost the short-context rows twice what
// they attend. A slot there the row does not attend holds -inf from the
// scores kernel and weighs exactly 0 -- which needs every slot of the span
// written, so qt must divide the scores kernel's query tile (16 rows for
// PagedAttnScoresMMA, 8*nt for PagedAttnScoresMMA70; the FMA scores write
// every slot): a key tile the scores kernel skipped is one none of its rows,
// so none of the tile's rows, attends.
//
// Splits 0 is the direct form, one split covering every key: the weights
// are normalised here, the sink joining the maximum and the denominator as in
// SoftmaxRowsSink, nothing goes to the partials, and the accumulate writes
// the output -- no merge. lanes 32 is a warp per item on a device that
// guarantees the width, 1 a thread. Parameters: pA (the scores), pN (not
// read), pOut (the weights), pPart (not written in the direct form), pTab (not
// read), pRow, and pSink for a direct form with a sink.
func PagedPrefillSoftmax(s FlashShape, lanes, qt int) (*ir.Kernel, error) {
	if err := prefillCheck("PagedPrefillSoftmax", s); err != nil {
		return nil, err
	}
	if lanes != 1 && lanes != 32 || qt < 1 || s.Rows%qt != 0 {
		return nil, fmt.Errorf("kernels: PagedPrefillSoftmax: lanes=%d (1 or 32), qt=%d tiling %d rows", lanes, qt, s.Rows)
	}
	direct := s.Splits == 0
	C, S := s.Chunk, splitsOf(s)
	items := s.Rows * s.Heads * S
	name := "pagedprefillsoftmax"
	if s.Map {
		name += "_map"
	}
	b := ir.New(name, [3]int{PagedPrefillSoftmaxWidth(lanes), 1, 1})
	pA := b.Param("pA", ir.F32)
	b.Param("pN", ir.U32)
	pOut, pPart := b.Param("pOut", ir.F32), b.Param("pPart", ir.F32)
	b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	var pSink ir.Value
	if direct && s.Sink {
		pSink = b.Param("pSink", ir.F32)
	}
	rm := newRowMap(b, s)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	add := func(a, c ir.Value) ir.Value { return b.Add(ir.U32, a, c) }
	var item, lane, step ir.Value
	if lanes == 32 {
		item = b.Min(ir.U32, b.CTAID(), u(items-1))
		lane, step = urem(b, b.TID(), 32), u(32)
	} else {
		item = b.Min(ir.U32, add(b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), u(items-1))
		lane, step = u(0), u(1)
	}
	sp := urem(b, item, S)
	row := udiv(b, item, s.Heads*S)
	c0 := add(rm.base(b, pRow, row), b.Mul(ir.U32, sp, u(C)))
	cEnd := add(c0, u(C))
	in := func(t ir.Value) ir.Value { return b.Sub(ir.U32, b.Min(ir.U32, b.Max(ir.U32, t, c0), cEnd), c0) }
	dr := rm.row(b, row)
	a := in(pagedDesc(b, pRow, dr, PRowStart))
	e := b.Max(ir.U32, in(pagedDesc(b, pRow, dr, PRowEnd)), a)
	// The query tile's slots in this chunk, in whole blocks of 8: [z0, z1).
	first := b.Mul(ir.U32, udiv(b, row, qt), u(qt))
	ts := in(pagedDesc(b, pRow, rm.row(b, first), PRowStart))
	te := b.Max(ir.U32, in(pagedDesc(b, pRow, rm.row(b, add(first, u(qt-1))), PRowEnd)), ts)
	z0 := b.Mul(ir.U32, udiv(b, ts, 8), u(8))
	z1 := b.Min(ir.U32, b.Mul(ir.U32, udiv(b, add(te, u(7)), 8), u(8)), u(C))
	z1 = b.Select(ir.U32, b.Lt(ir.U32, ts, te), z1, z0)
	pb := b.Mul(ir.U32, item, u(C))
	// count is how many of [lo, hi) this lane takes: lo+lane, lo+lane+step, ...
	count := func(lo, hi ir.Value) ir.Value {
		n := b.Sub(ir.U32, hi, lo)
		if lanes == 1 {
			return n
		}
		// ceil((n - lane)/32) as a shift: a warp is short at a short context
		// and a division is most of its setup.
		return b.Shr(ir.U32, add(b.Sub(ir.U32, b.Max(ir.U32, n, lane), lane), u(31)), u(5))
	}
	minus := b.ConstF32(-math.MaxFloat32)
	zero := b.ConstF32(0)
	first0 := add(add(pb, a), lane)
	b.LoopN(count(a, e))
	mx := b.Phi(ir.F32, minus)
	mi := b.Phi(ir.U32, first0)
	b.SetPhi(mx, b.Max(ir.F32, mx, b.Load(ir.F32, pA, mi, 0)))
	b.SetPhi(mi, add(mi, step))
	b.EndLoop()
	m := mx
	if lanes == 32 {
		m = butterfly(b, mx, func(x, y ir.Value) ir.Value { return b.Max(ir.F32, x, y) })
	}
	if direct {
		// The sum first, then the normalised weights.
		b.LoopN(count(a, e))
		sum := b.Phi(ir.F32, zero)
		si := b.Phi(ir.U32, first0)
		b.SetPhi(sum, b.Add(ir.F32, sum, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pA, si, 0), m))))
		b.SetPhi(si, add(si, step))
		b.EndLoop()
		l := sum
		if lanes == 32 {
			l = butterfly(b, sum, func(x, y ir.Value) ir.Value { return b.Add(ir.F32, x, y) })
		}
		mm := m
		if s.Sink {
			// The sink joins the maximum and the denominator; a row with no
			// key (m the stand-in) is then all sink.
			sv := b.Load(ir.F32, pSink, urem(b, urem(b, item, s.Heads*S), s.Heads), 0)
			mm = b.Max(ir.F32, m, sv)
			l = b.Add(ir.F32, b.Mul(ir.F32, l, b.Exp(b.Sub(ir.F32, m, mm))), b.Exp(b.Sub(ir.F32, sv, mm)))
		}
		inv := b.Div(ir.F32, b.ConstF32(1), b.Max(ir.F32, l, b.ConstF32(1e-30)))
		startW := add(add(pb, z0), lane)
		b.LoopN(count(z0, z1))
		wi := b.Phi(ir.U32, startW)
		b.Store(pOut, wi, b.Mul(ir.F32, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pA, wi, 0), mm)), inv), 0)
		b.SetPhi(wi, add(wi, step))
		b.EndLoop()
		return b.Done(), nil
	}
	startW := add(add(pb, z0), lane)
	b.LoopN(count(z0, z1))
	sum := b.Phi(ir.F32, zero)
	si := b.Phi(ir.U32, startW)
	w := b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pA, si, 0), m))
	b.Store(pOut, si, w, 0)
	b.SetPhi(sum, b.Add(ir.F32, sum, w))
	b.SetPhi(si, add(si, step))
	b.EndLoop()
	l := sum
	if lanes == 32 {
		l = butterfly(b, sum, func(x, y ir.Value) ir.Value { return b.Add(ir.F32, x, y) })
	}
	padded := (stagedValue(s) + 31) / 32 * 32
	pp := b.Mul(ir.U32, item, u(padded+64))
	if lanes == 32 {
		b.Store(pPart, add(pp, lane), m, int64(padded))
		b.Store(pPart, add(pp, lane), l, int64(padded+32))
	} else {
		for r := 0; r < 32; r++ {
			b.Store(pPart, pp, m, int64(padded+r))
			b.Store(pPart, pp, l, int64(padded+32+r))
		}
	}
	return b.Done(), nil
}

// PagedAttnAccTiledThreads is how many threads PagedAttnAccTiled takes
// (groups of 128).
func PagedAttnAccTiledThreads(s FlashShape, qt int) int {
	per := stagedValue(s)
	if s.F16 {
		per /= 2
	}
	return s.Rows / qt * s.Heads * splitsOf(s) * per
}

// PagedAttnAccTiled is AttnAccTiled over paged KV, one split at a time: a
// thread per (query tile of qt rows, head, split, value element) -- a word of
// two for packed f16 -- walks the keys of its split the tile attends, so one
// V load serves qt rows -- the qt PagedPrefillSoftmax was built with. Keys go
// in blocks of eight from a multiple of eight, which lie in one page: a
// block's page id is read once, a block ahead. The first and the last block
// are masked to the split's part of the tile's keys, V and weights both (a
// V there may be a page's unwritten tail, a weight a slot the softmax did
// not write); the blocks between are wholly inside, where every row's
// weight is written, 0 where it does not attend. It writes the numerators
// into the merge's partials, or in the direct form (Splits 0) the output,
// [row][head][value]. Parameters: pA (the weights), pV, pN (not read), pPart
// (the output in the direct form), pTab, pRow.
func PagedAttnAccTiled(s FlashShape, qt int) (*ir.Kernel, error) {
	if err := prefillCheck("PagedAttnAccTiled", s); err != nil {
		return nil, err
	}
	if qt < 1 || s.Rows%qt != 0 {
		return nil, fmt.Errorf("kernels: PagedAttnAccTiled: qt=%d does not tile rows=%d", qt, s.Rows)
	}
	const blk = 8
	P, C, S := s.Page, s.Chunk, splitsOf(s)
	kvRow, gqa := s.KVHeads*s.Dim, s.Heads/s.KVHeads
	vd := stagedValue(s)
	per := vd
	if s.F16 {
		per = vd / 2
	}
	name := "pagedacctiled"
	if s.Map {
		name += "_map"
	}
	b := ir.New(name, [3]int{128, 1, 1})
	pA, pV := b.Param("pA", ir.F32), b.Param("pV", ir.F32)
	b.Param("pN", ir.U32)
	pPart := b.Param("pPart", ir.F32)
	pTab, pRow := b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	rm := newRowMap(b, s)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	add := func(a, c ir.Value) ir.Value { return b.Add(ir.U32, a, c) }
	mul := func(a, c ir.Value) ir.Value { return b.Mul(ir.U32, a, c) }
	flat := b.Min(ir.U32, add(mul(b.CTAID(), b.NTID()), b.TID()), u(s.Rows/qt*s.Heads*S*per-1))
	j := urem(b, flat, per)
	rest := udiv(b, flat, per)
	spl := urem(b, rest, S)
	rest = udiv(b, rest, S)
	h := urem(b, rest, s.Heads)
	qrow := mul(udiv(b, rest, s.Heads), u(qt))
	sp := rm.span(b, pRow, qrow, add(qrow, u(qt-1)))
	// The split's part of the tile's keys, [st, en), walked from the block
	// holding st.
	c0 := add(sp.base, mul(spl, u(C)))
	st := b.Min(ir.U32, b.Max(ir.U32, sp.lo, c0), add(c0, u(C)))
	en := b.Max(ir.U32, b.Min(ir.U32, sp.hi, add(c0, u(C))), st)
	a8 := add(c0, mul(udiv(b, b.Sub(ir.U32, st, c0), blk), u(blk)))
	blocks := udiv(b, add(b.Sub(ir.U32, en, a8), u(blk-1)), blk)
	blocks = b.Select(ir.U32, b.Lt(ir.U32, st, en), blocks, u(0))
	col := j
	if s.MLAValueTail {
		col = add(j, u(s.Dim-s.MLA))
	}
	if s.MLA == 0 {
		hb := mul(udiv(b, h, gqa), u(s.Dim))
		if s.F16 {
			col = add(b.Shr(ir.U32, hb, u(1)), j)
		} else {
			col = add(hb, j)
		}
	}
	rowW := kvRow
	if s.F16 {
		rowW = kvRow / 2
	}
	blockPage := func(t ir.Value) ir.Value { return pageOf(b, pTab, sp.tab, sp.clamp(b, t, blk), P) }
	zero := b.ConstF32(0)
	// Row qrow+r's weight of slot k: ((qrow+r)*Heads + h)*S*C + spl*C + k.
	wb := add(mul(add(mul(qrow, u(s.Heads)), h), u(S*C)), mul(spl, u(C)))
	nacc := qt
	if s.F16 {
		nacc = 2 * qt
	}
	// block adds the 8 keys from bs, in page pid, to acc. A masked block keeps
	// only the keys of [st, en) -- V and weight both selected, since past the
	// split the weights are slots the softmax did not write -- with its weight
	// slot clamped into the chunk; the blocks between are wholly inside.
	block := func(bs, pid ir.Value, acc []ir.Value, masked bool) []ir.Value {
		next := append([]ir.Value(nil), acc...)
		vb := add(mul(add(mul(pid, u(P)), pageRem(b, bs, P)), u(rowW)), col)
		slot := b.Sub(ir.U32, bs, c0)
		if masked {
			slot = b.Min(ir.U32, slot, u(C-blk))
		}
		wk := add(wb, slot)
		for kk := 0; kk < blk; kk++ {
			var live ir.Value
			if masked {
				live = b.Lt(ir.U32, b.Sub(ir.U32, add(bs, u(kk)), st), b.Sub(ir.U32, en, st))
				if pagedFault == "vmask" {
					live = sp.in(b, add(bs, u(kk)))
				}
			}
			raw := b.Load(ir.F32, pV, vb, int64(kk*rowW))
			var v0, v1 ir.Value
			if s.F16 {
				word := b.Bitcast(ir.U32, raw)
				v0, v1 = b.CvtF16H(word), b.CvtF16H(b.Shr(ir.U32, word, u(16)))
				if masked {
					v0, v1 = b.Select(ir.F32, live, v0, zero), b.Select(ir.F32, live, v1, zero)
				}
			} else {
				v0 = raw
				if masked {
					v0 = b.Select(ir.F32, live, raw, zero)
				}
			}
			for r := 0; r < qt; r++ {
				w := b.Load(ir.F32, pA, wk, int64(r*s.Heads*S*C+kk))
				if masked {
					w = b.Select(ir.F32, live, w, zero)
				}
				if s.F16 {
					next[2*r] = b.Fma(w, v0, next[2*r])
					next[2*r+1] = b.Fma(w, v1, next[2*r+1])
				} else {
					next[r] = b.Fma(w, v0, next[r])
				}
			}
		}
		return next
	}
	acc := make([]ir.Value, nacc)
	for i := range acc {
		acc[i] = zero
	}
	acc = block(a8, blockPage(a8), acc, true)
	// The blocks between the first and the last, each page id a block ahead.
	mid := b.Sub(ir.U32, b.Max(ir.U32, blocks, u(2)), u(2))
	b1 := add(a8, u(blk))
	pid0 := blockPage(b1)
	b.LoopN(mid)
	ph := make([]ir.Value, nacc)
	for i := range ph {
		ph[i] = b.Phi(ir.F32, acc[i])
	}
	bs := b.Phi(ir.U32, b1)
	pid := b.Phi(ir.U32, pid0)
	next := block(bs, pid, ph, false)
	for i := range ph {
		b.SetPhi(ph[i], next[i])
	}
	b.SetPhi(pid, blockPage(add(bs, u(blk))))
	b.SetPhi(bs, add(bs, u(blk)))
	b.EndLoop()
	// The last block, masked; from the phis, which hold their inits when the
	// loop runs no block. With one block or none it is past the keys and adds 0.
	last := add(a8, mul(add(mid, u(1)), u(blk)))
	acc = block(last, blockPage(last), ph, true)
	// The numerators into the partials, row qrow+r's at
	// (((qrow+r)*Heads + h)*S + spl)*(padded+64); in the direct form the
	// output, [row][head][value].
	padded := (vd + 31) / 32 * 32
	pb := mul(add(mul(add(mul(qrow, u(s.Heads)), h), u(S)), spl), u(padded+64))
	rstep := int64(s.Heads * S * (padded + 64))
	if s.Splits == 0 {
		pb, rstep = mul(add(mul(qrow, u(s.Heads)), h), u(vd)), int64(s.Heads*vd)
	}
	for r := 0; r < qt; r++ {
		o := int64(r) * rstep
		at := pb
		if rm.on {
			// The direct form's output, at the slot's row.
			at, o = mul(add(mul(rm.row(b, add(qrow, u(r))), u(s.Heads)), h), u(vd)), 0
		}
		if s.F16 {
			b.Store(pPart, add(at, mul(j, u(2))), acc[2*r], o)
			b.Store(pPart, add(at, mul(j, u(2))), acc[2*r+1], o+1)
		} else {
			b.Store(pPart, add(at, j), acc[r], o)
		}
	}
	return b.Done(), nil
}

// PagedAttnScoresMMAWarps is how many warps PagedAttnScoresMMA launches
// (groups of 128 threads).
func PagedAttnScoresMMAWarps(s FlashShape, nt int) int {
	return s.Rows / 16 * s.Heads * (splitsOf(s) * s.Chunk / (8 * nt))
}

// PagedAttnScoresMMA is AttnScoresMMA over paged KV: the warp matrix
// instruction on 16 query rows by 8*nt keys, A the queries and B a page's
// transposed K, which is the column-major operand as it stands -- the page
// stride takes the cache's. The key tile lies in one page and reads its id
// once. A tile none of the warp's rows attends is not computed (its slots are
// never read: the softmax reads only the slots a row attends). Operands
// binary16, sums float32, the scale (and softcap) on the result.
// Parameters: pQ, pK, pN (not read), pOut (the plane), pTab, pRow.
func PagedAttnScoresMMA(s FlashShape, nt int) (*ir.Kernel, error) {
	if err := prefillCheck("PagedAttnScoresMMA", s); err != nil {
		return nil, err
	}
	W := splitsOf(s) * s.Chunk
	if s.MLA > 0 || s.Dim%16 != 0 || s.Rows%16 != 0 || nt < 1 || 64%(8*nt) != 0 {
		return nil, fmt.Errorf("kernels: PagedAttnScoresMMA: transposed K, Dim and Rows multiples of 16, "+
			"8*nt dividing 64: %+v nt=%d", s, nt)
	}
	P, kvRow, gqa := s.Page, s.KVHeads*s.Dim, s.Heads/s.KVHeads
	sh := ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}
	na, nb, nc := sh.Frags()
	b := ir.New("pagedscoresmma", [3]int{128, 1, 1})
	pQ, pK := b.Param("pQ", ir.F32), b.Param("pK", ir.F32)
	b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	pTab, pRow := b.Param("pTab", ir.U32), b.Param("pRow", ir.U32)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	add := func(a, c ir.Value) ir.Value { return b.Add(ir.U32, a, c) }
	mul := func(a, c ir.Value) ir.Value { return b.Mul(ir.U32, a, c) }
	ktiles := W / (8 * nt)
	tid := add(mul(b.CTAID(), b.NTID()), b.TID())
	lane := b.And(ir.U32, tid, u(31))
	g := b.Shr(ir.U32, lane, u(2))
	two := mul(b.And(ir.U32, lane, u(3)), u(2))
	warp := b.Min(ir.U32, udiv(b, tid, 32), u(ktiles*s.Heads*(s.Rows/16)-1))
	// The key tile varies fastest, so neighbouring warps read neighbouring keys.
	kt := urem(b, warp, ktiles)
	rest := udiv(b, warp, ktiles)
	h := urem(b, rest, s.Heads)
	qrow := mul(udiv(b, rest, s.Heads), u(16))
	sp := prefillSpanOf(b, pRow, qrow, add(qrow, u(15)))
	// A's rows: query rows qrow+g and qrow+g+8, elements 2t (and +8).
	qbase := add(mul(add(mul(add(qrow, g), u(s.Heads)), h), u(s.Dim)), two)
	// B's columns: keys T+g (+8j) of the tile at T, elements 2t (and +8).
	kvh := mul(udiv(b, h, gqa), u(s.Dim))
	slot0 := mul(kt, u(8*nt))
	T := add(sp.base, slot0)
	Tc := sp.clamp(b, T, 8*nt)
	pid := pageOf(b, pTab, sp.tab, Tc, P)
	kbase := add(add(add(mul(pid, u(P*kvRow)), mul(add(kvh, two), u(P))), pageRem(b, Tc, P)), g)
	rows := [2]ir.Value{add(qrow, g), add(qrow, add(g, u(8)))}
	var ks, ke [2]ir.Value
	for i, r := range rows {
		ks[i], ke[i] = pagedDesc(b, pRow, r, PRowStart), pagedDesc(b, pRow, r, PRowEnd)
	}
	live := b.Select(ir.U32, b.Lt(ir.U32, T, sp.hi), b.Select(ir.U32, b.Lt(ir.U32, sp.lo, add(T, u(8*nt))), u(1), u(0)), u(0))
	b.LoopN(live)
	zero := b.ConstF32(0)
	accs := make([][]ir.Value, nt)
	for j := range accs {
		accs[j] = make([]ir.Value, nc)
		for c := range accs[j] {
			accs[j][c] = zero
		}
	}
	for kk := 0; kk < s.Dim; kk += 16 {
		af := make([]ir.Value, na)
		for i := range af {
			off := int64(kk)
			if i >= 2 {
				off += 8
			}
			if i%2 == 1 {
				off += 8 * int64(s.Heads*s.Dim)
			}
			af[i] = b.PackF16(b.Load(ir.F32, pQ, qbase, off), b.Load(ir.F32, pQ, qbase, off+1))
		}
		for j := 0; j < nt; j++ {
			bf := make([]ir.Value, nb)
			for i := range bf {
				e := int64(kk)
				if i >= 1 {
					e += 8
				}
				o := e*int64(P) + int64(j*8)
				bf[i] = b.PackF16(b.Load(ir.F32, pK, kbase, o), b.Load(ir.F32, pK, kbase, o+int64(P)))
			}
			accs[j] = b.MMA(sh, af, bf, accs[j])
		}
	}
	negInf := b.Bitcast(ir.F32, u(0xFF800000))
	for j := 0; j < nt; j++ {
		for c := 0; c < nc; c++ {
			// D component c: row g (+8 when c >= 2), key column 8j + 2t + c%2.
			ri := c / 2
			slot := add(add(slot0, two), u(j*8+c%2))
			ok := rowAttends(b, add(sp.base, slot), ks[ri], ke[ri])
			out := add(mul(add(mul(rows[ri], u(s.Heads)), h), u(W)), slot)
			b.Store(pOut, out, b.Select(ir.F32, ok, softcapped(b, accs[j][c], s), negInf), 0)
		}
	}
	b.EndLoop()
	return b.Done(), nil
}
