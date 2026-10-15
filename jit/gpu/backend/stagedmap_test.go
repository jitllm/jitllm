package backend_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestStagedPrefillRunTable holds the staged prefill's run-table form
// (FlashShape.Map) to each sequence run alone through the plain form, bit
// for bit, as TestFlashPrefillRunTable holds the flash forms: two sequences
// with their own tables, histories and partition bases (pBase), their rows
// scattered among each other and among decoding rows the table does not
// name, each padded to the forms' query tile with its last row; the decoding
// rows untouched. Every matrix instruction the staged form takes -- the FMA
// tiles, m16n8k16 scores with the FMA and the m16n8k8 accumulate, sm_70's
// m8n8k4 pair -- in the direct form and in passes (a pass of Splits*Chunk
// keys, each run's pass j from its own base, the partials at the rows folded
// by FlashAttentionMergeRun and the mapped merge writing the rows), where
// the two sequences need different numbers of passes. The violation -- B's
// base one block past its first key -- must part, and it is checked to move
// B's own arithmetic, not only a row nothing reads.
func TestStagedPrefillRunTable(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	for _, d := range devs {
		defer d.Close()
	}
	type formSet struct {
		sh    *ir.MMAShape
		forms []prefillForm
	}
	volta, m16 := ir.MMAVolta, ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}
	sets := []formSet{
		{nil, []prefillForm{
			{name: "q2k2", scoresQT: 2, scoresKT: 2, accQT: 2, smLanes: 32},
			{name: "q4k2", scoresQT: 4, scoresKT: 2, accQT: 4, smLanes: 1},
		}},
		{&m16, []prefillForm{
			{name: "mma2", mmaNT: 2, accQT: 8, smLanes: 32},
			{name: "mma2acc2x2", mmaNT: 2, accMMA: [2]int{2, 2}, smLanes: 32},
		}},
		{&volta, []prefillForm{
			{name: "v2x4", voltaMT: 2, voltaNT: 4, voltaAccMT: 1, smLanes: 1}, // the tier's
		}},
	}
	ran := 0
	for _, set := range sets {
		on := devs
		if set.sh != nil {
			on = mmaDevices(t, devs, *set.sh)
		}
		for _, d := range on {
			for _, sh := range []kernels.FlashShape{
				{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: 64},
				{Heads: 8, KVHeads: 4, Dim: 128, Scale: .088, Page: 64, F16: true, Softcap: 50},
				{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: 64, Sink: true},
			} {
				for _, f := range set.forms {
					for _, passW := range []int{0, 128} {
						t.Run(fmt.Sprintf("%s/%s/%s/d%d/sink%v/pass%d", d.API(), d.Name(), f.name, sh.Dim, sh.Sink, passW), func(t *testing.T) {
							stagedTableCase(t, d, sh, f, passW)
						})
						ran++
					}
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no run-table case ran")
	}
}

// formTile is the query tile every kernel of f divides: the scores' and the
// accumulate's.
func (f prefillForm) formTile() int {
	sq := f.scoresQT
	switch {
	case f.mmaNT > 0:
		sq = 16
	case f.voltaMT > 0:
		sq = 8 * f.voltaNT
	}
	a, b := sq, f.accTile()
	x, y := a, b
	for y != 0 {
		x, y = y, x%y
	}
	return a / x * b
}

// stagePass is one launch of a staged prefill: its descriptors and, with a
// run table, its slots' partition bases.
type stagePass struct{ desc, base []uint32 }

// passesOf is a run's passes of width keys from base: each row's keys
// clipped to the pass. A width of 0 is the direct form, one launch.
func passesOf(desc []uint32, base, hi, width int) [][]uint32 {
	if width == 0 {
		return [][]uint32{desc}
	}
	var out [][]uint32
	for at := base; at < hi; at += width {
		out = append(out, passDesc(desc, at, at+width))
	}
	return out
}

func stagedTableCase(t *testing.T, d backend.Device, s kernels.FlashShape, f prefillForm, passW int) {
	t.Helper()
	wg := f.formTile()
	pg := s.Page
	up := func(n int) int { return (n + wg - 1) / wg * wg }
	ca, cb := prefillCase{"a", 3*pg + 17, 40, up(40), 0}, prefillCase{"b", 2*pg + 5, 23, up(23), pg/2 + 7}
	ra, rb := prefillRows(ca), prefillRows(cb)
	kvRow := s.KVHeads * s.Dim
	p := newPrefillPool(rand.New(rand.NewSource(int64(s.Dim))), pg, kvRow, s.F16, false, false, ra)
	tabB := append([]uint32(nil), p.tab...)
	tabB[0], tabB[1] = tabB[1], tabB[0]
	descOf := func(rows []pagedRow, tabOff int) []uint32 {
		var out []uint32
		for _, r := range rows {
			out = append(out, uint32(tabOff), uint32(r.ks), uint32(r.ke), uint32(r.wp))
		}
		return out
	}
	per := s.Heads * s.Dim
	qOf := func(seed int64, n int) []float32 {
		rng := rand.New(rand.NewSource(seed))
		q := make([]float32, n*per)
		for i := range q {
			q[i] = float32(rng.NormFloat64())
		}
		return q
	}
	qa, qb := qOf(11, ca.rows), qOf(12, cb.rows)
	baseOf := func(rows []pagedRow) int { return rows[0].ks / 64 * 64 }
	hiOf := func(rows []pagedRow) int {
		h := 0
		for _, r := range rows {
			h = max(h, r.ke)
		}
		return h
	}
	if passW == 0 {
		// One chunk covering either sequence's span from its own base.
		s.Chunk = max(directChunk(ra), directChunk(rb))
	} else {
		s.Chunk, s.Splits = passW/2, 2 // B needs one pass, A two
	}
	alone := func(q []float32, tab []uint32, rows []pagedRow, n int, need bool) []float64 {
		var ps []stagePass
		for _, dsc := range passesOf(descOf(rows, 0), baseOf(rows), hiOf(rows), passW) {
			ps = append(ps, stagePass{desc: dsc})
		}
		if passW > 0 && need && len(ps) < 2 {
			t.Fatalf("%d pass: A does not need passes", len(ps))
		}
		return runStaged(t, d, s, f, p, q, tab, ps, nil, n)
	}
	refA := alone(qa, p.tab, ra, ca.rows, true)
	refB := alone(qb, tabB, rb, cb.rows, false)

	type src struct{ seq, r int }
	var phys []src
	for r := 0; r < ca.nrow; r++ {
		phys = append(phys, src{0, r})
	}
	for r := 0; r < cb.nrow; r++ {
		phys = append(phys, src{1, r})
	}
	for r := 0; r < 5; r++ {
		phys = append(phys, src{2, r})
	}
	rand.New(rand.NewSource(5)).Shuffle(len(phys), func(i, j int) { phys[i], phys[j] = phys[j], phys[i] })
	at := map[src]int{}
	tab := append(append([]uint32(nil), p.tab...), tabB...)
	q := make([]float32, len(phys)*per)
	var desc []uint32
	for i, s := range phys {
		at[s] = i
		switch s.seq {
		case 0:
			copy(q[i*per:], qa[s.r*per:(s.r+1)*per])
			desc = append(desc, descOf(ra[s.r:s.r+1], 0)...)
		case 1:
			copy(q[i*per:], qb[s.r*per:(s.r+1)*per])
			desc = append(desc, descOf(rb[s.r:s.r+1], len(p.tab))...)
		default:
			desc = append(desc, 0, 0, 0, 0)
		}
	}
	var mp []uint32
	for seq, n := range []int{ca.nrow, cb.nrow} {
		for j := 0; j < up(n); j++ {
			mp = append(mp, uint32(at[src{seq, min(j, n-1)}]))
		}
	}
	// The step's passes: as many as the longer run needs, each run's pass
	// j from its own base (the shorter run's last passes empty).
	np := 1
	if passW > 0 {
		np = max((hiOf(ra)-baseOf(ra)+passW-1)/passW, (hiOf(rb)-baseOf(rb)+passW-1)/passW)
	}
	step := func(bShift int) []float64 {
		var ps []stagePass
		for j := 0; j < np; j++ {
			sp := stagePass{desc: append([]uint32(nil), desc...)}
			for slot, r := range mp {
				ab := baseOf(ra)
				if slot >= up(ca.nrow) {
					ab = baseOf(rb)
				}
				lo := ab + j*passW
				if passW > 0 {
					o := int(r) * kernels.PRowWords
					for _, fld := range []int{kernels.PRowStart, kernels.PRowEnd} {
						sp.desc[o+fld] = uint32(min(max(int(sp.desc[o+fld]), lo), lo+passW))
					}
				}
				if slot >= up(ca.nrow) {
					lo += bShift
				}
				sp.base = append(sp.base, uint32(lo))
			}
			ps = append(ps, sp)
		}
		return runStaged(t, d, s, f, p, q, tab, ps, mp, len(phys))
	}
	got := step(0)
	same := func(ref, got []float64, r, i int) bool {
		for e := 0; e < per; e++ {
			if math.Float64bits(ref[r*per+e]) != math.Float64bits(got[i*per+e]) {
				return false
			}
		}
		return true
	}
	for _, sr := range phys {
		i := at[sr]
		switch sr.seq {
		case 0:
			if !same(refA, got, sr.r, i) {
				t.Fatalf("A's row %d (step row %d) differs from A alone", sr.r, i)
			}
		case 1:
			if !same(refB, got, sr.r, i) {
				t.Fatalf("B's row %d (step row %d) differs from B alone", sr.r, i)
			}
		default:
			for e := 0; e < per; e++ {
				if !math.IsNaN(got[i*per+e]) {
					t.Fatalf("decoding row %d was written: the table names it nowhere", i)
				}
			}
		}
	}
	// The violation: B's base a block past its first key, so its first keys
	// fall outside its partition. It must part B, and leave A as it was --
	// so it moved B's own arithmetic through the base table, not something
	// shared.
	bad := step(64)
	parted := false
	for r := 0; r < cb.nrow && !parted; r++ {
		parted = !same(refB, bad, r, at[src{1, r}])
	}
	if !parted {
		t.Fatal("B's base moved past its first key and still matched B alone: the base table is not read")
	}
	for r := 0; r < ca.nrow; r++ {
		if !same(refA, bad, r, at[src{0, r}]) {
			t.Fatalf("B's base moved and A's row %d changed: the violation is not B's alone", r)
		}
	}
	t.Logf("%d launches a step; violation parted B and left A", np)
}

// runStaged runs the staged form f over rows rows of q, one launch per
// pass (direct when s.Splits is 0, else each pass's partials folded by
// FlashAttentionMergeRun and finished by FlashAttentionMergeWide), the run
// table mp when set, and returns the rows' output, NaN where nothing wrote.
func runStaged(t *testing.T, d backend.Device, s kernels.FlashShape, f prefillForm, p *pagedPool,
	q []float32, tab []uint32, passes []stagePass, mp []uint32, rows int) []float64 {
	t.Helper()
	s.Map = mp != nil
	s.Rows = rows
	if s.Map {
		s.Rows = len(mp)
	}
	// The sink joins the direct form's softmax, or the passes' last merge.
	direct := s.Splits == 0
	sink := s.Sink
	s.Sink = sink && direct
	ms := kernels.FlashShape{Heads: s.Heads, KVHeads: s.KVHeads, Dim: s.Dim, Rows: rows, Splits: max(1, s.Splits)}
	one := ms
	one.Splits, one.Sink = 1, sink
	fin := one
	if s.Map {
		fin.Rows, fin.Map = len(mp), true
	}
	compile := func(k *ir.Kernel, err error) backend.Kernel {
		if err != nil {
			t.Fatal(err)
		}
		if ok, why := backend.GuaranteedLanes(d, k.Lanes); !ok {
			t.Skip(why)
		}
		c, err := d.Compile(k)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(c.Close)
		return c
	}
	var sk, sm, ak backend.Kernel
	var skG, akG int
	switch {
	case f.mmaNT > 0:
		sk = compile(kernels.PagedAttnScoresMMA(s, f.mmaNT))
		skG = (kernels.PagedAttnScoresMMAWarps(s, f.mmaNT) + 3) / 4
	case f.voltaMT > 0:
		sk = compile(kernels.PagedAttnScoresMMA70(s, f.voltaMT, f.voltaNT))
		skG = (kernels.PagedAttnScoresMMA70Warps(s, f.voltaMT, f.voltaNT) + 3) / 4
	default:
		sk = compile(kernels.PagedAttnScoresTiled(s, f.scoresQT, f.scoresKT))
		skG = (kernels.PagedAttnScoresTiledThreads(s, f.scoresQT, f.scoresKT) + 127) / 128
	}
	sm = compile(kernels.PagedPrefillSoftmax(s, f.smLanes, f.accTile()))
	switch {
	case f.voltaMT > 0:
		ak = compile(kernels.PagedAttnAccMMA70(s, f.accMT(), f.voltaNT))
		akG = (kernels.PagedAttnAccMMA70Warps(s, f.accMT(), f.voltaNT) + 3) / 4
	case f.accMMA[0] > 0:
		ak = compile(kernels.PagedAttnAccMMA(s, f.accMMA[0], f.accMMA[1]))
		akG = (kernels.PagedAttnAccMMAWarps(s, f.accMMA[0], f.accMMA[1]) + 3) / 4
	default:
		ak = compile(kernels.PagedAttnAccTiled(s, f.accQT))
		akG = (kernels.PagedAttnAccTiledThreads(s, f.accQT) + 127) / 128
	}
	var first, fold, fk backend.Kernel
	if !direct {
		first, fold = compile(kernels.FlashAttentionMergeRun(ms, true)), compile(kernels.FlashAttentionMergeRun(ms, false))
		fk = compile(kernels.FlashAttentionMergeWide(fin))
	}
	g := newGPU(t, d)
	defer g.free()
	_, sinks := prefillInputs(s)
	nOut := rows * s.Heads * s.Dim
	nan := func(n int) backend.Buf { return g.up(f32bytes(nanFill(n))) }
	qo, ko, vo, oo := g.up(f32bytes(q)), g.up(f32bytes(p.k)), g.up(p.vBytes()), nan(nOut+32)
	no := g.up(u32bytes([]uint32{0xFFFFFFFF, 0xFFFFFFFF}))
	to := g.up(u32bytes(tab))
	plane := kernels.PagedPrefillPlane(s)
	sc, pr := nan(plane), nan(plane)
	part, run := nan(8), [2]backend.Buf{}
	if !direct {
		part = nan(kernels.FlashPartialFloats(ms))
		run = [2]backend.Buf{nan(kernels.FlashPartialFloats(one)), nan(kernels.FlashPartialFloats(one))}
	}
	so := g.up(f32bytes(sinks))
	var ro []backend.Buf
	var tails [][]backend.Buf
	var mo backend.Buf
	if s.Map {
		mo = g.up(u32bytes(mp))
	}
	for _, ps := range passes {
		ro = append(ro, g.up(u32bytes(ps.desc)))
		var tail []backend.Buf
		if s.Map {
			tail = []backend.Buf{mo, g.up(u32bytes(ps.base))}
		}
		tails = append(tails, tail)
	}
	padded := (s.Dim + 31) / 32 * 32
	var err error
	d.Session(func(se backend.Session) {
		launch := func(c backend.Kernel, groups, width int, args ...backend.Buf) {
			if err == nil {
				err = se.Launch(c, groups, width, args...)
			}
		}
		for j, r := range ro {
			tail := tails[j]
			dst := oo
			if !direct {
				dst = part
			}
			launch(sk, skG, 128, append([]backend.Buf{qo, ko, no, sc, to, r}, tail...)...)
			smArgs := []backend.Buf{sc, no, pr, part, to, r}
			if s.Sink {
				smArgs = append(smArgs, so)
			}
			launch(sm, kernels.PagedPrefillSoftmaxGroups(s, f.smLanes), kernels.PagedPrefillSoftmaxWidth(f.smLanes), append(smArgs, tail...)...)
			launch(ak, akG, 128, append([]backend.Buf{pr, vo, no, dst, to, r}, tail...)...)
			if direct {
				continue
			}
			rg := (rows*s.Heads*padded + 127) / 128
			if j == 0 {
				launch(first, rg, 128, part, run[0])
			} else {
				launch(fold, rg, 128, part, run[(j+1)%2], run[j%2])
			}
		}
		if !direct {
			fa := []backend.Buf{run[(len(ro)+1)%2], oo}
			if sink {
				fa = append(fa, so)
			}
			if s.Map {
				fa = append(fa, mo)
			}
			launch(fk, (fin.Rows*s.Heads*s.Dim+127)/128, 128, fa...)
		}
		if err == nil {
			err = se.Sync()
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	got := readF32(t, oo, nOut+32)
	for i := nOut; i < len(got); i++ {
		if !math.IsNaN(got[i]) {
			t.Fatalf("output guard %d overwritten", i)
		}
	}
	return got[:nOut]
}
