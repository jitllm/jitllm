package backend_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// The gates for the paged PREFILL kernels (docs/design/device-kv-paging.md):
// the rows of one sequence's prompt chunk, each attending its own causal (and
// windowed or chunked) keys through one shuffled table. Every pool word the
// chunk does not attend is NaN -- the positions below its first key, the
// pages wholly below it (mapped to a poison page), every page's tail past the
// last key -- and so is every scratch plane and the output, so a read or a
// store outside the contract shows. Each path is held to the float64 oracle,
// to its contiguous twin on the same data where the twin exists, and fails
// through a wrong table and through descriptors one key short.

// prefillCase is one chunk: nrow real rows from position pos0, padded to rows,
// and the mask -- window > 0 a sliding window of that many keys, window < 0 an
// aligned chunk of -window keys (Llama 4), 0 causal.
type prefillCase struct {
	name                     string
	pos0, nrow, rows, window int
}

// prefillCases are the chunks every prefill gate runs: the first chunk of a
// prompt, a later one crossing page edges and ending mid-page, padded rows,
// a sliding window from mid-page, an aligned chunk window, and a chunk whose
// first key is its own first position.
func prefillCases(page, rows int) []prefillCase {
	return []prefillCase{
		{"first", 0, rows, rows, 0},
		{"later", 3*page + 17, rows, rows, 0},
		{"padded", page - 3, rows - 5, rows, 0},
		{"window", 2*page + 5, rows, rows, page/2 + 7},
		{"chunked", page + 30, rows, rows, -64},
		{"one", 0, 1, rows, 0},
	}
}

// prefillRows is the case's descriptors' keys: row r at position pos0+r
// attends [ks, pos0+r+1); a padded row is empty at the last real row's end.
func prefillRows(c prefillCase) []pagedRow {
	out := make([]pagedRow, c.rows)
	for r := range out {
		if r >= c.nrow {
			e := out[c.nrow-1].ke
			out[r] = pagedRow{e, e, e}
			continue
		}
		p := c.pos0 + r
		ks := 0
		switch {
		case c.window > 0:
			ks = max(0, p+1-c.window)
		case c.window < 0:
			ks = p / -c.window * -c.window
		}
		out[r] = pagedRow{ks, p + 1, p}
	}
	return out
}

// newPrefillPool lays one sequence's history into shuffled pages of a pool:
// the keys [ks of row 0, ke of the last row) are written, everything else is
// NaN, pages wholly below the first key map to a poison page, and the table
// is followed by a poison entry. Every row's descriptor names the same table.
// kh/vh give every row the sequence's history, for pagedWant.
func newPrefillPool(rng *rand.Rand, page, kvRow int, f16, mla, ordered bool, rows []pagedRow) *pagedPool {
	p := &pagedPool{page: page, kvRow: kvRow, f16: f16, mla: mla}
	lo, hi := rows[0].ks, 0
	for _, r := range rows {
		hi = max(hi, r.ke)
	}
	npg := (max(hi, 1) + page - 1) / page
	p.pages = npg + 3
	perm := rng.Perm(p.pages)
	if ordered {
		for i := range perm {
			perm[i] = (i + p.pages - 2) % p.pages
		}
	}
	p.dummy, p.poison = perm[0], perm[1]
	n := p.pages * page * kvRow
	p.k, p.v = make([]float32, n), make([]float32, n)
	for i := range p.k {
		p.k[i], p.v[i] = nan32, nan32
	}
	for t := 0; t < page; t++ {
		for e := 0; e < kvRow; e++ {
			p.k[p.kAt(p.dummy, t, e)], p.v[p.vAt(p.dummy, t, e)] = 0, 0
		}
	}
	for j := 0; j < npg; j++ {
		if (j+1)*page <= lo {
			p.tab = append(p.tab, uint32(p.poison))
			continue
		}
		p.tab = append(p.tab, uint32(perm[2+j]))
	}
	kh, vh := make([]float32, hi*kvRow), make([]float32, hi*kvRow)
	for t := lo; t < hi; t++ {
		pid := int(p.tab[t/page])
		for e := 0; e < kvRow; e++ {
			kv, vv := float32(rng.NormFloat64()), float32(rng.NormFloat64())
			if f16 {
				vv = float32(quant.DecodeHalf(quant.EncodeHalf(vv)))
			}
			if mla {
				vv = kv
			}
			kh[t*kvRow+e], vh[t*kvRow+e] = kv, vv
			p.k[p.kAt(pid, t, e)], p.v[p.vAt(pid, t, e)] = kv, vv
		}
	}
	p.tab = append(p.tab, uint32(p.poison))
	for _, r := range rows {
		p.kh, p.vh = append(p.kh, kh), append(p.vh, vh)
		p.desc = append(p.desc, 0, uint32(r.ks), uint32(r.ke), uint32(r.wp))
	}
	p.wrong = make([]uint32, len(p.tab))
	for i, x := range p.tab {
		p.wrong[i] = (x + 1) % uint32(p.pages)
	}
	return p
}

// prefillForm is one paged prefill path's kernels: the scores (or the one
// flash kernel), the softmax and the accumulate, with their launches.
type prefillForm struct {
	name       string
	scoresQT   int // PagedAttnScoresTiled's query and key tiles
	scoresKT   int
	mmaNT      int // PagedAttnScoresMMA's key tiles a warp, 0 for none
	voltaMT    int // PagedAttnScoresMMA70 and PagedAttnAccMMA70's tiles, 0 for neither
	voltaNT    int
	voltaAccMT int    // PagedAttnAccMMA70's own dim tile, 0 for voltaMT (the tier's is 1)
	accQT      int    // PagedAttnAccTiled's query tile
	accMMA     [2]int // PagedAttnAccMMA's dim and query tiles (mt, nt), zero for the tiled accumulate
	smLanes    int
	groupMerge bool // FlashAttentionMerge rather than its wide form
}

// pagedParams reports whether k takes the paged ABI's pTab and pRow, in that
// order, after its existing parameters (an optional sink may follow).
func pagedParams(k *ir.Kernel) bool {
	for i := 0; i+1 < len(k.Params); i++ {
		if k.Params[i].Name == "pTab" && k.Params[i+1].Name == "pRow" {
			return i >= 4
		}
	}
	return false
}

// accTile is the accumulate's query tile, which the softmax is built with.
func (f prefillForm) accTile() int {
	if f.voltaMT > 0 {
		return 8 * f.voltaNT
	}
	if f.accMMA[0] > 0 {
		return 8 * f.accMMA[1]
	}
	return f.accQT
}

// directChunk is the one chunk the direct form (Splits 0) needs: the
// launch's whole span, in multiples of 64.
func directChunk(rows []pagedRow) int {
	return (prefillSplits(rows, 64, 0)*64 + 63) / 64 * 64
}

// accMT is the Volta accumulate's dim tile.
func (f prefillForm) accMT() int {
	if f.voltaAccMT > 0 {
		return f.voltaAccMT
	}
	return f.voltaMT
}

// prefillSplits is the fewest splits of chunk keys that cover the case, plus
// extra.
func prefillSplits(rows []pagedRow, chunk, extra int) int {
	base, hi := rows[0].ks/64*64, 0
	for _, r := range rows {
		hi = max(hi, r.ke)
	}
	return max(1, (hi-base+chunk-1)/chunk) + extra
}

// pagedPrefillStaged runs one staged form over the pool through tab and desc
// and returns the output ([row][head][value]) and its NMSE against the
// oracle, or an error for a non-finite output or an overrun.
func pagedPrefillStaged(t *testing.T, d backend.Device, s kernels.FlashShape, f prefillForm, p *pagedPool, tab, desc []uint32, rows []pagedRow) ([]float64, float64, error) {
	t.Helper()
	s.Rows = len(rows)
	vd := s.Dim
	if s.MLA > 0 {
		vd = s.MLA
	}
	ms := s
	ms.Dim, ms.MLA, ms.Page, ms.Chunk = vd, 0, 0, 0
	var ks []*ir.Kernel
	build := func(k *ir.Kernel, err error) {
		if err != nil {
			t.Fatal(err)
		}
		ks = append(ks, k)
	}
	switch {
	case f.mmaNT > 0:
		build(kernels.PagedAttnScoresMMA(s, f.mmaNT))
	case f.voltaMT > 0:
		build(kernels.PagedAttnScoresMMA70(s, f.voltaMT, f.voltaNT))
	default:
		build(kernels.PagedAttnScoresTiled(s, f.scoresQT, f.scoresKT))
	}
	build(kernels.PagedPrefillSoftmax(s, f.smLanes, f.accTile()))
	switch {
	case f.voltaMT > 0:
		build(kernels.PagedAttnAccMMA70(s, f.accMT(), f.voltaNT))
	case f.accMMA[0] > 0:
		build(kernels.PagedAttnAccMMA(s, f.accMMA[0], f.accMMA[1]))
	default:
		build(kernels.PagedAttnAccTiled(s, f.accQT))
	}
	direct := s.Splits == 0 // the accumulate writes the output, no merge
	switch {
	case direct:
	case f.groupMerge:
		build(kernels.FlashAttentionMerge(ms))
	default:
		build(kernels.FlashAttentionMergeWide(ms))
	}
	var cs []backend.Kernel
	for i, k := range ks {
		// The kernels under test are the paged ones.
		if i < 3 && !pagedParams(k) {
			t.Fatalf("%s is not a paged kernel: %v", k.Name, k.Params)
		}
		if ok, why := backend.GuaranteedLanes(d, k.Lanes); !ok {
			t.Skip(why)
		}
		c, err := d.Compile(k)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		cs = append(cs, c)
	}
	g := newGPU(t, d)
	defer g.free()
	q, sinks := prefillInputs(s)
	nanBuf := func(n int) backend.Buf { return g.up(f32bytes(nanFill(n))) }
	nOut := s.Rows * s.Heads * vd
	qo, ko, oo := g.up(f32bytes(q)), g.up(f32bytes(p.k)), nanBuf(nOut+32)
	vo := ko
	if s.MLA == 0 {
		vo = g.up(p.vBytes())
	}
	no := g.up(u32bytes([]uint32{0xFFFFFFFF, 0xFFFFFFFF}))
	to, ro := g.up(u32bytes(tab)), g.up(u32bytes(desc))
	plane := kernels.PagedPrefillPlane(s)
	sc, pr := nanBuf(plane), nanBuf(plane)
	part := nanBuf(kernels.FlashPartialFloats(ms))
	so := g.up(f32bytes(sinks))
	var err error
	d.Session(func(se backend.Session) {
		launch := func(c backend.Kernel, groups, width int, args ...backend.Buf) {
			if err == nil {
				err = se.Launch(c, groups, width, args...)
			}
		}
		switch {
		case f.mmaNT > 0:
			launch(cs[0], (kernels.PagedAttnScoresMMAWarps(s, f.mmaNT)+3)/4, 128, qo, ko, no, sc, to, ro)
		case f.voltaMT > 0:
			launch(cs[0], (kernels.PagedAttnScoresMMA70Warps(s, f.voltaMT, f.voltaNT)+3)/4, 128, qo, ko, no, sc, to, ro)
		default:
			launch(cs[0], (kernels.PagedAttnScoresTiledThreads(s, f.scoresQT, f.scoresKT)+127)/128, 128, qo, ko, no, sc, to, ro)
		}
		if direct && s.Sink {
			launch(cs[1], kernels.PagedPrefillSoftmaxGroups(s, f.smLanes), kernels.PagedPrefillSoftmaxWidth(f.smLanes), sc, no, pr, part, to, ro, so)
		} else {
			launch(cs[1], kernels.PagedPrefillSoftmaxGroups(s, f.smLanes), kernels.PagedPrefillSoftmaxWidth(f.smLanes), sc, no, pr, part, to, ro)
		}
		dst := part
		if direct {
			dst = oo
		}
		switch {
		case f.voltaMT > 0:
			launch(cs[2], (kernels.PagedAttnAccMMA70Warps(s, f.accMT(), f.voltaNT)+3)/4, 128, pr, vo, no, dst, to, ro)
		case f.accMMA[0] > 0:
			launch(cs[2], (kernels.PagedAttnAccMMAWarps(s, f.accMMA[0], f.accMMA[1])+3)/4, 128, pr, vo, no, dst, to, ro)
		default:
			launch(cs[2], (kernels.PagedAttnAccTiledThreads(s, f.accQT)+127)/128, 128, pr, vo, no, dst, to, ro)
		}
		ma := []backend.Buf{part, oo}
		if s.Sink {
			ma = append(ma, so)
		}
		switch {
		case direct:
		case f.groupMerge:
			launch(cs[3], s.Rows*s.Heads, 32, ma...)
		default:
			launch(cs[3], (nOut+127)/128, 128, ma...)
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
			return nil, 0, fmt.Errorf("output guard %d overwritten", i)
		}
	}
	nmse, err := prefillNMSE(s, p, rows, q, sinks, got[:nOut])
	return got[:nOut], nmse, err
}

// prefillInputs is a shape's queries and per-head sink logits, the same for
// every arm of a comparison.
func prefillInputs(s kernels.FlashShape) (q, sinks []float32) {
	rng := rand.New(rand.NewSource(97))
	q = make([]float32, s.Rows*s.Heads*s.Dim)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	sinks = make([]float32, s.Heads)
	for h := range sinks {
		sinks[h] = float32(h%4) - 1.5
	}
	return q, sinks
}

// prefillNMSE is got's NMSE against pagedWant, or an error for a non-finite
// element or a degenerate oracle.
func prefillNMSE(s kernels.FlashShape, p *pagedPool, rows []pagedRow, q, sinks []float32, got []float64) (float64, error) {
	want := pagedWant(s, p, rows, q, sinks)
	var se, ss float64
	for i, w := range want {
		a := got[i]
		if math.IsNaN(a) || math.IsInf(a, 0) {
			return math.Inf(1), fmt.Errorf("non-finite output %d (row %d)", i, i/(len(want)/len(rows)))
		}
		se += (a - w) * (a - w)
		ss += w * w
	}
	if ss == 0 {
		return math.Inf(1), fmt.Errorf("the oracle is all zero")
	}
	return se / ss, nil
}

// shortDesc is desc with every row's keyEnd one short: the last key dropped.
func shortDesc(desc []uint32) []uint32 {
	out := append([]uint32(nil), desc...)
	for i := kernels.PRowEnd; i < len(out); i += kernels.PRowWords {
		if out[i] > out[i-1] {
			out[i]--
		}
	}
	return out
}

// prefillShapes are the geometries every staged prefill gate sweeps: GQA,
// sinks and softcap, packed f16 V, an odd head width, a wide head, and MLA's
// row-major region with the value its prefix, at a latent's widths.
var prefillShapes = []kernels.FlashShape{
	{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125},
	{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Sink: true, Softcap: 3},
	{Heads: 6, KVHeads: 6, Dim: 48, Scale: .14, F16: true, Sink: true},
	{Heads: 4, KVHeads: 2, Dim: 7, Scale: .3},
	{Heads: 8, KVHeads: 1, Dim: 256, Scale: .06},
	{Heads: 8, KVHeads: 2, Dim: 256, Scale: .0625}, // Qwen3.5's full-attention blocks
	{Heads: 8, KVHeads: 4, Dim: 128, Scale: .088, F16: true, Softcap: 20},
	{Heads: 4, KVHeads: 1, Dim: 96, Scale: .1, MLA: 64},
	{Heads: 16, KVHeads: 1, Dim: 576, Scale: .04, MLA: 512},
}

// TestPagedPrefillStaged holds the staged paged prefill -- the tiled FMA
// scores, the per-split softmax, the tiled accumulate and the merge -- to the
// oracle over every prefill case, shape, chunk and split count, both softmax
// forms and both merges; the pages shuffled and in order; and demands failure
// through a wrong table and through keyEnds one short.
func TestPagedPrefillStaged(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	forms := []prefillForm{
		{name: "q1k1", scoresQT: 1, scoresKT: 1, accQT: 1, smLanes: 1},
		{name: "q4k2", scoresQT: 4, scoresKT: 2, accQT: 4, smLanes: 32, groupMerge: true},
		{name: "q2k4", scoresQT: 2, scoresKT: 4, accQT: 8, smLanes: 32},
		// The tier's own form: its default scores tile (defAttnQTile 4,
		// defAttnKTile 2) beside its accumulate's query tile of 8.
		{name: "q4k2a8", scoresQT: 4, scoresKT: 2, accQT: 8, smLanes: 32},
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API()+"/"+d.Name(), func(t *testing.T) {
			for _, page := range []int{64, 256} {
				for _, sh := range prefillShapes {
					for _, c := range prefillCases(page, 16) {
						rows := prefillRows(c)
						for fi, f := range forms {
							for _, extra := range []int{-1, 0, 2} {
								s := sh
								s.Page, s.Chunk = page, []int{64, 256}[fi%2]
								if extra < 0 {
									s.Chunk = directChunk(rows) // the direct form
								} else {
									s.Splits = prefillSplits(rows, s.Chunk, extra)
								}
								chunk := s.Chunk
								t.Run(fmt.Sprintf("P%d/%s/h%d-%d/d%d/mla%d/f16%v/sink%v/%s/C%d/S%d", page, c.name, s.Heads, s.KVHeads, s.Dim, s.MLA, s.F16, s.Sink, f.name, chunk, s.Splits), func(t *testing.T) {
									seed := int64(page + s.Dim + c.pos0)
									p := newPrefillPool(rand.New(rand.NewSource(seed)), page, s.KVHeads*s.Dim, s.F16, s.MLA > 0, false, rows)
									_, nmse, err := pagedPrefillStaged(t, d, s, f, p, p.tab, p.desc, rows)
									if err != nil || nmse > 1e-10 {
										t.Fatalf("shuffled pages: NMSE %.3g, %v", nmse, err)
									}
									if extra == 0 {
										ord := newPrefillPool(rand.New(rand.NewSource(seed)), page, s.KVHeads*s.Dim, s.F16, s.MLA > 0, true, rows)
										if _, n2, err := pagedPrefillStaged(t, d, s, f, ord, ord.tab, ord.desc, rows); err != nil || n2 > 1e-10 {
											t.Fatalf("pages in order: NMSE %.3g, %v", n2, err)
										}
									}
									if _, bad, err := pagedPrefillStaged(t, d, s, f, p, p.wrong, p.desc, rows); err == nil && bad < 1e-6 {
										t.Fatalf("a wrong table passed (NMSE %.3g): the table is not read", bad)
									}
									if _, bad, err := pagedPrefillStaged(t, d, s, f, p, p.tab, shortDesc(p.desc), rows); err == nil && bad < 1e-6 {
										t.Fatalf("keyEnds one short passed (NMSE %.3g): the descriptor is not what bounds the keys", bad)
									}
								})
								ran++
							}
						}
					}
				}
			}
		})
	}
	if ran == 0 {
		t.Fatal("no staged prefill case ran")
	}
}
