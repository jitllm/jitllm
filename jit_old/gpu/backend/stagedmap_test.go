package backend_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestStagedPrefillRunTable holds the staged prefill's run-table form
// (FlashShape.Map: the FMA scores, the softmax and the tiled accumulate, the
// direct form) to each sequence run alone through the plain form, bit for
// bit, as TestFlashPrefillRunTable holds the flash forms: two sequences with
// their own tables, histories and partition bases (pBase), their rows
// scattered among each other and among decoding rows the table does not
// name, each padded to the forms' query tile with its last row; the decoding
// rows untouched; the violation -- B's base one block past its first key -- parts.
func TestStagedPrefillRunTable(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	for _, d := range devs {
		defer d.Close()
	}
	for _, d := range devs {
		for _, sh := range []mainkernels.FlashShape{
			{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: 64},
			{Heads: 8, KVHeads: 4, Dim: 128, Scale: .088, Page: 64, F16: true, Softcap: 50},
			{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: 64, Sink: true},
		} {
			for _, f := range []prefillForm{
				{scoresQT: 2, scoresKT: 2, accQT: 2, smLanes: 32},
				{scoresQT: 4, scoresKT: 2, accQT: 4, smLanes: 1},
			} {
				t.Run(fmt.Sprintf("%s/%s/d%d/sink%v/q%d-%d", d.API(), d.Name(), sh.Dim, sh.Sink, f.scoresQT, f.smLanes), func(t *testing.T) {
					stagedTableCase(t, d, sh, f)
				})
			}
		}
	}
}

func stagedTableCase(t *testing.T, d backend.Device, s mainkernels.FlashShape, f prefillForm) {
	t.Helper()
	wg := max(f.scoresQT, f.accQT) // every kernel's query tile divides it
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
	// One chunk covering either sequence's span from its own base.
	chunk := max(directChunk(ra), directChunk(rb))
	s.Chunk = chunk
	refA := runStaged(t, d, s, f, p, qa, p.tab, descOf(ra, 0), nil, nil, ca.rows)
	refB := runStaged(t, d, s, f, p, qb, tabB, descOf(rb, 0), nil, nil, cb.rows)

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
	baseOf := func(rows []pagedRow) uint32 { return uint32(rows[0].ks / 64 * 64) }
	step := func(bBase uint32) []float64 {
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
		var mp, base []uint32
		for seq, n := range []int{ca.nrow, cb.nrow} {
			b := baseOf(ra)
			if seq == 1 {
				b = bBase
			}
			for j := 0; j < up(n); j++ {
				mp = append(mp, uint32(at[src{seq, min(j, n-1)}]))
				base = append(base, b)
			}
		}
		return runStaged(t, d, s, f, p, q, tab, desc, mp, base, len(phys))
	}
	got := step(baseOf(rb))
	same := func(ref []float64, r, i int) bool {
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
			if !same(refA, sr.r, i) {
				t.Fatalf("A's row %d (step row %d) differs from A alone", sr.r, i)
			}
		case 1:
			if !same(refB, sr.r, i) {
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
	// fall outside its partition.
	got = step(baseOf(rb) + 64)
	parted := false
	for r := 0; r < cb.nrow && !parted; r++ {
		parted = !same(refB, r, at[src{1, r}])
	}
	if !parted {
		t.Fatal("B's base moved past its first key and still matched B alone: the base table is not read")
	}
}

// runStaged runs the staged direct form (Map when mp is set) over rows rows
// of q and returns the rows' output, NaN where nothing wrote.
func runStaged(t *testing.T, d backend.Device, s mainkernels.FlashShape, f prefillForm, p *pagedPool,
	q []float32, tab, desc, mp, base []uint32, rows int) []float64 {
	t.Helper()
	s.Splits, s.Map = 0, mp != nil
	s.Rows = rows
	if s.Map {
		s.Rows = len(mp)
	}
	ks := []func() (*ir.Kernel, error){
		func() (*ir.Kernel, error) { return mainkernels.PagedAttnScoresTiled(s, f.scoresQT, f.scoresKT) },
		func() (*ir.Kernel, error) { return mainkernels.PagedPrefillSoftmax(s, f.smLanes, f.accTile()) },
		func() (*ir.Kernel, error) { return mainkernels.PagedAttnAccTiled(s, f.accQT) },
	}
	var cs []backend.Kernel
	for _, mk := range ks {
		k, err := mk()
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
		defer c.Close()
		cs = append(cs, c)
	}
	g := newGPU(t, d)
	defer g.free()
	_, sinks := prefillInputs(s)
	nOut := rows * s.Heads * s.Dim
	nan := func(n int) backend.Buf { return g.up(f32bytes(nanFill(n))) }
	qo, ko, vo, oo := g.up(f32bytes(q)), g.up(f32bytes(p.k)), g.up(p.vBytes()), nan(nOut+32)
	no := g.up(u32bytes([]uint32{0xFFFFFFFF, 0xFFFFFFFF}))
	to, ro := g.up(u32bytes(tab)), g.up(u32bytes(desc))
	plane := mainkernels.PagedPrefillPlane(s)
	sc, pr, part := nan(plane), nan(plane), nan(8)
	so := g.up(f32bytes(sinks))
	var tail []backend.Buf
	if s.Map {
		tail = []backend.Buf{g.up(u32bytes(mp)), g.up(u32bytes(base))}
	}
	var err error
	d.Session(func(se backend.Session) {
		launch := func(c backend.Kernel, groups, width int, args ...backend.Buf) {
			if err == nil {
				err = se.Launch(c, groups, width, append(args, tail...)...)
			}
		}
		launch(cs[0], (mainkernels.PagedAttnScoresTiledThreads(s, f.scoresQT, f.scoresKT)+127)/128, 128, qo, ko, no, sc, to, ro)
		sm := []backend.Buf{sc, no, pr, part, to, ro}
		if s.Sink {
			sm = append(sm, so)
		}
		launch(cs[1], mainkernels.PagedPrefillSoftmaxGroups(s, f.smLanes), mainkernels.PagedPrefillSoftmaxWidth(f.smLanes), sm...)
		launch(cs[2], (mainkernels.PagedAttnAccTiledThreads(s, f.accQT)+127)/128, 128, pr, vo, no, oo, to, ro)
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
