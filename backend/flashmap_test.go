package backend_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestFlashPrefillRunTable holds a flash prefill's run-table form (Map) --
// one launch over the prompt chunks of a ragged step, each sequence's rows
// filling whole workgroups through the table (docs/design/ragged-prefill.md)
// -- to each sequence run alone through the plain form, bit for bit: two
// sequences with their own page tables and histories, their rows scattered
// among each other and among decoding rows the table does not name, each
// padded to a workgroup with its last row. The decoding rows must come back
// untouched, and the violation -- the second sequence's rows reading the
// first's table -- must part from its own run.
func TestFlashPrefillRunTable(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	for _, d := range devs {
		defer d.Close()
	}
	ran := 0
	for _, fk := range []struct {
		form flashForm
		k    int
	}{{flash80Form, 16}, {flash80K8Form, 8}} {
		for _, d := range mmaDevices(t, devs, ir.MMAShape{M: 16, N: 8, K: fk.k, Kind: ir.MMAF16}) {
			for _, fs := range []kernels.FlashPrefill70Shape{
				{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: 64},
				{Heads: 8, KVHeads: 4, Dim: 128, Scale: .088, Page: 64, F16: true, Softcap: 50},
				{Heads: 8, KVHeads: 4, Dim: 256, Scale: .0625, Page: 64},
				{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: 64, Splits: 1},
			} {
				t.Run(fmt.Sprintf("k%d/%s/%s/d%d/S%d", fk.k, d.API(), d.Name(), fs.Dim, fs.Splits), func(t *testing.T) {
					runTableCase(t, d, fs, fk.form)
				})
				ran++
			}
		}
	}
	for _, d := range devs {
		if ok, _ := backend.LowersTile(d, tile8f32); !ok {
			continue
		}
		for _, fs := range []kernels.FlashPrefill70Shape{
			{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: 64},
			{Heads: 8, KVHeads: 4, Dim: 128, Scale: .088, Page: 64, F16: true, Softcap: 50},
			{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: 64, Splits: 1},
		} {
			t.Run(fmt.Sprintf("tile/%s/%s/d%d/S%d", d.API(), d.Name(), fs.Dim, fs.Splits), func(t *testing.T) {
				runTableCase(t, d, fs, flashTileForm)
			})
			ran++
		}
	}
	for _, d := range mmaDevices(t, devs, ir.MMAVolta()) {
		for _, fs := range []kernels.FlashPrefill70Shape{
			{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: 64},
			{Heads: 8, KVHeads: 4, Dim: 128, Scale: .088, Page: 64, F16: true, Softcap: 50},
			{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: 64, Splits: 1},
		} {
			t.Run(fmt.Sprintf("v70/%s/%s/d%d/S%d", d.API(), d.Name(), fs.Dim, fs.Splits), func(t *testing.T) {
				runTableCase(t, d, fs, flash70Form)
			})
			ran++
		}
	}
	if ran == 0 {
		t.Skip("no device lowers m16n8k16, m16n8k8, m8n8k4 or the collective tiles")
	}
}

func runTableCase(t *testing.T, d backend.Device, fs kernels.FlashPrefill70Shape, form flashForm) {
	t.Helper()
	wg := kernels.FlashPrefill80Rows(fs)
	switch form {
	case flashTileForm:
		tl, _ := kernels.FlashTileFor(fs.Dim)
		wg = tl.Rows()
	case flash70Form:
		wg = kernels.FlashPrefill70Rows
	}
	pg := fs.Page
	// Two sequences: A's 40 rows at a later base, B's 23 crossing a page edge.
	up := func(n int) int { return (n + wg - 1) / wg * wg }
	ca, cb := prefillCase{"a", 3*pg + 17, 40, up(40), 0}, prefillCase{"b", pg - 3, 23, up(23), 0}
	ra, rb := prefillRows(ca), prefillRows(cb)
	kvRow := fs.KVHeads * fs.Dim
	p := newPrefillPool(rand.New(rand.NewSource(int64(fs.Dim))), pg, kvRow, fs.F16, false, false, ra)
	descOf := func(rows []pagedRow, tabOff int) []uint32 {
		var out []uint32
		for _, r := range rows {
			out = append(out, uint32(tabOff), uint32(r.ks), uint32(r.ke), uint32(r.wp))
		}
		return out
	}
	per := fs.Heads * fs.Dim
	qOf := func(seed int64, n int) []float32 {
		rng := rand.New(rand.NewSource(seed))
		q := make([]float32, n*per)
		for i := range q {
			q[i] = float32(rng.NormFloat64())
		}
		return q
	}
	qa, qb := qOf(11, ca.rows), qOf(12, cb.rows)
	// Each sequence alone, plain form: the reference.
	refA := runFlash(t, d, fs, form, p, qa, p.tab, descOf(ra, 0), nil, ca.rows)
	// B's own table: A's with its first two pages swapped, so B's history
	// is real keys, and different ones.
	tabB := append([]uint32(nil), p.tab...)
	tabB[0], tabB[1] = tabB[1], tabB[0]
	refB := runFlash(t, d, fs, form, p, qb, tabB, descOf(rb, 0), nil, cb.rows)

	// The step: A's and B's real rows and five decoding rows, shuffled.
	type src struct{ seq, r int } // seq 0 A, 1 B, 2 a decoding row
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
	step := func(bOff int) ([]float64, []src) {
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
				desc = append(desc, descOf(rb[s.r:s.r+1], bOff)...)
			default:
				desc = append(desc, 0, 0, 0, 0)
			}
		}
		// The run table: each sequence's rows in position order, padded
		// to a workgroup with its last.
		var mp []uint32
		for seq, n := range []int{ca.nrow, cb.nrow} {
			for j := 0; j < (n+wg-1)/wg*wg; j++ {
				mp = append(mp, uint32(at[src{seq, min(j, n-1)}]))
			}
		}
		return runFlash(t, d, fs, form, p, q, tab, desc, mp, len(phys)), phys
	}
	got, _ := step(len(p.tab))
	same := func(ref []float64, r int, i int) bool {
		for e := 0; e < per; e++ {
			a, b := ref[r*per+e], got[i*per+e]
			if math.Float64bits(a) != math.Float64bits(b) {
				return false
			}
		}
		return true
	}
	for _, s := range phys {
		i := at[s]
		switch s.seq {
		case 0:
			if !same(refA, s.r, i) {
				t.Fatalf("A's row %d (step row %d) differs from A alone", s.r, i)
			}
		case 1:
			if !same(refB, s.r, i) {
				t.Fatalf("B's row %d (step row %d) differs from B alone", s.r, i)
			}
		default:
			for e := 0; e < per; e++ {
				if !math.IsNaN(got[i*per+e]) {
					t.Fatalf("decoding row %d was written: the table names it nowhere", i)
				}
			}
		}
	}
	// The violation: B's rows read A's table.
	got, _ = step(0)
	parted := false
	for r := 0; r < cb.nrow && !parted; r++ {
		parted = !same(refB, r, at[src{1, r}])
	}
	if !parted {
		t.Fatal("B's rows read A's table and still matched B alone: the table offset is not read")
	}
}

// runFlash launches one flash prefill (Map when mp is set) over rows rows of q
// with tab and desc, and returns the rows' output, NaN where nothing wrote.
func runFlash(t *testing.T, d backend.Device, fs kernels.FlashPrefill70Shape, form flashForm, p *pagedPool,
	q []float32, tab, desc, mp []uint32, rows int) []float64 {
	t.Helper()
	fs.Map = mp != nil
	fs.Rows = rows
	if fs.Map {
		fs.Rows = len(mp)
	}
	k, groups, threads := flashPrefillBuild(t, fs, form)
	c, err := d.Compile(k)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	g := newGPU(t, d)
	defer g.free()
	nOut := rows * fs.Heads * fs.Dim
	oo := g.up(f32bytes(nanFill(nOut + 32)))
	args := []backend.Buf{g.up(f32bytes(q)), g.up(f32bytes(p.k)), g.up(p.vBytes()),
		g.up(u32bytes([]uint32{0xFFFFFFFF, 0xFFFFFFFF})), oo, g.up(u32bytes(tab)), g.up(u32bytes(desc))}
	if fs.Map {
		args = append(args, g.up(u32bytes(mp)))
	}
	// With a split the kernel writes partials, and the wide merge -- with
	// the same table where the launch has one -- adds the sink and writes
	// the rows.
	var merge backend.Kernel
	var margs []backend.Buf
	if fs.Splits > 0 {
		ms := kernels.FlashShape{Heads: fs.Heads, KVHeads: fs.KVHeads, Dim: fs.Dim, Rows: fs.Rows,
			Splits: fs.Splits, Sink: true, Map: fs.Map}
		mk, err := kernels.FlashAttentionMergeWide(ms)
		if err != nil {
			t.Fatal(err)
		}
		if merge, err = d.Compile(mk); err != nil {
			t.Fatal(err)
		}
		defer merge.Close()
		// Partials at the rows: sized for every row the table can name.
		part := g.up(f32bytes(nanFill(kernels.FlashPartialFloats(kernels.FlashShape{Heads: fs.Heads, Dim: fs.Dim,
			Rows: max(rows, fs.Rows), Splits: fs.Splits}))))
		args[4] = part
		_, sinks := prefillInputs(ms)
		margs = []backend.Buf{part, oo, g.up(f32bytes(sinks))}
		if fs.Map {
			margs = append(margs, args[len(args)-1])
		}
	}
	d.Session(func(se backend.Session) {
		err = se.Launch(c, groups, threads, args...)
		if err == nil && merge != nil {
			err = se.Launch(merge, (fs.Rows*fs.Heads*fs.Dim+127)/128, 128, margs...)
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
