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

// pagedStaged runs the staged paged decode -- PagedStagedScores,
// PagedStagedSoftmax, PagedStagedAcc and FlashAttentionMerge -- over rows
// through tab, with scoreLanes/smLanes selecting each kernel's form, and
// returns the NMSE against the oracle. Every scratch plane starts NaN.
func pagedStaged(t *testing.T, d backend.Device, s mainkernels.FlashShape, scoreLanes, smLanes int, p *pagedPool, tab []uint32, rows []pagedRow) (float64, error) {
	t.Helper()
	s.Rows = len(rows)
	vd := s.Dim
	if s.MLA > 0 {
		vd = s.MLA
	}
	ms := s
	ms.Dim, ms.MLA, ms.Page, ms.Chunk = vd, 0, 0, 0
	type built struct {
		k    *ir.Kernel
		err  error
		name string
	}
	var ks []built
	for _, f := range []func() (*ir.Kernel, error){
		func() (*ir.Kernel, error) { return mainkernels.PagedStagedScores(s, scoreLanes) },
		func() (*ir.Kernel, error) { return mainkernels.PagedStagedSoftmax(s, smLanes) },
		func() (*ir.Kernel, error) { return mainkernels.PagedStagedAcc(s) },
		func() (*ir.Kernel, error) { return mainkernels.FlashAttentionMergeWide(ms) },
	} {
		k, err := f()
		ks = append(ks, built{k, err, ""})
	}
	var cs []backend.Kernel
	for i, b := range ks {
		if b.err != nil {
			t.Fatal(b.err)
		}
		if i < 3 && b.k.Params[len(b.k.Params)-1].Name != "pRow" {
			t.Fatalf("%s is not a paged kernel", b.k.Name)
		}
		if ok, why := backend.GuaranteedLanes(d, b.k.Lanes); !ok {
			t.Skip(why)
		}
		c, err := d.Compile(b.k)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		cs = append(cs, c)
	}
	g := newGPU(t, d)
	defer g.free()
	rng := rand.New(rand.NewSource(97))
	q := make([]float32, s.Rows*s.Heads*s.Dim)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	sinks := make([]float32, s.Heads)
	for h := range sinks {
		sinks[h] = float32(h%4) - 1.5
	}
	nanBuf := func(n int) backend.Buf {
		a := make([]float32, n)
		for i := range a {
			a[i] = nan32
		}
		return g.up(f32bytes(a))
	}
	nOut := s.Rows * s.Heads * vd
	qo, ko, oo := g.up(f32bytes(q)), g.up(p.kBytes()), nanBuf(nOut+32)
	vo := ko
	if s.MLA == 0 {
		vo = g.up(p.vBytes())
	}
	no := g.up(u32bytes([]uint32{0xFFFFFFFF, 0xFFFFFFFF}))
	to, ro := g.up(u32bytes(tab)), g.up(u32bytes(p.desc))
	plane := mainkernels.PagedStagedPlane(s)
	sc, pr := nanBuf(plane), nanBuf(plane)
	part := nanBuf(mainkernels.FlashPartialFloats(ms))
	total := mainkernels.PagedStagedScoreItems(s)
	groups := (total + 127) / 128
	if scoreLanes == 32 {
		groups = (total + 3) / 4
	}
	items := s.Rows * s.Heads * s.Splits
	smGroups := (items + 63) / 64
	if smLanes == 32 {
		smGroups = (items + 1) / 2
	}
	var err error
	d.Session(func(se backend.Session) {
		launch := func(c backend.Kernel, groups, width int, args ...backend.Buf) {
			if err == nil {
				err = se.Launch(c, groups, width, args...)
			}
		}
		launch(cs[0], groups, 128, qo, ko, no, sc, to, ro)
		launch(cs[1], smGroups, 64, sc, no, pr, part, to, ro)
		launch(cs[2], (mainkernels.PagedStagedAccThreads(s)+127)/128, 128, pr, vo, no, part, to, ro)
		ma := []backend.Buf{part, oo}
		if s.Sink {
			ma = append(ma, g.up(f32bytes(sinks)))
		}
		launch(cs[3], (s.Rows*s.Heads*vd+127)/128, 128, ma...)
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
			return 0, fmt.Errorf("output guard %d overwritten", i)
		}
	}
	want := pagedWant(s, p, rows, q, sinks)
	var se, ss float64
	for i, w := range want {
		a := got[i]
		if math.IsNaN(a) || math.IsInf(a, 0) {
			return math.Inf(1), fmt.Errorf("non-finite output %d (row %d)", i, i/(s.Heads*vd))
		}
		se += (a - w) * (a - w)
		ss += w * w
	}
	return se / max(ss, 1e-30), nil
}

// TestPagedStagedDecode holds the staged paged decode to the oracle: shuffled
// NaN-poisoned pages, windows from mid-page, several rows (the ragged decode)
// with a padded one, GQA, sinks and softcap applied once, f16 V, MLA's
// row-major region with V its prefix, both forms of the scores and the
// softmax, and split counts from the fewest that cover the rows to more than
// they need (empty splits). A wrong table must fail.
func TestPagedStagedDecode(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API()+"/"+d.Name(), func(t *testing.T) {
			for _, page := range []int{64, 256} {
				for _, sh := range []mainkernels.FlashShape{
					{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125},
					{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Sink: true, Softcap: 3},
					{Heads: 6, KVHeads: 6, Dim: 48, Scale: .14, F16: true, Sink: true},
					{Heads: 4, KVHeads: 2, Dim: 7, Scale: .3},
					{Heads: 4, KVHeads: 1, Dim: 96, Scale: .1, MLA: 64},
					{Heads: 16, KVHeads: 1, Dim: 576, Scale: .04, MLA: 512},
					{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, F16K: true, Softcap: 3},
					{Heads: 4, KVHeads: 2, Dim: 34, Scale: .17, F16K: true, F16: true},
					{Heads: 8, KVHeads: 2, Dim: 256, Scale: .06, F16K: true, Sink: true},
				} {
					sh.Page = page
					for name, rows := range pagedCases(page) {
						longest := 0
						for _, r := range rows {
							longest = max(longest, r.ke)
						}
						for _, chunk := range []int{64, 256} {
							need := (longest + chunk - 1) / chunk
							for _, splits := range []int{max(1, need), need + 3} {
								for _, lanes := range [][2]int{{1, 1}, {1, 32}, {32, 32}} {
									s := sh
									s.Chunk, s.Splits = chunk, splits
									t.Run(fmt.Sprintf("P%d/%s/h%d-%d/d%d/mla%d/f16%v/f16k%v/sink%v/C%d/S%d/lanes%v", page, name, s.Heads, s.KVHeads, s.Dim, s.MLA, s.F16, s.F16K, s.Sink, chunk, splits, lanes), func(t *testing.T) {
										rng := rand.New(rand.NewSource(int64(page + s.Dim)))
										p := newPagedPoolOrder(rng, page, s.KVHeads*s.Dim, s.F16, rows, false, s.MLA > 0)
										if s.F16K {
											p.withF16K()
										}
										nmse, err := pagedStaged(t, d, s, lanes[0], lanes[1], p, p.tab, rows)
										if err != nil || nmse > 1e-10 {
											t.Fatalf("shuffled pages: NMSE %.3g, %v", nmse, err)
										}
										if bad, err := pagedStaged(t, d, s, lanes[0], lanes[1], p, p.wrong, rows); err == nil && bad < 1e-6 {
											t.Fatalf("a wrong table passed (NMSE %.3g): the table is not read", bad)
										}
									})
									ran++
								}
							}
						}
					}
				}
			}
		})
	}
	if ran == 0 {
		t.Fatal("no staged case ran")
	}
}
