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

// contigCache is the pool's history laid out as the contiguous kernels read
// it: K transposed at stride L+1 ([kv dim][position]; row-major for MLA, whose
// value is the row's prefix), V row-major, over positions [0, L). Positions
// below the chunk's first key are 0 -- finite, since a contiguous kernel reads
// from position 0 and relies on the mask.
func contigCache(p *pagedPool, rows []pagedRow) (kc, vc []float32, L, kStride int) {
	for _, r := range rows {
		L = max(L, r.ke)
	}
	kh, vh := p.kh[0], p.vh[0]
	if p.mla {
		return kh[:L*p.kvRow], nil, L, 0
	}
	kStride = L + 1
	kc = make([]float32, p.kvRow*kStride)
	for t := 0; t < L; t++ {
		for e := 0; e < p.kvRow; e++ {
			kc[e*kStride+t] = kh[t*p.kvRow+e]
		}
	}
	return kc, vh[:L*p.kvRow], L, kStride
}

// contigCounts is pN for the contiguous kernels: the uniform width, then each
// row's causal count (a padded row's the last real row's, as the tier sets).
func contigCounts(rows []pagedRow, L int) []uint32 {
	pn := []uint32{uint32(L)}
	for _, r := range rows {
		pn = append(pn, uint32(r.ke))
	}
	return pn
}

// contigStaged runs form f's contiguous twin -- AttnScoresTiledW,
// AttnScoresMMAW or AttnScoresMMA70, SoftmaxRowsSink over the context-wide
// plane, AttnAccTiled or AttnAccMMA70 -- on the same history and queries with
// the case's window baked in, and returns its output [row][head][value].
func contigStaged(t *testing.T, d backend.Device, s kernels.FlashShape, f prefillForm, p *pagedPool, rows []pagedRow, window int) []float64 {
	t.Helper()
	R, H := len(rows), s.Heads
	s.Rows = R
	kvRow, gqa, vd := s.KVHeads*s.Dim, s.Heads/s.KVHeads, prefillValue(s)
	kc, vc, L, kStride := contigCache(p, rows)
	sstride := (L + 128 + 3) / 4 * 4
	var sk, sm, ak *ir.Kernel
	var err error
	var skG, akG, accQT int
	switch {
	case f.mmaNT > 0:
		sk, err = kernels.AttnScoresMMAW(H, s.Dim, kvRow, gqa, sstride, s.Scale, R, kStride, f.mmaNT, window)
		skG = (R/16*H*((L+8*f.mmaNT-1)/(8*f.mmaNT))*32 + 127) / 128
		accQT = f.accQT
	case f.voltaMT > 0:
		sk, err = kernels.AttnScoresMMA70(H, s.Dim, kvRow, gqa, sstride, s.Scale, R, kStride, f.voltaMT, f.voltaNT, window)
		skG = (kernels.AttnScoresMMA70Warps(H, R, L, f.voltaMT, f.voltaNT)*32 + 127) / 128
		accQT = 8 * f.voltaNT
	default:
		sk, err = kernels.AttnScoresTiledW(H, s.Dim, kvRow, gqa, sstride, s.Scale, R, f.scoresQT, f.scoresKT, kStride, window)
		skG = (R/f.scoresQT*H*((L+f.scoresKT-1)/f.scoresKT) + 127) / 128
		accQT = f.accQT
	}
	if err != nil {
		t.Fatal(err)
	}
	if sm, err = kernels.SoftmaxRowsSink(H, sstride, 32, R, accQT, s.Sink); err != nil {
		t.Fatal(err)
	}
	if f.voltaMT > 0 {
		ak, err = kernels.AttnAccMMA70(H, vd, kvRow, gqa, sstride, R, f.accMT(), f.voltaNT)
		akG = (kernels.AttnAccMMA70Warps(H, vd, R, f.accMT(), f.voltaNT)*32 + 127) / 128
	} else {
		ak, err = kernels.AttnAccTiled(H, vd, kvRow, gqa, sstride, R, accQT)
		akG = (R*H*vd/accQT + 127) / 128
	}
	if err != nil {
		t.Fatal(err)
	}
	var cs []backend.Kernel
	for _, k := range []*ir.Kernel{sk, sm, ak} {
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
	ko := g.up(f32bytes(kc))
	vo := ko
	if vc != nil {
		vo = g.up(f32bytes(vc))
	}
	qo, no := g.up(f32bytes(q)), g.up(u32bytes(contigCounts(rows, L)))
	sc, pr := g.up(f32bytes(make([]float32, R*H*sstride))), g.up(f32bytes(make([]float32, R*H*sstride)))
	out := g.up(f32bytes(nanFill(R * H * vd)))
	so := g.up(f32bytes(sinks))
	d.Session(func(se backend.Session) {
		launch := func(c backend.Kernel, groups, width int, args ...backend.Buf) {
			if err == nil {
				err = se.Launch(c, groups, width, args...)
			}
		}
		launch(cs[0], skG, 128, qo, ko, no, sc)
		if s.Sink {
			launch(cs[1], R*H, 32, sc, no, pr, so)
		} else {
			launch(cs[1], R*H, 32, sc, no, pr)
		}
		launch(cs[2], akG, 128, pr, vo, no, out)
		if err == nil {
			err = se.Sync()
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	return readF32(t, out, R*H*vd)
}

// realRowsNMSE is got's NMSE against ref over the rows that attend a key: a
// padded row is empty in the paged form and the last real row in the
// contiguous one.
func realRowsNMSE(got, ref []float64, rows []pagedRow) float64 {
	per := len(got) / len(rows)
	var se, ss float64
	for r, row := range rows {
		if row.ks == row.ke {
			continue
		}
		for i := r * per; i < (r+1)*per; i++ {
			se += (got[i] - ref[i]) * (got[i] - ref[i])
			ss += ref[i] * ref[i]
		}
	}
	return se / max(ss, 1e-30)
}

// TestPagedPrefillMatchesContiguous holds each staged paged prefill form to
// its contiguous twin on the same history and queries, at every case the twin
// can express (no softcap, which the tier applies as its own kernel there;
// no packed f16 V, which the contiguous accumulates do not read).
func TestPagedPrefillMatchesContiguous(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	for _, d := range devs {
		defer d.Close()
	}
	ran := 0
	sh16 := ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}
	volta := ir.MMAVolta()
	for _, pass := range []struct {
		sh    *ir.MMAShape
		bound float64
		forms []prefillForm
	}{
		{nil, 1e-10, []prefillForm{
			{name: "q1k1", scoresQT: 1, scoresKT: 1, accQT: 1, smLanes: 32},
			{name: "q4k2", scoresQT: 4, scoresKT: 2, accQT: 4, smLanes: 32},
		}},
		{&sh16, 1e-5, []prefillForm{{name: "mma2", mmaNT: 2, accQT: 4, smLanes: 32}}},
		{&volta, 1e-5, []prefillForm{{name: "v1x2", voltaMT: 1, voltaNT: 2, smLanes: 32},
			{name: "v2x4", voltaMT: 2, voltaNT: 4, voltaAccMT: 1, smLanes: 32}}},
	} {
		pdevs := devs
		if pass.sh != nil {
			pdevs = mmaDevices(t, devs, *pass.sh)
		}
		for _, d := range pdevs {
			t.Run(d.API()+"/"+d.Name(), func(t *testing.T) {
				for _, sh := range prefillShapes {
					if sh.Softcap > 0 || sh.F16 {
						continue
					}
					for _, f := range pass.forms {
						if f.mmaNT > 0 && (sh.MLA > 0 || sh.Dim%16 != 0) ||
							f.voltaMT > 0 && (sh.Dim%4 != 0 || prefillValue(sh)%(32*f.accMT()) != 0) {
							continue
						}
						for _, c := range prefillCases(256, 32) {
							rows := prefillRows(c)
							s := sh
							s.Page, s.Chunk, s.Splits = 256, 64, prefillSplits(rows, 64, 1)
							if len(c.name)%2 == 0 {
								s.Chunk, s.Splits = directChunk(rows), 0 // the direct form, on half the cases
							}
							t.Run(fmt.Sprintf("%s/h%d-%d/d%d/mla%d/sink%v/%s/S%d", c.name, s.Heads, s.KVHeads, s.Dim, s.MLA, s.Sink, f.name, s.Splits), func(t *testing.T) {
								p := newPrefillPool(rand.New(rand.NewSource(int64(s.Dim+c.pos0))), 256, s.KVHeads*s.Dim, false, s.MLA > 0, false, rows)
								got, nmse, err := pagedPrefillStaged(t, d, s, f, p, p.tab, p.desc, rows)
								if err != nil || nmse > pass.bound {
									t.Fatalf("paged against the oracle: NMSE %.3g, %v", nmse, err)
								}
								ref := contigStaged(t, d, s, f, p, rows, c.window)
								for _, x := range ref {
									if math.IsNaN(x) || math.IsInf(x, 0) {
										t.Fatalf("the contiguous twin is non-finite: the comparison proves nothing")
									}
								}
								if n := realRowsNMSE(got, ref, rows); n > pass.bound {
									t.Fatalf("paged against contiguous: NMSE %.3g", n)
								} else {
									t.Logf("paged/oracle %.3g, paged/contiguous %.3g", nmse, n)
								}
							})
							ran++
						}
					}
				}
			})
		}
	}
	if ran == 0 {
		t.Fatal("no comparison ran")
	}
}
