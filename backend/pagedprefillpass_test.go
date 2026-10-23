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

// passDesc is desc clipped to the keys [lo, hi): every row's keyStart and
// keyEnd moved into the range, which keeps them non-decreasing and leaves a
// row with no key there empty. It is how a pass over part of a history is
// described; nothing in a kernel changes.
func passDesc(desc []uint32, lo, hi int) []uint32 {
	out := append([]uint32(nil), desc...)
	cl := func(x uint32) uint32 { return uint32(min(max(int(x), lo), hi)) }
	for i := 0; i < len(out); i += kernels.PRowWords {
		out[i+kernels.PRowStart], out[i+kernels.PRowEnd] = cl(out[i+kernels.PRowStart]), cl(out[i+kernels.PRowEnd])
	}
	return out
}

// pagedPrefillPasses runs one paged prefill path over the chunk in passes of
// width keys (each a launch of s.Splits splits), folding each pass's partials
// into a running partial with FlashAttentionMergeRun and finishing with
// FlashAttentionMerge at one split -- the plane bounded by the pass, whatever
// the history -- and returns the NMSE against the oracle and the passes run.
// dropFirst skips the first pass, the violation. flash runs
// FlashPrefill70 (or FlashPrefillTile when tile) at s.Splits; otherwise f's
// staged kernels.
func pagedPrefillPasses(t *testing.T, d backend.Device, s kernels.FlashShape, f prefillForm, flash, tile bool, p *pagedPool, rows []pagedRow, width int, dropFirst bool) (float64, int, error) {
	t.Helper()
	R := len(rows)
	s.Rows = R
	vd := prefillValue(s)
	ms := kernels.FlashShape{Heads: s.Heads, KVHeads: s.KVHeads, Dim: vd, Rows: R, Splits: s.Splits, Sink: s.Sink}
	one := ms
	one.Splits = 1
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
	var sk, sm, ak, fk backend.Kernel
	var skG, akG, fG, fW int
	fs := kernels.FlashPrefill70Shape{Heads: s.Heads, KVHeads: s.KVHeads, Dim: s.Dim, Rows: R, Scale: s.Scale, Softcap: s.Softcap,
		Page: s.Page, Splits: s.Splits, F16: s.F16}
	if flash {
		k, g, w := flashPrefillBuild(t, fs, formOf(tile))
		fk, fG, fW = compile(k, nil), g, w
	} else {
		switch {
		case f.voltaMT > 0:
			sk = compile(kernels.PagedAttnScoresMMA70(s, f.voltaMT, f.voltaNT))
			skG = (kernels.PagedAttnScoresMMA70Warps(s, f.voltaMT, f.voltaNT) + 3) / 4
			ak = compile(kernels.PagedAttnAccMMA70(s, f.accMT(), f.voltaNT))
			akG = (kernels.PagedAttnAccMMA70Warps(s, f.accMT(), f.voltaNT) + 3) / 4
		default:
			sk = compile(kernels.PagedAttnScoresTiled(s, f.scoresQT, f.scoresKT))
			skG = (kernels.PagedAttnScoresTiledThreads(s, f.scoresQT, f.scoresKT) + 127) / 128
			ak = compile(kernels.PagedAttnAccTiled(s, f.accQT))
			akG = (kernels.PagedAttnAccTiledThreads(s, f.accQT) + 127) / 128
		}
		sm = compile(kernels.PagedPrefillSoftmax(s, f.smLanes, f.accTile()))
	}
	first, fold := compile(kernels.FlashAttentionMergeRun(ms, true)), compile(kernels.FlashAttentionMergeRun(ms, false))
	fin := compile(kernels.FlashAttentionMergeWide(one))
	g := newGPU(t, d)
	defer g.free()
	q, sinks := prefillInputs(s)
	nOut := R * s.Heads * vd
	qo, ko, oo := g.up(f32bytes(q)), g.up(f32bytes(p.k)), g.up(f32bytes(nanFill(nOut+32)))
	vo := ko
	if s.MLA == 0 {
		vo = g.up(p.vBytes())
	}
	no, to := g.up(u32bytes([]uint32{0xFFFFFFFF, 0xFFFFFFFF})), g.up(u32bytes(p.tab))
	nanBuf := func(n int) backend.Buf { return g.up(f32bytes(nanFill(n))) }
	plane := kernels.PagedPrefillPlane(s)
	sc, pr, part := nanBuf(plane), nanBuf(plane), nanBuf(kernels.FlashPartialFloats(ms))
	run := [2]backend.Buf{nanBuf(kernels.FlashPartialFloats(one)), nanBuf(kernels.FlashPartialFloats(one))}
	padded := (vd + 31) / 32 * 32
	lo, hi := rows[0].ks/64*64, 0
	for _, r := range rows {
		hi = max(hi, r.ke)
	}
	// Every upload before the session: a pass's clipped descriptors, the sinks.
	var descs []backend.Buf
	for at := lo; at < hi; at += width {
		descs = append(descs, g.up(u32bytes(passDesc(p.desc, at, at+width))))
	}
	so := g.up(f32bytes(sinks))
	passes := 0
	var err error
	d.Session(func(se backend.Session) {
		launch := func(c backend.Kernel, groups, width int, args ...backend.Buf) {
			if err == nil {
				err = se.Launch(c, groups, width, args...)
			}
		}
		for i, ro := range descs {
			if dropFirst && i == 0 {
				continue
			}
			if flash {
				launch(fk, fG, fW, qo, ko, vo, no, part, to, ro)
			} else {
				launch(sk, skG, 128, qo, ko, no, sc, to, ro)
				launch(sm, kernels.PagedPrefillSoftmaxGroups(s, f.smLanes), kernels.PagedPrefillSoftmaxWidth(f.smLanes), sc, no, pr, part, to, ro)
				launch(ak, akG, 128, pr, vo, no, part, to, ro)
			}
			rg := (R*s.Heads*padded + 127) / 128
			if passes == 0 {
				launch(first, rg, 128, part, run[0])
			} else {
				launch(fold, rg, 128, part, run[(passes+1)%2], run[passes%2])
			}
			passes++
		}
		ma := []backend.Buf{run[(passes+1)%2], oo}
		if s.Sink {
			ma = append(ma, so)
		}
		launch(fin, (R*s.Heads*vd+127)/128, 128, ma...)
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
			return 0, passes, fmt.Errorf("output guard %d overwritten", i)
		}
	}
	n, err := prefillNMSE(s, p, rows, q, sinks, got[:nOut])
	return n, passes, err
}

// TestPagedPrefillPasses runs the paged prefill paths over a history several
// times one launch's width, in passes folded by FlashAttentionMergeRun: the
// staged plane is then Rows*Heads*Splits*Chunk for a pass, not for the
// context. It holds the result to the oracle and demands failure when a pass
// is dropped (the fold then covers part of the keys).
func TestPagedPrefillPasses(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	for _, d := range devs {
		defer d.Close()
	}
	volta := map[backend.Device]bool{}
	for _, d := range mmaDevices(t, devs, ir.MMAVolta()) {
		volta[d] = true
	}
	ran := 0
	for _, d := range devs {
		t.Run(d.API()+"/"+d.Name(), func(t *testing.T) {
			type path struct {
				name        string
				f           prefillForm
				flash, tile bool
			}
			paths := []path{{name: "tiled", f: prefillForm{scoresQT: 2, scoresKT: 2, accQT: 4, smLanes: 1}}}
			if volta[d] {
				paths = append(paths, path{name: "volta", f: prefillForm{voltaMT: 1, voltaNT: 2, smLanes: 32}},
					path{name: "flash70", flash: true})
			}
			if d.API() == "msl" {
				paths = append(paths, path{name: "flashtile", flash: true, tile: true})
			}
			for _, pa := range paths {
				for _, sh := range []kernels.FlashShape{
					{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Sink: true, Softcap: 5},
					{Heads: 8, KVHeads: 4, Dim: 128, Scale: .088, F16: true},
				} {
					for _, c := range []prefillCase{{"later", 3*256 + 17, 64, 64, 0}, {"window", 2*256 + 5, 64, 64, 200}} {
						const page = 256
						rows := prefillRows(c)
						s := sh
						s.Page, s.Chunk, s.Splits = page, 128, 2
						t.Run(fmt.Sprintf("%s/%s/d%d/f16%v", pa.name, c.name, s.Dim, s.F16), func(t *testing.T) {
							p := newPrefillPool(rand.New(rand.NewSource(int64(s.Dim+c.pos0))), page, s.KVHeads*s.Dim, s.F16, false, false, rows)
							bound := 1e-10
							if pa.flash || pa.f.voltaMT > 0 {
								bound = 1e-5
							}
							n, passes, err := pagedPrefillPasses(t, d, s, pa.f, pa.flash, pa.tile, p, rows, s.Splits*s.Chunk, false)
							if err != nil || n > bound {
								t.Fatalf("%d passes: NMSE %.3g, %v", passes, n, err)
							}
							if passes < 2 {
								t.Fatalf("%d pass: the history does not need passes, so the fold is not tested", passes)
							}
							bad, _, err := pagedPrefillPasses(t, d, s, pa.f, pa.flash, pa.tile, p, rows, s.Splits*s.Chunk, true)
							if err == nil && bad < 1e-4 {
								t.Fatalf("dropping the first pass passed (NMSE %.3g)", bad)
							}
							t.Logf("%d passes: NMSE %.3g; first pass dropped %.3g %v", passes, n, bad, err)
						})
						ran++
					}
				}
			}
		})
	}
	if ran == 0 {
		t.Fatal("no pass case ran")
	}
}
