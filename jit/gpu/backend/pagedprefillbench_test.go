//go:build jitllmbench

package backend_test

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/dev/bench"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestPagedPrefillAB is the paged prefill kernels' performance bar
// (docs/design/device-kv-paging.md): each paged prefill path against the
// contiguous one the tier runs on that device, a 512-row chunk at several
// context depths, same pass, A/A control first, two ABBA passes.
//
//	staged  the scores, the softmax and the accumulate: the tier's tiles --
//	        m16n8k16 scores where the device has them, m8n8k4 on sm_70, the
//	        FMA tile elsewhere; the paged arm adds its merge
//	flash   FlashPrefill70 on sm_70, FlashPrefillTile where the device lowers
//	        the collective tiles (Metal, sm_75+)
//
// Each arm owns `layers` histories and a launch reads the next, so K and V
// come from DRAM. The contiguous K is transposed at stride n+1, as the tier's;
// the paged pages are shuffled, P=256. The rate is attention GFLOP/s: two
// products of 2*Dim flops for every (row, head, attended key).
//
//	JITLLM_PAGED_API, _DEV, _MODEL, _PAGE  as TestPagedAttentionAB
//	JITLLM_PAGED_DEPTH=4096   one context depth (the chunk's last key)
//	JITLLM_PAGED_PKERN=staged|flash|vs    one path; vs is paged staged (A)
//	                          against the paged one-kernel form (B)
//	JITLLM_PAGED_CHUNK=256    the staged paged arm's keys a split
//	JITLLM_PAGED_FSPLITS=0    the paged flash arm's key splits
//	JITLLM_PAGED_PART=scores|softmax|acc  one kernel of the staged arms (the
//	                          paged softmax arm includes the merge)
func TestPagedPrefillAB(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Fatal("no GPU")
	}
	depths := []int{512, 4096, 16384}
	if dd := os.Getenv("JITLLM_PAGED_DEPTH"); dd != "" {
		var x int
		fmt.Sscanf(dd, "%d", &x)
		depths = []int{x}
	}
	page := 256
	fmt.Sscanf(os.Getenv("JITLLM_PAGED_PAGE"), "%d", &page)
	models := []struct {
		name                     string
		heads, kv, dim, attnLays int
	}{
		{"Llama-3.2-1B", 32, 8, 64, 16},
		{"Qwen3-1.7B", 16, 8, 128, 28},
	}
	for _, d := range devs {
		defer d.Close()
		if api := os.Getenv("JITLLM_PAGED_API"); api != "" && api != d.API() {
			continue
		}
		if dn := os.Getenv("JITLLM_PAGED_DEV"); dn != "" && !strings.Contains(d.Name(), dn) {
			continue
		}
		volta := len(mmaDevices(t, []backend.Device{d}, ir.MMAVolta())) > 0 && strings.Contains(d.Name(), "sm_70")
		mma16 := len(mmaDevices(t, []backend.Device{d}, ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16})) > 0
		tiles, _ := backend.LowersTile(d, tile8f32)
		for _, kern := range []string{"staged", "flash", "vs"} {
			if k := os.Getenv("JITLLM_PAGED_PKERN"); k != "" && k != kern {
				continue
			}
			if kern != "staged" && !volta && !tiles {
				continue
			}
			for _, m := range models {
				if mn := os.Getenv("JITLLM_PAGED_MODEL"); mn != "" && !strings.Contains(m.name, mn) {
					continue
				}
				for _, n := range depths {
					t.Run(fmt.Sprintf("%s/%s/%s/%s/n%d", d.API(), d.Name(), kern, m.name, n), func(t *testing.T) {
						prefillAB(t, d, kern, volta, mma16, m.heads, m.kv, m.dim, m.attnLays, n, page)
					})
				}
			}
		}
	}
}

// prefillArm is one arm's launches for a chunk against layer l's history.
type prefillArm func(se backend.Session, l int) error

func prefillAB(t *testing.T, d backend.Device, kern string, volta, mma16 bool, heads, nkv, dim, modelLayers, n, page int) {
	// A 512-row chunk, halved while the contiguous arm's two context-wide
	// planes would pass 1.2 GB (the card is 4 GB and shared); JITLLM_PAGED_ROWS
	// pins it.
	R := 512
	for R > 64 && float64(R*heads*(n+68))*8 > 1.2e9 {
		R /= 2
	}
	fmt.Sscanf(os.Getenv("JITLLM_PAGED_ROWS"), "%d", &R)
	kvRow, gqa := nkv*dim, heads/nkv
	pos0 := n - R // the chunk's rows are positions n-R .. n-1
	perLayer := n * kvRow * 8
	layers := max(1, min(modelLayers, (256<<20)/perLayer))
	scale := float32(1 / math.Sqrt(float64(dim)))
	compile := func(k *ir.Kernel, e error) backend.Kernel {
		if e != nil {
			t.Fatal(e)
		}
		if ok, why := backend.GuaranteedLanes(d, k.Lanes); !ok {
			t.Skip(why)
		}
		c, e := d.Compile(k)
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(c.Close)
		return c
	}
	g := newGPU(t, d)
	defer g.free()
	rng := rand.New(rand.NewSource(int64(n)))
	rnd := func(m int) []float32 {
		a := make([]float32, m)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		return a
	}
	// The chunk: row r at position pos0+r attends [0, pos0+r+1).
	rows := make([]pagedRow, R)
	counts := []uint32{uint32(n)}
	desc := []uint32{}
	for r := range rows {
		rows[r] = pagedRow{0, pos0 + r + 1, pos0 + r}
		counts = append(counts, uint32(pos0+r+1))
		desc = append(desc, 0, 0, uint32(pos0+r+1), uint32(pos0+r))
	}
	q := g.up(f32bytes(rnd(R * heads * dim)))
	nb, db := g.up(u32bytes(counts)), g.up(u32bytes(desc))
	var ck, cv, pk, pv, tabs []backend.Buf
	npages := (n + page - 1) / page
	for l := 0; l < layers; l++ {
		ck = append(ck, g.up(f32bytes(rnd(kvRow*(n+1)))))
		cv = append(cv, g.up(f32bytes(rnd(n*kvRow))))
		pk = append(pk, g.up(f32bytes(rnd(npages*page*kvRow))))
		pv = append(pv, g.up(f32bytes(rnd(npages*page*kvRow))))
		perm := rng.Perm(npages)
		tab := make([]uint32, npages)
		for i := range tab {
			tab[i] = uint32(perm[i])
		}
		tabs = append(tabs, g.up(u32bytes(tab)))
	}
	out := g.up(make([]byte, R*heads*dim*4))
	part := os.Getenv("JITLLM_PAGED_PART")
	on := func(p string) bool { return part == "" || part == p }
	var contig, paged prefillArm
	var what string
	// vs pits the paged staged form (arm A) against the paged one-kernel
	// form (arm B) on one device: the tier's choice between them.
	var stagedPaged prefillArm
	if kern == "staged" || kern == "vs" {
		sstride := (n + 64 + 3) &^ 3
		chunk, passW := 256, 4096
		fmt.Sscanf(os.Getenv("JITLLM_PAGED_CHUNK"), "%d", &chunk)
		fmt.Sscanf(os.Getenv("JITLLM_PAGED_PASSW"), "%d", &passW)
		// The paged arm runs in passes of at most passW keys, folded by
		// FlashAttentionMergeRun: its plane is a pass's, not the context's.
		W := min(n, passW)
		s := kernels.FlashShape{Heads: heads, KVHeads: nkv, Dim: dim, Rows: R, Scale: scale, Page: page, Chunk: chunk,
			Splits: (W + chunk - 1) / chunk}
		W = s.Splits * chunk
		// JITLLM_PAGED_DIRECT=1: the direct form (Splits 0, one chunk of the
		// whole span, normalised in the softmax, no merge) where one pass
		// covers the context.
		direct := os.Getenv("JITLLM_PAGED_DIRECT") == "1" && n <= W
		if direct {
			s.Chunk, s.Splits = (n+63)/64*64, 0
			W = s.Chunk
		}
		ms := kernels.FlashShape{Heads: heads, KVHeads: nkv, Dim: dim, Rows: R, Splits: s.Splits}
		one := ms
		one.Splits = 1
		var passDescs []backend.Buf
		for at := 0; at < n; at += W {
			passDescs = append(passDescs, g.up(u32bytes(passDesc(desc, at, at+W))))
		}
		lanes := 1
		if ok, _ := backend.GuaranteedLanes(d, 32); ok {
			lanes = 32
		}
		var cs, csm, ca, ps, psm, pa, pm backend.Kernel
		var csG, caG, psG, paG int
		atile := 8
		switch {
		case volta:
			const mt, nt = 2, 4
			atile = 8 * nt
			what = fmt.Sprintf("m8n8k4 %dx%d, acc m8n8k4 1x%d", mt, nt, nt)
			cs = compile(kernels.AttnScoresMMA70(heads, dim, kvRow, gqa, sstride, scale, R, n+1, mt, nt, 0))
			csG = (kernels.AttnScoresMMA70Warps(heads, R, n, mt, nt)*32 + 127) / 128
			ca = compile(kernels.AttnAccMMA70(heads, dim, kvRow, gqa, sstride, R, 1, nt))
			caG = (kernels.AttnAccMMA70Warps(heads, dim, R, 1, nt)*32 + 127) / 128
			ps = compile(kernels.PagedAttnScoresMMA70(s, mt, nt))
			psG = (kernels.PagedAttnScoresMMA70Warps(s, mt, nt) + 3) / 4
			pa = compile(kernels.PagedAttnAccMMA70(s, 1, nt))
			paG = (kernels.PagedAttnAccMMA70Warps(s, 1, nt) + 3) / 4
		case mma16:
			const nt = 2
			what = fmt.Sprintf("m16n8k16 nt=%d, acc FMA qt=%d", nt, atile)
			cs = compile(kernels.AttnScoresMMA(heads, dim, kvRow, gqa, sstride, scale, R, n+1, nt))
			csG = (R/16*heads*((n+8*nt-1)/(8*nt))*32 + 127) / 128
			ps = compile(kernels.PagedAttnScoresMMA(s, nt))
			psG = (kernels.PagedAttnScoresMMAWarps(s, nt) + 3) / 4
		default:
			const qt, kt = 4, 2
			what = fmt.Sprintf("FMA %dx%d, acc FMA qt=%d", qt, kt, atile)
			cs = compile(kernels.AttnScoresTiled(heads, dim, kvRow, gqa, sstride, scale, R, qt, kt, n+1))
			csG = (R/qt*heads*((n+kt-1)/kt) + 127) / 128
			ps = compile(kernels.PagedAttnScoresTiled(s, qt, kt))
			psG = (kernels.PagedAttnScoresTiledThreads(s, qt, kt) + 127) / 128
		}
		if !volta {
			ca = compile(kernels.AttnAccTiled(heads, dim, kvRow, gqa, sstride, R, atile))
			caG = (R*heads*dim/atile + 127) / 128
			pa = compile(kernels.PagedAttnAccTiled(s, atile))
			paG = (kernels.PagedAttnAccTiledThreads(s, atile) + 127) / 128
		}
		csm = compile(kernels.SoftmaxRows(heads, sstride, lanes, R, atile))
		psm = compile(kernels.PagedPrefillSoftmax(s, lanes, atile))
		if !direct {
			pm = compile(kernels.FlashAttentionMergeWide(ms))
		}
		var mFirst, mFold, mFin backend.Kernel
		if len(passDescs) > 1 {
			mFirst, mFold = compile(kernels.FlashAttentionMergeRun(ms, true)), compile(kernels.FlashAttentionMergeRun(ms, false))
			mFin = compile(kernels.FlashAttentionMergeWide(one))
		}
		smG, smW := R*heads, 32
		if lanes == 1 {
			smG, smW = (R*heads+63)/64, 64
		}
		what += fmt.Sprintf(", R=%d, paged C=%d S=%d in %d pass(es)", R, s.Chunk, s.Splits, len(passDescs))
		sc, pr := g.up(make([]byte, R*heads*sstride*4)), g.up(make([]byte, R*heads*sstride*4))
		plane := kernels.PagedPrefillPlane(s)
		psc, ppr := g.up(make([]byte, plane*4)), g.up(make([]byte, plane*4))
		pp := g.up(make([]byte, kernels.FlashPartialFloats(ms)*4))
		run := [2]backend.Buf{g.up(make([]byte, kernels.FlashPartialFloats(one)*4)), g.up(make([]byte, kernels.FlashPartialFloats(one)*4))}
		rg := (R*heads*((dim+31)/32*32) + 127) / 128
		contig = func(se backend.Session, l int) error {
			var e error
			if on("scores") {
				e = se.Launch(cs, csG, 128, q, ck[l], nb, sc)
			}
			if e == nil && on("softmax") {
				e = se.Launch(csm, smG, smW, sc, nb, pr)
			}
			if e == nil && on("acc") {
				e = se.Launch(ca, caG, 128, pr, cv[l], nb, out)
			}
			return e
		}
		paged = func(se backend.Session, l int) error {
			var e error
			for j, pd := range passDescs {
				if e == nil && on("scores") {
					e = se.Launch(ps, psG, 128, q, pk[l], nb, psc, tabs[l], pd)
				}
				if e == nil && on("softmax") {
					e = se.Launch(psm, kernels.PagedPrefillSoftmaxGroups(s, lanes), kernels.PagedPrefillSoftmaxWidth(lanes), psc, nb, ppr, pp, tabs[l], pd)
				}
				if e == nil && on("acc") {
					dst := pp
					if direct {
						dst = out
					}
					e = se.Launch(pa, paG, 128, ppr, pv[l], nb, dst, tabs[l], pd)
				}
				if e == nil && on("softmax") && len(passDescs) > 1 {
					if j == 0 {
						e = se.Launch(mFirst, rg, 128, pp, run[0])
					} else {
						e = se.Launch(mFold, rg, 128, pp, run[(j+1)%2], run[j%2])
					}
				}
			}
			if e == nil && on("softmax") {
				if len(passDescs) > 1 {
					e = se.Launch(mFin, (R*heads*dim+127)/128, 128, run[(len(passDescs)+1)%2], out)
				} else {
					if !direct {
						e = se.Launch(pm, (R*heads*dim+127)/128, 128, pp, out)
					}
				}
			}
			return e
		}
		stagedPaged = paged
	}
	if kern == "flash" || kern == "vs" {
		if dim != 64 && dim != 128 {
			t.Skip("FlashPrefill70 takes head widths 64 and 128")
		}
		splits := 0
		fmt.Sscanf(os.Getenv("JITLLM_PAGED_FSPLITS"), "%d", &splits)
		cf := kernels.FlashPrefill70Shape{Heads: heads, KVHeads: nkv, Dim: dim, Rows: R, KStride: n + 1, Scale: scale}
		pf := cf
		pf.KStride, pf.Page, pf.Splits = 0, page, splits
		tile := !volta
		kc, cG, cW := flashPrefillBuild(t, cf, formOf(tile))
		kp, pG, pW := flashPrefillBuild(t, pf, formOf(tile))
		fc, fp := compile(kc, nil), compile(kp, nil)
		what = fmt.Sprintf("%s, paged splits %d", kc.Name, splits)
		dst := out
		var merge backend.Kernel
		if splits > 0 {
			ms := kernels.FlashShape{Heads: heads, KVHeads: nkv, Dim: dim, Rows: R, Splits: splits}
			merge = compile(kernels.FlashAttentionMergeWide(ms))
			dst = g.up(make([]byte, kernels.FlashPartialFloats(ms)*4))
		}
		contig = func(se backend.Session, l int) error {
			return se.Launch(fc, cG, cW, q, ck[l], cv[l], nb, out)
		}
		paged = func(se backend.Session, l int) error {
			e := se.Launch(fp, pG, pW, q, pk[l], pv[l], nb, dst, tabs[l], db)
			if e == nil && merge != nil {
				e = se.Launch(merge, (R*heads*dim+127)/128, 128, dst, out)
			}
			return e
		}
		if kern == "vs" {
			contig = stagedPaged
			what = "paged staged (A) vs paged " + kc.Name + " (B)"
		}
	}
	arm := func(f prefillArm) func(int) {
		return func(iters int) {
			d.Session(func(se backend.Session) {
				for i := 0; i < iters; i++ {
					if e := f(se, i%layers); e != nil {
						panic(e)
					}
				}
				if e := se.Sync(); e != nil {
					panic(e)
				}
			})
		}
	}
	a := bench.Case{Name: "contiguous", Fn: arm(contig)}
	b := bench.Case{Name: "paged", Fn: arm(paged)}
	// Calibrate to about 300 ms a round, or JITLLM_PAGED_ROUNDMS.
	t0 := time.Now()
	a.Fn(2)
	ms := 300
	fmt.Sscanf(os.Getenv("JITLLM_PAGED_ROUNDMS"), "%d", &ms)
	iters := max(2, min(400, int(time.Duration(ms)*time.Millisecond/max(time.Since(t0)/2, time.Microsecond))))
	for until := time.Now().Add(2 * time.Second); time.Now().Before(until); {
		a.Fn(iters)
		b.Fn(iters)
	}
	// Two products of 2*Dim flops per (row, head, attended key).
	flops := 0.0
	for _, r := range rows {
		flops += float64(r.ke-r.ks) * float64(heads) * 4 * float64(dim)
	}
	rate := func(dt time.Duration) string {
		return fmt.Sprintf("%.0f GFLOP/s", flops*float64(iters)/dt.Seconds()/1e9)
	}
	aa := bench.AB(a, a, 20, iters)
	t.Logf("%s; %d layers, %d iters; A/A control %s", what, layers, iters, aa)
	for pass := 0; pass < 2; pass++ {
		r := bench.AB(a, b, 20, iters)
		t.Logf("pass%d %s (>1: paged faster); contiguous %s, paged %s", pass, r, rate(r.AMedian), rate(r.BMedian))
	}
}
