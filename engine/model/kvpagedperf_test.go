//go:build jitllmbench && (amd64 || arm64)

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than by a skip in every default run:
//
//	go test -tags jitllmbench -run <Name> ./<pkg>
//
// JITLLM_* variables select its parameters.

package model

import (
	"fmt"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestPagedKVCost measures what walking the KV pages costs, paired and
// interleaved. The arms differ only in page size: at P >= maxSeq a layer has
// one page and one kernel call; at P = 256 the same window takes ceil(n/256)
// calls. It measures at depth, where the arms differ, and asserts the page
// counts.
//
// ABBA in one process, A/A self-control first, two passes, median of per-round
// ratios, rejected above IQR/median 0.10.
func TestPagedKVCost(t *testing.T) {
	// Gate on core busy-ness, not PSI (see coreBusy); one contended core
	// stalls every region's barrier.
	if b, err := coreBusy(time.Second); err == nil && b > 20 {
		t.Skipf("the cores this measurement pins to are %.1f%% busy; a rate row needs "+
			"them idle (PSI cannot see this -- see coreBusy)", b)
	}
	name := envOr("JITLLM_KVPERF_MODEL", "tinyllama-1.1b-q3_K_M.gguf")
	depth := envInt("JITLLM_KVPERF_DEPTH", 2048)
	page := envInt("JITLLM_KVPERF_PAGE", 256)
	rounds := envInt("JITLLM_KVPERF_ROUNDS", 12)
	warm, meas := 8, 32

	path := jlmOf(t, testmodels.Path(name))
	m, err := Open(path)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()
	// Clamp the depth to the model's context; otherwise kvPagePositions clamps
	// the one-page arm and the two arms become the same configuration.
	if n := m.Cfg.NCtx; n > 0 && depth+warm+meas+8 > n {
		depth = n - (warm + meas + 8)
	}
	if depth < 4*page {
		t.Skipf("%s has room for a depth of only %d, which is under four %d-position "+
			"pages -- the paged arm would hold too few pages to differ from one page",
			name, depth, page)
	}
	maxSeq := depth + warm + meas + 8

	// One sample: a fresh session, prefilled to `depth`, then `meas` timed
	// tokens after `warm` untimed ones. It returns the rate and how many pages
	// the layer with the most of them actually holds.
	sample := func(p int) (float64, int) {
		defer m.setKVPageForTest(p)()
		s := m.NewInstance(maxSeq)
		defer s.Close()
		ids := make([]int32, depth)
		for i := range ids {
			ids[i] = int32(1 + i%256)
		}
		if _, err := s.Prefill(ids); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < warm; i++ {
			if _, err := s.Forward(int32(7 + i)); err != nil {
				t.Fatal(err)
			}
		}
		t0 := time.Now()
		for i := 0; i < meas; i++ {
			if _, err := s.Forward(int32(11 + i)); err != nil {
				t.Fatal(err)
			}
		}
		d := time.Since(t0)
		pages := 0
		for li := range s.kv.layers {
			if n := s.kv.layers[li].mem.Len(); n > pages {
				pages = n
			}
		}
		return float64(meas) / d.Seconds(), pages
	}

	// ABBA so a drifting box cancels within a round. Core busy-ness is read at
	// the start and finish of every pass so a tenant arriving mid-run shows.
	pass := func(a, b int) ([]float64, float64, float64, int, int, float64, float64) {
		b0, _ := coreBusy(300 * time.Millisecond)
		var ratios []float64
		var ra, rb []float64
		var pa, pb int
		for i := 0; i < rounds; i++ {
			x1, p1 := sample(a)
			y1, q1 := sample(b)
			y2, _ := sample(b)
			x2, _ := sample(a)
			pa, pb = p1, q1
			ratios = append(ratios, x1/y1, x2/y2)
			ra = append(ra, x1, x2)
			rb = append(rb, y1, y2)
		}
		b1, _ := coreBusy(300 * time.Millisecond)
		return ratios, median(ra), median(rb), pa, pb, b0, b1
	}

	// The A/A self-control first: it shows whether the harness discriminates.
	ctl, _, _, _, _, cb0, cb1 := pass(maxSeq, maxSeq)
	cm, craw := medIQR(ctl)
	ci := craw / cm
	t.Logf("A/A self-control (one page BOTH arms)   %.4f   IQR/median %.3f   n=%d   cores %.1f%% -> %.1f%%",
		cm, ci, len(ctl), cb0, cb1)

	bytesPerTok := float64(m.BytesPerToken())
	for p := 1; p <= 2; p++ {
		r, ra, rb, pgA, pgB, pb0, pb1 := pass(page, maxSeq)
		med, raw := medIQR(r)
		iqr := raw / med
		gate := "GATES"
		if iqr > 0.10 {
			gate = "REJECTED (IQR/median over 0.10)"
		}
		if pb0 > 20 || pb1 > 20 {
			gate = fmt.Sprintf("REJECTED (cores %.1f%% -> %.1f%% busy; a tenant was here)", pb0, pb1)
		}
		if pgA < 2 {
			t.Fatalf("the paged arm holds %d page(s) at depth %d with P=%d -- the arms are "+
				"the same configuration and the ratio means nothing", pgA, depth, page)
		}
		if pgB != 1 {
			t.Fatalf("the control arm holds %d pages, not 1 -- it is not the unpaged cache", pgB)
		}
		t.Logf("pass %d  P=%d (%d pages) / one page (%d)   %.4f   IQR/median %.3f   n=%d   cores %.1f%% -> %.1f%%   %s",
			p, page, pgA, pgB, med, iqr, len(r), pb0, pb1, gate)
		t.Logf("        paged %.2f tok/s = %.1f GB/s   one page %.2f tok/s = %.1f GB/s   "+
			"(%.0f B/token)",
			ra, ra*bytesPerTok/1e9, rb, rb*bytesPerTok/1e9, bytesPerTok)
	}
}

// TestPrefixCacheSpeedup prices what a cached prefix is worth: prompt time with
// the store holding it against prompt time computing it. The arms do different
// amounts of work by design, so the check is whether the measured ratio matches
// the fraction of positions skipped, which is predicted from the count.
//
// ABBA in one process, A/A self-control first, two passes, IQR/median gated at
// 0.10, cores checked at each pass's start and finish.
func TestPrefixCacheSpeedup(t *testing.T) {
	if b, err := coreBusy(time.Second); err == nil && b > 20 {
		t.Skipf("the cores this pins to are %.1f%% busy; a rate row needs them idle", b)
	}
	name := envOr("JITLLM_KVPREFIX_MODEL", "tinyllama-1.1b-q3_K_M.gguf")
	n := envInt("JITLLM_KVPREFIX_TOKENS", 1024)
	rounds := envInt("JITLLM_KVPREFIX_ROUNDS", 6)

	path := jlmOf(t, testmodels.Path(name))
	m, err := Open(path)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	prompt := make([]int32, n)
	for i := range prompt {
		prompt[i] = int32(300 + (i*7)%2000)
	}
	fs, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	const key = "prefix/speedup"

	// Fill the store once, outside every timed arm.
	restored := 0
	func() {
		s := m.NewInstance(n + 16)
		defer s.Close()
		s.SetKVStore(fs)
		s.SetCacheKey(key)
		if _, err := s.Prefill(prompt); err != nil {
			t.Fatal(err)
		}
	}()

	// Warm the process first: the first instance emits and tunes kernels and
	// faults weights in, which would land in the earliest rounds as dispersion.
	for i := 0; i < 2; i++ {
		func() {
			s := m.NewInstance(n + 16)
			defer s.Close()
			if _, err := s.Prefill(prompt); err != nil {
				t.Fatal(err)
			}
		}()
	}

	sample := func(cached bool) float64 {
		s := m.NewInstance(n + 16)
		defer s.Close()
		if cached {
			s.SetKVStore(fs)
			s.SetCacheKey(key)
		}
		t0 := time.Now()
		var err error
		if cached {
			_, err = s.PrefillCached(prompt)
			restored = s.KVRestored()
		} else {
			_, err = s.Prefill(prompt)
		}
		if err != nil {
			t.Fatal(err)
		}
		return time.Since(t0).Seconds()
	}

	pass := func(a, b bool) ([]float64, float64, float64, float64, float64) {
		b0, _ := coreBusy(300 * time.Millisecond)
		var r, ta, tb []float64
		for i := 0; i < rounds; i++ {
			x1, y1 := sample(a), sample(b)
			y2, x2 := sample(b), sample(a)
			// above 1 means the SECOND arm is faster
			r = append(r, x1/y1, x2/y2)
			ta = append(ta, x1, x2)
			tb = append(tb, y1, y2)
		}
		b1, _ := coreBusy(300 * time.Millisecond)
		return r, median(ta), median(tb), b0, b1
	}

	ctl, _, _, c0, c1 := pass(false, false)
	cm, craw := medIQR(ctl)
	t.Logf("A/A self-control (cold BOTH arms)   %.4f   IQR/median %.3f   n=%d   cores %.1f%% -> %.1f%%",
		cm, craw/cm, len(ctl), c0, c1)

	for p := 1; p <= 2; p++ {
		r, cold, warm, b0, b1 := pass(false, true)
		med, raw := medIQR(r)
		gate := "GATES"
		if raw/med > 0.10 {
			gate = "REJECTED (IQR/median over 0.10)"
		}
		if b0 > 20 || b1 > 20 {
			gate = "REJECTED (a tenant was on the cores)"
		}
		// The prediction, from the position count rather than the clock.
		want := float64(n) / float64(n-restored)
		t.Logf("pass %d  cold / cached   %.4fx   IQR/median %.3f   n=%d   cores %.1f%% -> %.1f%%   %s",
			p, med, raw/med, len(r), b0, b1, gate)
		t.Logf("        cold %.0f ms   cached %.0f ms   %d of %d positions from the store "+
			"(%.0f%%), so the skip alone predicts %.2fx",
			cold*1000, warm*1000, restored, n, 100*float64(restored)/float64(n), want)
	}
	if restored == 0 {
		t.Fatal("nothing was restored, so both arms ran the whole prompt and this measured nothing")
	}
}
