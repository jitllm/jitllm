package model

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/metal"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestDeviceDecodeRate prices a device tier against the host on the same model
// in the same process, so "is the GPU being used" has a number rather than an
// assumption.
func TestDeviceDecodeRate(t *testing.T) {
	if os.Getenv("JITLLM_RATE") == "" {
		t.Skip("set JITLLM_RATE=1")
	}
	p := testmodels.Resolve(os.Getenv("JITLLM_MODEL"))
	if p == "" {
		t.Skip("set JITLLM_MODEL")
	}
	spec := os.Getenv("JITLLM_DEV")
	if spec == "" {
		spec = "metal"
	}
	m, err := Open(jlmOf(t, p))
	if err != nil {
		t.Skipf("cannot load: %v", err)
	}
	defer m.Close()

	prof := false
	rate := func(g *tier.GPU, max int) (float64, int) {
		s := m.NewState(256)
		defer s.Close()
		blocks := 0
		if g != nil {
			s.SetDeviceLayers(g, max)
			for _, n := range g.Placed() {
				blocks += n
			}
		}
		for i := 0; i < 4; i++ {
			if _, err := s.Forward(int32(1 + i)); err != nil {
				t.Fatal(err)
			}
		}
		const n = 32
		// The profile starts after placement: around the whole call it is
		// mostly PrepLayer and the weight upload.
		if prof && g != nil {
			fh, err := os.Create(os.Getenv("JITLLM_PROF"))
			if err != nil {
				t.Fatal(err)
			}
			if err := pprof.StartCPUProfile(fh); err != nil {
				t.Fatal(err)
			}
			defer func() { pprof.StopCPUProfile(); fh.Close() }()
		}
		st := time.Now()
		for i := 0; i < n; i++ {
			if _, err := s.Forward(int32(1 + i%16)); err != nil {
				t.Fatal(err)
			}
		}
		return float64(n) / time.Since(st).Seconds(), blocks
	}

	// Bytes and a wall, not only a rate: a fraction of a measured bandwidth is
	// checkable, and above the roofline is a harness bug.
	wb := float64(m.WeightBytes())
	// WeightBytes is the file; BytesPerToken is what a token reads, which is
	// what every bandwidth claim uses (they differ on a tied model).
	bpt := float64(m.BytesPerToken())
	// JITLLM_WALL is the read wall to quote against, in GB/s. It is not measured
	// here: mvbench's streamWall depends on its own working set (AGENTS.md).
	wall := 0.0
	if v := os.Getenv("JITLLM_WALL"); v != "" {
		fmt.Sscanf(v, "%g", &wall)
	}
	pct := func(r float64) string {
		if wall <= 0 {
			return ""
		}
		return fmt.Sprintf(" = %.0f%% of the %.1f GB/s wall", 100*r*bpt/1e9/wall, wall)
	}
	// And what a host token allocates: the allocator is part of the token and
	// no kernel counter sees it.
	var ms0, ms1 runtime.MemStats
	rate(nil, 0) // warm: page-in and codegen allocate once
	runtime.ReadMemStats(&ms0)
	host, _ := rate(nil, 0)
	runtime.ReadMemStats(&ms1)
	t.Logf("  host arm allocations: %.1f objects and %.1f KiB per token, %d GC cycle(s), heap released %+.1f MiB",
		float64(ms1.Mallocs-ms0.Mallocs)/36, float64(ms1.TotalAlloc-ms0.TotalAlloc)/1024/36,
		ms1.NumGC-ms0.NumGC, (float64(ms1.HeapReleased)-float64(ms0.HeapReleased))/(1<<20))
	// Knobs. tier reads no environment, so a pinned setting has to be passed as
	// an option.
	opts := []tier.Option{tier.WithDevices(spec)}
	if v := os.Getenv("JITLLM_DEV_SPLIT"); v != "" {
		n := 0
		fmt.Sscanf(v, "%d", &n)
		opts = append(opts, tier.WithSplit(n))
		t.Logf("  split PINNED to %d", n)
	}
	if os.Getenv("JITLLM_DEV_NOGROUP") != "" {
		opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.NoGroupSplit = true }))
		t.Logf("  in-group split reduction REFUSED; the global partial buffer and its Reduce launch stand")
	}
	if os.Getenv("JITLLM_DEV_VERBOSE") != "" {
		// There is no GroupSplit counter, so the tuner's own "in-group" lines
		// are the evidence that the knob selected a different kernel.
		opts = append(opts, tier.WithVerbose(true))
	}
	// The row tile. RowtSplitTiles is what says the tile-and-split combination
	// ran: a shape the tuner sends to Split == 1 tiles without it.
	if v := os.Getenv("JITLLM_DEV_ROWT"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			t.Fatalf("JITLLM_DEV_ROWT=%q is not a positive integer", v)
		}
		opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.DecodeRowt = n }))
	}
	// The unretained command buffer is a measurement, not a setting: it prices
	// whether fixed per-token device time is work or retain/release bookkeeping.
	// Safe only while buffers outlive the submission, which holds on a fully
	// resident model and not across a page-in.
	if os.Getenv("JITLLM_DEV_UNRETAINED") != "" {
		backend.SetMetalUnretained(true)
	}
	if v := os.Getenv("JITLLM_DEV_HEADSPLIT"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			t.Fatalf("JITLLM_DEV_HEADSPLIT=%q is not a positive integer", v)
		}
		opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.HeadSplit = n }))
	}
	if os.Getenv("JITLLM_DEV_PERLAYER") != "" {
		opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.PerLayerSubmit = true }))
	}
	if v := os.Getenv("JITLLM_DEV_MVREPEAT"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			t.Fatalf("JITLLM_DEV_MVREPEAT=%q is not a non-negative integer", v)
		}
		opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.HeadMatvecRepeat = n }))
	}
	if v := os.Getenv("JITLLM_DEV_NORMREPEAT"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			t.Fatalf("JITLLM_DEV_NORMREPEAT=%q is not a non-negative integer", v)
		}
		opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.HeadNormRepeat = n }))
	}
	if os.Getenv("JITLLM_DEV_RESIDENCY") != "" {
		backend.SetMetalResidency(true)
	}
	if os.Getenv("JITLLM_DEV_FORCEGROUP") != "" {
		opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.ForceGroupSplit = true }))
		t.Logf("  in-group split reduction FORCED on every eligible shape")
	}
	g, err := openTier(t, opts...)
	if err != nil || g == nil {
		t.Logf("%s: not present (%v)", spec, err)
		t.Logf("host %.2f tok/s = %.1f GB/s%s over %.0f MiB of weights",
			host, host*bpt/1e9, pct(host), wb/(1<<20))
		return
	}
	defer g.Close()
	// Profile the device arm (the device submission path).
	if os.Getenv("JITLLM_PROF") != "" {
		prof = true
		rate(g, -1)
		prof = false
		t.Logf("  cpu profile written to %s", os.Getenv("JITLLM_PROF"))
	}
	// Probe that the device arm actually runs on the device: a declined block
	// leaves the host doing it, correctly and slowly.
	probe := m.NewState(256)
	probe.SetDeviceLayers(g, -1)
	for i := 0; i < 8; i++ {
		if _, err := probe.Forward(int32(1 + i)); err != nil {
			t.Fatal(err)
		}
	}
	probe.Close()
	dev, blocks := rate(g, -1)
	c0, w0, b0 := metal.SubmitStats()
	gns0, gbuf0 := metal.GPUBusy()
	gap0, gapN0 := metal.GPUIdle()
	hg0, hgN0 := metal.HostGap()
	rt0, rtN0 := metal.RoundTrip()
	_, _, dl0 := metal.SubmitTimes()
	bar0 := metal.Barriers()
	devMs0 := time.Now()
	rate(g, -1)
	devWall := time.Since(devMs0)
	gns1, gbuf1 := metal.GPUBusy()
	gap1, gapN1 := metal.GPUIdle()
	hg1, hgN1 := metal.HostGap()
	if hgN1 > hgN0 {
		t.Logf("  host from a Wait's return to the next commit: %.3f ms a token over %d",
			float64(hg1-hg0)/1e6/float64(hgN1-hgN0), hgN1-hgN0)
	}
	if rt1, rtN1 := metal.RoundTrip(); rtN1 > rtN0 {
		t.Logf("  commit-to-start plus wake-up latency: %.3f ms a round trip over %d",
			float64(rt1-rt0)/1e6/float64(rtN1-rtN0), rtN1-rtN0)
	}
	if gapN1 > gapN0 {
		t.Logf("  GPU idle between command buffers: %.3f ms/token over %d gaps (the device's own clock)",
			float64(gap1-gap0)/1e6/36, gapN1-gapN0)
	}
	_, _, dl1 := metal.SubmitTimes()
	if bar := metal.Barriers() - bar0; bar > 0 {
		t.Logf("  concurrent encoder: %.0f buffer barriers a token between %.0f dispatches",
			float64(bar)/36, float64(dl1-dl0)/36)
	}
	c1, w1, b1 := metal.SubmitStats()
	if c1 > c0 {
		t.Logf("  submission over 36 tokens: %d commits, %d Wait calls, %d BLOCKED (%.1f commits/token)",
			c1-c0, w1-w0, b1-b0, float64(c1-c0)/36)
	}
	e1, wn1, l1 := metal.SubmitTimes()
	sn, sns := metal.SoloStats()
	if sn > 0 {
		t.Logf("  SOLO dispatches (a command buffer each, committed AND waited): "+
			"%.1f/token costing %.3f ms/token", float64(sn)/float64(c1), float64(sns)/1e6/float64(c1))
	}
	if snN, snNs := metal.SessionStats(); snN > 0 {
		t.Logf("  inside backend Session: %.1f sessions/token, %.3f ms/token TOTAL "+
			"(this bounds submission from above)",
			float64(snN)/float64(c1), float64(snNs)/1e6/float64(c1))
	}
	if nn0, cn0 := metal.PhaseStats(); nn0+cn0 > 0 {
		t.Logf("  Session phases: %.3f ms/token making the command buffer, "+
			"%.3f ms/token endEncoding+commit",
			float64(nn0)/1e6/float64(c1), float64(cn0)/1e6/float64(c1))
	}
	if l1 > 0 {
		// The counters are bracketed round the timed arm; dividing cumulative
		// launches by cumulative commits once averaged in the warm-ups and the
		// host arm and undercounted dispatches tenfold.
		const nTok = 36
		t.Logf("  CPU split: %.3f ms/token encoding %.0f dispatches, %.3f ms/token blocked",
			float64(e1)/1e6/float64(c1), float64(dl1-dl0)/nTok, float64(wn1)/1e6/float64(c1))
	}
	// Sweep the placement to separate per-layer cost from bandwidth: both grow
	// with blocks on the card, and one full-offload number cannot tell them apart.
	if os.Getenv("JITLLM_SWEEP") != "" {
		for _, n := range []int{0, 1, 2, 4, 8, m.Cfg.NLayer / 2, m.Cfg.NLayer} {
			if n > m.Cfg.NLayer {
				continue
			}
			// Report the limit, not g.Placed(): a GPU keeps a block once owned,
			// so Placed() reads the full count on every arm after the first.
			//
			// The GPU's own clock and the dispatch count are bracketed round each
			// arm, so GPU busy against layers is a regression: its slope is one
			// block's streaming time (known from the container) plus per-layer
			// non-streaming time, read off rather than fitted.
			gn0, gb0 := metal.GPUBusy()
			_, _, dl0 := metal.SubmitTimes()
			r, _ := rate(g, n)
			gn1, gb1 := metal.GPUBusy()
			_, _, dl1 := metal.SubmitTimes()
			const nTok = 36
			gpu, disp := 0.0, 0.0
			if gb1 > gb0 {
				gpu = float64(gn1-gn0) / 1e6 / nTok
				disp = float64(dl1-dl0) / nTok
			}
			t.Logf("  %3d of %d layers on the card: %8.2f tok/s  (%.3f ms/token, "+
				"GPU %.3f ms/token over %.0f dispatches)",
				n, m.Cfg.NLayer, r, 1000/r, gpu, disp)
		}
	}
	// The token split in two with no fitted parameter: CPU timers bound only
	// submission, while GPUBusy is the device's own clock, so token minus GPU
	// busy is the off-device remainder as a measurement. The container's byte
	// counts give the sweep's slope and intercept their meaning (one block's
	// streaming time, the dense read's).
	if pg, dn := m.PageSize(), m.DenseBytes(); pg > 0 && wall > 0 {
		// The dense region is not the dense read: an untied model's region
		// holds token_embd and output.weight, and a token reads the whole head
		// but one embedding row. BytesPerToken minus the paged part is the
		// dense read.
		denseRead := float64(bpt) - float64(pg)*float64(m.Cfg.NLayer)
		t.Logf("  container: %d block(s) of %.1f MB = %.1f MB paged, %.1f MB dense "+
			"region of which %.1f MB is READ per token; at the wall that is "+
			"%.3f ms per block and %.3f ms for the dense read",
			m.Cfg.NLayer, float64(pg)/1e6, float64(pg)*float64(m.Cfg.NLayer)/1e6,
			float64(dn)/1e6, denseRead/1e6,
			float64(pg)/(wall*1e9)*1000, denseRead/(wall*1e9)*1000)
	}
	// Under one submission per block the last command buffer is the head's
	// alone, so its GPU clock prices the head sequence in situ, once and cold.
	if lastNs := metal.GPUBusyLast(); lastNs > 0 {
		t.Logf("  last command buffer: %.3f ms of GPU time (under one submission "+
			"per block this is the HEAD SEQUENCE, measured in situ)",
			float64(lastNs)/1e6)
	}
	if gbuf1 > gbuf0 {
		// 36 tokens: rate() warms 4 and times 32.
		const nTok = 36
		gpuMs := float64(gns1-gns0) / 1e6 / nTok
		tokMs := float64(devWall.Nanoseconds()) / 1e6 / nTok
		t.Logf("  GPU clock: %.3f ms/token executing over %.1f command buffer(s)/token; "+
			"the arm's own wall is %.3f ms/token -> %.3f ms (%.0f%%) NOT on the device",
			gpuMs, float64(gbuf1-gbuf0)/nTok, tokMs, tokMs-gpuMs,
			100*(tokMs-gpuMs)/tokMs)
	}
	// What the tile actually reached, beside the rate (see JITLLM_DEV_ROWT).
	if st := g.Stats(); st.RowtTiles+st.RowtPlain+st.RowtSplitTiles > 0 {
		t.Logf("  decode matvecs: %d tiled (%d of them WITH a split), %d plain",
			st.RowtTiles, st.RowtSplitTiles, st.RowtPlain)
	}
	t.Logf("%s: host %.2f tok/s = %.1f GB/s%s, device %.2f tok/s = %.1f GB/s%s, "+
		"%d of %d blocks placed = %.2fx; %d B/token over %.0f MiB of weights",
		spec, host, host*bpt/1e9, pct(host), dev, dev*bpt/1e9, pct(dev),
		blocks, m.Cfg.NLayer, dev/host, m.BytesPerToken(), wb/(1<<20))
}
