//go:build linux

package model

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestStreamedExpertBankMatchesTheResidentOne compares a mixture's routed bank
// assembled per token (StreamExperts) against the same bank held resident on
// the card. The kernel cannot tell a compact bank from the full one (see
// backend.TestIndexedMatVecMatchesACompactBank), so the bar is bit equality:
// sheets uploaded in the wrong order would be finite and fluent. The tuner is
// pinned, and the stream counters show which arm ran.
func TestStreamedExpertBankMatchesTheResidentOne(t *testing.T) {
	// wideMoE is Qwen3-Next's mixture shape (512 experts, 10 routed), so the
	// compact bank is genuinely smaller than the resident one.
	//
	// The page budget must evict: fill() reads only the routed sheets
	// (nn.LayerWeights.EnsureExperts), and on a fully resident container
	// nothing is read, so a wrong sheet offset would be invisible.
	probe := hybridModelOpt(t, hyOpt{moe: true, wideMoE: true})
	page := probe.container.H.PageSize
	probe.Close()
	m := hybridModelOpt(t, hyOpt{moe: true, wideMoE: true, budget: 2 * page})
	defer m.Close()
	if !m.container.CanEvict() {
		t.Fatalf("the budget is %d bytes over pages of %d and nothing is evicted: "+
			"the narrowed expert read would never run", m.container.Budget(), page)
	}
	if m.Cfg.NExpertUsed >= m.Cfg.NExpert || m.Cfg.NExpertUsed < 2 {
		t.Fatalf("the fixture routes %d of %d experts; streaming needs 2 <= used < total",
			m.Cfg.NExpertUsed, m.Cfg.NExpert)
	}

	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	run := func(stream func(*tier.Config)) ([][]float32, tier.Stats, int, uint64) {
		t.Helper()
		opts := []tier.Option{tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff)}
		if stream != nil {
			opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.StreamExperts = true; stream(c) }))
		}
		g, err := openTier(t, opts...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		s := m.NewState(16)
		defer s.Close()
		s.SetDeviceLayers(g, m.Cfg.NLayer)
		placed := s.GPULayers()
		if placed == 0 {
			t.Skipf("the device took no block of this fixture: %s", g.Err())
		}
		out := make([][]float32, len(ids))
		for i, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatalf("stream=%v: %v", stream != nil, err)
			}
			out[i] = append([]float32(nil), l...)
		}
		// Read while the tier is open: after Close the driver reclaims the
		// context and the ledger is gone.
		return out, g.Stats(), placed, g.Bytes()
	}

	want, rs, rp, rb := run(nil)
	// Four streamed arms: the fixture's sheets are small, so the default
	// gathers them; StreamDirectBytes 1 sends every sheet straight from its
	// frame; four groups put reads in flight behind the transfers. Each
	// must be the resident answer bit for bit, and each must have taken its
	// path (StreamDirect, StreamGathered, StreamOverlaps).
	arms := []struct {
		name    string
		cfg     func(*tier.Config)
		direct  bool
		overlap bool
	}{
		{"gathered", func(c *tier.Config) { c.StreamGroups = 1; c.StreamCacheSlots = -1 }, false, false},
		{"gathered, four groups", func(c *tier.Config) { c.StreamGroups = 4; c.StreamCacheSlots = -1 }, false, true},
		{"direct, pageable", func(c *tier.Config) {
			c.StreamDirectBytes = 1
			c.StreamGroups = 1
			c.StreamNoPin = true
			c.StreamCacheSlots = -1
		}, true, false},
		{"direct", func(c *tier.Config) { c.StreamDirectBytes = 1; c.StreamGroups = 1; c.StreamCacheSlots = -1 }, true, false},
		{"direct, a piece a half", func(c *tier.Config) {
			c.StreamDirectBytes = 1
			c.StreamGroups = 1
			c.StreamPinHalf = 1
			c.StreamCacheSlots = -1
		}, true, false},
		{"expert cache", func(c *tier.Config) { c.StreamCacheSlots = 13; c.StreamGroups = 1 }, true, false},
		{"expert cache, four groups", func(c *tier.Config) { c.StreamCacheSlots = 13; c.StreamGroups = 4 }, true, true},
		{"expert cache sized after placement", func(c *tier.Config) { c.StreamGroups = 1 }, true, false},
		{"expert cache, prefetch", func(c *tier.Config) { c.StreamCacheSlots = 13; c.StreamGroups = 1; c.StreamPrefetch = true }, true, false},
		{"direct, four groups", func(c *tier.Config) { c.StreamDirectBytes = 1; c.StreamGroups = 4; c.StreamCacheSlots = -1 }, true, true},
	}
	for _, arm := range arms {
		t.Run(arm.name, func(t *testing.T) {
			streamedMatches(t, run, arm.cfg, arm.direct, arm.overlap, ids, want, rs, rp, rb)
		})
	}
}

func streamedMatches(t *testing.T, run func(func(*tier.Config)) ([][]float32, tier.Stats, int, uint64),
	cfg func(*tier.Config), direct, overlap bool, ids []int32, want [][]float32, rs tier.Stats, rp int, rb uint64) {
	got, ss, sp, sb := run(cfg)
	if direct && (ss.StreamDirect == 0 || ss.StreamGathered != 0) {
		t.Fatalf("%d planes sent direct and %d gathered: the direct path did not run alone",
			ss.StreamDirect, ss.StreamGathered)
	}
	if !direct && (ss.StreamGathered == 0 || ss.StreamDirect != 0) {
		t.Fatalf("%d planes gathered and %d sent direct: the gathered path did not run alone",
			ss.StreamGathered, ss.StreamDirect)
	}
	// A direct arm on a device with page-locked memory goes through it unless
	// told not to; the pageable arm must not.
	if pinned := direct && !noPin(cfg) && ss.StreamPinned == 0 && ss.StreamDirect > 0; pinned && pinsHost(t) {
		t.Fatalf("%d planes sent direct and none through page-locked memory", ss.StreamDirect)
	}
	if noPin(cfg) && ss.StreamPinned != 0 {
		t.Fatalf("the pageable arm sent %d planes through page-locked memory", ss.StreamPinned)
	}
	// A cache of 13 sheets for a selection of 10 out of 512 both hits and
	// evicts over eight tokens; a cache arm that did neither tested the plain
	// bank.
	var c tier.Config
	cfg(&c)
	if c.StreamCacheSlots >= 0 && (ss.StreamCacheHits == 0 || ss.StreamCacheMisses == 0 || ss.StreamCacheShort != 0) {
		t.Fatalf("expert cache: %d hits, %d misses, %d blocks without room: the cache did not run",
			ss.StreamCacheHits, ss.StreamCacheMisses, ss.StreamCacheShort)
	}
	if c.StreamCacheSlots >= 0 {
		t.Logf("expert cache: %d sheets, %d hits, %d misses", ss.StreamCacheSize, ss.StreamCacheHits, ss.StreamCacheMisses)
	}
	if c.StreamCacheSlots < 0 && ss.StreamCacheSize != 0 {
		t.Fatalf("an arm with the cache off has a %d-sheet cache", ss.StreamCacheSize)
	}
	if c.StreamPrefetch && (ss.StreamPrefetched == 0 || ss.ProbeExperts == 0) {
		t.Fatalf("the prefetch arm read %d experts ahead and scored %d: it did not run",
			ss.StreamPrefetched, ss.ProbeExperts)
	}
	if !c.StreamPrefetch && ss.StreamPrefetched != 0 {
		t.Fatalf("an arm without the prefetch read %d experts ahead", ss.StreamPrefetched)
	}
	if overlap != (ss.StreamOverlaps > 0) {
		t.Fatalf("%d reads issued behind a transfer, want overlap %v", ss.StreamOverlaps, overlap)
	}
	t.Logf("resident: %d blocks placed, %d streamed, %d fills, %d device bytes",
		rp, rs.StreamBlocks, rs.StreamFills, rb)
	t.Logf("streamed: %d blocks placed, %d streamed, %d fills, %d device bytes (%.2fx less)",
		sp, ss.StreamBlocks, ss.StreamFills, sb, float64(rb)/float64(sb))

	// The device bytes are asserted: a tier that uploaded the full bank and
	// also assembled a compact one would give equal logits.
	if sb >= rb {
		t.Errorf("the streamed arm holds %d device bytes against the resident arm's %d: "+
			"the bank is still on the card", sb, rb)
	}

	// The selection check, both ways: the streamed arm must stream, and the
	// resident arm must not (the knob must not leak between tiers).
	if ss.StreamBlocks == 0 || ss.StreamFills == 0 {
		t.Fatalf("the streamed arm placed %d streamed blocks and filled %d times: "+
			"the configuration under test never ran", ss.StreamBlocks, ss.StreamFills)
	}
	if rs.StreamBlocks != 0 || rs.StreamFills != 0 {
		t.Errorf("the RESIDENT arm reports %d streamed blocks and %d fills",
			rs.StreamBlocks, rs.StreamFills)
	}
	if sp != rp {
		t.Errorf("the streamed arm placed %d blocks and the resident one %d: "+
			"the two arms are not the same placement", sp, rp)
	}
	// One fill per streamed block per token, and nothing else: a block that
	// filled twice has uploaded a bank it then overwrote, and one that filled
	// none ran against whatever the buffer held.
	if n := ss.StreamBlocks * len(ids); ss.StreamFills != n {
		t.Errorf("%d fills over %d tokens with %d streamed blocks, want %d",
			ss.StreamFills, len(ids), ss.StreamBlocks, n)
	}

	for p := range ids {
		for i := range want[p] {
			w, gv := want[p][i], got[p][i]
			if math.IsNaN(float64(gv)) || math.IsInf(float64(gv), 0) {
				t.Fatalf("pos %d logit %d is %g streamed against %g resident: "+
					"not finite, which no tolerance can see", p, i, gv, w)
			}
			if w != gv {
				t.Fatalf("pos %d logit %d: streamed %v, resident %v -- a compact "+
					"bank runs the same kernel over the same bytes, so this is "+
					"the sheets or their order, not rounding", p, i, gv, w)
			}
		}
	}
}

// noPin reports whether an arm's configuration turns page-locked transfers off.
func noPin(cfg func(*tier.Config)) bool {
	var c tier.Config
	cfg(&c)
	return c.StreamNoPin
}

// pinsHost reports whether the gate's device can page-lock host memory, which
// is CUDA's alone (backend.HostPinner).
func pinsHost(t *testing.T) bool {
	g, err := openTier(t, tier.WithDevices("gpu:0"))
	if err != nil || g == nil {
		return false
	}
	defer g.Close()
	return strings.HasPrefix(strings.ToLower(g.Name()), "cuda") || strings.Contains(g.Name(), "[ptx]")
}

// TestAutoStreamPlacesWhatCannotFit: on a device budget below a mixture
// block's resident size and above its base, the default placement streams
// the block (nn.AutoStreamer) instead of leaving it on the host, with the
// resident answer bit for bit; NoAutoStream leaves it home. The fixture's
// blocks are each refused by size first, so the auto path is the only way
// onto the card.
func TestAutoStreamPlacesWhatCannotFit(t *testing.T) {
	probe := hybridModelOpt(t, hyOpt{moe: true, wideMoE: true})
	page := probe.container.H.PageSize
	probe.Close()
	m := hybridModelOpt(t, hyOpt{moe: true, wideMoE: true, budget: 2 * page})
	defer m.Close()
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	var biggest, base uint64
	for li := 0; li < m.Cfg.NLayer; li++ {
		total, bank := m.blockBytes(li)
		if bank > 0 && total > biggest {
			biggest, base = total, total-bank
		}
	}
	if biggest == 0 {
		t.Fatal("the fixture has no mixture block")
	}
	trials := 0
	run := func(opts ...tier.Option) ([][]float32, tier.Stats, int) {
		t.Helper()
		opts = append([]tier.Option{tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff)}, opts...)
		g, err := openTier(t, opts...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		s := m.NewState(16)
		defer s.Close()
		s.SetDeviceLayers(g, m.Cfg.NLayer)
		// An auto-streamed placement arms the trial against the host; a
		// forced or resident one does not.
		if trial := s.seam != nil; trial != (s.autoStreamed > 0) {
			t.Fatalf("%d blocks auto-streamed and the stream trial armed is %v", s.autoStreamed, trial)
		}
		trials += s.autoStreamed
		out := make([][]float32, len(ids))
		for i, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out[i] = append([]float32(nil), l...)
		}
		return out, g.Stats(), s.GPULayers()
	}
	_, rs, rp := run()
	if rp == 0 || rs.StreamBlocks != 0 {
		t.Fatalf("the resident arm placed %d blocks, %d streamed", rp, rs.StreamBlocks)
	}
	// Between the base and the whole block, with room for scratch: every
	// mixture block is refused resident and fits streamed.
	budget := tier.WithBudget(biggest - 1)
	t.Logf("mixture block %d bytes, base %d; device budget %d", biggest, base, biggest-1)
	// The auto arm sizes its cache after placement, so it holds the same
	// blocks as the forced arm, which streams them with no cache at all.
	// The fixture's scratch leaves this budget no room for a cache, so the
	// cache's sizing is TestStreamedExpertBankMatchesTheResidentOne's.
	// By default an auto-streamed block runs its experts on the host
	// (TestHybridExpertsMatchTheHost holds that path); the bit-equality arm
	// sends the sheets, as the forced placement does.
	if _, hs, _ := run(budget); hs.HybridRuns == 0 || hs.StreamFills != 0 {
		t.Fatalf("the default auto-streamed run made %d hybrid block-steps and %d fills, want hybrid alone",
			hs.HybridRuns, hs.StreamFills)
	}
	got, ss, sp := run(budget, tier.WithConfig(func(c *tier.Config) { c.NoHybrid = true }))
	// The bar is the forced streamed placement on the same budget: the same
	// blocks on the card, so bit equality holds (a resident arm places more
	// blocks and differs in the device-host band).
	want, fs, fp := run(budget, tier.WithConfig(func(c *tier.Config) { c.StreamExperts = true; c.NoAutoStream = true }))
	if fp != sp || fs.StreamBlocks != ss.StreamBlocks {
		t.Fatalf("auto placed %d blocks (%d streamed), forced streaming %d (%d)", sp, ss.StreamBlocks, fp, fs.StreamBlocks)
	}
	if ss.StreamBlocks == 0 || ss.StreamFills == 0 {
		t.Fatalf("auto: %d blocks placed, %d streamed, %d fills: the block was not streamed (%s)",
			sp, ss.StreamBlocks, ss.StreamFills, "the size decline left it home")
	}
	for p := range ids {
		for i := range want[p] {
			if want[p][i] != got[p][i] {
				t.Fatalf("pos %d logit %d: auto-streamed %v, forced %v", p, i, got[p][i], want[p][i])
			}
		}
	}
	if trials == 0 {
		t.Fatal("no placement auto-streamed, so the stream trial was never armed")
	}
	// The trial runs through to a decision: the blocks go home and come back
	// streamed between its runs, and every token stays finite.
	{
		g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff), budget)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		s := m.NewState(64)
		s.SetDeviceLayers(g, m.Cfg.NLayer)
		// The CLI turns seam tuning off after placement; that must not
		// disarm the trial.
		s.SetSeamTuning(false)
		if s.seam == nil {
			t.Fatal("the stream trial was not armed, or seam tuning off disarmed it")
		}
		s.seam.warmup, s.seam.perRun, s.seam.rounds = 1, 2, 1
		for i := 0; i < 60 && !s.seam.settled; i++ {
			l, err := s.Forward(int32(1 + i%8))
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range l {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("token %d of the trial is not finite", i)
				}
			}
		}
		if !s.seam.settled {
			t.Fatal("the stream trial did not settle in 60 tokens")
		}
		t.Logf("stream trial settled on %s: %s", s.seam.name(s.seam.best), s.seam.why)
		s.Close()
		g.Close()
	}
	// A budget that holds one mixture block resident but not every one: the
	// rest are refused for room, and stream instead of going home.
	{
		_, rs2, rp2 := run(tier.WithBudget(biggest * 3 / 2))
		if rs2.StreamBlocks == 0 || rp2 < 2 {
			t.Fatalf("a card with room for one resident block: %d placed, %d streamed -- the "+
				"blocks refused for room went to the host", rp2, rs2.StreamBlocks)
		}
	}
	_, ns, _ := run(budget, tier.WithConfig(func(c *tier.Config) { c.NoAutoStream = true }))
	if ns.StreamBlocks != 0 {
		t.Fatalf("NoAutoStream streamed %d blocks", ns.StreamBlocks)
	}
	t.Logf("auto: %d blocks placed, %d streamed, a %d-sheet cache, %d hits %d misses; NoAutoStream: %d streamed",
		sp, ss.StreamBlocks, ss.StreamCacheSize, ss.StreamCacheHits, ss.StreamCacheMisses, ns.StreamBlocks)
}
