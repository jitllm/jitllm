package model

import (
	"errors"
	"hash/fnv"
	"strconv"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// windowFixtures are the fixtures with windowed layers: exaone4's sliding
// window of 4, cohere2's sliding layers, llama4's chunked ones, gpt-oss's
// alternating window and OLMo 3's. DeepSeek V4's every block slides, and its
// compressed blocks' entries are the history that grows beside the windows.
var windowFixtures = []string{"synth-exaone4.gguf", "synth-cohere2.gguf", "synth-llama4.gguf", "synth-gptoss.gguf",
	"synth-olmo3.gguf"}

// TestWindowedLayersHoldTheirWindow holds a windowed layer's history to its
// window over a long decode, on the host and on every device: the paging
// principle says memory is bounded by what attention reads, and a sliding
// layer reads its window, not the context.
//
// On the host the shipping arm is compared, logits bit for bit at every
// step, against a control that keeps every page (kvCache.keepWindow): a
// released page is one nothing reads again, so the answer cannot move. Its
// windowed layers must stay within the window and the slack, and a full layer
// beside them must still hold the whole history (the selection check that
// release did not simply stop everything growing). The violation releases the
// page the window starts in (kvCache.relFault) and must change a logit or fail.
//
// On a device every block and the head are placed and each step's logits are
// held to the host control; the device's releases are counted
// (Stats.KVWindowReleased), exactly: every windowed layer gives back every page
// below the window start of the last position less nn.KVWindowSlack.
func TestWindowedLayersHoldTheirWindow(t *testing.T) {
	specs := testdevices.Specs("cuda", "vulkan", "metal")
	for _, name := range windowFixtures {
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(name)), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			c := m.Cfg
			var win, full []int
			for li := 0; li < c.NLayer; li++ {
				if kvPagePositions(c, li, 0) == 0 {
					continue // a recurrent layer keeps no history
				}
				if c.SWA(li) && c.SWAWindow > 0 {
					win = append(win, li)
				} else {
					full = append(full, li)
				}
			}
			// DeepSeek V4 has no full layer: its compressed blocks' entries
			// (kvCache.ent) are what must keep growing while the windows release.
			var ents []int
			for li := 0; li < c.NLayer; li++ {
				if c.CompRateAt(li) > 0 {
					ents = append(ents, li)
				}
			}
			if len(win) == 0 || len(full) == 0 && len(ents) == 0 {
				t.Fatalf("%d windowed and %d full layer(s), %d with entries: this fixture does not have "+
					"both, so the gate proves nothing on it", len(win), len(full), len(ents))
			}
			maxSeq := min(c.NCtx, 1536)
			n := maxSeq - 8
			toks := make([]int32, n)
			for i := range toks {
				toks[i] = int32(3 + (i*7)%61)
			}
			// run decodes toks one at a time, hashing every step's logits, and
			// reports the peak resident pages of the windowed layers.
			run := func(g *tier.GPU, keep, fault bool) (sums []uint64, last []float32, peak int, s *State, err error) {
				s = m.NewState(maxSeq)
				if g != nil {
					if err := s.SetDevice(g); err != nil {
						t.Fatal(err)
					}
				}
				s.kv.keepWindow, s.kv.relFault = keep, fault
				for _, tk := range toks {
					lg, err := s.Forward(tk)
					if err != nil {
						return sums, last, peak, s, err
					}
					sums = append(sums, hashLogits(lg))
					last = append(last[:0], lg...)
					for _, li := range win {
						pg := &s.kv.layers[li]
						r := 0
						for j := range pg.k {
							if pg.resident(j) {
								r++
							}
						}
						peak = max(peak, r)
					}
				}
				return sums, last, peak, s, nil
			}
			ctl, ctlLast, ctlPeak, cs, err := run(nil, true, false)
			if err != nil {
				t.Fatal(err)
			}
			cs.Close()
			got, _, peak, hs, err := run(nil, false, false)
			if err != nil {
				t.Fatal(err)
			}
			wl := &hs.kv.layers[win[0]]
			bound := (wl.win+nn.KVWindowSlack)/wl.p + 3
			// The history that must still be whole: a full layer's positions,
			// or a compressed block's entries.
			fl, held := (*kvPages)(nil), n
			if len(full) > 0 {
				fl = &hs.kv.layers[full[0]]
			} else {
				fl, held = &hs.kv.ent[ents[0]], n/hs.kv.entRate(ents[0])
			}
			fullPages := 0
			for j := range fl.k {
				if fl.resident(j) {
					fullPages++
				}
			}
			hs.Close()
			t.Logf("host: %d positions, window %d (chunked %v), page %d: windowed layers peak at %d page(s) "+
				"(bound %d) against %d kept by the control; a full layer (or a block's entries) holds %d "+
				"page(s) of %d",
				n, wl.win, wl.chunked, wl.p, peak, bound, ctlPeak, fullPages, fl.p)
			if peak > bound {
				t.Fatalf("a windowed layer held %d pages, past its window and slack's %d", peak, bound)
			}
			if ctlPeak <= bound {
				t.Fatalf("the control held only %d page(s): the decode never outgrew the window, so this "+
					"gate shows nothing about releasing behind it", ctlPeak)
			}
			if want := (held + fl.p - 1) / fl.p; fullPages != want {
				t.Fatalf("a full layer (or a block's entries) holds %d page(s) of the %d its history needs",
					fullPages, want)
			}
			for i := range ctl {
				if got[i] != ctl[i] {
					t.Fatalf("step %d: the logits moved when pages behind the window were released", i)
				}
			}
			bad, _, _, bs, err := run(nil, false, true)
			bs.Close()
			if err == nil && equalSums(bad, ctl) {
				t.Fatal("releasing the page the window starts in changed nothing: the gate cannot tell a " +
					"window read from a released page")
			}
			t.Logf("violation (the window's own page released): %v", violationWhat(err, bad, ctl))

			for _, spec := range specs {
				t.Run(spec, func(t *testing.T) { windowOnDevice(t, m, spec, run, win, n, maxSeq, ctlLast) })
			}
		})
	}
}

// windowRun is one decode of TestWindowedLayersHoldTheirWindow: on g (nil for
// the host), keeping every page (keep) or releasing the window's own (fault).
type windowRun func(g *tier.GPU, keep, fault bool) (sums []uint64, last []float32, peak int, s *State, err error)

// windowOnDevice is the device arm: every block and the head placed, the
// shipping tier against one that keeps every page (tier.Config.KVKeepWindow),
// bit for bit at every step, with the device's split tuner off so the two
// tiers reduce in one order.
func windowOnDevice(t *testing.T, m *Model, spec string, run windowRun, win []int, n, maxSeq int, host []float32) {
	c := m.Cfg
	open := func(keep bool) *tier.GPU {
		o := append([]tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff),
			tier.WithConfig(func(c *tier.Config) { c.KVKeepWindow = keep })}, testTierOpts(t)...)
		g, err := tier.OpenWith(o...)
		if err != nil || g == nil {
			noDevice(t, spec, err)
		}
		return g
	}
	gc := open(true)
	ctl, _, _, cs, err := run(gc, false, false)
	if err != nil {
		t.Fatal(err)
	}
	composedBlocksRan(t, composedOf(m, gc), 0, n*cs.GPULayers(), gc)
	cs.Close()
	gc.Close()
	g := open(false)
	defer g.Close()
	r0 := g.Stats().KVWindowReleased
	got, last, _, ds, err := run(g, false, false)
	if err != nil {
		t.Fatal(err)
	}
	placed, head := ds.GPULayers(), ds.HeadOnDevice()
	ds.Close()
	composedBlocksRan(t, composedOf(m, g), 0, n*placed, g)
	if placed != c.NLayer || !head {
		t.Skipf("CARD TOO SMALL: %d of %d blocks, head %v -- this arm proved nothing", placed, c.NLayer, head)
	}
	rel := g.Stats().KVWindowReleased - r0
	P := devKVPage(maxSeq)
	want := 0
	for _, li := range win {
		// A KV-sharing block (Gemma 3n, Gemma 4's E-models) holds no history
		// of its own on the device: its source releases for both.
		if c.KVShared(li) {
			continue
		}
		want += keyStartOf(c, li, max(0, n-1-nn.KVWindowSlack)) / P
	}
	t.Logf("%s: %d of %d blocks placed, page %d: %d page(s) released behind windows (want %d); the "+
		"last logits NMSE %.3e from the host's", g.Name(), placed, c.NLayer, P, rel, want, logitNMSE(last, host))
	if want == 0 {
		t.Fatal("no page lay behind a window on the device: the decode is too short to show release")
	}
	if rel != want {
		t.Fatalf("the device released %d windowed page(s), the window says %d", rel, want)
	}
	for i := range ctl {
		if got[i] != ctl[i] {
			t.Fatalf("step %d: the device's logits moved when pages behind the window were released", i)
		}
	}
}

// devKVPage is the device pool's page for a session of maxSeq positions
// (tier's kvPage): the smallest power of two from 64 that covers it, at most
// 256.
func devKVPage(maxSeq int) int {
	p := 64
	for p < 256 && p < maxSeq {
		p *= 2
	}
	return p
}

// keyStartOf is the first key layer li attends at pos.
func keyStartOf(c *Config, li, pos int) int {
	w0, _ := c.AttnWindow(li, pos)
	return w0
}

// hashLogits is a step's logits bit for bit.
func hashLogits(lg []float32) uint64 {
	h := fnv.New64a()
	if len(lg) > 0 {
		h.Write(unsafe.Slice((*byte)(unsafe.Pointer(&lg[0])), 4*len(lg)))
	}
	return h.Sum64()
}

func equalSums(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// violationWhat says how a violation failed: an error, or the first step whose
// logits moved.
func violationWhat(err error, bad, ctl []uint64) string {
	if err != nil {
		if errors.Is(err, ErrNoPage) {
			return "the read found the page gone: " + err.Error()
		}
		return err.Error()
	}
	for i := range bad {
		if i >= len(ctl) || bad[i] != ctl[i] {
			return "the logits move at step " + strconv.Itoa(i)
		}
	}
	return "nothing moved"
}
