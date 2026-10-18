package model

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestHybridDecodeIsCaptured: a fully placed hybrid's decode must be recorded
// as a CUDA graph, like every other resident model's. A write from inside emit
// (such as bs.dRows) is a memcpy during stream capture, which invalidates the
// capture and silently disables graphs.
func TestHybridDecodeIsCaptured(t *testing.T) {
	m := hybridModelOpt(t, hyOpt{tiled: true})
	g, err := openTier(t, tier.WithDevices(testdevices.Like(t, "cuda:0")), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		t.Skipf("no cuda device: %v", err)
	}
	defer g.Close()
	s := m.NewState(32)
	defer s.Close()
	s.SetDeviceLayers(g, m.Cfg.NLayer)
	if s.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("%d of %d blocks placed; this gate is about a fully resident hybrid: %s",
			s.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	for _, id := range []int32{1, 2, 3, 4, 5, 6} {
		if _, err := s.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	st := g.Stats()
	if strings.Contains(st.LastErr, "graphs disabled") {
		t.Fatalf("the capture failed: %s", st.LastErr)
	}
	if st.Captures == 0 {
		t.Fatalf("six decode tokens of a fully resident hybrid recorded nothing (LastErr %q)", st.LastErr)
	}
	t.Logf("%d capture(s) over six tokens", st.Captures)
}

// TestDeltaKeyTiledIsTheDeltaRulesPairing is the host gate on
// jlm.FlagDeltaKeyTiled: the same weights read with the value heads paired to
// key heads TILED (vh % kHeads, Qwen3.5's file order) and GROUPED (vh / rep,
// qwen3next's) must give DIFFERENT logits, and the tiled arm must be the
// pairing the flag asked for. The fixture has two value heads per key head,
// where the pairings differ (at one they coincide). Correctness against the
// reference is Qwen3.5-4B/-9B teacher-forced against llama.cpp; this keeps a
// later change from dropping the flag.
func TestDeltaKeyTiledIsTheDeltaRulesPairing(t *testing.T) {
	grouped := hybridModelOpt(t, hyOpt{})
	tiled := hybridModelOpt(t, hyOpt{tiled: true})
	if grouped.Cfg.DeltaKeyTiled || !tiled.Cfg.DeltaKeyTiled {
		t.Fatalf("the flag did not survive the container: grouped=%v tiled=%v",
			grouped.Cfg.DeltaKeyTiled, tiled.Cfg.DeltaKeyTiled)
	}
	g := tiled.Cfg.delta()
	if g.rep < 2 {
		t.Fatalf("the fixture has %d value head(s) per key head; the pairings coincide and this proves nothing", g.rep)
	}
	for vh := 0; vh < g.vHeads; vh++ {
		if want := vh % g.kHeads; g.keyHead(vh) != want {
			t.Fatalf("tiled: value head %d reads key head %d, want %d", vh, g.keyHead(vh), want)
		}
		if want := vh / g.rep; grouped.Cfg.delta().keyHead(vh) != want {
			t.Fatalf("grouped: value head %d reads key head %d, want %d", vh, grouped.Cfg.delta().keyHead(vh), want)
		}
	}

	run := func(m *Model) [][]float32 {
		s := m.NewState(16)
		defer s.Close()
		var out [][]float32
		for _, id := range []int32{1, 2, 3, 4, 5} {
			lg, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	a, b := run(grouped), run(tiled)
	var worst float64
	for p := range a {
		var num, den float64
		for i := range a[p] {
			d := float64(a[p][i] - b[p][i])
			num += d * d
			den += float64(a[p][i]) * float64(a[p][i])
		}
		if math.IsNaN(num) || math.IsInf(num, 0) {
			t.Fatalf("pos %d: non-finite logits", p)
		}
		worst = math.Max(worst, num/den)
	}
	if worst < 1e-6 {
		t.Fatalf("tiled and grouped pairings read the same logits (worst NMSE %.3e): the flag "+
			"never reached the delta rule", worst)
	}
	t.Logf("tiled against grouped: worst logit NMSE %.3e over 5 positions", worst)
}
