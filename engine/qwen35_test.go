package engine

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestHybridDecodeIsCaptured: a fully placed hybrid's decode must be recorded
// as a CUDA graph, like every other resident model's. A write from inside emit
// (such as bs.dRows) is a memcpy during stream capture, which invalidates the
// capture and silently disables graphs.
func TestHybridDecodeIsCaptured(t *testing.T) {
	m, openOpts := hybridModelOpt(t, hyOpt{tiled: true})
	g, err := openGPU(t, gpu.WithDevices(testdevices.Like(t, "cuda:0")), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		t.Skipf("no cuda device: %v", err)
	}
	defer g.Close()
	s := testInstance(t, m, 32, openOpts...)
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
// reference is Qwen3.5-4B (two value heads a key head) against the GGUF oracle
// in TestGreedyMatchesLlamaCpp; this keeps a later change from dropping the
// flag.
func TestDeltaKeyTiledIsTheDeltaRulesPairing(t *testing.T) {
	grouped, openOpts := hybridModelOpt(t, hyOpt{})
	tiled, openOpts2 := hybridModelOpt(t, hyOpt{tiled: true})
	if keyTiled(grouped) || !keyTiled(tiled) {
		t.Fatalf("the flag did not survive the container: grouped=%v tiled=%v",
			keyTiled(grouped), keyTiled(tiled))
	}
	g := deltaOf(tiled)
	if g.Rep < 2 {
		t.Fatalf("the fixture has %d value head(s) per key head; the pairings coincide and this proves nothing", g.Rep)
	}
	// The built scan states the pairing: q and k along the value head's
	// cycle of key heads where tiled, its group where grouped.
	pairing := func(m *model.Model) graph.ScanAxis {
		for li := range m.Cfg.NLayer {
			for _, o := range m.Composing().BlockGraph(li).Ops {
				if o.Kind == graph.OpStateScan {
					if o.Attr.NKVHead != g.KHeads || o.Attr.Words[0] != o.Attr.Words[1] {
						t.Fatalf("block %d: a scan over %d key heads, q along %#x and k along %#x", li, o.Attr.NKVHead, o.Attr.Words[0], o.Attr.Words[1])
					}
					return graph.ScanAxis(o.Attr.Words[1])
				}
			}
		}
		t.Fatal("the fixture states no scan")
		return 0
	}
	if p := pairing(tiled); p != graph.ScanCycle|graph.ScanCol {
		t.Fatalf("tiled: the keys along %#x, want the key heads' cycle", p)
	}
	if p := pairing(grouped); p != graph.ScanGroup|graph.ScanCol {
		t.Fatalf("grouped: the keys along %#x, want the key heads' groups", p)
	}
	// The pairing is the scan's index and costs no op of its own: a tiled
	// block states the grouped block's ops.
	for li := range tiled.Cfg.NLayer {
		if a, b := len(tiled.Composing().BlockGraph(li).Ops), len(grouped.Composing().BlockGraph(li).Ops); a != b {
			t.Fatalf("block %d: %d ops tiled, %d grouped", li, a, b)
		}
	}

	run := func(m *model.Model, io []Option) [][]float32 {
		s := testInstance(t, m, 16, io...)
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
	a, b := run(grouped, openOpts), run(tiled, openOpts2)
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
