package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// gateUp2Models are the dense gated FFNs the Gate2 gate decodes: gate and up
// one format in both.
var gateUp2Models = []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf"}

// Off PTX, where Gate2 is not bit-identical, its decode may sit from the two
// launches by denseStepBand times the model's own
// order-only control on that device (orderControl, the step gate's), the
// worst step's NMSE. A swapped gate and up sit orders above it
// (TestGateUp2SeesASwap).
//
// gateUp2Devices are the backends the gate runs on, each where this host has
// one.
var gateUp2Devices = testdevices.Specs("cuda", "vulkan", "metal")

// gateUp2Decode decodes steps teacher-forced tokens after a prompt on one
// fresh tier over devs, twice: with Gate2 off and with it forced on, the
// switch thrown between the arms on the same tier, so both run the splits
// its tuner chose once and reduce in one order. It returns each arm's
// logits for every step, the Gate2 arm's stats and the blocks placed; nil
// where the host has no such device. layers < 0 places every block.
func gateUp2Decode(t *testing.T, m *Model, devs string, ids, toks []int32, noGraph bool, layers int, cache string, extra ...tier.Option) (off, on [][]float32, st tier.Stats, placed int) {
	t.Helper()
	g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(devs), tier.WithTuneCache(cache),
		tier.WithConfig(func(c *tier.Config) { c.NoGraph = noGraph })}, extra...)...)
	if err != nil || g == nil {
		return nil, nil, tier.Stats{}, 0
	}
	defer g.Close()
	arm := func(gu2 bool) [][]float32 {
		g.GateUp2, g.GateUp2Force = gu2, gu2
		s := m.NewState(len(ids) + len(toks) + 4)
		defer s.Close()
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if layers >= 0 {
			s.SetGPULayers(layers)
		}
		placed = s.GPULayers()
		l, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		var out [][]float32
		for _, tok := range toks {
			out = append(out, append([]float32(nil), l...))
			if l, err = s.Forward(tok); err != nil {
				t.Fatal(err)
			}
		}
		s.SetGPULayers(0)
		return out
	}
	// A first pass settles the tuner: the arms after it run its answers.
	first := arm(false)
	off = arm(false)
	if n, d := gateUp2NMSE(off, first); d != 0 {
		t.Logf("%s: the two-launch arm against itself: NMSE %.3g, %d logits differ", devs, n, d)
	}
	before := g.Stats().GateUp2Runs
	if before != 0 {
		t.Fatalf("%s: the two-launch arm ran %d Gate2 FFNs", devs, before)
	}
	on = arm(true)
	st = g.Stats()
	return off, on, st, placed
}

// gateUp2NMSE is the NMSE of got against want over every step, and the
// count of logits whose bits differ.
func gateUp2NMSE(got, want [][]float32) (float64, int) {
	var num, den float64
	diff := 0
	for i := range got {
		for j := range got[i] {
			if math.Float32bits(got[i][j]) != math.Float32bits(want[i][j]) {
				diff++
			}
			d := float64(got[i][j]) - float64(want[i][j])
			num, den = num+d*d, den+float64(want[i][j])*float64(want[i][j])
		}
	}
	if den == 0 {
		return math.Inf(1), diff
	}
	return num / den, diff
}

// gateUp2Worst is the worst one step's NMSE of got against want.
func gateUp2Worst(got, want [][]float32) float64 {
	w := 0.0
	for i := range got {
		n, _ := gateUp2NMSE(got[i:i+1], want[i:i+1])
		if math.IsNaN(n) || n > w {
			w = n
		}
	}
	return w
}

// gateUp2Model opens name, or reports it missing.
func gateUp2Model(t *testing.T, name string) *Model {
	t.Helper()
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
			return nil
		}
	}
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestGateUp2MatchesTwoLaunches decodes real dense models with the FFN's
// gate and up as one Gate2 launch (forced) and without, on every backend
// this host has, and holds the two to the same logits: bit for bit on CUDA,
// whose PTX fixes every rounding, and inside the reduction-order band
// (the bound on gateUp2Devices) on Vulkan and Metal, where the driver may contract the one
// kernel's sums differently from the two's. Every block placed with and
// without the captured graph, and a partial seam. Stats().GateUp2Runs is the
// selection check.
func TestGateUp2MatchesTwoLaunches(t *testing.T) {
	ran := 0
	for _, name := range gateUp2Models {
		m := gateUp2Model(t, name)
		if m == nil {
			t.Logf("MODEL MISSING: %s", name)
			continue
		}
		t.Run(name, func(t *testing.T) {
			defer m.Close()
			ids := m.Vocab.Encode("The river rose every spring until the bridges were islands, and", true)
			toks := make([]int32, 16)
			for i := range toks {
				toks[i] = int32(100 + 37*i)
			}
			ctrl := map[string]float64{}
			for _, dev := range gateUp2Devices {
				cache := t.TempDir()
				for _, c := range []struct {
					seam    int
					noGraph bool
				}{{-1, false}, {-1, true}, {m.Cfg.NLayer / 2, false}} {
					off, on, st, placed := gateUp2Decode(t, m, dev, ids, toks, c.noGraph, c.seam, cache)
					if off == nil {
						t.Logf("%s: no such device here", dev)
						break
					}
					want := m.Cfg.NLayer
					if c.seam >= 0 {
						want = c.seam
					}
					if placed != want {
						t.Fatalf("%s seam %d: the device took %d blocks, want %d", dev, c.seam, placed, want)
					}
					if st.GateUp2Runs == 0 {
						t.Fatalf("%s seam %d: no FFN ran gate and up as one Gate2 launch (%s)", dev, c.seam, st.LastErr)
					}
					nmse, diff := gateUp2NMSE(on, off)
					worst := gateUp2Worst(on, off)
					bound := 0.0
					if dev != "cuda" {
						if ctrl[dev] == 0 {
							ctrl[dev] = orderControl(t, m, dev)
						}
						bound = denseStepBand * ctrl[dev]
					}
					if dev == "cuda" && diff != 0 || math.IsNaN(worst) || worst > bound && dev != "cuda" {
						t.Fatalf("%s seam %d graph %v: Gate2 against two launches: worst step NMSE %.3g (bound %.3g), %d logits differ",
							dev, c.seam, !c.noGraph, worst, bound, diff)
					}
					t.Logf("%s seam %d graph %v: %d Gate2 FFNs, NMSE %.3g over %d steps, worst step %.3g (bound %.3g), %d logits differ",
						dev, c.seam, !c.noGraph, st.GateUp2Runs, nmse, len(on), worst, bound, diff)
					ran++
				}
			}
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "no model or device ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}
