//go:build linux

package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestHybridExpertsMatchTheHost runs Kimi-K3-0.40B with every block on the
// device and the routed experts on the host (tier.Config.HybridExperts), and
// holds it to the host-only run: the same greedy token at every position and
// logits within the device's f32 band (RULE 11c), the hybrid path counted as
// taken. A latent mixture is the case that matters: the experts' input is the
// device's latent projection, and their sum goes back into its latent.
func TestHybridExpertsMatchTheHost(t *testing.T) {
	path, ok := existingModel(testmodels.Path("kimik3/Kimi-K3-0.40B.Q8_0.gguf"))
	if !ok {
		t.Skip("MODEL MISSING: kimik3/Kimi-K3-0.40B.Q8_0.gguf -- this gate proved nothing")
	}
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := []int32{1008, 10484, 318, 15383, 387, 17374, 13, 646}
	host := func() [][]float32 {
		s := m.NewState(32)
		defer s.Close()
		var out [][]float32
		for _, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), l...))
		}
		return out
	}
	want := host()
	stats := func(cfg func(*tier.Config)) ([][]float32, tier.Stats, int) {
		var st tier.Stats
		var got [][]float32
		var placed int
		func() {
			s := m.NewState(32)
			defer s.Close()
			g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff), tier.WithConfig(cfg))
			if err != nil || g == nil {
				noDevice(t, "gpu:0", err)
			}
			defer g.Close()
			s.SetDeviceLayers(g, m.Cfg.NLayer)
			placed = s.GPULayers()
			for _, id := range ids {
				l, err := s.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				got = append(got, append([]float32(nil), l...))
			}
			st = g.Stats()
		}()
		return got, st, placed
	}
	got, st, placed := stats(func(c *tier.Config) { c.StreamExperts, c.HybridExperts, c.StreamCacheSlots = true, true, -1 })
	if placed == 0 || st.HybridRuns == 0 || st.StreamFills != 0 {
		t.Fatalf("%d blocks placed, %d hybrid block-steps, %d fills: the hybrid path did not run alone",
			placed, st.HybridRuns, st.StreamFills)
	}
	argmax := func(v []float32) int {
		b := 0
		for i := range v {
			if v[i] > v[b] {
				b = i
			}
		}
		return b
	}
	for p := range want {
		var num, den float64
		for i := range want[p] {
			if math.IsNaN(float64(got[p][i])) || math.IsInf(float64(got[p][i]), 0) {
				t.Fatalf("pos %d logit %d is not finite", p, i)
			}
			d := float64(got[p][i] - want[p][i])
			num += d * d
			den += float64(want[p][i]) * float64(want[p][i])
		}
		nmse := num / den
		if nmse > 1e-5 || argmax(got[p]) != argmax(want[p]) {
			t.Fatalf("pos %d: NMSE %.2e, argmax %d hybrid against %d host", p, nmse, argmax(got[p]), argmax(want[p]))
		}
		if p == len(want)-1 {
			t.Logf("%d blocks placed, %d hybrid block-steps; last position NMSE %.2e", placed, st.HybridRuns, nmse)
		}
	}
}
