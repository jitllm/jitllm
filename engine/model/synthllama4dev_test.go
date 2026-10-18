package model

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestSynthLlama4OnEveryDevice runs the Llama 4 fixture with every block and the
// head on each device present -- cuda, vulkan, metal -- against the host.
// Twelve positions cross two chunk boundaries and three
// temperature steps, layer 3 is unrotated, layers 1 and 3 are input-weighted
// top-1 mixtures, and every rotated layer carries the weightless q/k norm. Both
// device paths are held: decode one row at a time, and a prefill of the whole
// chunk through the batched kernels (windowed tiles, the per-row temperature,
// the batched mixture).
//
// Each violation edits the config only for the device session (the host arm
// uses the real config), so the device arm must see every feature.
func TestSynthLlama4OnEveryDevice(t *testing.T) {
	m, g := openSynthLlama4(t)
	defer m.Close()
	cfg := *m.Cfg
	decode := func(st *State) [][]float32 {
		var out [][]float32
		for _, id := range g.IDs {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	prefill := func(st *State) [][]float32 {
		lg, err := st.Prefill(g.IDs)
		if err != nil {
			t.Fatal(err)
		}
		return [][]float32{append([]float32(nil), lg...)}
	}
	onHost := func(c *Config, run func(*State) [][]float32) [][]float32 {
		m.Cfg = c
		defer func() { m.Cfg = &cfg }()
		st := m.NewState(len(g.IDs) + 1)
		defer st.Close()
		return run(st)
	}
	cmp := func(want, got [][]float32) (float64, int) {
		worst, flips := 0.0, 0
		for p := range want {
			var num, den float64
			for i := range want[p] {
				d := float64(got[p][i] - want[p][i])
				num, den = num+d*d, den+float64(want[p][i])*float64(want[p][i])
			}
			nmse := num / den
			if math.IsNaN(nmse) || den == 0 {
				return math.Inf(1), len(want)
			}
			worst = math.Max(worst, nmse)
			if Greedy(got[p]) != Greedy(want[p]) {
				flips++
			}
		}
		return worst, flips
	}
	c0 := cfg
	wantDec, wantPre := onHost(&c0, decode), onHost(&c0, prefill)
	// The NoPE layer's own arm: with the temperature off on BOTH sides, rotating
	// the unrotated layer is the only difference left, so the device cannot hide
	// behind declining the temperature on rotated layers.
	cold := cfg
	cold.AttnTempScale = 0
	wantCold := onHost(&cold, decode)

	ran := 0
	for _, spec := range stepDevices() {
		probe, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || probe == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		probe.Close()
		ran++
		t.Run(spec, func(t *testing.T) {
			// A fresh tier per arm: a warm tier's scratch is the first plan's,
			// so planConflict would refuse a violation instead of running it.
			onDevice := func(c *Config, run func(*State) [][]float32) ([][]float32, string) {
				gpu, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil {
					t.Fatal(err)
				}
				defer gpu.Close()
				m.Cfg = c
				regraph(m)
				defer func() { m.Cfg = &cfg; regraph(m) }()
				st := m.NewState(len(g.IDs) + 1)
				defer st.Close()
				st.SetDeviceLayers(gpu, -1)
				if n := st.GPULayers(); n != c.NLayer {
					return nil, fmt.Sprintf("took %d of %d blocks: %v %v", n, c.NLayer,
						st.DeviceDeclines(), gpu.Err())
				}
				got := run(st)
				if n := st.GPULayers(); n != c.NLayer {
					t.Fatalf("the session fell to the host (%d blocks left): %v", n, gpu.Err())
				}
				return got, ""
			}
			clean := func(name string, run func(*State) [][]float32, want [][]float32) float64 {
				c := cfg
				got, why := onDevice(&c, run)
				if why != "" {
					t.Fatalf("%s: %s", name, why)
				}
				worst, flips := cmp(want, got)
				if !(worst < 1e-2) || flips > 1 {
					t.Fatalf("%s: worst logit NMSE %.3e, %d argmax flips, against the host", name, worst, flips)
				}
				t.Logf("%s, %d blocks and the head on %s: worst logit NMSE %.3e, %d flips",
					name, c.NLayer, spec, worst, flips)
				return worst
			}
			worst := clean("decode", decode, wantDec)
			worst = math.Max(worst, clean("prefill", prefill, wantPre))
			// One warm tier, two plans: the scratch bakes the first plan's
			// chunk, temperature table and kernels (planUnion widens none), so
			// a second session with another Llama 4 shape must be refused.
			func() {
				gpu, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil {
					t.Fatal(err)
				}
				defer gpu.Close()
				first := m.NewState(len(g.IDs) + 1)
				defer first.Close()
				first.SetDeviceLayers(gpu, -1)
				if first.GPULayers() != cfg.NLayer {
					t.Fatalf("the first session placed %d blocks", first.GPULayers())
				}
				other := cfg
				other.SWAChunked = false
				m.Cfg = &other
				defer func() { m.Cfg = &cfg }()
				second := m.NewState(len(g.IDs) + 1)
				defer second.Close()
				second.SetDeviceLayers(gpu, -1)
				if n := second.GPULayers(); n != 0 {
					t.Errorf("a session with a sliding window was placed on a tier built for a "+
						"chunk: %d blocks", n)
				}
			}()
			for _, v := range []struct {
				name string
				mut  func(c *Config)
				want [][]float32
			}{
				{"sliding window instead of a chunk", func(c *Config) { c.SWAChunked = false }, wantDec},
				{"no QK L2 norm", func(c *Config) { c.QKL2Norm = false }, wantDec},
				{"weight on the expert's output", func(c *Config) { c.ExpertWeightIn = false }, wantDec},
				{"no attention temperature", func(c *Config) { c.AttnTempScale = 0 }, wantDec},
				{"every layer rotated (temperature off on both sides)", func(c *Config) {
					c.AttnTempScale, c.NoPEGlobal = 0, false
				}, wantCold},
			} {
				c := cfg
				v.mut(&c)
				got, why := onDevice(&c, decode)
				if why != "" {
					t.Errorf("%s: the device declined the violation rather than running it: %s", v.name, why)
					continue
				}
				bad, _ := cmp(v.want, got)
				if !(bad > 10*worst) || !(bad > 1e-3) {
					t.Errorf("%s: the device arm cannot see it -- worst NMSE %.3e against a clean %.3e",
						v.name, bad, worst)
				}
				t.Logf("violation %q: worst NMSE %.3e", v.name, bad)
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}
