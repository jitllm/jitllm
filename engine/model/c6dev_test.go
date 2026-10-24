package model

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// c6DevViolations are the kit's features taken out of the model while only the
// device session runs; the host arm is computed once from the clean model. A
// tier that ignored a feature would match the host on the clean arm and fail
// nothing, so each must move the device's logits.
var c6DevViolations = map[string][]c6Violation{
	"synth-phi2": {
		{"RMSNorm for LayerNorm", rmsForLayerNorm},
		{"no norm biases", noNormBiases},
		{"no FFN biases", noFFNBiases},
		{"serial residual", serialFromShared},
		{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
		{"no output norm bias", func(m *Model) func() {
			was := m.outNormB
			m.outNormB = nil
			return func() { m.outNormB = was }
		}},
	},
	"synth-starcoder": {
		{"no position table", noPosTable},
		{"rotary instead of none", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.NoPosEnc = false }) }},
		{"RMSNorm for LayerNorm", rmsForLayerNorm},
		{"no norm biases", noNormBiases},
		{"no FFN biases", noFFNBiases},
	},
	"synth-stablelm": {
		{"RMSNorm for LayerNorm", rmsForLayerNorm},
		{"parallel residual", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Parallel = true }) }},
		{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
	},
	"synth-stablelm-par": {
		{"serial residual", serialFromShared},
	},
	"synth-starcoder2": {
		{"RMSNorm for LayerNorm", rmsForLayerNorm},
		{"parallel residual", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Parallel = true }) }},
		{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
		{"no FFN biases", noFFNBiases},
	},
	"synth-falcon": {
		{"serial residual", serialFromShared},
		{"RMSNorm for LayerNorm", rmsForLayerNorm},
	},
	"synth-falcon40": {
		{"the two norms swapped", swapNorms},
		{"one norm for both branches", oneNormForBoth},
	},
	"synth-nemotron": {
		{"GELU for relu2", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Act = nn.ActGELU }) }},
		{"SiLU for relu2", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Act = nn.ActSiLU }) }},
		{"RMSNorm for LayerNorm", rmsForLayerNorm},
		{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
	},
	"synth-dbrx": {
		{"no clip_qkv", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ClampKQV = 0 }) }},
		{"clip_qkv at twice its value", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ClampKQV *= 2 }) }},
		{"RMSNorm for LayerNorm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LayerNorm = false }) }},
		{"the routed weights not renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
		}},
	},
	"synth-commandr": {
		{"RMSNorm for LayerNorm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LayerNorm = false }) }},
		{"serial residual", serialFromShared},
		{"NEOX rotary for NORM", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = true }) }},
	},
	"synth-cohere2": {
		{"rotary on the global layer", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.NoPEGlobal = false }) }},
		{"no sliding window", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.SWAWindow = 0 }) }},
		{"serial residual", serialFromShared},
	},
}

// TestC6GreedyOnDeviceHonoursTheHeadBias holds the two device argmax paths to
// phi-2's head bias. The bias is added on the host (finishLogits), so a device
// argmax picks from logits it never reached: ForwardGreedy must keep the
// argmax on the host for such a model, and ForwardBatchGreedy -- which has no
// host logits at all -- must refuse it.
func TestC6GreedyOnDeviceHonoursTheHeadBias(t *testing.T) {
	m, g := openC6(t, "synth-phi2")
	defer m.Close()
	if m.outB == nil {
		t.Fatal("the fixture lost its head bias")
	}
	want := make([]int32, len(g.IDs))
	host := m.NewState(len(g.IDs) + 1)
	for p, id := range g.IDs {
		lg, err := host.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		want[p] = Greedy(lg)
	}
	host.Close()
	gpu, err := tier.OpenWith(tier.WithDeviceTune(tier.TuneOff))
	if err != nil || gpu == nil {
		noDevice(t, "device", err)
	}
	defer gpu.Close()
	st := m.NewState(len(g.IDs) + 1)
	defer st.Close()
	st.SetDevice(gpu)
	if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
		t.Fatalf("placed %d of %d blocks, head %v: %v", st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), gpu.Err())
	}
	for p, id := range g.IDs {
		got, err := st.ForwardGreedy(id)
		if err != nil {
			t.Fatal(err)
		}
		if got != want[p] {
			t.Errorf("pos %d: ForwardGreedy on the device %d, the host's biased argmax %d", p, got, want[p])
		}
	}
	b := m.NewBatch(2, len(g.IDs)+1)
	defer b.Close()
	b.SetDevice(gpu)
	if _, err := b.ForwardBatchGreedy([]int32{g.IDs[0], g.IDs[1]}); err == nil {
		t.Error("ForwardBatchGreedy ran a model with a head bias; its per-row argmax never sees the bias")
	}
}

// TestC6OnEveryDevice runs each classic-transformer fixture with every block
// and the head on each device present -- cuda, vulkan, metal -- against the
// host, decode one row at a time and a prefill of the whole prompt
// through the batched kernels, and then each feature removed from the device
// session alone.
func TestC6OnEveryDevice(t *testing.T) {
	for _, fx := range c6Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openC6(t, fx.name)
			defer m.Close()
			fixtureOnEveryDevice(t, m, g, c6DevViolations[fx.name])
		})
	}
}

// fixtureOnEveryDevice is TestC6OnEveryDevice's body for one opened fixture:
// every block and the head on each device present against the host, decode
// and a whole-prompt prefill, then each of vs on the device session alone.
func fixtureOnEveryDevice(t *testing.T, m *Model, g *c6Golden, vs []c6Violation) {
	t.Helper()
	fixtureOnEveryDeviceWithin(t, m, g, vs, 1e-5)
}

// fixtureOnEveryDeviceWithin is fixtureOnEveryDevice with the prefill arm's
// bound named: the binary16 attention tiles' band, which a graph that
// renormalises a small branch (Gemma 4's mixture block) amplifies.
func fixtureOnEveryDeviceWithin(t *testing.T, m *Model, g *c6Golden, vs []c6Violation, preBound float64) {
	t.Helper()
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
	onHost := func(run func(*State) [][]float32) [][]float32 {
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
	wantDec, wantPre := onHost(decode), onHost(prefill)
	nl := m.Cfg.NLayer

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
			// onDevice runs run on a fresh tier of spec, every block and the
			// head placed; steps is its one-row steps, every one of which the
			// composer runs where it serves blocks the front end states.
			onDevice := func(run func(*State) [][]float32, steps int) ([][]float32, string) {
				gpu, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
					tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
				if err != nil {
					t.Fatal(err)
				}
				defer gpu.Close()
				st := m.NewState(len(g.IDs) + 1)
				defer st.Close()
				st.SetDeviceLayers(gpu, -1)
				if n := st.GPULayers(); n != nl {
					return nil, fmt.Sprintf("took %d of %d blocks: %v %v", n, nl,
						st.DeviceDeclines(), gpu.Err())
				}
				if !st.HeadOnDevice() {
					return nil, "the head stayed on the host: " + gpu.Err()
				}
				got := run(st)
				if n := st.GPULayers(); n != nl {
					t.Fatalf("the session fell to the host (%d blocks left): %v", n, gpu.Err())
				}
				composedBlocksRan(t, composedOf(m, gpu), 0, steps*nl, gpu)
				return got, ""
			}
			worst := 0.0
			for _, arm := range []struct {
				name  string
				run   func(*State) [][]float32
				want  [][]float32
				steps int
			}{{"decode", decode, wantDec, len(g.IDs)}, {"prefill", prefill, wantPre, 0}} {
				got, why := onDevice(arm.run, arm.steps)
				if why != "" {
					t.Fatalf("%s: %s", arm.name, why)
				}
				w, flips := cmp(arm.want, got)
				// A prompt's chunk takes Q K^T through binary16 tiles where the
				// device has them (CUDA's m16n8k16, Metal's FlashPrefillTile),
				// and P V in float32 on both: up to 4.7e-07 on CUDA and 6.7e-06
				// on Metal (synth-glm4), where decode's f32 reads 1e-12. With P
				// and V in binary16 as well, Metal's tile put synth-apertus --
				// q/k-normed, so its softmax is nearly one-hot -- at 1.3e-04 and
				// an argmax flip, which this bound and the zero-flip rule refuse.
				// The violations below land at 1e-01 and up.
				bound := 1e-6
				if arm.name == "prefill" {
					bound = preBound
				}
				if !(w < bound) || flips > 0 {
					t.Fatalf("%s: worst logit NMSE %.3e, %d argmax flips, against the host", arm.name, w, flips)
				}
				t.Logf("%s, %d blocks and the head on %s: worst logit NMSE %.3e", arm.name, nl, spec, w)
				worst = math.Max(worst, w)
			}
			for _, v := range vs {
				undo := v.mut(m)
				got, why := onDevice(decode, len(g.IDs))
				undo()
				if why != "" {
					t.Errorf("%s: the device declined the violation rather than running it: %s", v.name, why)
					continue
				}
				bad, _ := cmp(wantDec, got)
				if !(bad > 100*worst) || !(bad > 1e-4) {
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
