package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestComposedHeadTokens holds the tokens a device head selects -- the greedy
// one (nn.Head.ArgmaxOnly) and the sampler's (nn.Head.SampleK) -- to the
// logits it returns, for a head with a bias, a softcap and a scale. Where the
// device served the head's graph it has applied them all and selects every
// token itself, a bias and a scale included; where it ran its own head it
// declines the requests that a stage left to the host would make wrong, and
// the host selects from the logits it finished. Either way the tokens are the
// host's own, and the counts say which arm ran: a gate that cannot tell the
// two would certify a device that selected before the bias, or a host that
// never asked.
func TestComposedHeadTokens(t *testing.T) {
	const gen = 24
	ran := 0
	for _, fx := range headFixtures {
		t.Run(fx.name, func(t *testing.T) {
			for _, dev := range stepDevices() {
				for _, parts := range []tier.Parts{tier.ComposedHead, 0} {
					name := dev + "/own head"
					if parts != 0 {
						name = dev + "/composed head"
					}
					t.Run(name, func(t *testing.T) {
						g, err := tier.OpenWith(tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
							tier.WithConfig(func(c *tier.Config) { c.Composed = parts }))
						if err != nil || g == nil {
							noDevice(t, dev, err)
						}
						defer g.Close()
						m := fx.open(t)
						defer m.Close()
						// A head with a bias takes no softcap: the tier leaves the
						// pair to the host.
						biased := m.outB != nil
						defer cfgMut(m, func(c *Config) {
							if !biased {
								c.FinalSoftcap = 1.5
							}
							c.LogitScale = 8
						})()
						m.opt.devSample = DeviceSampleOn
						stages := m.headStages()
						ids := headTokens(m)
						newState := func() *State {
							st := m.NewState(len(ids) + gen + 4)
							st.SetDeviceFallback(false)
							if err := st.SetDevice(g); err != nil {
								t.Fatal(err)
							}
							if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
								t.Fatalf("wanted every block and the head on %s, got %d of %d blocks and the head %v: %s", dev, st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
							}
							return st
						}
						// Greedy: the token the device selects against Greedy over
						// the logits of the same position.
						a, b := newState(), newState()
						defer a.Close()
						defer b.Close()
						served := 0
						for p, id := range ids {
							lg, err := a.Forward(id)
							if err != nil {
								t.Fatal(err)
							}
							want := Greedy(lg)
							b.head.Token = -1
							got, err := b.ForwardGreedy(id)
							if err != nil {
								t.Fatal(err)
							}
							if got != want {
								t.Fatalf("pos %d: ForwardGreedy %d, Greedy over the logits %d", p, got, want)
							}
							if b.head.Token >= 0 {
								served++
							}
						}
						wantServed := len(ids)
						if parts == 0 && stages&nn.HeadBias != 0 {
							wantServed = 0
						}
						if served != wantServed {
							t.Fatalf("the device selected %d of %d greedy tokens, want %d (stages %03b, parts %d)", served, len(ids), wantServed, stages, parts)
						}
						// The sampler: the same seeded draws over the device's
						// candidates and over the logits it returned.
						k := min(40, m.Cfg.NVocab/4)
						newSampler := func() *Sampler {
							return &Sampler{Temp: 1.3, TopK: k, TopP: 0.9, MinP: 0.01, RepeatPen: 1.3, RepeatLastN: 32, Seed: 4242}
						}
						// draws is gen seeded draws made two ways, from the device's
						// candidates (ForwardSample) and from the logits it returned
						// (Sample over Forward's): how many parted, and how many the
						// device selected. hide clears the head's stages from the
						// first, so that nothing tells the device a stage is the
						// host's to apply.
						draws := func(hide bool) (parted int, onDev int64) {
							c, d := newState(), newState()
							defer c.Close()
							defer d.Close()
							if hide {
								c.head.Stages = 0
							}
							sc, sd := newSampler(), newSampler()
							lc, err := c.Prefill(ids)
							if err != nil {
								t.Fatal(err)
							}
							ld, err := d.Prefill(ids)
							if err != nil {
								t.Fatal(err)
							}
							nc, nd := sc.Sample(lc), sd.Sample(ld)
							for range gen {
								if nc != nd {
									parted++
								}
								sc.Observe(nd)
								sd.Observe(nd)
								if nc, err = c.ForwardSample(nd, sc); err != nil {
									t.Fatal(err)
								}
								lg, err := d.Forward(nd)
								if err != nil {
									t.Fatal(err)
								}
								nd = sd.Sample(lg)
							}
							return parted, c.SampledOnDevice
						}
						parted, onDev := draws(false)
						if parted != 0 {
							t.Fatalf("%d of %d draws from the device's candidates parted from the draws over its logits", parted, gen)
						}
						wantSampled := int64(gen)
						if parts == 0 && stages&(nn.HeadBias|nn.HeadScale) != 0 {
							wantSampled = 0
						}
						if onDev != wantSampled {
							t.Fatalf("the device selected %d of %d sampled tokens, want %d (stages %03b, parts %d)", onDev, gen, wantSampled, stages, parts)
						}
						t.Logf("%d greedy tokens (%d on the device) and %d sampled tokens (%d on the device), stages %03b", len(ids), served, gen, onDev, stages)
						// The violation: the tier's own head, which leaves the bias
						// and the scale to the host, asked to select as though it
						// had applied them. It does, before them, and the draws part.
						if parts == 0 && stages&(nn.HeadBias|nn.HeadScale) != 0 {
							parted, onDev := draws(true)
							t.Logf("violation, the stages hidden from the device: %d of %d draws parted, %d selected on the device", parted, gen, onDev)
							if onDev != int64(gen) || parted == 0 {
								t.Fatalf("the gate cannot see a device that selects before the host's stages: %d of %d draws parted, %d selected on the device", parted, gen, onDev)
							}
						}
						ran++
					})
				}
			}
		})
	}
	if ran == 0 {
		t.Fatal("no arm ran")
	}
}
