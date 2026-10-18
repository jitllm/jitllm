package model

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// bitnetDevViolations are BitNet's features taken out of the model while only
// the device session runs; the host arm is computed once from the clean
// model. A tier that ignored one would match the host on the clean arm. An
// inner norm taken away is a block the tier refuses by name (the plan says
// the block has one), so the norms' weights are made ones instead: the
// device must read the block's own.
var bitnetDevViolations = []c6Violation{
	{"the attention inner norm's weight all ones", func(m *Model) func() {
		return eachLayer(m, func(l *layer) { l.attnSubNorm = onesLike(l.attnSubNorm) })
	}},
	{"the FFN inner norm's weight all ones", func(m *Model) func() {
		return eachLayer(m, func(l *layer) { l.ffnSubNorm = onesLike(l.ffnSubNorm) })
	}},
	bitnetViolations[2], // SiLU for relu2
	bitnetViolations[3], // NORM rotary
	// The per-token scale is the session's activation window, which the
	// device takes from the host's JIT (nn.LayerPlan.ActWin).
	bitnetViolations[4],
}

// TestBitNetOnEveryDevice runs each BitNet fixture with every block and the
// head on each device present -- cuda, vulkan, metal -- against the host,
// decode a row at a time and a prefill of the whole prompt, then a partial
// seam that leaves the last block and the head home, then each feature
// removed from the device session alone.
//
// The device quantizes the same activations per token as the host, but sums
// in another order, so an activation within a few ulps of an int8 rounding
// boundary can land on the other side: bitnetNMSE takes such a flip, and the
// zero-flip positions must read the reduction band.
//
// A prompt's chunk is the exception, measured: CUDA takes Q K^T through
// binary16 tiles (the warp MMA and FlashPrefill70) where decode is f32, and
// every int8 rounding downstream of that attention turns its 1e-5 band into
// flips -- 4e-3..8e-3 on these fixtures, against 5e-14 with those tiles
// refused (NoMMA, NoVolta, NoFlashPrefill). So the prefill arm runs twice:
// with the tiles at bitnetDevPrefill, and without them at bitnetNMSE, which
// is the arm that proves the tier's wiring of BitNet exact.
func TestBitNetOnEveryDevice(t *testing.T) {
	for _, fx := range bitnetFixtures {
		t.Run(fx.model, func(t *testing.T) {
			m, g := openBitNet(t, fx.model, fx.golden, fx.typ)
			defer m.Close()
			bitnetOnEveryDevice(t, m, g)
		})
	}
}

// bitnetDevPrefill is the shipping prefill arm's bound: binary16 Q K^T
// tiles under per-token int8 activations (see TestBitNetOnEveryDevice).
const bitnetDevPrefill = 2e-2

func bitnetOnEveryDevice(t *testing.T, m *Model, g *c6Golden) {
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
	// worst is the worst NMSE, and band how many positions read inside the
	// device's reduction band (1e-6): a flip moves a position far past it.
	cmp := func(want, got [][]float32) (worst float64, band int) {
		for p := range want {
			var num, den float64
			for i := range want[p] {
				d := float64(got[p][i] - want[p][i])
				num, den = num+d*d, den+float64(want[p][i])*float64(want[p][i])
			}
			nmse := num / den
			if math.IsNaN(nmse) || den == 0 {
				return math.Inf(1), 0
			}
			worst = math.Max(worst, nmse)
			if nmse < 1e-6 {
				band++
			}
		}
		return worst, band
	}
	wantDec, wantPre := onHost(decode), onHost(prefill)
	nl := m.Cfg.NLayer

	ran := 0
	for _, spec := range stepDevices() {
		probe, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || probe == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		probe.Close()
		ran++
		t.Run(spec, func(t *testing.T) {
			// layers -1 places every block and the head; n > 0 the first n
			// and leaves the rest, the head included, on the host.
			onDevice := func(run func(*State) [][]float32, layers int, f32Attn bool) ([][]float32, string) {
				opts := append([]tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff)},
					testTierOpts(t)...)
				if f32Attn {
					opts = append(opts, tier.WithConfig(func(c *tier.Config) {
						c.NoMMA, c.NoVolta, c.NoFlashPrefill = true, true, true
					}))
				}
				gpu, err := openTier(t, opts...)
				if err != nil {
					t.Fatal(err)
				}
				defer gpu.Close()
				st := m.NewState(len(g.IDs) + 1)
				defer st.Close()
				st.SetDeviceLayers(gpu, layers)
				want := nl
				if layers > 0 {
					want = layers
				}
				if n := st.GPULayers(); n != want {
					return nil, fmt.Sprintf("took %d of %d blocks: %v %v", n, want, st.DeviceDeclines(), gpu.Err())
				}
				if layers < 0 && !st.HeadOnDevice() {
					return nil, "the head stayed on the host: " + gpu.Err()
				}
				got := run(st)
				if n := st.GPULayers(); n != want {
					t.Fatalf("the session fell to the host (%d blocks left): %v", n, gpu.Err())
				}
				return got, ""
			}
			for _, arm := range []struct {
				name    string
				run     func(*State) [][]float32
				want    [][]float32
				layers  int
				f32Attn bool
				bound   float64
			}{
				{"decode", decode, wantDec, -1, false, bitnetNMSE},
				{"prefill, f32 attention", prefill, wantPre, -1, true, bitnetNMSE},
				{"prefill", prefill, wantPre, -1, false, bitnetDevPrefill},
				{"decode, partial seam", decode, wantDec, nl - 1, false, bitnetNMSE},
				{"prefill, partial seam, f32 attention", prefill, wantPre, nl - 1, true, bitnetNMSE},
			} {
				got, why := onDevice(arm.run, arm.layers, arm.f32Attn)
				if why != "" {
					t.Fatalf("%s: %s", arm.name, why)
				}
				w, band := cmp(arm.want, got)
				if arm.bound != bitnetNMSE {
					band = len(arm.want) // the binary16 tiles' band is not the reduction band
				}
				if !(w < arm.bound) || band < len(arm.want)/4 {
					t.Fatalf("%s: worst logit NMSE %.3e, %d of %d positions in the reduction band, against the host",
						arm.name, w, band, len(arm.want))
				}
				t.Logf("%s on %s: worst logit NMSE %.3e, %d of %d positions in the reduction band",
					arm.name, spec, w, band, len(arm.want))
			}
			for _, v := range bitnetDevViolations {
				undo := v.mut(m)
				got, why := onDevice(decode, nl, false) // every block; the head may stay home
				undo()
				if why != "" {
					t.Errorf("%s: the device declined the violation rather than running it: %s", v.name, why)
					continue
				}
				bad, _ := cmp(wantDec, got)
				if !(bad > 10*bitnetNMSE) {
					t.Errorf("%s: the device arm cannot see it -- worst NMSE %.3e", v.name, bad)
				}
				t.Logf("violation %q: worst NMSE %.3e", v.name, bad)
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// onesLike is a weight as long as w, every element one.
func onesLike(w []float32) []float32 {
	o := make([]float32, len(w))
	for i := range o {
		o[i] = 1
	}
	return o
}
