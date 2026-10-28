//go:build jitllmfault

package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestComposedTowerOnDeviceSeesCausalRows runs the composed device tower's
// one-block arm with its rows attending causally ("piccausal") and demands
// it part from the host far past the clean reading (RULE 10).
func TestComposedTowerOnDeviceSeesCausalRows(t *testing.T) {
	m, tw := openTower(t, noTune)
	defer m.Close()
	c := tw.Cfg
	px := make([]float32, c.ImageSz*c.ImageSz*3)
	for i := range px {
		px[i] = float32((i*37)%255)/127.5 - 1
	}
	hs := tw.testState()
	want0, _ := towerEncodeAt(t, hs, px)
	hs.Close()
	defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
	tw.opt.towerGPULayers = 1
	ran := 0
	for _, spec := range stepDevices() {
		g, err := tier.OpenWith(tier.WithDevices(spec), composedOpt(towerParts))
		if err != nil || g == nil {
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer g.Close()
			read := func(fault string) float64 {
				tier.SetRowsFault(fault)
				defer tier.SetRowsFault("")
				s := tw.testState()
				defer s.Close()
				s.SetDevice(g)
				s0, _, _ := g.Stats().ServedPart(tier.ComposedTower)
				got0, _ := towerEncodeAt(t, s, px)
				if n, _, _ := g.Stats().ServedPart(tier.ComposedTower); n-s0 != 1 {
					t.Fatalf("%s: the composer served %d tower block runs, want the one placed (%s)", spec, n-s0, g.Err())
				}
				nm, _ := nmseOf(got0, want0)
				return nm
			}
			clean, bad := read(""), read("piccausal")
			t.Logf("%s: after block 0, clean NMSE %.3e, causal rows %.3e", spec, clean, bad)
			if !(bad > 100*clean) || !(bad > 5e-4) {
				t.Fatalf("%s: the gate cannot see causal rows: %.3e against a clean %.3e", spec, bad, clean)
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device on this host: the gate proved nothing")
	}
}
