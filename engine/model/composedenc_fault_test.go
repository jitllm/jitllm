//go:build jitllmfault

package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestComposedEncoderOnDeviceSeesCausalRows runs the composed device
// encoder's one-block arm with its rows attending causally ("piccausal") and
// demands it part from the host far past the clean reading (RULE 10).
func TestComposedEncoderOnDeviceSeesCausalRows(t *testing.T) {
	gs, ms := encoderGoldens(t)
	for i, g := range gs {
		t.Run(g.Model, func(t *testing.T) {
			m := ms[i]
			he, err := m.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			want0, _ := encodeTexts(t, m, he, g)
			he.Close()
			ran := 0
			for _, spec := range stepDevices() {
				probe, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || probe == nil {
					continue
				}
				probe.Close()
				ran++
				got0, _ := encDeviceArm(t, m, g, spec, 1)
				tier.SetRowsFault("piccausal")
				bad0, _ := encDeviceArm(t, m, g, spec, 1)
				tier.SetRowsFault("")
				n0, nbad := worstNMSE(got0, want0), worstNMSE(bad0, want0)
				t.Logf("%s: clean NMSE %.3e after block 0, causal rows %.3e", spec, n0, nbad)
				if !(nbad > 100*n0) || !(nbad > 5e-4) {
					t.Errorf("%s: the gate cannot see causal rows: NMSE %.3e against a clean %.3e", spec, nbad, n0)
				}
			}
			if ran == 0 {
				t.Fatal("no device on this host: the gate proved nothing")
			}
		})
	}
}
