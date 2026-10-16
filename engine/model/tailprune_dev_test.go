package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// tailPlaced is how many blocks the device takes of a model when asked for
// every one.
func tailPlaced(t *testing.T, path, spec string) (placed, layers, shared int) {
	m, err := Open(jlmOf(t, path), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := tier.OpenWith(tier.WithDevices(spec))
	if err != nil || g == nil {
		t.Fatalf("%s: %v", spec, err)
	}
	defer g.Close()
	st := m.NewState(64)
	defer st.Close()
	if err := st.SetDeviceLayers(g, -1); err != nil {
		t.Fatal(err)
	}
	return st.GPULayers(), m.Cfg.NLayer, int(m.Cfg.NKVShared)
}

// TestTailPruneOnTheDevice: every block the device takes placed, and a seam
// inside that run (the tail partly or wholly on the host), give the unpruned
// prompt's decode.
func TestTailPruneOnTheDevice(t *testing.T) {
	spec := testdevices.One("gpu")
	for _, name := range []string{"synth-gemma4-e", "synth-gemma4-e-ragged"} {
		path := tailModelPath(t, name)
		placed, n, T := tailPlaced(t, path, spec)
		t.Logf("%s on %s: %d of %d blocks placed, %d share KV", name, spec, placed, n, T)
		for _, dev := range []int{placed, placed - T} {
			if dev < 1 {
				continue
			}
			a := tailExec(t, path, false, 7, 40, dev, spec, false)
			b := tailExec(t, path, true, 7, 40, dev, spec, false)
			if b.would != 40*T || b.ran != T {
				t.Errorf("%s, %d blocks placed: row-blocks %d -> %d, want %d -> %d", name, dev, b.would, b.ran, 40*T, T)
			}
			maxd := 0.0
			for i := range a.dec {
				maxd = max(maxd, float64(max(a.dec[i]-b.dec[i], b.dec[i]-a.dec[i])))
			}
			if maxd > 1e-3 {
				t.Errorf("%s, %d blocks placed: decode logits differ by %.3g", name, dev, maxd)
			}
		}
	}
}
