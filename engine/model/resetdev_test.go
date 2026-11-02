package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestResetZeroesTheDeviceRecurrentState holds a reused instance to a fresh one
// on a hybrid with every block on the device: a prompt after Reset must not
// start from the last prompt's recurrent summary, which a placed linear block
// keeps on the card (Instance.Reset zeroes it there, as Retire does a row's). A
// decision model asks its questions on one instance, Reset between them, and on
// d1 (lfm2) the second question's answers moved 0.27 of a logit before.
func TestResetZeroesTheDeviceRecurrentState(t *testing.T) {
	for _, name := range []string{"synth-lfm2", "synth-mamba2"} {
		t.Run(name, func(t *testing.T) {
			m, g := openSSM(t, name)
			defer m.Close()
			half := len(g.IDs) / 2
			first, second := g.IDs[:half], g.IDs[half:]
			ran := 0
			for _, spec := range stepDevices() {
				gpux, err := gpu.OpenWith(append([]gpu.Option{gpu.WithDevices(spec),
					gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
				if err != nil || gpux == nil {
					continue
				}
				ran++
				t.Run(spec, func(t *testing.T) {
					defer gpux.Close()
					prefill := func(st *Instance, ids []int32) []float32 {
						lg, err := st.Prefill(ids)
						if err != nil {
							t.Fatal(err)
						}
						return append([]float32(nil), lg...)
					}
					placed := func() *Instance {
						st := m.NewInstance(len(g.IDs) + 1)
						st.SetDeviceLayers(gpux, -1)
						if n := st.GPULayers(); n != m.Cfg.NLayer {
							t.Fatalf("placed %d of %d blocks: %v", n, m.Cfg.NLayer, gpux.Err())
						}
						return st
					}
					fresh := placed()
					want := prefill(fresh, second)
					fresh.Close()
					st := placed()
					defer st.Close()
					prefill(st, first)
					st.Reset()
					got := prefill(st, second)
					d, _ := nmse32(want, got)
					t.Logf("%s: the second prompt after Reset against a fresh instance: NMSE %.3e", spec, d)
					if !(d < 1e-12) {
						t.Fatalf("%s: after Reset the second prompt reads NMSE %.3e from a fresh instance's: "+
							"the device kept the first prompt's recurrent state", spec, d)
					}
				})
			}
			if ran == 0 {
				noDevice(t, "device", nil)
			}
		})
	}
}
