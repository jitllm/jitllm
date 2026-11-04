package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestResetClearsTheRecurrentSummary checks Reset makes a hybrid's sequence
// fresh, forgetting the recurrent summary (a linear block's whole history) as
// well as positions and pages, on the CPU and with every block on each test
// device. The bar is bit equality against a never-used instance on the same
// placement, since the arithmetic is identical.
func TestResetClearsTheRecurrentSummary(t *testing.T) {
	m := hybridModelOpt(t, hyOpt{moe: true, layers: 8})
	defer m.Close()
	t.Run("cpu", func(t *testing.T) { resetCase(t, m, nil) })
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || g == nil {
				noDevice(t, dev, err)
			}
			defer g.Close()
			resetCase(t, m, g)
		})
	}
}

// resetCase is the gate on g's devices, or on the CPU where g is nil.
func resetCase(t *testing.T, m *Model, g *gpu.GPU) {
	place := func(s *Instance) {
		if g == nil {
			return
		}
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if s.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("the device took %d of %d blocks: %s", s.GPULayers(), m.Cfg.NLayer, g.Err())
		}
	}
	fresh := m.NewInstance(64)
	defer fresh.Close()
	if !fresh.recurrent() {
		t.Fatal("this fixture has no recurrent blocks, so the gate proves nothing")
	}
	place(fresh)

	ids := []int32{1, 2, 3, 4, 5}
	want, err := fresh.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	wantCopy := append([]float32(nil), want...)

	// A second state, dirtied with a DIFFERENT sequence, then reset.
	used := m.NewInstance(64)
	defer used.Close()
	place(used)
	if _, err := used.Prefill([]int32{7, 8, 9, 10, 11, 12}); err != nil {
		t.Fatal(err)
	}
	used.Reset()
	got, err := used.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != len(wantCopy) {
		t.Fatalf("logit widths differ: %d against %d", len(got), len(wantCopy))
	}
	diff, worst := 0, float32(0)
	for i := range got {
		if got[i] != wantCopy[i] {
			diff++
			if d := got[i] - wantCopy[i]; d > worst || -d > worst {
				worst = d
				if worst < 0 {
					worst = -worst
				}
			}
		}
	}
	if diff != 0 {
		t.Errorf("%d of %d logits differ after Reset (worst |d| %g): the sequence "+
			"still carries the previous conversation's recurrent summary",
			diff, len(got), worst)
	}
}
