package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestSynthLlama4Q8DecodeOnDevice holds the device's composed DECODE mixture on
// the quantized llama4 fixture against the host. Llama 4 weights the expert's
// input (ActMulWeighted), which a form that weights the expert's output would
// get wrong without moving the argmax, so the bound is an NMSE of 1e-6.
func TestSynthLlama4Q8DecodeOnDevice(t *testing.T) {
	m, g := openSynthLlama4Q8(t)
	defer m.Close()
	host := m.NewInstance(len(g.IDs) + 1)
	want := runIDs(t, host, g.IDs)
	host.Close()
	gpux, err := openGPU(t, gpu.WithDevices(testdevices.Like(t, "cuda:0")), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || gpux == nil {
		t.Skipf("no cuda device (%v)", err)
	}
	defer gpux.Close()
	st := m.NewInstance(len(g.IDs) + 1)
	defer st.Close()
	st.SetDeviceLayers(gpux, -1)
	if n := st.GPULayers(); n != m.Cfg.NLayer {
		t.Fatalf("the device took %d of %d blocks: %v %v", n, m.Cfg.NLayer, st.DeviceDeclines(), gpux.Err())
	}
	got := runIDs(t, st, g.IDs)
	if n := st.GPULayers(); n != m.Cfg.NLayer {
		t.Fatalf("the session fell to the host: %v", gpux.Err())
	}
	if n, d := gpux.Stats().ServedPart(gpu.ComposedBlock); n == 0 || d != 0 {
		t.Fatalf("%d block steps composed and %d declined: the composed mixture was not selected", n, d)
	}
	worst := 0.0
	for p := range want {
		nmse := llama4Cmp(f64s(want[p]), got[p])
		worst = math.Max(worst, nmse)
		if math.IsNaN(nmse) || !(nmse < 1e-6) {
			t.Fatalf("pos %d: NMSE %.3e, argmax %d against the host's %d",
				p, nmse, Greedy(got[p]), Greedy(want[p]))
		}
	}
	t.Logf("decode on cuda matches the host at %d positions, worst NMSE %.3e", len(want), worst)
}
