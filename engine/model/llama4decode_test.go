package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestSynthLlama4Q8DecodeOnDevice holds the device's DECODE mixture on the
// quantized llama4 fixture against the host, with the expert matvecs pinned to
// the one configuration tier.fuseMoE would fuse (split 4, in-group). Llama 4
// weights the expert's input (ActMulWeighted), which the fused up epilogue does
// not carry, so fuseMoE must decline it. The F32 fixture cannot reach this (a
// float bank is never fused). The bound is an NMSE of 1e-6 rather than an
// argmax because removing the exclusion leaves the argmax unchanged.
func TestSynthLlama4Q8DecodeOnDevice(t *testing.T) {
	m, g := openSynthLlama4Q8(t)
	defer m.Close()
	host := m.NewState(len(g.IDs) + 1)
	want := runIDs(t, host, g.IDs)
	host.Close()
	gpu, err := openTier(t, tier.WithDevices(testdevices.Like(t, "cuda:0")), tier.WithDeviceTune(tier.TuneOff),
		tier.WithSplit(4), tier.WithConfig(func(c *tier.Config) { c.ForceIndexedGroup = true }))
	if err != nil || gpu == nil {
		t.Skipf("no cuda device (%v)", err)
	}
	defer gpu.Close()
	st := m.NewState(len(g.IDs) + 1)
	defer st.Close()
	st.SetDeviceLayers(gpu, -1)
	if n := st.GPULayers(); n != m.Cfg.NLayer {
		t.Fatalf("the device took %d of %d blocks: %v %v", n, m.Cfg.NLayer, st.DeviceDeclines(), gpu.Err())
	}
	got := runIDs(t, st, g.IDs)
	if n := st.GPULayers(); n != m.Cfg.NLayer {
		t.Fatalf("the session fell to the host: %v", gpu.Err())
	}
	if n := gpu.Stats().IndexedGroup; n == 0 {
		t.Fatal("no expert matvec ran in-group, so the fusable configuration was never selected")
	}
	worst := 0.0
	for p := range want {
		nmse := llama4Cmp(f64s(want[p]), got[p])
		worst = math.Max(worst, nmse)
		if math.IsNaN(nmse) || !(nmse < 1e-6) {
			t.Fatalf("pos %d: NMSE %.3e, argmax %d against the host's %d (%d fused expert launches)",
				p, nmse, Greedy(got[p]), Greedy(want[p]), gpu.Stats().MoEFused)
		}
	}
	t.Logf("decode on cuda matches the host at %d positions, worst NMSE %.3e, %d fused expert launches",
		len(want), worst, gpu.Stats().MoEFused)
}
