package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestShrinkingTheSeamBringsTheSummaryHome: shrinking the seam on a hybrid must
// bring the recurrent summary home, not free it. A linear block's history is a
// running summary the host cannot re-derive, so a release that moved only the
// KV cache would resume the host from a summary that never saw a token (fluent
// wrong text). SetGPULayers(0) takes the same ReleaseLayers path a demotion
// does, through migrateKV/migrateRec, without a fault injector. The seam moves
// mid-run; the tail is compared against a pure-host run.
func TestShrinkingTheSeamBringsTheSummaryHome(t *testing.T) {
	defer fusedRunsCheck(t, true)()
	m := hybridModelOpt(t, hyOpt{moe: true, layers: 8})
	defer m.Close()
	if m.Cfg.NLayer < 4 {
		t.Fatalf("the fixture has %d layers; this gate wants a real hybrid", m.Cfg.NLayer)
	}

	ids := []int32{1, 2, 3, 4, 5, 6}
	host := m.NewState(16)
	defer host.Close()
	want := make([][]float32, len(ids))
	for i, id := range ids {
		l, err := host.Forward(id)
		if err != nil {
			t.Fatalf("host: %v", err)
		}
		want[i] = append([]float32(nil), l...)
	}

	// Pinned: see hybridDeviceNMSEOn.
	g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	dev := m.NewState(16)
	defer dev.Close()
	dev.SetDeviceLayers(g, m.Cfg.NLayer)
	runs := dev.DeviceBlocks()
	placed := 0
	for _, r := range runs {
		placed += r[1] - r[0]
	}
	if placed == 0 {
		t.Skipf("the device took no block of this fixture: %s", g.Err())
	}
	// The configuration check is on layer kind: a placement of only attention
	// blocks exercises migrateKV and says nothing about the summary.
	lin := 0
	for _, r := range runs {
		for li := r[0]; li < r[1]; li++ {
			if m.Cfg.LayerKind(li).Recurrent() {
				lin++
			}
		}
	}
	if lin == 0 {
		t.Skipf("%d blocks placed and none of them linear: nothing to bring home", placed)
	}
	t.Logf("%d block(s) placed, %d of them linear", placed, lin)

	half := len(ids) / 2
	got := make([][]float32, len(ids))
	for i := 0; i < half; i++ {
		l, err := dev.Forward(ids[i])
		if err != nil {
			t.Fatalf("device: %v", err)
		}
		got[i] = append([]float32(nil), l...)
	}
	// The migration under test.
	if n := dev.SetGPULayers(0); n != 0 {
		t.Fatalf("SetGPULayers(0) left %d blocks placed", n)
	}
	for i := half; i < len(ids); i++ {
		l, err := dev.Forward(ids[i])
		if err != nil {
			t.Fatalf("host after shrink: %v", err)
		}
		got[i] = append([]float32(nil), l...)
	}

	// The tail is the assertion, the head the control: the device half carries
	// its own small int8 perturbation, while a lost summary is a step change
	// after the shrink.
	worst, at := 0.0, -1
	for p := range ids {
		var num, den float64
		for i := range want[p] {
			if math.IsNaN(float64(got[p][i])) || math.IsInf(float64(got[p][i]), 0) {
				t.Fatalf("pos %d logit %d is %g: not finite, which no tolerance can see",
					p, i, got[p][i])
			}
			d := float64(got[p][i] - want[p][i])
			num += d * d
			den += float64(want[p][i]) * float64(want[p][i])
		}
		nmse := num / den
		t.Logf("pos %d: logit NMSE %.3e%s", p, nmse, map[bool]string{true: "  <- after the shrink"}[p >= half])
		if p >= half && nmse > worst {
			worst, at = nmse, p
		}
	}
	if worst > 1e-3 {
		t.Errorf("pos %d after the shrink: logit NMSE %.3e -- the recurrent summary "+
			"did not come home, so the host resumed from a state that never saw "+
			"the first %d tokens", at, worst, half)
	}
}
