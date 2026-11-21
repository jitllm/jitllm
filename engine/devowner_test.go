package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// ownerBound is what a model with every block on the host but its lone
// matvecs on the device reads against the host alone, with headroom: the f32
// reduction-order band of a few routers and gates (1e-4..5e-4 measured over
// eight blocks). Another model's blocks in place of
// this one's read ~1.
const ownerBound = 1e-2

// TestATierServesASecondModelOnItsOwnTier offers two models of the same shape
// and different weights to one tier. A tier keys blocks by index, so the
// second model's block li offered beside the first's would be taken for it:
// the second model would run the first's weights, fluent and wrong. The second
// model's instance is given a tier of its own over the same card (GPU.ForModel):
// it places its blocks there and answers as its host run does, the first
// answers as before, and closing the first leaves the second as it was. A tier
// that refused it (the second model on the host) or mixed the two fails.
func TestATierServesASecondModelOnItsOwnTier(t *testing.T) {
	ids := []int32{1, 2, 3, 4}
	a, openOpts := hybridModelOpt(t, hyOpt{moe: true, layers: 4})
	b, openOpts2 := hybridModelOpt(t, hyOpt{moe: true, layers: 4, seed: 11})
	n := b.Cfg.NLayer

	logits := func(s *Instance) [][]float32 {
		var out [][]float32
		for _, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), l...))
		}
		return out
	}
	worst := func(got, want [][]float32) float64 {
		w := 0.0
		for p := range want {
			var num, den float64
			for i := range want[p] {
				d := float64(got[p][i] - want[p][i])
				num += d * d
				den += float64(want[p][i]) * float64(want[p][i])
			}
			// `NaN > bound` is false, so a NaN is mapped above every bound
			// rather than compared.
			e := num / den
			if math.IsNaN(e) {
				return math.Inf(1)
			}
			w = max(w, e)
		}
		return w
	}
	hb := testInstance(t, b, 16, openOpts2...)
	want := logits(hb)
	// The four tokens again at positions 4..7: what a session that ran the
	// first four reads next.
	wantNext := logits(hb)
	hb.Close()
	wantA := func() [][]float32 {
		ha := testInstance(t, a, 16, openOpts...)
		defer ha.Close()
		return logits(ha)
	}()
	if d := worst(want, wantA); d < 0.1 {
		t.Fatalf("the two models read NMSE %.3e apart on the host: mixing them could not be seen", d)
	}

	// Pinned: see hybridDeviceNMSEOn.
	g, err := openGPU(t, gpu.WithDevices("gpu:0"), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	sa := testInstance(t, a, 16, openOpts...)
	if err := sa.SetDeviceLayers(g, n); err != nil {
		t.Fatal(err)
	}
	if sa.GPULayers() == 0 {
		t.Skipf("the device took no block of the first model: %s", g.Err())
	}

	sb := testInstance(t, b, 16, openOpts2...)
	defer sb.Close()
	if err := sb.SetDeviceLayers(g, n); err != nil {
		t.Fatal(err)
	}
	if sb.GPULayers() == 0 {
		t.Fatalf("the second model placed no block while the first holds %d: %v %s", sa.GPULayers(), sb.DeviceDeclines(), g.Err())
	}
	if d := worst(logits(sb), want); d > ownerBound {
		t.Fatalf("the second model placed %d block(s) beside the first's and reads NMSE %.3e against "+
			"its own host run: it ran the first model's weights", sb.GPULayers(), d)
	}
	if d := worst(logits(sa), wantA); d > ownerBound {
		t.Fatalf("the first model reads NMSE %.3e against its host run beside the second", d)
	}
	t.Logf("the first holds %d block(s) and the second %d on one card", sa.GPULayers(), sb.GPULayers())

	// The first closes: the second keeps its tier and its answer, and a new
	// instance of it reaches the same tier.
	sa.Close()
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	for dev, held := range g.Placed() {
		if held != 0 {
			t.Fatalf("device %d still holds %d of the first model's block(s) after it closed", dev, held)
		}
	}
	if d := worst(logits(sb), wantNext); d > ownerBound {
		t.Fatalf("the second model reads NMSE %.3e after the first closed", d)
	}
	sc := testInstance(t, b, 16, openOpts2...)
	defer sc.Close()
	if err := sc.SetDeviceLayers(g, n); err != nil {
		t.Fatal(err)
	}
	if sc.GPULayers() == 0 {
		t.Fatalf("a second instance of the second model placed nothing: %v", sc.DeviceDeclines())
	}
	if d := worst(logits(sc), want); d > ownerBound {
		t.Fatalf("a second instance of the second model reads NMSE %.3e against its host run", d)
	}
}
