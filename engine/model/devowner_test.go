package model

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// ownerBound is what a model with every block on the host but its lone
// matvecs on the device reads against the host alone, with headroom: the f32
// reduction-order band of a few routers and gates (TestPagedWeightsAreNeverServedStale
// reads 1e-4..5e-4 over eight blocks). Another model's blocks in place of
// this one's read ~1.
const ownerBound = 1e-2

// TestATierServesOneModel offers two models of the same shape and different
// weights to one tier. A tier keys blocks by index, so the second model's
// block li offered beside the first's was taken for it: the second model ran
// the first's weights, fluent and wrong. The second must be refused by name,
// run on the host, and get the tier once the first is closed.
func TestATierServesOneModel(t *testing.T) {
	ids := []int32{1, 2, 3, 4}
	a := hybridModelOpt(t, hyOpt{moe: true, layers: 4})
	b := hybridModelOpt(t, hyOpt{moe: true, layers: 4, seed: 11})
	n := b.Cfg.NLayer

	logits := func(s *State) [][]float32 {
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
	hb := b.NewState(16)
	want := logits(hb)
	hb.Close()
	if d := worst(want, func() [][]float32 {
		ha := a.NewState(16)
		defer ha.Close()
		return logits(ha)
	}()); d < 0.1 {
		t.Fatalf("the two models read NMSE %.3e apart on the host: mixing them could not be seen", d)
	}

	// Pinned: see hybridDeviceNMSEOn.
	g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	sa := a.NewState(16)
	if err := sa.SetDeviceLayers(g, n); err != nil {
		t.Fatal(err)
	}
	if sa.GPULayers() == 0 {
		t.Skipf("the device took no block of the first model: %s", g.Err())
	}

	sb := b.NewState(16)
	if err := sb.SetDeviceLayers(g, n); err != nil {
		t.Fatal(err)
	}
	got := logits(sb)
	d := worst(got, want)
	if sb.GPULayers() != 0 || sb.head != nil || d > ownerBound {
		t.Fatalf("the second model placed %d block(s) (head %v) beside the first's and reads "+
			"NMSE %.3e against its own host run: it ran the first model's weights",
			sb.GPULayers(), sb.head != nil, d)
	}
	if e := g.Err(); !strings.Contains(e, "serves one model") {
		t.Fatalf("the refusal does not say why: %q", e)
	}
	t.Logf("refused while the first holds %d block(s): %s; NMSE %.3e on the host",
		sa.GPULayers(), g.Err(), d)
	sb.Close()

	// The first model is unaffected by the refusal, and closing it frees the
	// tier for the second.
	logits(sa)
	sa.Close()
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	for dev, held := range g.Placed() {
		if held != 0 {
			t.Fatalf("device %d still holds %d block(s) after the first model closed", dev, held)
		}
	}
	sb = b.NewState(16)
	defer sb.Close()
	if err := sb.SetDeviceLayers(g, n); err != nil {
		t.Fatal(err)
	}
	if sb.GPULayers() == 0 {
		t.Fatalf("the tier stayed refused after the first model closed: %s", g.Err())
	}
	if d := worst(logits(sb), want); d > ownerBound {
		t.Fatalf("after the first model closed the second placed %d block(s) and reads "+
			"NMSE %.3e against its host run", sb.GPULayers(), d)
	} else {
		t.Logf("after the first closed: %d of %d block(s) placed, NMSE %.3e", sb.GPULayers(), n, d)
	}
}
