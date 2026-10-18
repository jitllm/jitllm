package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestFinalSoftcapReachesTheDeviceHead is the gate on gemma2's final logit
// softcap when the device runs the output projection: the folded path (every
// block placed, the head in the same submission) and Prefill's device-head tail
// returned the tier's logits without the cap. The cap is perturbed rather
// than waited for: an F32 fixture agrees with the host to ~1e-12, so setting
// FinalSoftcap from the host's own logits makes the cap the only possible
// difference.
func TestFinalSoftcapReachesTheDeviceHead(t *testing.T) {
	m, err := Open(hfContainer(t, "synth-deepseek-dense"), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	ids := []int32{5, 11, 23, 41, 67, 89}
	raw := teacherForce(t, m, ids)
	amax := 0.0
	for _, l := range raw {
		for _, v := range l {
			amax = math.Max(amax, math.Abs(float64(v)))
		}
	}
	if amax == 0 {
		t.Fatal("the host's logits are all zero -- a cap would compare nothing")
	}
	// Half the largest logit: tanh is far from linear there, so an uncapped
	// row differs by a large fraction of its own norm.
	m.Cfg.FinalSoftcap = float32(amax / 2)
	want := teacherForce(t, m, ids)

	nmse := func(got, w []float32) float64 {
		var num, den float64
		for i := range w {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				return math.Inf(1)
			}
			d := float64(got[i] - w[i])
			num += d * d
			den += float64(w[i]) * float64(w[i])
		}
		return num / den
	}
	const bound = 1e-8

	// The folded path: every block placed, the head in the same submission.
	dev := m.NewState(len(ids) + 1)
	defer dev.Close()
	dev.SetDevice(g)
	if dev.GPULayers() != m.Cfg.NLayer || !dev.HeadOnDevice() {
		t.Skipf("wanted every block and the head on the device, got %d of %d, head %v: %s",
			dev.GPULayers(), m.Cfg.NLayer, dev.HeadOnDevice(), g.Err())
	}
	var devTok []int32
	for p, id := range ids {
		got, err := dev.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		if n := nmse(got, want[p]); !(n < bound) {
			t.Fatalf("Forward, head folded on the device, pos %d: NMSE %.3e against the capped "+
				"host (%.3e against the UNCAPPED one) -- the final softcap was not applied",
				p, n, nmse(got, raw[p]))
		}
		devTok = append(devTok, Greedy(got))
	}
	dev.SetGPULayers(0)

	// ForwardGreedy under the cap: the device must serve the token (the argmax
	// runs after nn.Head.Softcap) and match Greedy over Forward's logits. The
	// count of device-served tokens says the configuration was selected.
	gs := m.NewState(len(ids) + 1)
	defer gs.Close()
	gs.SetDevice(g)
	if gs.GPULayers() != m.Cfg.NLayer || !gs.HeadOnDevice() {
		t.Skipf("the greedy State did not get every block and the head: %s", g.Err())
	}
	served := 0
	for p, id := range ids {
		// -1 before every call: ForwardGreedy's host arm never writes Token,
		// so a value left from an earlier token would count as served.
		gs.head.Token = -1
		tok, err := gs.ForwardGreedy(id)
		if err != nil {
			t.Fatal(err)
		}
		if gs.head != nil && gs.head.Token >= 0 {
			served++
		}
		if tok != devTok[p] {
			t.Fatalf("ForwardGreedy, capped, pos %d: token %d, Greedy over the device's capped "+
				"logits %d", p, tok, devTok[p])
		}
	}
	if served != len(ids) {
		t.Fatalf("the device served %d of %d capped greedy tokens: the rest read every logit "+
			"back and capped them on the host", served, len(ids))
	}

	// Prefill's device-head tail.
	ps := m.NewState(len(ids) + 1)
	defer ps.Close()
	ps.SetDevice(g)
	if !ps.HeadOnDevice() {
		t.Skipf("the prefill State did not get the head on the device: %s", g.Err())
	}
	got, err := ps.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	last := len(ids) - 1
	if n := nmse(got, want[last]); !(n < bound) {
		t.Fatalf("Prefill, head on the device: NMSE %.3e against the capped host (%.3e against "+
			"the UNCAPPED one) -- the final softcap was not applied", n, nmse(got, raw[last]))
	}
	t.Logf("cap %.3g: folded Forward and device-head Prefill both agree with the capped host", m.Cfg.FinalSoftcap)
}
