//go:build jitllmfault

package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestDemotedDevicePrefillFinishesOnTheHost fails the device partway through a
// prompt whose every block it held, and demands the host finish it with the
// same logits a host-only prefill gives. A device-only prompt allocates no host
// rows (growDeviceBatch), so the host restart after demotion must allocate
// them (growBatch). Injected failures: the batched call (1), then the per-row
// retry after one success (3).
//
// Build with -tags jitllmfault; tier reads JITLLM_DEVICE_FAIL_AT at Open.
func TestDemotedDevicePrefillFinishesOnTheHost(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS) -- this gate proved nothing", err)
	}
	// Both arms on the same arithmetic: the tuner off (a pack width is a
	// summation order) and an f32 KV cache (a device steps an f16 one down to
	// f32 before it places a block, so the demoted State holds f32).
	m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("The capital of France is Paris, and the capital of Germany is Berlin, "+
		"and the capital of Italy is", true)

	host := m.NewState(len(ids) + 8)
	defer host.Close()
	want, err := host.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)

	t.Setenv("JITLLM_DEVICE_FAIL_AT", "1,3")
	g, err := tier.Open(0)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	dev := m.NewState(len(ids) + 8)
	defer dev.Close()
	dev.SetDevice(g)
	if dev.GPULayers() != m.Cfg.NLayer {
		t.Skipf("the device took %d of %d blocks; this gate needs a device-only prompt", dev.GPULayers(), m.Cfg.NLayer)
	}
	got, err := dev.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	// The selection check: the injected failures must have demoted the blocks,
	// or the host restart this gate is about never ran.
	if dev.DeviceDemotions() == 0 || dev.GPULayers() != 0 {
		t.Fatalf("%d demotion(s) and %d blocks still on the device: the host restart was not reached",
			dev.DeviceDemotions(), dev.GPULayers())
	}
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	// The restart runs the WHOLE chunk on the host from the embeddings, so it
	// is the host arm's arithmetic exactly.
	if argmaxID(got) != argmaxID(want) || num/den > 1e-9 {
		t.Fatalf("after the demotion: argmax %d (host %d), logit NMSE %.3e", argmaxID(got), argmaxID(want), num/den)
	}
}
