package engine

import (
	"os"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestForwardSampleMatchesSampleOverForward decodes the same prompt twice with
// every block and the head on the device and the same seeded sampler (a
// temperature, top-k, top-p, min-p and a repeat penalty), once through
// ForwardSample and once through Forward and Sample, and demands the same
// tokens -- and that the device selected every one of them (SampledOnDevice,
// the tier's SampleReads), because a call that quietly read the row back is
// Sample(Forward) compared with itself. The device draws with the composed
// walk and the host arm with the host sampler, two implementations of one
// draw; the walk's own definition is jit's TestComposedCPUSampleWalk.
func TestForwardSampleMatchesSampleOverForward(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "gemma-2-2b-it-Q4_K_M"} {
		t.Run(name, func(t *testing.T) { forwardSampleMatches(t, name) })
	}
}

func sampleGateSampler() *Sampler {
	return &Sampler{Temp: 1.3, TopK: 40, TopP: 0.6, MinP: 0.02, RepeatPen: 1.3, RepeatLastN: 32, Seed: 4242}
}

func forwardSampleMatches(t *testing.T, name string) {
	p := testmodels.Path(name + ".jlm")
	if src := testmodels.Path(name + ".gguf"); fileExists(src) {
		p = jlmOf(t, src)
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := model.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("Once upon a time", true)
	const n = 40
	run := func(device bool, mode DeviceSampleMode) (out []int32, onDev int64, reads int, allocs float64) {
		g := swaTier(t)
		defer g.Close()
		st := testInstance(t, m, 256, WithDeviceSample(mode))
		defer st.Close()
		st.SetDevice(g)
		if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
			t.Skipf("CARD TOO SMALL: placed %d of %d blocks, head on the device %v (%s) -- this gate proved nothing here",
				st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
		}
		sm := sampleGateSampler()
		logits, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		next, err := st.Sample(sm, logits)
		if err != nil {
			t.Fatal(err)
		}
		reads0 := g.Stats().SampleReads
		for i := 0; i < n; i++ {
			out = append(out, next)
			sm.Observe(next)
			if device {
				if next, err = st.ForwardSample(next, sm); err != nil {
					t.Fatal(err)
				}
			} else {
				if logits, err = st.Forward(next); err != nil {
					t.Fatal(err)
				}
				if next, err = st.Sample(sm, logits); err != nil {
					t.Fatal(err)
				}
			}
		}
		onDev, reads = st.SampledOnDevice, g.Stats().SampleReads-reads0
		if device {
			// Warm: the same step again allocates nothing on the engine's side.
			allocs = testing.AllocsPerRun(4, func() {
				sm.Observe(next)
				if next, err = st.ForwardSample(next, sm); err != nil {
					t.Fatal(err)
				}
			})
		}
		return out, onDev, reads, allocs
	}
	want, _, _, _ := run(false, DeviceSampleOn)
	// Forced off: every token reads the row back.
	off, offDev, offReads, offAllocs := run(true, DeviceSampleOff)
	if offDev != 0 || offReads != 0 || !slices.Equal(off, want) || offAllocs != 0 {
		t.Fatalf("forced off: %d tokens selected on the device (tier %d), %.1f warm allocations, tokens equal %v",
			offDev, offReads, offAllocs, slices.Equal(off, want))
	}
	// Auto: every token drawn on the device, as forced on.
	auto, autoDev, _, autoAllocs := run(true, DeviceSampleAuto)
	if autoDev != n || !slices.Equal(auto, want) || autoAllocs != 0 {
		t.Fatalf("auto: %d of %d tokens drawn on the device, %.1f warm allocations, tokens equal %v",
			autoDev, n, autoAllocs, slices.Equal(auto, want))
	}
	got, onDev, reads, allocs := run(true, DeviceSampleOn)
	if onDev != n || reads != n {
		t.Fatalf("the device selected %d of %d sampled tokens (tier: %d): the rest read the row back",
			onDev, n, reads)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("ForwardSample %v\nSample(Forward) %v", got, want)
	}
	t.Logf("%d tokens identical, all drawn on the device; read back per token 4 bytes (the row: %d); "+
		"warm allocations per token %.1f", n, 4*m.Cfg.NVocab, allocs)
	if allocs != 0 {
		t.Fatalf("a warm ForwardSample allocates %.1f times per token", allocs)
	}
}
