package model

import (
	"math"
	"os"
	"slices"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestForwardSampleMatchesSampleOverForward decodes the same prompt twice with
// every block and the head on the device and the same seeded sampler (a
// temperature, top-k, top-p, min-p and a repeat penalty), once through
// ForwardSample and once through Forward and Sample, and demands the same
// tokens -- and that the device selected every one of them (SampledOnDevice,
// the tier's SampleReads), because a call that quietly read the row back is
// Sample(Forward) compared with itself.
//
// The candidates are then held to the definition rather than to the shared
// draw: two thousand seeds drawn from the last step's candidates must all land
// in what llama.cpp's order keeps of them -- top-p, then min-p, over their
// softmax at temperature 1 -- computed here in float64. The host and device
// arms share drawBounded, so only this half sees a broken cut.
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
	m, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("Once upon a time", true)
	// Past one whole probe (4*sampleProbeRounds tokens), so the auto arm
	// decides inside the run.
	const n = 4*sampleProbeRounds + 8
	var lastVals []float32
	var lastIDs []uint32
	var probes int64
	run := func(device bool, mode DeviceSampleMode) (out []int32, onDev int64, reads int, allocs float64) {
		m.opt.devSample = mode
		m.sampleChoices = sampleChoices{}
		g := swaTier(t)
		defer g.Close()
		st := m.NewState(256)
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
		next := sm.Sample(logits)
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
				next = sm.Sample(logits)
			}
		}
		onDev, reads, probes = st.SampledOnDevice, g.Stats().SampleReads-reads0, st.SampleProbes
		if device {
			if mode == DeviceSampleOn {
				lastVals = slices.Clone(st.sampleVals[:sm.TopK])
				lastIDs = slices.Clone(st.sampleIDs[:sm.TopK])
			}
			if mode == DeviceSampleAuto {
				// The warm steps below are probe steps: a fresh probe.
				c := st.sampleChoiceFor(st.sampleKeyOf(sm))
				c.step, c.dev, c.host = 0, 0, 0
			}
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
	// Auto: a probe ran, decided, and every token is the host's.
	auto, autoDev, _, autoAllocs := run(true, DeviceSampleAuto)
	if probes != 4*sampleProbeRounds || !slices.Equal(auto, want) || autoAllocs != 0 {
		t.Fatalf("auto: %d probe tokens (want %d), %.1f warm allocations in a probe, tokens equal %v",
			probes, 4*sampleProbeRounds, autoAllocs, slices.Equal(auto, want))
	}
	t.Logf("auto: %d probe tokens, %d of %d selected on the device", probes, autoDev, n)
	got, onDev, reads, allocs := run(true, DeviceSampleOn)
	if onDev != n || reads != n {
		t.Fatalf("the device selected %d of %d sampled tokens (tier: %d): the rest read the row back",
			onDev, n, reads)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("ForwardSample %v\nSample(Forward) %v", got, want)
	}
	k := sampleGateSampler().TopK
	t.Logf("%d tokens identical, all selected on the device; read back per token %d bytes (the row: %d); "+
		"warm allocations per token %.1f", n, 8*k, 4*m.Cfg.NVocab, allocs)
	if allocs != 0 {
		t.Fatalf("a warm ForwardSample allocates %.1f times per token", allocs)
	}

	// The definition: every draw lands in the nucleus.
	nucleus := topPNucleus(lastVals, lastIDs, sampleGateSampler())
	for seed := int64(0); seed < 2000; seed++ {
		s := sampleGateSampler()
		s.Seed = seed
		tok := s.SampleFrom(lastVals, lastIDs)
		if !nucleus[tok] {
			t.Fatalf("seed %d drew %d, outside the top-p nucleus %v of the candidates", seed, tok, nucleus)
		}
	}
}

// topPNucleus is the tokens a top-k/top-p/min-p draw over these candidates
// may return, by llama.cpp's order in float64: their softmax at temperature
// 1, the smallest prefix whose mass reaches top-p, then min-p's floor against
// the best -- the temperature shapes only the draw over what survives --
// widened by one past the boundary, where float32 rounding may legitimately
// keep one more.
func topPNucleus(vals []float32, ids []uint32, s *Sampler) map[int32]bool {
	p := make([]float64, len(vals))
	sum := 0.0
	for i, v := range vals {
		p[i] = math.Exp(float64(v) - float64(vals[0]))
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	if !sort.SliceIsSorted(p, func(a, b int) bool { return p[a] > p[b] }) {
		panic("candidates not best first")
	}
	cut, mass := len(p), 0.0
	for i := range p {
		mass += p[i]
		if mass >= s.TopP {
			cut = i + 1
			break
		}
	}
	keep := cut
	for i := range cut {
		if p[i] < s.MinP*p[0] {
			keep = i
			break
		}
	}
	out := map[int32]bool{}
	for i := range min(keep+1, len(p)) {
		out[int32(ids[i])] = true
	}
	return out
}
