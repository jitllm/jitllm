package engine

import (
	"testing"

	"github.com/jitllm/jitllm/nn"
)

// TestSpeculationComposesOnce: an instance's speculations share one draft
// session, so only its first composes: a second and third Speculator on the
// instance, each started and run a round, ask the composer for nothing
// (Instance.Account's Asks), and draw the tokens the first drew for the same
// seed.
func TestSpeculationComposesOnce(t *testing.T) {
	m, openOpts := openSpecModel(t, "synth-qwen35-hybrid-mtp.gguf", WithKVF16(false))
	defer m.Close()
	cd, err := OpenCPU()
	if err != nil {
		t.Fatal(err)
	}
	defer cd.Close()
	in := testInstance(t, m, 32, append(append([]Option(nil), openOpts...), WithCPU(cd))...)
	defer in.Close()
	prompt := m.Vocab.Encode("The capital of France is", true)
	round := func() []int32 {
		in.Reset()
		sp, err := in.Speculate(WithSpecDraft(2))
		if err != nil {
			t.Fatal(err)
		}
		defer sp.Close()
		sm := &Sampler{Temp: 1, TopK: 6, Seed: 7}
		y, err := sp.Start(prompt, sm)
		if err != nil {
			t.Fatal(err)
		}
		ys, err := sp.Next(sm)
		if err != nil {
			t.Fatal(err)
		}
		return append([]int32{y}, ys...)
	}
	first := round()
	c0 := in.Account(nn.CPU).Asks
	for i := range 2 {
		got := round()
		if n := in.Account(nn.CPU).Asks - c0; n != 0 {
			t.Fatalf("speculation %d asked the composer %d times", i+2, n)
		}
		if len(got) != len(first) {
			t.Fatalf("speculation %d drew %v, the first %v", i+2, got, first)
		}
		for j := range got {
			if got[j] != first[j] {
				t.Fatalf("speculation %d drew %v, the first %v", i+2, got, first)
			}
		}
	}
	t.Logf("three speculations, the draft composed once; each drew %v", first)
}
